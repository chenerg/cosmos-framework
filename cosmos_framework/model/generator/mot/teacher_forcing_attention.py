# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Teacher-forcing GEN attention: block-span gather plus a dense-mask oracle."""

from collections import defaultdict

import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingBlockAttentionGroup,
    build_teacher_forcing_packed_kv_metadata,
)
from cosmos_framework.model.attention import attention

# Packed TND duplicates overlapping UND/history KV. Chunk so one call stays
# near 1 GiB of bf16 K+V (262144 tokens * 8 heads * 128 dim * 2 bytes * 2).
_PACKED_KV_TOKEN_BUDGET = 262144
_MAX_TND_SEGMENTS = 1024


def teacher_forcing_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor,
    *,
    scale: float | None = None,
    mask_is_prevalidated: bool = False,
) -> torch.Tensor:
    """Attend GEN queries once over unified UND/clean/noisy keys.

    ``allowed_mask[q, k] == True`` means key ``k`` is visible to query ``q``.
    The implementation uses one SDPA softmax and never exposes or merges LSE.
    """

    output = attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        backend="masked_sdpa",
        backend_kwargs={
            "allowed_mask": allowed_mask,
            "validate_allowed_mask": not mask_is_prevalidated,
        },
        scale=scale,
    )
    return output.squeeze(0)


def teacher_forcing_per_sample_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_masks: tuple[torch.Tensor, ...],
    *,
    sample_lens: tuple[int, ...],
    gen_sample_lens: tuple[int, ...],
    scale: float | None = None,
    masks_are_prevalidated: bool = False,
) -> torch.Tensor:
    """Run Scheme-B dense attention independently for each packed sample."""

    num_samples = len(sample_lens)
    if num_samples == 0 or len(gen_sample_lens) != num_samples or len(allowed_masks) != num_samples:
        raise ValueError("per-sample teacher-forcing metadata must contain one entry per packed sample")
    if sum(sample_lens) != key.shape[0] or value.shape[0] != key.shape[0]:
        raise ValueError("per-sample KV lengths must cover the complete packed key/value sequence")
    if sum(gen_sample_lens) != query.shape[0]:
        raise ValueError("per-sample GEN lengths must cover the complete packed query sequence")

    outputs: list[torch.Tensor] = []
    query_offset = 0
    kv_offset = 0
    for sample_len, gen_len, allowed_mask in zip(sample_lens, gen_sample_lens, allowed_masks, strict=True):
        query_end = query_offset + gen_len
        kv_end = kv_offset + sample_len
        outputs.append(
            teacher_forcing_dense_attention(
                query[query_offset:query_end],
                key[kv_offset:kv_end],
                value[kv_offset:kv_end],
                allowed_mask,
                scale=scale,
                mask_is_prevalidated=masks_are_prevalidated,
            )
        )
        query_offset = query_end
        kv_offset = kv_end
    return torch.cat(outputs, dim=0)


def _expand_gqa(key: torch.Tensor, value: torch.Tensor, num_query_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = num_query_heads // key.shape[-3]
    if repeats == 1:
        return key, value
    return key.repeat_interleave(repeats, dim=-3), value.repeat_interleave(repeats, dim=-3)


def _unmasked_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None,
) -> torch.Tensor:
    """Run one unmasked SDPA call. ``query``/``key``/``value`` are ``[B, S, H, D]``."""

    query_heads = query.transpose(1, 2)
    key_heads, value_heads = _expand_gqa(key.transpose(1, 2), value.transpose(1, 2), query_heads.shape[1])
    output = torch.nn.functional.scaled_dot_product_attention(
        query_heads,
        key_heads,
        value_heads,
        dropout_p=0.0,
        scale=scale,
    )
    return output.transpose(1, 2)


def _gather_kv_slices(
    tensor: torch.Tensor,
    kv_slices: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    parts = [tensor[start:end] for start, end in kv_slices if start < end]
    if not parts:
        raise ValueError("teacher-forcing block attention requires at least one visible key")
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def _validate_block_attention_groups(
    groups: tuple[TeacherForcingBlockAttentionGroup, ...],
    *,
    num_queries: int,
    num_keys: int,
) -> None:
    if num_queries < 0 or num_keys < 0:
        raise ValueError("query/key lengths must be non-negative")
    if not groups:
        if num_queries != 0:
            raise ValueError("teacher-forcing block groups are empty but GEN queries are not")
        return

    query_ranges = sorted((group.query_start, group.query_end) for group in groups)
    cursor = 0
    for start, end in query_ranges:
        if start != cursor or end <= start:
            raise ValueError("teacher-forcing block groups must cover every GEN query without overlap")
        cursor = end
    if cursor != num_queries:
        raise ValueError(f"teacher-forcing block groups cover {cursor} GEN queries, expected {num_queries}")

    for group in groups:
        if not group.kv_slices:
            raise ValueError("teacher-forcing block groups must contain at least one KV slice")
        for start, end in group.kv_slices:
            if start < 0 or end > num_keys or start > end:
                raise ValueError(
                    f"teacher-forcing KV slice [{start}, {end}) is outside the packed key length {num_keys}"
                )


def _use_packed_varlen_attention(query: torch.Tensor) -> bool:
    return query.device.type in {"npu", "cuda"} and query.dtype in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }


def _chunk_group_bounds(
    cumulative_seqlen_q: torch.Tensor,
    cumulative_seqlen_kv: torch.Tensor,
) -> list[tuple[int, int]]:
    """Split packed groups so each TND call stays within segment and KV budgets."""

    cu_q = cumulative_seqlen_q.tolist()
    cu_kv = cumulative_seqlen_kv.tolist()
    num_groups = len(cu_q) - 1
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < num_groups:
        end = start
        while end < num_groups:
            next_end = end + 1
            kv_tokens = cu_kv[next_end] - cu_kv[start]
            if end > start and (next_end - start > _MAX_TND_SEGMENTS or kv_tokens > _PACKED_KV_TOKEN_BUDGET):
                break
            end = next_end
        if end == start:
            end = start + 1
        bounds.append((start, end))
        start = end
    return bounds


def _packed_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    kv_gather_index: torch.Tensor,
    cumulative_seqlen_q: torch.Tensor,
    cumulative_seqlen_kv: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    """Run unmasked varlen attention over pre-gathered block KV segments."""

    device = query.device
    kv_gather_index = kv_gather_index.to(device=device)
    cumulative_seqlen_q = cumulative_seqlen_q.to(device=device, dtype=torch.int32)
    cumulative_seqlen_kv = cumulative_seqlen_kv.to(device=device, dtype=torch.int32)
    gathered_key = key.index_select(0, kv_gather_index)
    gathered_value = value.index_select(0, kv_gather_index)
    outputs: list[torch.Tensor] = []
    bounds = _chunk_group_bounds(cumulative_seqlen_q, cumulative_seqlen_kv)
    cu_q_host = cumulative_seqlen_q.tolist()
    cu_kv_host = cumulative_seqlen_kv.tolist()
    for group_start, group_end in bounds:
        query_start = cu_q_host[group_start]
        query_end = cu_q_host[group_end]
        kv_start = cu_kv_host[group_start]
        kv_end = cu_kv_host[group_end]
        cu_q = cumulative_seqlen_q[group_start : group_end + 1] - query_start
        cu_kv = cumulative_seqlen_kv[group_start : group_end + 1] - kv_start
        max_seqlen_q = max(cu_q_host[index + 1] - cu_q_host[index] for index in range(group_start, group_end))
        max_seqlen_kv = max(cu_kv_host[index + 1] - cu_kv_host[index] for index in range(group_start, group_end))
        chunk = attention(
            query[query_start:query_end].unsqueeze(0),
            gathered_key[kv_start:kv_end].unsqueeze(0),
            gathered_value[kv_start:kv_end].unsqueeze(0),
            scale=scale,
            cumulative_seqlen_Q=cu_q,
            cumulative_seqlen_KV=cu_kv,
            max_seqlen_Q=max_seqlen_q,
            max_seqlen_KV=max_seqlen_kv,
        )
        outputs.append(chunk.squeeze(0))
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def _batched_unmasked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    groups: tuple[TeacherForcingBlockAttentionGroup, ...],
    *,
    scale: float | None,
) -> torch.Tensor:
    """CPU/fp64 path: stack equal-shaped groups into unmasked SDPA."""

    buckets: dict[tuple[int, int], list[TeacherForcingBlockAttentionGroup]] = defaultdict(list)
    for group in groups:
        query_len = group.query_end - group.query_start
        kv_len = sum(end - start for start, end in group.kv_slices)
        if kv_len < 1:
            raise ValueError("every teacher-forcing GEN query block must have at least one visible key")
        buckets[query_len, kv_len].append(group)

    output = query.new_empty(query.shape)
    for kv_len, bucket in ((kv_len, bucket) for (_, kv_len), bucket in buckets.items()):
        batch_limit = max(1, _PACKED_KV_TOKEN_BUDGET // kv_len)
        for chunk_start in range(0, len(bucket), batch_limit):
            chunk = bucket[chunk_start : chunk_start + batch_limit]
            query_batch = torch.stack(
                [query[group.query_start : group.query_end] for group in chunk],
                dim=0,
            )
            key_batch = torch.stack([_gather_kv_slices(key, group.kv_slices) for group in chunk], dim=0)
            value_batch = torch.stack([_gather_kv_slices(value, group.kv_slices) for group in chunk], dim=0)
            chunk_output = _unmasked_sdpa(query_batch, key_batch, value_batch, scale=scale)
            for group_index, group in enumerate(chunk):
                output[group.query_start : group.query_end] = chunk_output[group_index]
    return output


def teacher_forcing_block_gather_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    groups: tuple[TeacherForcingBlockAttentionGroup, ...],
    *,
    scale: float | None = None,
    kv_gather_index: torch.Tensor | None = None,
    cumulative_seqlen_q: torch.Tensor | None = None,
    cumulative_seqlen_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend each GEN block to 2-3 contiguous visible KV slices without a dense mask.

    On NPU/CUDA this packs every block into one or a few unmasked varlen
    attention calls (TND ``npu_fusion_attention`` on Ascend). CPU/fp64 tests
    keep a batched SDPA path that matches the dense-mask oracle.
    """

    if key.shape[0] != value.shape[0]:
        raise ValueError("teacher-forcing block attention requires key and value to share the sequence length")
    _validate_block_attention_groups(groups, num_queries=query.shape[0], num_keys=key.shape[0])
    if query.shape[0] == 0:
        return query.new_zeros(query.shape)

    if not _use_packed_varlen_attention(query):
        return _batched_unmasked_attention(query, key, value, groups, scale=scale)

    if kv_gather_index is None or cumulative_seqlen_q is None or cumulative_seqlen_kv is None:
        kv_gather_index, cumulative_seqlen_q, cumulative_seqlen_kv = build_teacher_forcing_packed_kv_metadata(groups)
    return _packed_varlen_attention(
        query,
        key,
        value,
        kv_gather_index=kv_gather_index,
        cumulative_seqlen_q=cumulative_seqlen_q,
        cumulative_seqlen_kv=cumulative_seqlen_kv,
        scale=scale,
    )
