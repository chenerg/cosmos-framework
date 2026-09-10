# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Layout metadata helpers for teacher-forcing causal video training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence


class TeacherForcingStream(IntEnum):
    """Token stream identifiers used by teacher-forcing attention."""

    UND = -1
    CLEAN = 0
    NOISY = 1


def _empty_long_tensor() -> torch.LongTensor:
    return torch.empty(0, dtype=torch.long)  # type: ignore[return-value]


@dataclass(frozen=True)
class TeacherForcingGeometry:
    """Per-sample block geometry shared by noise sampling and attention."""

    block_sizes: tuple[int, ...]
    history_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.block_sizes:
            raise ValueError("teacher-forcing geometry cannot be empty")
        if len(self.block_sizes) != len(self.history_blocks):
            raise ValueError("teacher-forcing geometry must contain one block_size and history_blocks value per sample")
        if any(block_size < 1 for block_size in self.block_sizes):
            raise ValueError(f"teacher-forcing block_size values must be >= 1, got {self.block_sizes}")
        if any(history < 1 for history in self.history_blocks):
            raise ValueError(f"teacher-forcing history_blocks values must be >= 1, got {self.history_blocks}")


def shared_teacher_forcing_geometry(num_samples: int, block_size: int, history_blocks: int) -> TeacherForcingGeometry:
    """Repeat one S/K pair across every packed sample."""

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    return TeacherForcingGeometry(
        block_sizes=(block_size,) * num_samples,
        history_blocks=(history_blocks,) * num_samples,
    )


@dataclass(frozen=True)
class TeacherForcingBlockAttentionGroup:
    """One GEN query block and the packed KV slices visible to it.

    ``query_start``/``query_end`` index the concatenated GEN (full-mode) tokens.
    ``kv_slices`` index the packed dual-stream sequence and are concatenated in
    packed order so the gathered keys match the dense-mask visible set.
    """

    query_start: int
    query_end: int
    kv_slices: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TeacherForcingLayout:
    """Immutable geometry shared by packing, attention, and output recovery."""

    geometry: TeacherForcingGeometry
    original_sample_lens: tuple[int, ...]
    sample_lens: tuple[int, ...]
    split_lens: tuple[int, ...]
    attn_modes: tuple[str, ...]
    source_sequence_indexes: torch.LongTensor
    sample_ids: torch.LongTensor
    stream_ids: torch.LongTensor
    block_ids: torch.LongTensor
    gen_query_indexes: torch.LongTensor
    clean_token_indexes: torch.LongTensor
    noisy_output_indexes: torch.LongTensor
    # Action-stream metadata (empty for vision-only layouts). Vision clean slots
    # stay in ``clean_token_indexes``; action clean slots live here so the
    # network can scatter each modality's clean payload independently.
    clean_action_token_indexes: torch.LongTensor = field(default_factory=_empty_long_tensor)
    noisy_action_output_indexes: torch.LongTensor = field(default_factory=_empty_long_tensor)
    # Per-sample GEN-local ``(start, end)`` spans, indexed by causal block id.
    # Clean and noisy streams share these offsets inside each sample.
    block_token_spans: tuple[tuple[tuple[int, int], ...], ...] = ()

    def to(self, device: torch.device | str) -> TeacherForcingLayout:
        """Return a copy with all tensor metadata moved to ``device``."""

        return replace(
            self,
            source_sequence_indexes=self.source_sequence_indexes.to(device=device),
            sample_ids=self.sample_ids.to(device=device),
            stream_ids=self.stream_ids.to(device=device),
            block_ids=self.block_ids.to(device=device),
            gen_query_indexes=self.gen_query_indexes.to(device=device),
            clean_token_indexes=self.clean_token_indexes.to(device=device),
            noisy_output_indexes=self.noisy_output_indexes.to(device=device),
            clean_action_token_indexes=self.clean_action_token_indexes.to(device=device),
            noisy_action_output_indexes=self.noisy_action_output_indexes.to(device=device),
        )


@dataclass
class TeacherForcingData:
    """Runtime data attached to a dual-stream ``PackedSequence``."""

    layout: TeacherForcingLayout
    clean_vision_tokens: list[torch.Tensor]
    # Clean action payloads (one per packed sample) for V+A causal training.
    # None for vision-only teacher forcing.
    clean_action_tokens: list[torch.Tensor] | None = None

    def to_cuda(self) -> None:
        """Move clean payloads and layout tensors to CUDA/NPU in-place."""

        self.layout = self.layout.to("cuda")
        self.clean_vision_tokens = [token.cuda() for token in self.clean_vision_tokens]
        if self.clean_action_tokens is not None:
            self.clean_action_tokens = [token.cuda() for token in self.clean_action_tokens]


def _validate_inclusive_range(name: str, minimum: int, maximum: int) -> None:
    if minimum < 1:
        raise ValueError(f"{name}_min must be >= 1, got {minimum}")
    if maximum < 1:
        raise ValueError(f"{name}_max must be >= 1, got {maximum}")
    if minimum > maximum:
        raise ValueError(f"{name} range must satisfy min <= max, got min={minimum}, max={maximum}")


def sample_teacher_forcing_parameters(
    *,
    block_size_min: int = 1,
    block_size_max: int = 4,
    history_blocks_min: int = 1,
    history_blocks_max: int = 32,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Sample one block size and history window shared by the whole forward."""

    _validate_inclusive_range("block_size", block_size_min, block_size_max)
    _validate_inclusive_range("history_blocks", history_blocks_min, history_blocks_max)

    block_size = int(torch.randint(block_size_min, block_size_max + 1, (1,), generator=generator, device="cpu").item())
    history_blocks = int(
        torch.randint(history_blocks_min, history_blocks_max + 1, (1,), generator=generator, device="cpu").item()
    )
    return block_size, history_blocks


def sample_teacher_forcing_geometry(
    *,
    num_samples: int,
    block_size_min: int = 1,
    block_size_max: int = 4,
    history_blocks_min: int = 1,
    history_blocks_max: int = 32,
    generator: torch.Generator | None = None,
) -> TeacherForcingGeometry:
    """Independently sample block geometry for every packed sample."""

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    _validate_inclusive_range("block_size", block_size_min, block_size_max)
    _validate_inclusive_range("history_blocks", history_blocks_min, history_blocks_max)
    block_sizes = torch.randint(
        block_size_min,
        block_size_max + 1,
        (num_samples,),
        generator=generator,
        device="cpu",
    )
    history_blocks = torch.randint(
        history_blocks_min,
        history_blocks_max + 1,
        (num_samples,),
        generator=generator,
        device="cpu",
    )
    return TeacherForcingGeometry(
        block_sizes=tuple(int(value) for value in block_sizes.tolist()),
        history_blocks=tuple(int(value) for value in history_blocks.tolist()),
    )


def build_teacher_forcing_frame_block_ids(num_frames: int, block_size: int) -> torch.LongTensor:
    """Assign the first latent to a singleton block and chunk the remaining latents."""

    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    block_ids = torch.zeros(num_frames, dtype=torch.long)
    if num_frames > 1:
        block_ids[1:] = 1 + torch.div(
            torch.arange(num_frames - 1, dtype=torch.long),
            block_size,
            rounding_mode="floor",
        )
    return block_ids  # type: ignore[return-value]


def map_action_sigmas_from_vision_schedule(
    vision_sigmas: torch.Tensor,
    *,
    action_sample_indices: Sequence[int],
    action_lengths: Sequence[int],
    num_vision_latent_frames: Sequence[int],
    temporal_compression_factor: int,
) -> torch.Tensor:
    """Expand per-frame vision σ onto each action step via latent-frame alignment.

    Returns a dense tensor of shape ``[n_action, T_action_max]``. Unused tail
    entries for shorter action sequences are zero.
    """

    if len(action_sample_indices) != len(action_lengths):
        raise ValueError(
            "action_sample_indices and action_lengths must contain one entry per action sample, "
            f"got {len(action_sample_indices)} and {len(action_lengths)}"
        )
    if not action_sample_indices:
        return vision_sigmas.new_zeros((0, 0))
    t_action_max = max(action_lengths)
    mapped = vision_sigmas.new_zeros((len(action_sample_indices), t_action_max))
    for dense_index, (batch_index, action_len) in enumerate(zip(action_sample_indices, action_lengths, strict=True)):
        t_vis = num_vision_latent_frames[batch_index]
        latent_frames = assign_action_steps_to_latent_frames(action_len, t_vis, temporal_compression_factor)
        mapped[dense_index, :action_len] = vision_sigmas[batch_index, latent_frames]
    return mapped


def assign_action_steps_to_latent_frames(
    num_action_steps: int,
    num_latent_frames: int,
    temporal_compression_factor: int,
) -> list[int]:
    """Map each action step to its VAE latent frame under offset-0 alignment.

    Action step ``j`` sits at raw frame ``j`` (``action_start_frame_offset=0``).
    The causal Wan VAE maps raw frame 0 to latent frame 0 and raw frames
    ``cf*(k-1)+1 .. cf*k`` to latent frame ``k``, so
    ``latent_frame(j) = ceil(j / cf)``.
    """

    if temporal_compression_factor < 1:
        raise ValueError(f"temporal_compression_factor must be >= 1, got {temporal_compression_factor}")
    latent_frames = [
        (step + temporal_compression_factor - 1) // temporal_compression_factor for step in range(num_action_steps)
    ]
    if latent_frames and latent_frames[-1] > num_latent_frames - 1:
        raise ValueError(
            f"action steps extend beyond the video: step {num_action_steps - 1} maps to latent frame "
            f"{latent_frames[-1]} but the video only has {num_latent_frames} latent frames"
        )
    return latent_frames


def build_teacher_forcing_layout(
    *,
    und_token_counts: Sequence[int],
    vision_token_shapes: Sequence[tuple[int, int, int]],
    geometry: TeacherForcingGeometry | None = None,
    block_size: int | None = None,
    history_blocks: int | None = None,
    action_token_counts: Sequence[int] | None = None,
    temporal_compression_factor: int | None = None,
) -> TeacherForcingLayout:
    """Build batch metadata for dual-stream teacher-forcing samples.

    Vision-only samples expand to ``[UND | clean vision | noisy vision]``.
    When ``action_token_counts`` is provided, each GEN stream is interleaved
    per causal block: ``[UND | clean: V_b0 A_b0 V_b1 A_b1 ... | noisy: ...]``.
    Action steps are assigned to blocks through their physical VAE latent
    frame (``assign_action_steps_to_latent_frames``), assuming
    ``action_start_frame_offset=0``. Latent frame 0 is a singleton block;
    remaining frames are chunked with each sample's ``block_size``.
    """

    if len(und_token_counts) != len(vision_token_shapes):
        raise ValueError(
            "und_token_counts and vision_token_shapes must contain the same number of samples, "
            f"got {len(und_token_counts)} and {len(vision_token_shapes)}"
        )
    if not und_token_counts:
        raise ValueError("teacher-forcing layout cannot be built from an empty batch")
    if geometry is None:
        if block_size is None or history_blocks is None:
            raise ValueError("build_teacher_forcing_layout requires geometry or both block_size and history_blocks")
        geometry = shared_teacher_forcing_geometry(len(und_token_counts), block_size, history_blocks)
    elif len(geometry.block_sizes) != len(und_token_counts):
        raise ValueError(
            "teacher-forcing geometry must contain one entry per packed sample, "
            f"got {len(geometry.block_sizes)} and {len(und_token_counts)}"
        )
    if action_token_counts is not None:
        if len(action_token_counts) != len(und_token_counts):
            raise ValueError(
                "action_token_counts must contain one entry per sample, "
                f"got {len(action_token_counts)} and {len(und_token_counts)}"
            )
        if temporal_compression_factor is None:
            raise ValueError("temporal_compression_factor is required when action_token_counts is provided")

    original_sample_lens: list[int] = []
    sample_lens: list[int] = []
    split_lens: list[int] = []
    attn_modes: list[str] = []
    source_sequence_indexes: list[int] = []
    sample_ids: list[int] = []
    stream_ids: list[int] = []
    block_ids: list[int] = []
    gen_query_indexes: list[int] = []
    clean_token_indexes: list[int] = []
    clean_action_token_indexes: list[int] = []
    noisy_output_indexes: list[int] = []
    noisy_action_output_indexes: list[int] = []
    block_token_spans: list[tuple[tuple[int, int], ...]] = []

    original_offset = 0
    new_offset = 0
    for sample_id, (und_count, vision_shape, sample_block_size) in enumerate(
        zip(und_token_counts, vision_token_shapes, geometry.block_sizes, strict=True)
    ):
        if und_count < 1:
            raise ValueError(f"und_token_counts[{sample_id}] must be >= 1, got {und_count}")
        if len(vision_shape) != 3:
            raise ValueError(f"vision_token_shapes[{sample_id}] must contain exactly (T, H, W), got {vision_shape}")
        num_frames, height, width = vision_shape
        if num_frames < 1 or height < 1 or width < 1:
            raise ValueError(f"vision_token_shapes[{sample_id}] must be positive, got {vision_shape}")

        action_count = 0
        if action_token_counts is not None:
            action_count = action_token_counts[sample_id]
            if action_count < 0:
                raise ValueError(f"action_token_counts[{sample_id}] must be >= 0, got {action_count}")

        spatial_tokens = height * width
        vision_count = num_frames * spatial_tokens
        gen_len = vision_count + action_count
        original_sample_len = und_count + gen_len
        new_sample_len = und_count + 2 * gen_len

        und_source = list(range(original_offset, original_offset + und_count))
        vision_source_start = original_offset + und_count
        action_source_start = vision_source_start + vision_count

        vision_frame_block_ids = build_teacher_forcing_frame_block_ids(num_frames, sample_block_size)
        num_blocks = int(vision_frame_block_ids[-1].item()) + 1
        action_block_of_step: list[int] = []
        if action_count > 0:
            assert temporal_compression_factor is not None
            action_latent_frames = assign_action_steps_to_latent_frames(
                action_count, num_frames, temporal_compression_factor
            )
            action_block_of_step = vision_frame_block_ids[torch.tensor(action_latent_frames, dtype=torch.long)].tolist()

        # Interleave the GEN stream per causal block: vision frames of block b
        # (in frame order) followed by the action steps of block b (in step
        # order). With no action this degenerates to the plain frame order.
        gen_source: list[int] = []
        gen_block_ids: list[int] = []
        vision_positions_in_gen: list[int] = []  # frame-major token order
        action_positions_in_gen: list[int] = []  # step order
        sample_block_spans: list[tuple[int, int] | None] = [None] * num_blocks
        next_action_step = 0
        vision_frame_block_ids_list = vision_frame_block_ids.tolist()
        for block_id in range(num_blocks):
            frame_indexes = [frame_idx for frame_idx, bid in enumerate(vision_frame_block_ids_list) if bid == block_id]
            if not frame_indexes:
                continue
            block_gen_start = len(gen_source)
            frame_lo = frame_indexes[0]
            frame_hi = frame_indexes[-1] + 1
            num_block_vision = (frame_hi - frame_lo) * spatial_tokens
            vision_positions_in_gen.extend(range(len(gen_source), len(gen_source) + num_block_vision))
            gen_source.extend(
                range(vision_source_start + frame_lo * spatial_tokens, vision_source_start + frame_hi * spatial_tokens)
            )
            gen_block_ids.extend([block_id] * num_block_vision)
            while next_action_step < action_count and action_block_of_step[next_action_step] == block_id:
                action_positions_in_gen.append(len(gen_source))
                gen_source.append(action_source_start + next_action_step)
                gen_block_ids.append(block_id)
                next_action_step += 1
            sample_block_spans[block_id] = (block_gen_start, len(gen_source))
        if next_action_step != action_count:
            raise ValueError(
                f"sample {sample_id}: {action_count - next_action_step} action steps were not assigned to any block"
            )
        if any(span is None or span[0] >= span[1] for span in sample_block_spans):
            raise ValueError(f"sample {sample_id}: every causal block must contain at least one GEN token")
        filled_block_spans = tuple(span for span in sample_block_spans if span is not None)

        clean_start = new_offset + und_count
        noisy_start = clean_start + gen_len
        new_sample_end = new_offset + new_sample_len

        original_sample_lens.append(original_sample_len)
        sample_lens.append(new_sample_len)
        split_lens.extend((und_count, 2 * gen_len))
        attn_modes.extend(("causal", "full"))
        source_sequence_indexes.extend(und_source + gen_source + gen_source)
        sample_ids.extend([sample_id] * new_sample_len)
        stream_ids.extend(
            [int(TeacherForcingStream.UND)] * und_count
            + [int(TeacherForcingStream.CLEAN)] * gen_len
            + [int(TeacherForcingStream.NOISY)] * gen_len
        )
        block_ids.extend([-1] * und_count + gen_block_ids + gen_block_ids)
        gen_query_indexes.extend(range(clean_start, new_sample_end))
        clean_token_indexes.extend(clean_start + position for position in vision_positions_in_gen)
        clean_action_token_indexes.extend(clean_start + position for position in action_positions_in_gen)
        # Noisy outputs stay in the original packed order: vision (frame order)
        # then action (step order) per sample.
        noisy_output_indexes.extend(noisy_start + position for position in vision_positions_in_gen)
        noisy_output_indexes.extend(noisy_start + position for position in action_positions_in_gen)
        noisy_action_output_indexes.extend(noisy_start + position for position in action_positions_in_gen)
        block_token_spans.append(filled_block_spans)

        original_offset += original_sample_len
        new_offset = new_sample_end

    return TeacherForcingLayout(
        geometry=geometry,
        original_sample_lens=tuple(original_sample_lens),
        sample_lens=tuple(sample_lens),
        split_lens=tuple(split_lens),
        attn_modes=tuple(attn_modes),
        source_sequence_indexes=torch.tensor(source_sequence_indexes, dtype=torch.long),
        sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        stream_ids=torch.tensor(stream_ids, dtype=torch.long),
        block_ids=torch.tensor(block_ids, dtype=torch.long),
        gen_query_indexes=torch.tensor(gen_query_indexes, dtype=torch.long),
        clean_token_indexes=torch.tensor(clean_token_indexes, dtype=torch.long),
        noisy_output_indexes=torch.tensor(noisy_output_indexes, dtype=torch.long),
        clean_action_token_indexes=torch.tensor(clean_action_token_indexes, dtype=torch.long),
        noisy_action_output_indexes=torch.tensor(noisy_action_output_indexes, dtype=torch.long),
        block_token_spans=tuple(block_token_spans),
    )


def build_dense_teacher_forcing_gen_mask(
    layout: TeacherForcingLayout,
) -> torch.BoolTensor:
    """Build the reference GEN-query mask over all dual-stream KV tokens."""

    query_indexes = layout.gen_query_indexes[:, None]
    query_sample_ids = layout.sample_ids[query_indexes]
    query_stream_ids = layout.stream_ids[query_indexes]
    query_block_ids = layout.block_ids[query_indexes]

    is_gen_query = (query_stream_ids == int(TeacherForcingStream.CLEAN)) | (
        query_stream_ids == int(TeacherForcingStream.NOISY)
    )
    if not bool(is_gen_query.all()):
        raise ValueError("gen_query_indexes must contain only CLEAN or NOISY GEN queries")

    key_sample_ids = layout.sample_ids[None, :]
    key_stream_ids = layout.stream_ids[None, :]
    key_block_ids = layout.block_ids[None, :]

    same_sample = query_sample_ids == key_sample_ids
    key_is_und = key_stream_ids == int(TeacherForcingStream.UND)
    key_is_clean = key_stream_ids == int(TeacherForcingStream.CLEAN)
    key_is_noisy = key_stream_ids == int(TeacherForcingStream.NOISY)

    sample_history_blocks = torch.tensor(
        layout.geometry.history_blocks,
        dtype=layout.block_ids.dtype,
        device=layout.block_ids.device,
    )
    query_history_blocks = sample_history_blocks[query_sample_ids]
    inside_history = key_block_ids >= query_block_ids - query_history_blocks
    clean_query_visible = key_is_clean & inside_history & (key_block_ids <= query_block_ids)
    noisy_query_visible = (key_is_clean & inside_history & (key_block_ids < query_block_ids)) | (
        key_is_noisy & (key_block_ids == query_block_ids)
    )

    allowed_by_stream = torch.where(
        query_stream_ids == int(TeacherForcingStream.CLEAN),
        clean_query_visible,
        noisy_query_visible,
    )
    return same_sample & (key_is_und | allowed_by_stream)


def build_per_sample_teacher_forcing_gen_masks(
    layout: TeacherForcingLayout,
) -> tuple[torch.BoolTensor, ...]:
    """Build one GEN-query dense mask per packed sample without a global 2D allocation."""

    masks: list[torch.BoolTensor] = []
    sample_offset = 0
    for sample_len, history_blocks in zip(layout.sample_lens, layout.geometry.history_blocks, strict=True):
        sample_slice = slice(sample_offset, sample_offset + sample_len)
        sample_stream_ids = layout.stream_ids[sample_slice]
        sample_block_ids = layout.block_ids[sample_slice]
        query_rows = (sample_stream_ids == int(TeacherForcingStream.CLEAN)) | (
            sample_stream_ids == int(TeacherForcingStream.NOISY)
        )
        query_stream_ids = sample_stream_ids[query_rows, None]
        query_block_ids = sample_block_ids[query_rows, None]
        key_stream_ids = sample_stream_ids[None, :]
        key_block_ids = sample_block_ids[None, :]

        key_is_und = key_stream_ids == int(TeacherForcingStream.UND)
        key_is_clean = key_stream_ids == int(TeacherForcingStream.CLEAN)
        key_is_noisy = key_stream_ids == int(TeacherForcingStream.NOISY)
        inside_history = key_block_ids >= query_block_ids - history_blocks
        clean_query_visible = key_is_clean & inside_history & (key_block_ids <= query_block_ids)
        noisy_query_visible = (key_is_clean & inside_history & (key_block_ids < query_block_ids)) | (
            key_is_noisy & (key_block_ids == query_block_ids)
        )
        allowed_by_stream = torch.where(
            query_stream_ids == int(TeacherForcingStream.CLEAN),
            clean_query_visible,
            noisy_query_visible,
        )
        mask = key_is_und | allowed_by_stream
        masks.append(mask)
        sample_offset += sample_len

    if sample_offset != layout.source_sequence_indexes.numel():
        raise ValueError("teacher-forcing sample lengths do not cover the complete packed sequence")
    return tuple(masks)


def build_teacher_forcing_block_attention_groups(
    layout: TeacherForcingLayout,
) -> tuple[TeacherForcingBlockAttentionGroup, ...]:
    """Build per-block GEN query groups with 2-3 contiguous visible KV slices.

    CLEAN block ``i`` sees UND plus CLEAN blocks ``[lo, i]``.
    NOISY block ``i`` sees UND, CLEAN blocks ``[lo, i)``, and NOISY block ``i``.
    ``lo = max(0, i - history_blocks)``. Noisy never sees the current clean block.
    """

    if len(layout.block_token_spans) != len(layout.sample_lens):
        raise ValueError(
            "teacher-forcing layout must contain one block-span list per packed sample, "
            f"got {len(layout.block_token_spans)} and {len(layout.sample_lens)}"
        )

    groups: list[TeacherForcingBlockAttentionGroup] = []
    packed_offset = 0
    gen_query_offset = 0
    for sample_id, (sample_len, und_count, history_blocks, spans) in enumerate(
        zip(
            layout.sample_lens,
            layout.split_lens[::2],
            layout.geometry.history_blocks,
            layout.block_token_spans,
            strict=True,
        )
    ):
        gen_split = layout.split_lens[2 * sample_id + 1]
        if gen_split % 2 != 0:
            raise ValueError(f"sample {sample_id}: GEN split length must be even (clean+noisy), got {gen_split}")
        gen_len = gen_split // 2
        if spans and spans[-1][1] != gen_len:
            raise ValueError(f"sample {sample_id}: block spans cover {spans[-1][1]} GEN tokens, expected {gen_len}")

        und_start = packed_offset
        und_end = packed_offset + und_count
        clean_start = und_end
        noisy_start = clean_start + gen_len
        und_slice = (und_start, und_end)
        # Emit all CLEAN groups then all NOISY groups so the concatenated GEN
        # query order is a contiguous partition of ``full_q``.
        sample_clean_groups: list[TeacherForcingBlockAttentionGroup] = []
        sample_noisy_groups: list[TeacherForcingBlockAttentionGroup] = []

        for block_id, (gen_start, gen_end) in enumerate(spans):
            if gen_start >= gen_end:
                raise ValueError(f"sample {sample_id} block {block_id} has an empty GEN span")
            lo = max(0, block_id - history_blocks)
            history_start = spans[lo][0]
            sample_clean_groups.append(
                TeacherForcingBlockAttentionGroup(
                    query_start=gen_query_offset + gen_start,
                    query_end=gen_query_offset + gen_end,
                    kv_slices=(und_slice, (clean_start + history_start, clean_start + gen_end)),
                )
            )
            noisy_kv: list[tuple[int, int]] = [und_slice]
            if history_start < gen_start:
                noisy_kv.append((clean_start + history_start, clean_start + gen_start))
            noisy_kv.append((noisy_start + gen_start, noisy_start + gen_end))
            sample_noisy_groups.append(
                TeacherForcingBlockAttentionGroup(
                    query_start=gen_query_offset + gen_len + gen_start,
                    query_end=gen_query_offset + gen_len + gen_end,
                    kv_slices=tuple(noisy_kv),
                )
            )
        groups.extend(sample_clean_groups)
        groups.extend(sample_noisy_groups)

        packed_offset += sample_len
        gen_query_offset += gen_split

    if packed_offset != layout.source_sequence_indexes.numel():
        raise ValueError("teacher-forcing sample lengths do not cover the complete packed sequence")
    return tuple(groups)


def build_teacher_forcing_packed_kv_metadata(
    groups: tuple[TeacherForcingBlockAttentionGroup, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build TND gather index and cumulative lengths for packed unmasked attention.

    Query segments follow ``query_start`` order, so packed queries are a view of
    the GEN tensor. Key/value tokens are gathered with ``kv_gather_index``.
    """

    kv_index: list[int] = []
    cu_q = [0]
    cu_kv = [0]
    for group in groups:
        cu_q.append(cu_q[-1] + (group.query_end - group.query_start))
        for start, end in group.kv_slices:
            kv_index.extend(range(start, end))
        cu_kv.append(len(kv_index))
    if cu_q[-1] == 0:
        raise ValueError("teacher-forcing packed KV metadata requires at least one GEN query")
    return (
        torch.tensor(kv_index, dtype=torch.long),
        torch.tensor(cu_q, dtype=torch.int32),
        torch.tensor(cu_kv, dtype=torch.int32),
    )


def visualize_dense_teacher_forcing_gen_mask(
    dense_gen_mask: torch.BoolTensor,
    layout: TeacherForcingLayout,
    output_path: str | Path,
    *,
    und_block_size: int = 256,
) -> Path:
    """Save a compact view of the complete teacher-forcing attention pattern.

    UND tokens are grouped into blocks of ``und_block_size`` for a compact
    causal lower-triangular overview. CLEAN/NOISY tokens are grouped by their
    causal blocks because all tokens in such a block have identical
    teacher-forcing visibility; within a block, vision and action tokens are
    kept as separate groups (modality recovered from
    ``clean_action_token_indexes`` / ``noisy_action_output_indexes``) so both
    modalities stay visible in the plot. The GEN rows come from
    ``dense_gen_mask``; the UND rows summarize the separate causal
    self-attention call.
    """

    if und_block_size < 1:
        raise ValueError(f"und_block_size must be >= 1, got {und_block_size}")
    if dense_gen_mask.dtype != torch.bool:
        raise TypeError(f"dense_gen_mask must be bool, got {dense_gen_mask.dtype}")
    expected_shape = (layout.gen_query_indexes.numel(), layout.source_sequence_indexes.numel())
    if tuple(dense_gen_mask.shape) != expected_shape:
        raise ValueError(f"dense_gen_mask shape must be {expected_shape}, got {tuple(dense_gen_mask.shape)}")

    # Recover per-token modality (0 = UND/vision, 1 = action) from the action
    # scatter indexes so action tokens interleaved in a causal block do not
    # collapse into the block's vision group.
    modality_ids = torch.zeros_like(layout.stream_ids)
    modality_ids[layout.clean_action_token_indexes] = 1
    modality_ids[layout.noisy_action_output_indexes] = 1

    def _group_starts(
        sample_ids: torch.Tensor,
        stream_ids: torch.Tensor,
        block_ids: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        starts = torch.ones(sample_ids.numel(), dtype=torch.bool, device=sample_ids.device)
        if sample_ids.numel() > 1:
            starts[1:] = (
                (sample_ids[1:] != sample_ids[:-1])
                | (stream_ids[1:] != stream_ids[:-1])
                | (block_ids[1:] != block_ids[:-1])
                | (modality_ids[1:] != modality_ids[:-1])
            )
        sample_offset = 0
        for sample_len, und_count in zip(layout.sample_lens, layout.split_lens[::2]):
            for block_start in range(und_block_size, und_count, und_block_size):
                starts[sample_offset + block_start] = True
            sample_offset += sample_len
        return torch.nonzero(starts, as_tuple=True)[0]

    gen_indexes = layout.gen_query_indexes
    representatives = _group_starts(layout.sample_ids, layout.stream_ids, layout.block_ids, modality_ids)
    representative_sample_ids = layout.sample_ids.index_select(0, representatives)
    representative_stream_ids = layout.stream_ids.index_select(0, representatives)
    representative_block_ids = layout.block_ids.index_select(0, representatives)
    representative_modality_ids = modality_ids.index_select(0, representatives)

    num_groups = representatives.numel()
    grouped_mask = torch.zeros((num_groups, num_groups), dtype=torch.bool, device=dense_gen_mask.device)
    und_rows = representative_stream_ids == int(TeacherForcingStream.UND)
    grouped_mask[und_rows] = (
        (representative_sample_ids[und_rows, None] == representative_sample_ids[None, :])
        & (representative_stream_ids[None, :] == int(TeacherForcingStream.UND))
        & (representatives[None, :] <= representatives[und_rows, None])
    )

    gen_rows = ~und_rows
    gen_representatives = representatives[gen_rows]
    gen_row_by_source = torch.full(
        (layout.source_sequence_indexes.numel(),),
        -1,
        dtype=torch.long,
        device=gen_indexes.device,
    )
    gen_row_by_source[gen_indexes] = torch.arange(gen_indexes.numel(), device=gen_indexes.device)
    gen_mask_rows = gen_row_by_source.index_select(0, gen_representatives)
    if bool((gen_mask_rows < 0).any()):
        raise ValueError("CLEAN/NOISY representatives must be present in gen_query_indexes")
    grouped_mask[gen_rows] = dense_gen_mask.index_select(0, gen_mask_rows).index_select(1, representatives)
    grouped_mask = grouped_mask.detach().to(device="cpu")

    query_sample_ids = representative_sample_ids.detach().cpu()
    query_stream_ids = representative_stream_ids.detach().cpu()
    query_block_ids = representative_block_ids.detach().cpu()
    query_modality_ids = representative_modality_ids.detach().cpu()
    key_representatives = representatives
    key_sample_ids = layout.sample_ids.index_select(0, key_representatives).detach().cpu()
    key_stream_ids = layout.stream_ids.index_select(0, key_representatives).detach().cpu()
    key_block_ids = layout.block_ids.index_select(0, key_representatives).detach().cpu()
    key_modality_ids = modality_ids.index_select(0, key_representatives).detach().cpu()

    # Pillow is intentionally imported only when the opt-in debug switch is on.
    from PIL import Image, ImageDraw

    num_rows, num_cols = grouped_mask.shape
    cell_size = max(1, min(18, 1400 // max(num_rows, num_cols, 1)))
    left_margin = 125
    top_margin = 135
    mask_width = max(num_cols * cell_size, 1)
    mask_height = max(num_rows * cell_size, 1)

    false_color = torch.tensor((24, 27, 35), dtype=torch.uint8)
    # Keyed by (stream_id, modality_id): action columns get their own hues so
    # video and action visibility can be told apart at a glance.
    visible_colors = {
        (int(TeacherForcingStream.UND), 0): torch.tensor((245, 166, 35), dtype=torch.uint8),
        (int(TeacherForcingStream.CLEAN), 0): torch.tensor((68, 190, 120), dtype=torch.uint8),
        (int(TeacherForcingStream.NOISY), 0): torch.tensor((79, 145, 245), dtype=torch.uint8),
        (int(TeacherForcingStream.CLEAN), 1): torch.tensor((0, 137, 132), dtype=torch.uint8),
        (int(TeacherForcingStream.NOISY), 1): torch.tensor((186, 104, 240), dtype=torch.uint8),
    }
    pixels = false_color.expand(num_rows, num_cols, 3).clone()
    for column, (stream_id, modality_id) in enumerate(zip(key_stream_ids.tolist(), key_modality_ids.tolist())):
        pixels[grouped_mask[:, column], column] = visible_colors[(int(stream_id), int(modality_id))]

    mask_image = Image.fromarray(pixels.numpy())
    if cell_size != 1:
        mask_image = mask_image.resize((mask_width, mask_height), resample=Image.Resampling.NEAREST)
    image = Image.new("RGB", (left_margin + mask_width + 15, top_margin + mask_height + 15), "white")
    image.paste(mask_image, (left_margin, top_margin))
    draw = ImageDraw.Draw(image)
    has_action = bool(modality_ids.any())
    draw.text((8, 8), "Teacher-forcing attention visibility (True = visible)", fill="black")
    if has_action:
        columns_hint = "[UND blocks | CLEAN V/A blocks | NOISY V/A blocks]"
    else:
        columns_hint = "[UND blocks | CLEAN blocks | NOISY blocks]"
    draw.text((8, 25), f"rows/columns: {columns_hint}", fill="black")
    draw.text(
        (8, 44),
        f"und_block={und_block_size}, vision_block={layout.geometry.block_sizes}, history={layout.geometry.history_blocks}",
        fill="black",
    )
    legend_entries = [
        ("UND", tuple(visible_colors[(int(TeacherForcingStream.UND), 0)].tolist())),
        ("CLEAN-V" if has_action else "CLEAN", tuple(visible_colors[(int(TeacherForcingStream.CLEAN), 0)].tolist())),
        ("NOISY-V" if has_action else "NOISY", tuple(visible_colors[(int(TeacherForcingStream.NOISY), 0)].tolist())),
    ]
    if has_action:
        legend_entries.insert(2, ("CLEAN-A", tuple(visible_colors[(int(TeacherForcingStream.CLEAN), 1)].tolist())))
        legend_entries.append(("NOISY-A", tuple(visible_colors[(int(TeacherForcingStream.NOISY), 1)].tolist())))
    legend_entries.append(("MASKED", tuple(false_color.tolist())))
    legend_x = 8
    for label, color in legend_entries:
        draw.rectangle((legend_x, 66, legend_x + 10, 76), fill=color)
        draw.text((legend_x + 14, 65), label, fill="black")
        legend_x += 20 + 6 * len(label)

    stream_names = {-1: "U", 0: "C", 1: "N"}

    und_block_ids: list[int] = []
    next_und_block_id: dict[int, int] = {}
    for sample_id, stream_id in zip(query_sample_ids.tolist(), query_stream_ids.tolist()):
        if stream_id == int(TeacherForcingStream.UND):
            und_block_ids.append(next_und_block_id.get(sample_id, 0))
            next_und_block_id[sample_id] = und_block_ids[-1] + 1
        else:
            und_block_ids.append(-1)

    def _label(sample_id: int, stream_id: int, block_id: int, und_block_id: int, modality_id: int) -> str:
        suffix = str(und_block_id) if stream_id == int(TeacherForcingStream.UND) else str(block_id)
        modality = "a" if modality_id == 1 else ""
        return f"s{sample_id}:{stream_names[stream_id]}{suffix}{modality}"

    # Draw all labels when cells are readable; otherwise retain sample boundary
    # labels and the color legend so large packed batches remain interpretable.
    label_stride = 1 if cell_size >= 8 else max(1, 48 // cell_size)
    for row in range(0, num_rows, label_stride):
        label = _label(
            int(query_sample_ids[row]),
            int(query_stream_ids[row]),
            int(query_block_ids[row]),
            und_block_ids[row],
            int(query_modality_ids[row]),
        )
        draw.text((4, top_margin + row * cell_size), label, fill="black")
    for column in range(0, num_cols, label_stride):
        label = _label(
            int(key_sample_ids[column]),
            int(key_stream_ids[column]),
            int(key_block_ids[column]),
            und_block_ids[column],
            int(key_modality_ids[column]),
        )
        label_image = Image.new("RGBA", (60, 12), (255, 255, 255, 0))
        ImageDraw.Draw(label_image).text((0, 0), label, fill="black")
        label_image = label_image.rotate(90, expand=True)
        image.paste(
            label_image,
            (left_margin + column * cell_size, top_margin - label_image.height - 2),
            label_image,
        )

    if cell_size >= 6:
        for row in range(1, num_rows):
            y = top_margin + row * cell_size
            draw.line((left_margin, y, left_margin + mask_width, y), fill=(90, 95, 105))
        for column in range(1, num_cols):
            x = left_margin + column * cell_size
            draw.line((x, top_margin, x, top_margin + mask_height), fill=(90, 95, 105))

    def _draw_boundaries(sample_ids: torch.Tensor, *, rows: bool) -> None:
        for index in range(1, sample_ids.numel()):
            if int(sample_ids[index]) == int(sample_ids[index - 1]):
                continue
            if rows:
                y = top_margin + index * cell_size
                draw.line((left_margin, y, left_margin + mask_width, y), fill=(220, 45, 45), width=2)
            else:
                x = left_margin + index * cell_size
                draw.line((x, top_margin, x, top_margin + mask_height), fill=(220, 45, 45), width=2)

    _draw_boundaries(query_sample_ids, rows=True)
    _draw_boundaries(key_sample_ids, rows=False)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _validate_teacher_forcing_packed_sequence(
    packed_sequence: PackedSequence,
) -> tuple[list[int], list[tuple[int, int, int]], list[int] | None]:
    """Validate the supported ``[UND | vision (| action)]`` source packing contract."""

    if packed_sequence.teacher_forcing is not None:
        raise ValueError("packed_sequence already contains teacher-forcing data")
    if packed_sequence.vision is None:
        raise ValueError("teacher-forcing expansion requires vision data")
    if packed_sequence.sound is not None:
        raise ValueError("teacher-forcing expansion does not support sound data")
    if packed_sequence.is_image_batch:
        raise ValueError("teacher-forcing expansion requires a video batch")
    if packed_sequence.vision_item_split_lens or packed_sequence.control_weights is not None:
        raise ValueError("teacher-forcing expansion does not support multi-item vision packing")
    if packed_sequence.num_action_tokens_per_supertoken != 0:
        raise ValueError("teacher-forcing expansion does not support temporal-causal supertoken packing")
    if packed_sequence.sequence_length != sum(packed_sequence.sample_lens):
        raise ValueError(
            "packed_sequence.sequence_length must equal sum(sample_lens), "
            f"got {packed_sequence.sequence_length} and {sum(packed_sequence.sample_lens)}"
        )
    if (
        packed_sequence.position_ids.ndim != 2
        or packed_sequence.position_ids.shape[-1] != packed_sequence.sequence_length
    ):
        raise ValueError(
            "packed_sequence.position_ids must have shape [axes, sequence_length], "
            f"got {tuple(packed_sequence.position_ids.shape)}"
        )

    vision = packed_sequence.vision
    action = packed_sequence.action
    num_samples = len(packed_sequence.sample_lens)
    if len(vision.token_shapes) != num_samples or len(vision.tokens) != num_samples:
        raise ValueError(
            "teacher-forcing expansion requires exactly one vision item per packed sample, "
            f"got {len(vision.token_shapes)} shapes, {len(vision.tokens)} payloads, and {num_samples} samples"
        )
    if action is not None and (len(action.token_shapes) != num_samples or len(action.tokens) != num_samples):
        raise ValueError(
            "teacher-forcing expansion requires exactly one action item per packed sample, "
            f"got {len(action.token_shapes)} shapes, {len(action.tokens)} payloads, and {num_samples} samples"
        )

    vision_token_shapes: list[tuple[int, int, int]] = []
    und_token_counts: list[int] = []
    action_token_counts: list[int] | None = [] if action is not None else None
    expected_text_indexes: list[int] = []
    expected_vision_indexes: list[int] = []
    expected_action_indexes: list[int] = []
    expected_split_lens: list[int] = []
    expected_attn_modes: list[str] = []
    sample_offset = 0
    for sample_id, (sample_len, token_shape) in enumerate(zip(packed_sequence.sample_lens, vision.token_shapes)):
        if len(token_shape) != 3:
            raise ValueError(f"vision.token_shapes[{sample_id}] must contain (T, H, W), got {token_shape}")
        num_frames, height, width = token_shape
        vision_count = num_frames * height * width

        action_count = 0
        if action is not None:
            action_shape = action.token_shapes[sample_id]
            if len(action_shape) != 1:
                raise ValueError(f"action.token_shapes[{sample_id}] must contain (T_action,), got {action_shape}")
            action_count = action_shape[0]
            if action_count < 1:
                raise ValueError(f"action.token_shapes[{sample_id}] must contain at least one step, got {action_shape}")
            assert action_token_counts is not None
            action_token_counts.append(action_count)

        und_count = sample_len - vision_count - action_count
        if und_count < 1:
            raise ValueError(
                f"sample {sample_id} must contain at least one UND token before its GEN tokens, got {und_count}"
            )
        vision_start = sample_offset + und_count
        action_start = vision_start + vision_count
        expected_text_indexes.extend(range(sample_offset, vision_start))
        expected_vision_indexes.extend(range(vision_start, action_start))
        expected_action_indexes.extend(range(action_start, sample_offset + sample_len))
        expected_split_lens.extend((und_count, vision_count + action_count))
        expected_attn_modes.extend(("causal", "full"))
        und_token_counts.append(und_count)
        vision_token_shapes.append((num_frames, height, width))

        if action_count > 0:
            # Enforce action_start_frame_offset == 0: with offset 0 the first
            # action step shares the temporal mRoPE coordinate of the first
            # vision latent frame (both integer and FPS-modulated paths).
            vision_first_temporal = float(packed_sequence.position_ids[0, vision_start])
            action_first_temporal = float(packed_sequence.position_ids[0, action_start])
            if vision_first_temporal != action_first_temporal:
                raise ValueError(
                    f"sample {sample_id}: teacher-forcing V+A training requires action_start_frame_offset=0 "
                    f"(first action temporal position {action_first_temporal} must equal first vision latent "
                    f"frame temporal position {vision_first_temporal})"
                )

        sample_offset += sample_len

    if not _sequence_indexes_match(packed_sequence.text_indexes, expected_text_indexes):
        raise ValueError("text and GEN sequence indexes must form the expected per-sample UND/GEN partition")
    if not _sequence_indexes_match(vision.sequence_indexes, expected_vision_indexes):
        raise ValueError("text and GEN sequence indexes must form the expected per-sample UND/GEN partition")
    if action is not None and not _sequence_indexes_match(action.sequence_indexes, expected_action_indexes):
        raise ValueError("action sequence indexes must directly follow each sample's vision tokens")
    if packed_sequence.split_lens != expected_split_lens or packed_sequence.attn_modes != expected_attn_modes:
        raise ValueError(
            "source attention splits must alternate one causal UND split and one full GEN split per sample"
        )
    if packed_sequence.text_ids.numel() != len(expected_text_indexes):
        raise ValueError(
            "text_ids must contain one payload per UND sequence index, "
            f"got {packed_sequence.text_ids.numel()} and {len(expected_text_indexes)}"
        )

    return und_token_counts, vision_token_shapes, action_token_counts


def _sequence_indexes_match(actual: torch.Tensor, expected: list[int]) -> bool:
    """Compare packed index tensors without ``.tolist()`` on the hot path."""

    if actual.numel() != len(expected):
        return False
    expected_t = torch.tensor(expected, dtype=actual.dtype)
    return bool(torch.equal(actual.detach().cpu(), expected_t))


def _build_source_to_stream_index(
    layout: TeacherForcingLayout,
    stream: TeacherForcingStream,
) -> torch.Tensor:
    """Map original packed indexes to expanded indexes of one stream.

    Returns a 1-D long table ``t`` where ``t[source]`` is the expanded index, or
    ``-1`` when ``source`` is not in ``stream``. Built with a single scatter so
    expansion does not call ``.item()`` once per token.
    """

    source = layout.source_sequence_indexes
    if source.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    table = source.new_full((int(source.max().item()) + 1,), -1)
    stream_indexes = torch.nonzero(layout.stream_ids == int(stream), as_tuple=True)[0]
    if stream_indexes.numel() == 0:
        return table
    table[source.index_select(0, stream_indexes)] = stream_indexes
    return table


def _remap_indexes(indexes: torch.Tensor | None, source_to_new: torch.Tensor, name: str) -> torch.Tensor | None:
    if indexes is None:
        return None
    if indexes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    cpu_indexes = indexes.detach().to(device=source_to_new.device, dtype=torch.long)
    in_range = (cpu_indexes >= 0) & (cpu_indexes < source_to_new.numel())
    if not bool(in_range.all()):
        bad = int(cpu_indexes[~in_range][0].item())
        raise ValueError(f"{name} contains an index outside the supported source stream: {bad}")
    remapped = source_to_new.index_select(0, cpu_indexes)
    missing = remapped < 0
    if bool(missing.any()):
        bad = int(cpu_indexes[missing][0].item())
        raise ValueError(f"{name} contains an index outside the supported source stream: {bad}")
    return remapped


def _validate_clean_payloads(
    name: str,
    clean_tokens: Sequence[torch.Tensor],
    noisy_tokens: Sequence[torch.Tensor],
) -> None:
    """Require one clean payload per noisy payload with identical shape/dtype/device."""

    if len(clean_tokens) != len(noisy_tokens):
        raise ValueError(
            f"clean_{name}_tokens must contain one payload per noisy {name} payload, "
            f"got {len(clean_tokens)} and {len(noisy_tokens)}"
        )
    for payload_id, (clean_token, noisy_token) in enumerate(zip(clean_tokens, noisy_tokens)):
        if clean_token.shape != noisy_token.shape:
            raise ValueError(
                f"clean/noisy {name} payload {payload_id} must have the same shape, "
                f"got {tuple(clean_token.shape)} and {tuple(noisy_token.shape)}"
            )
        if clean_token.dtype != noisy_token.dtype:
            raise ValueError(
                f"clean/noisy {name} payload {payload_id} must have the same dtype, "
                f"got {clean_token.dtype} and {noisy_token.dtype}"
            )
        if clean_token.device != noisy_token.device:
            raise ValueError(
                f"clean/noisy {name} payload {payload_id} must be on the same device, "
                f"got {clean_token.device} and {noisy_token.device}"
            )


def _remap_spans(spans, source_to_noisy: torch.Tensor, name: str) -> list:
    """Shift modality spans into the noisy stream, requiring per-span contiguity."""

    remapped_spans = []
    for span in spans:
        span_indexes = source_to_noisy[span.sequence_start : span.sequence_start + span.sequence_len]
        if span_indexes.numel() != span.sequence_len or bool((span_indexes < 0).any()):
            raise ValueError(f"{name} contains an index outside the supported source stream: {span.sequence_start}")
        start = int(span_indexes[0].item())
        expected = torch.arange(start, start + span.sequence_len, dtype=span_indexes.dtype, device=span_indexes.device)
        if not torch.equal(span_indexes, expected):
            raise ValueError(f"{name} span at source index {span.sequence_start} is not contiguous after remapping")
        remapped_spans.append(replace(span, sequence_start=start))
    return remapped_spans


def expand_packed_sequence_for_teacher_forcing(
    packed_sequence: PackedSequence,
    *,
    clean_vision_tokens: Sequence[torch.Tensor],
    geometry: TeacherForcingGeometry | None = None,
    block_size: int | None = None,
    history_blocks: int | None = None,
    clean_action_tokens: Sequence[torch.Tensor] | None = None,
    temporal_compression_factor: int | None = None,
) -> PackedSequence:
    """Return a packed sequence expanded into clean/noisy GEN streams.

    Vision-only sequences expand to ``[UND | clean V | noisy V]``. Sequences
    with action data expand to per-block interleaved dual streams and require
    ``clean_action_tokens`` plus ``temporal_compression_factor`` for the
    action-to-latent-frame block assignment.
    """

    und_token_counts, vision_token_shapes, action_token_counts = _validate_teacher_forcing_packed_sequence(
        packed_sequence
    )
    assert packed_sequence.vision is not None
    vision = packed_sequence.vision
    action = packed_sequence.action

    _validate_clean_payloads("vision", clean_vision_tokens, vision.tokens)
    if geometry is None:
        if block_size is None or history_blocks is None:
            raise ValueError("teacher-forcing expansion requires geometry or both block_size and history_blocks")
        geometry = shared_teacher_forcing_geometry(len(clean_vision_tokens), block_size, history_blocks)
    if action is not None:
        if clean_action_tokens is None:
            raise ValueError("clean_action_tokens is required when the packed sequence contains action data")
        _validate_clean_payloads("action", clean_action_tokens, action.tokens)
    elif clean_action_tokens is not None:
        raise ValueError("clean_action_tokens was provided but the packed sequence contains no action data")

    layout = build_teacher_forcing_layout(
        und_token_counts=und_token_counts,
        vision_token_shapes=vision_token_shapes,
        geometry=geometry,
        action_token_counts=action_token_counts,
        temporal_compression_factor=temporal_compression_factor,
    )
    source_to_und = _build_source_to_stream_index(layout, TeacherForcingStream.UND)
    source_to_noisy = _build_source_to_stream_index(layout, TeacherForcingStream.NOISY)

    remapped_text_indexes = _remap_indexes(packed_sequence.text_indexes, source_to_und, "text_indexes")
    remapped_ce_loss_indexes = _remap_indexes(
        packed_sequence.ce_loss_indexes,
        source_to_und,
        "ce_loss_indexes",
    )
    remapped_vision_indexes = _remap_indexes(vision.sequence_indexes, source_to_noisy, "vision.sequence_indexes")
    remapped_mse_loss_indexes = _remap_indexes(
        vision.mse_loss_indexes,
        source_to_noisy,
        "vision.mse_loss_indexes",
    )
    assert remapped_text_indexes is not None
    assert remapped_vision_indexes is not None
    assert remapped_mse_loss_indexes is not None

    expanded_vision = replace(
        vision,
        sequence_indexes=remapped_vision_indexes,
        mse_loss_indexes=remapped_mse_loss_indexes,
        spans=_remap_spans(vision.spans, source_to_noisy, "vision"),
        tokens=list(vision.tokens),
        token_shapes=list(vision.token_shapes),
        condition_mask=list(vision.condition_mask),
        noisy_frame_indexes=list(vision.noisy_frame_indexes),
    )

    expanded_action = None
    if action is not None:
        remapped_action_indexes = _remap_indexes(action.sequence_indexes, source_to_noisy, "action.sequence_indexes")
        remapped_action_mse_indexes = _remap_indexes(
            action.mse_loss_indexes,
            source_to_noisy,
            "action.mse_loss_indexes",
        )
        assert remapped_action_indexes is not None
        assert remapped_action_mse_indexes is not None
        expanded_action = replace(
            action,
            sequence_indexes=remapped_action_indexes,
            mse_loss_indexes=remapped_action_mse_indexes,
            spans=_remap_spans(action.spans, source_to_noisy, "action"),
            tokens=list(action.tokens),
            token_shapes=list(action.token_shapes),
            condition_mask=list(action.condition_mask),
            noisy_frame_indexes=list(action.noisy_frame_indexes),
        )

    teacher_forcing = TeacherForcingData(
        layout=layout,
        clean_vision_tokens=list(clean_vision_tokens),
        clean_action_tokens=list(clean_action_tokens) if clean_action_tokens is not None else None,
    )
    return replace(
        packed_sequence,
        sample_lens=list(layout.sample_lens),
        split_lens=list(layout.split_lens),
        attn_modes=list(layout.attn_modes),
        uses_single_timestep=False,
        sequence_length=sum(layout.sample_lens),
        text_indexes=remapped_text_indexes,
        position_ids=packed_sequence.position_ids[:, layout.source_sequence_indexes],
        ce_loss_indexes=remapped_ce_loss_indexes,
        vision=expanded_vision,
        action=expanded_action,
        teacher_forcing=teacher_forcing,
    )


def select_teacher_forcing_noisy_outputs(
    packed_output: torch.Tensor,
    layout: TeacherForcingLayout,
) -> torch.Tensor:
    """Select the complete noisy vision stream in original packed order."""

    expected_length = layout.source_sequence_indexes.numel()
    if packed_output.ndim < 1 or packed_output.shape[0] != expected_length:
        raise ValueError(
            "packed_output first dimension must equal the expanded sequence length, "
            f"got shape {tuple(packed_output.shape)} and expected {expected_length}"
        )
    return packed_output.index_select(0, layout.noisy_output_indexes.to(device=packed_output.device))
