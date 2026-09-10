# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalitySpan, as_frame_timesteps
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingData,
    TeacherForcingGeometry,
    TeacherForcingLayout,
    TeacherForcingStream,
    _build_source_to_stream_index,
    _remap_indexes,
    assign_action_steps_to_latent_frames,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_block_attention_groups,
    build_teacher_forcing_frame_block_ids,
    build_teacher_forcing_layout,
    expand_packed_sequence_for_teacher_forcing,
    map_action_sigmas_from_vision_schedule,
    sample_teacher_forcing_geometry,
    sample_teacher_forcing_parameters,
    select_teacher_forcing_noisy_outputs,
)


def test_teacher_forcing_stream_values_are_stable():
    assert int(TeacherForcingStream.UND) == -1
    assert int(TeacherForcingStream.CLEAN) == 0
    assert int(TeacherForcingStream.NOISY) == 1


def test_teacher_forcing_layout_is_frozen():
    empty = torch.empty(0, dtype=torch.long)
    layout = TeacherForcingLayout(
        geometry=TeacherForcingGeometry(block_sizes=(1,), history_blocks=(1,)),
        original_sample_lens=(),
        sample_lens=(),
        split_lens=(),
        attn_modes=(),
        source_sequence_indexes=empty,
        sample_ids=empty,
        stream_ids=empty,
        block_ids=empty,
        gen_query_indexes=empty,
        clean_token_indexes=empty,
        noisy_output_indexes=empty,
    )

    with pytest.raises(FrozenInstanceError):
        layout.geometry = TeacherForcingGeometry(block_sizes=(2,), history_blocks=(1,))


def test_sample_teacher_forcing_parameters_is_reproducible():
    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)

    draws_a = [sample_teacher_forcing_parameters(generator=generator_a) for _ in range(32)]
    draws_b = [sample_teacher_forcing_parameters(generator=generator_b) for _ in range(32)]

    assert draws_a == draws_b
    assert all(1 <= block_size <= 4 for block_size, _ in draws_a)
    assert all(1 <= history_blocks <= 32 for _, history_blocks in draws_a)


@pytest.mark.parametrize(
    ("kwargs", "invalid_field"),
    [
        ({"block_size_min": 0}, "block_size_min"),
        ({"block_size_min": 4, "block_size_max": 3}, "block_size"),
        ({"history_blocks_min": 0}, "history_blocks_min"),
        ({"history_blocks_min": 32, "history_blocks_max": 31}, "history_blocks"),
    ],
)
def test_sample_teacher_forcing_parameters_rejects_invalid_ranges(
    kwargs: dict[str, int],
    invalid_field: str,
):
    with pytest.raises(ValueError, match=invalid_field):
        sample_teacher_forcing_parameters(**kwargs)


def test_build_teacher_forcing_layout_maps_both_streams_to_the_original_tokens():
    layout = build_teacher_forcing_layout(
        und_token_counts=[2],
        vision_token_shapes=[(5, 1, 1)],
        block_size=2,
        history_blocks=1,
    )

    assert layout.original_sample_lens == (7,)
    assert layout.sample_lens == (12,)
    assert layout.split_lens == (2, 10)
    assert layout.attn_modes == ("causal", "full")
    assert layout.source_sequence_indexes.tolist() == [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
    assert layout.sample_ids.tolist() == [0] * 12
    assert layout.stream_ids.tolist() == [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # Frame 0 is a singleton block; remaining frames are chunked by S=2.
    assert layout.block_ids.tolist() == [-1, -1, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2]
    assert layout.geometry.block_sizes == (2,)
    assert layout.geometry.history_blocks == (1,)
    assert layout.gen_query_indexes.tolist() == list(range(2, 12))
    assert layout.clean_token_indexes.tolist() == list(range(2, 7))
    assert layout.noisy_output_indexes.tolist() == list(range(7, 12))
    # Frame 0 is a singleton; remaining frames chunk as (1,2) then (3,4).
    assert layout.block_token_spans == (((0, 1), (1, 3), (3, 5)),)


def test_build_teacher_forcing_layout_expands_spatial_tokens_and_isolates_sample_offsets():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(3, 1, 2), (2, 2, 1)],
        block_size=2,
        history_blocks=3,
    )

    assert layout.original_sample_lens == (7, 6)
    assert layout.sample_lens == (13, 10)
    assert layout.split_lens == (1, 12, 2, 8)
    assert layout.attn_modes == ("causal", "full", "causal", "full")
    assert layout.source_sequence_indexes.tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        9,
        10,
        11,
        12,
    ]
    # Sample 0: T=3,S=2 → blocks [0 | 1 1], 2 spatial tokens per frame.
    # Sample 1: T=2,S=2 → blocks [0 | 1], 2 spatial tokens per frame.
    assert layout.block_ids.tolist() == [
        -1,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        -1,
        -1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
    ]
    assert layout.sample_ids.tolist() == [0] * 13 + [1] * 10


@pytest.mark.parametrize(
    ("kwargs", "invalid_field"),
    [
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1), (2, 1, 1)],
                "block_size": 1,
                "history_blocks": 1,
            },
            "same number",
        ),
        (
            {
                "und_token_counts": [],
                "vision_token_shapes": [],
                "block_size": 1,
                "history_blocks": 1,
            },
            "empty",
        ),
        (
            {
                "und_token_counts": [0],
                "vision_token_shapes": [(2, 1, 1)],
                "block_size": 1,
                "history_blocks": 1,
            },
            "und_token_counts",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 0, 1)],
                "block_size": 1,
                "history_blocks": 1,
            },
            "vision_token_shapes",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1)],
                "block_size": 0,
                "history_blocks": 1,
            },
            "block_size",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1)],
                "block_size": 1,
                "history_blocks": 0,
            },
            "history_blocks",
        ),
    ],
)
def test_build_teacher_forcing_layout_rejects_invalid_geometry(
    kwargs: dict[str, object],
    invalid_field: str,
):
    with pytest.raises(ValueError, match=invalid_field):
        build_teacher_forcing_layout(**kwargs)


def test_dense_mask_matches_s1_k1_block_causal_matrix():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(3, 1, 1)],
        block_size=1,
        history_blocks=1,
    )

    mask = build_dense_teacher_forcing_gen_mask(layout)

    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0],  # C0 -> U,C0
            [1, 1, 1, 0, 0, 0, 0],  # C1 -> U,C0,C1
            [1, 0, 1, 1, 0, 0, 0],  # C2 -> U,C1,C2
            [1, 0, 0, 0, 1, 0, 0],  # N0 -> U,N0
            [1, 1, 0, 0, 0, 1, 0],  # N1 -> U,C0,N1
            [1, 0, 1, 0, 0, 0, 1],  # N2 -> U,C1,N2
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask, expected)
    assert not mask[3, 1]
    assert not mask[4, 2]
    assert not mask[5, 3]


def test_dense_mask_keeps_blocks_full_and_limits_clean_history_to_k_blocks():
    num_frames = 5
    spatial = 2
    block_size = 2
    history_blocks = 1
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(num_frames, 1, spatial)],
        block_size=block_size,
        history_blocks=history_blocks,
    )
    mask = build_dense_teacher_forcing_gen_mask(layout)
    frame_block_ids = build_teacher_forcing_frame_block_ids(num_frames, block_size).tolist()
    clean_columns = layout.clean_token_indexes
    noisy_columns = layout.noisy_output_indexes
    for frame_id in range(num_frames):
        block_id = frame_block_ids[frame_id]
        oldest_visible_block = max(0, block_id - history_blocks)
        expected_clean = [
            oldest_visible_block <= frame_block_ids[candidate] <= block_id for candidate in range(num_frames)
        ]
        expected_clean = [value for value in expected_clean for _ in range(spatial)]
        expected_prev_clean = [
            oldest_visible_block <= frame_block_ids[candidate] < block_id for candidate in range(num_frames)
        ]
        expected_prev_clean = [value for value in expected_prev_clean for _ in range(spatial)]
        expected_same_noisy = [frame_block_ids[candidate] == block_id for candidate in range(num_frames)]
        expected_same_noisy = [value for value in expected_same_noisy for _ in range(spatial)]
        for token in range(spatial):
            clean_row = frame_id * spatial + token
            noisy_row = num_frames * spatial + clean_row
            assert mask[clean_row, clean_columns].tolist() == expected_clean
            assert not mask[clean_row, noisy_columns].any()
            assert mask[noisy_row, clean_columns].tolist() == expected_prev_clean
            assert mask[noisy_row, noisy_columns].tolist() == expected_same_noisy


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
@pytest.mark.parametrize("history_blocks", [1, 32])
def test_dense_mask_matches_lingbot_full_video_boundaries(block_size: int, history_blocks: int):
    num_frames = 7
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(num_frames, 1, 1)],
        block_size=block_size,
        history_blocks=history_blocks,
    )
    mask = build_dense_teacher_forcing_gen_mask(layout)

    clean_columns = layout.clean_token_indexes
    noisy_columns = layout.noisy_output_indexes
    frame_block_ids = build_teacher_forcing_frame_block_ids(num_frames, block_size).tolist()
    for frame_id in range(num_frames):
        block_id = frame_block_ids[frame_id]
        oldest_visible_block = max(0, block_id - history_blocks)
        expected_clean_for_clean = [
            oldest_visible_block <= frame_block_ids[candidate] <= block_id for candidate in range(num_frames)
        ]
        expected_clean_for_noisy = [
            oldest_visible_block <= frame_block_ids[candidate] < block_id for candidate in range(num_frames)
        ]
        expected_noisy_for_noisy = [frame_block_ids[candidate] == block_id for candidate in range(num_frames)]

        clean_row = frame_id
        noisy_row = num_frames + frame_id
        assert mask[clean_row, clean_columns].tolist() == expected_clean_for_clean
        assert not mask[clean_row, noisy_columns].any()
        assert mask[noisy_row, clean_columns].tolist() == expected_clean_for_noisy
        assert mask[noisy_row, noisy_columns].tolist() == expected_noisy_for_noisy


def test_dense_mask_isolates_packed_samples():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(2, 1, 1), (2, 1, 1)],
        block_size=1,
        history_blocks=2,
    )

    mask = build_dense_teacher_forcing_gen_mask(layout)
    query_sample_ids = layout.sample_ids[layout.gen_query_indexes]

    assert not mask[query_sample_ids == 0][:, layout.sample_ids == 1].any()
    assert not mask[query_sample_ids == 1][:, layout.sample_ids == 0].any()


def test_dense_mask_rejects_und_queries_in_gen_query_indexes():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(1, 1, 1)],
        block_size=1,
        history_blocks=1,
    )
    corrupted = replace(layout, gen_query_indexes=torch.tensor([0], dtype=torch.long))

    with pytest.raises(ValueError, match="GEN queries"):
        build_dense_teacher_forcing_gen_mask(corrupted)


def test_teacher_forcing_api_is_exported():
    from cosmos_framework.data.generator import sequence_packing

    assert sequence_packing.TeacherForcingData is TeacherForcingData
    assert sequence_packing.TeacherForcingLayout is TeacherForcingLayout
    assert sequence_packing.build_teacher_forcing_layout is build_teacher_forcing_layout
    assert sequence_packing.build_dense_teacher_forcing_gen_mask is build_dense_teacher_forcing_gen_mask
    assert sequence_packing.expand_packed_sequence_for_teacher_forcing is expand_packed_sequence_for_teacher_forcing
    assert sequence_packing.sample_teacher_forcing_parameters is sample_teacher_forcing_parameters
    assert sequence_packing.sample_teacher_forcing_geometry is sample_teacher_forcing_geometry
    assert sequence_packing.TeacherForcingGeometry is TeacherForcingGeometry
    assert sequence_packing.select_teacher_forcing_noisy_outputs is select_teacher_forcing_noisy_outputs


def test_build_teacher_forcing_frame_block_ids_isolates_the_first_latent():
    assert build_teacher_forcing_frame_block_ids(1, 4).tolist() == [0]
    assert build_teacher_forcing_frame_block_ids(5, 2).tolist() == [0, 1, 1, 2, 2]
    assert build_teacher_forcing_frame_block_ids(4, 1).tolist() == [0, 1, 2, 3]


def test_sample_teacher_forcing_geometry_is_independent_per_sample():
    generator = torch.Generator().manual_seed(7)
    geometry = sample_teacher_forcing_geometry(
        num_samples=8,
        block_size_min=1,
        block_size_max=4,
        history_blocks_min=1,
        history_blocks_max=4,
        generator=generator,
    )
    assert len(geometry.block_sizes) == 8
    assert len(set(geometry.block_sizes)) > 1 or len(set(geometry.history_blocks)) > 1


def _make_packed_video_sequence() -> PackedSequence:
    noisy_0 = torch.tensor([[[[[10.0]], [[11.0]], [[12.0]]]]])
    noisy_1 = torch.tensor([[[[[20.0, 21.0]], [[22.0, 23.0]]]]])
    vision = ModalityData(
        sequence_indexes=torch.tensor([2, 3, 4, 6, 7, 8, 9]),
        timesteps=torch.tensor([0.4, 0.4, 0.7, 0.7, 0.7, 0.7]),
        mse_loss_indexes=torch.tensor([3, 4, 6, 7, 8, 9]),
        spans=[
            ModalitySpan(2, 1, 0, 0, 1, (1, 1, 1)),
            ModalitySpan(3, 1, 0, 1, 1, (1, 1, 1)),
            ModalitySpan(4, 1, 0, 2, 1, (1, 1, 1)),
            ModalitySpan(6, 2, 1, 0, 2, (1, 1, 2)),
            ModalitySpan(8, 2, 1, 2, 2, (1, 1, 2)),
        ],
        token_shapes=[(3, 1, 1), (2, 1, 2)],
        tokens=[noisy_0, noisy_1],
        condition_mask=[torch.tensor([[[1.0]], [[0.0]], [[0.0]]]), torch.zeros(2, 1, 1)],
        noisy_frame_indexes=[torch.tensor([1, 2]), torch.tensor([0, 1])],
    )
    return PackedSequence(
        sample_lens=[5, 5],
        split_lens=[2, 3, 1, 4],
        attn_modes=["causal", "full", "causal", "full"],
        is_image_batch=False,
        uses_single_timestep=True,
        sequence_length=10,
        text_ids=torch.tensor([101, 102, 103]),
        text_indexes=torch.tensor([0, 1, 5]),
        position_ids=torch.arange(30).reshape(3, 10),
        label_ids=torch.tensor([102, 103]),
        ce_loss_indexes=torch.tensor([0, 5]),
        ce_loss_weights=torch.tensor([1.0, 0.5]),
        vision=vision,
    )


def test_packed_sequence_has_no_teacher_forcing_data_by_default():
    assert PackedSequence().teacher_forcing is None


def test_expand_packed_sequence_preserves_noisy_contract_and_duplicates_rope():
    packed = _make_packed_video_sequence()
    clean_tokens = [torch.full_like(packed.vision.tokens[0], 1.0), torch.full_like(packed.vision.tokens[1], 2.0)]

    expanded = expand_packed_sequence_for_teacher_forcing(
        packed,
        clean_vision_tokens=clean_tokens,
        block_size=2,
        history_blocks=3,
    )

    assert expanded is not packed
    assert expanded.vision is not packed.vision
    assert expanded.teacher_forcing is not None
    layout = expanded.teacher_forcing.layout
    assert isinstance(expanded.teacher_forcing, TeacherForcingData)
    assert expanded.sample_lens == [8, 9]
    assert expanded.split_lens == [2, 6, 1, 8]
    assert expanded.attn_modes == ["causal", "full", "causal", "full"]
    assert expanded.sequence_length == 17
    assert expanded.uses_single_timestep is False
    assert expanded.text_ids is packed.text_ids
    assert expanded.text_indexes.tolist() == [0, 1, 8]
    assert expanded.ce_loss_indexes.tolist() == [0, 8]
    assert expanded.position_ids.tolist() == packed.position_ids[:, layout.source_sequence_indexes].tolist()
    assert torch.equal(
        expanded.position_ids[:, layout.clean_token_indexes],
        expanded.position_ids[:, layout.noisy_output_indexes],
    )

    assert layout.clean_token_indexes.tolist() == [2, 3, 4, 9, 10, 11, 12]
    assert layout.noisy_output_indexes.tolist() == [5, 6, 7, 13, 14, 15, 16]
    assert expanded.vision.sequence_indexes.tolist() == [5, 6, 7, 13, 14, 15, 16]
    assert expanded.vision.mse_loss_indexes.tolist() == [6, 7, 13, 14, 15, 16]
    assert expanded.vision.timesteps is packed.vision.timesteps
    assert expanded.vision.tokens == packed.vision.tokens
    assert expanded.vision.condition_mask == packed.vision.condition_mask
    assert expanded.vision.noisy_frame_indexes == packed.vision.noisy_frame_indexes
    assert [span.sequence_start for span in expanded.vision.spans] == [5, 6, 7, 13, 15]
    assert expanded.teacher_forcing.clean_vision_tokens == clean_tokens

    assert packed.sample_lens == [5, 5]
    assert packed.vision.sequence_indexes.tolist() == [2, 3, 4, 6, 7, 8, 9]
    assert packed.teacher_forcing is None


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda packed: setattr(packed, "vision", None), "vision"),
        (
            lambda packed: setattr(
                packed,
                "action",
                ModalityData(sequence_indexes=torch.tensor([0])),
            ),
            "action",
        ),
        (lambda packed: setattr(packed, "is_image_batch", True), "video"),
        (lambda packed: packed.text_indexes.__setitem__(0, 9), "partition"),
        (lambda packed: packed.sample_lens.__setitem__(0, 6), "sequence_length"),
        (lambda packed: packed.split_lens.__setitem__(0, 1), "attention splits"),
        (lambda packed: packed.attn_modes.__setitem__(1, "causal"), "attention splits"),
    ],
)
def test_expand_packed_sequence_rejects_unsupported_layouts(mutate, error: str):
    packed = _make_packed_video_sequence()
    mutate(packed)
    clean_tokens = [torch.zeros(1, 1, 3, 1, 1), torch.zeros(1, 1, 2, 1, 2)]

    with pytest.raises((TypeError, ValueError), match=error):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=clean_tokens,
            block_size=1,
            history_blocks=1,
        )


def test_expand_packed_sequence_rejects_clean_payload_shape_mismatch():
    packed = _make_packed_video_sequence()
    clean_tokens = [torch.zeros(1, 1, 2, 1, 1), torch.zeros(1, 1, 2, 1, 2)]

    with pytest.raises(ValueError, match="shape"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=clean_tokens,
            block_size=1,
            history_blocks=1,
        )


def test_expand_packed_sequence_rejects_clean_payload_dtype_mismatch():
    packed = _make_packed_video_sequence()
    assert packed.vision is not None
    packed.vision.tokens = [token.to(torch.bfloat16) for token in packed.vision.tokens]
    clean_tokens = [torch.ones_like(token, dtype=torch.float32) for token in packed.vision.tokens]

    with pytest.raises(ValueError, match="dtype"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=clean_tokens,
            block_size=1,
            history_blocks=1,
        )


def test_map_action_sigmas_from_vision_schedule_follows_latent_frames():
    vision_sigmas = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]])
    mapped = map_action_sigmas_from_vision_schedule(
        vision_sigmas,
        action_sample_indices=[0, 1],
        action_lengths=[5, 3],
        num_vision_latent_frames=[3, 2],
        temporal_compression_factor=2,
    )
    # Sample 0: latent frames [0,1,1,2,2] → σ [0.1, 0.2, 0.2, 0.3, 0.3]
    # Sample 1: latent frames [0,1,1] → σ [0.4, 0.5, 0.5], padded to T=5
    torch.testing.assert_close(mapped[0], torch.tensor([0.1, 0.2, 0.2, 0.3, 0.3]))
    torch.testing.assert_close(mapped[1], torch.tensor([0.4, 0.5, 0.5, 0.0, 0.0]))


def test_assign_action_steps_to_latent_frames_uses_offset_zero_ceiling():
    assert assign_action_steps_to_latent_frames(5, 3, 2) == [0, 1, 1, 2, 2]
    # LIBERO-like: 16 action steps, 5 latent frames, cf=4.
    assert assign_action_steps_to_latent_frames(16, 5, 4) == [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    assert assign_action_steps_to_latent_frames(0, 3, 2) == []


def test_assign_action_steps_to_latent_frames_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="beyond the video"):
        assign_action_steps_to_latent_frames(7, 3, 2)
    with pytest.raises(ValueError, match="temporal_compression_factor"):
        assign_action_steps_to_latent_frames(3, 3, 0)


def test_build_teacher_forcing_layout_interleaves_action_per_block():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(3, 1, 1)],
        block_size=2,
        history_blocks=1,
        action_token_counts=[5],
        temporal_compression_factor=2,
    )

    # Vision frame blocks [0 | 1 1]; action latent frames [0,1,1,2,2] -> blocks
    # [0,1,1,1,1]. Interleaved GEN order: [V0 A0 | V1 V2 A1 A2 A3 A4].
    assert layout.original_sample_lens == (9,)
    assert layout.sample_lens == (17,)
    assert layout.split_lens == (1, 16)
    assert layout.source_sequence_indexes.tolist() == [0] + [1, 4, 2, 3, 5, 6, 7, 8] * 2
    assert layout.stream_ids.tolist() == [-1] + [0] * 8 + [1] * 8
    assert layout.block_ids.tolist() == [-1] + [0, 0, 1, 1, 1, 1, 1, 1] * 2
    assert layout.gen_query_indexes.tolist() == list(range(1, 17))
    assert layout.clean_token_indexes.tolist() == [1, 3, 4]
    assert layout.clean_action_token_indexes.tolist() == [2, 5, 6, 7, 8]
    assert layout.noisy_output_indexes.tolist() == [9, 11, 12, 10, 13, 14, 15, 16]
    assert layout.noisy_action_output_indexes.tolist() == [10, 13, 14, 15, 16]


def test_build_teacher_forcing_layout_without_action_keeps_empty_action_fields():
    layout = build_teacher_forcing_layout(
        und_token_counts=[2],
        vision_token_shapes=[(5, 1, 1)],
        block_size=2,
        history_blocks=1,
    )

    assert layout.clean_action_token_indexes.numel() == 0
    assert layout.noisy_action_output_indexes.numel() == 0


def test_build_teacher_forcing_layout_rejects_action_without_compression_factor():
    with pytest.raises(ValueError, match="temporal_compression_factor"):
        build_teacher_forcing_layout(
            und_token_counts=[1],
            vision_token_shapes=[(3, 1, 1)],
            block_size=1,
            history_blocks=1,
            action_token_counts=[3],
        )


def test_dense_mask_applies_block_rules_across_vision_and_action():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(3, 1, 1)],
        block_size=1,
        history_blocks=1,
        action_token_counts=[5],
        temporal_compression_factor=2,
    )
    mask = build_dense_teacher_forcing_gen_mask(layout)

    # GEN order per stream: [V0 A0 | V1 A1 A2 | V2 A3 A4] with blocks [0,0,1,1,1,2,2,2].
    # Columns: 0=UND, clean 1..8, noisy 9..16. Rows follow gen_query_indexes (position-1).
    noisy_a1_row = 11  # noisy A1 at position 12, block 1
    assert mask[noisy_a1_row].nonzero(as_tuple=True)[0].tolist() == [0, 1, 2, 11, 12, 13]
    clean_a1_row = 3  # clean A1 at position 4, block 1
    assert mask[clean_a1_row].nonzero(as_tuple=True)[0].tolist() == [0, 1, 2, 3, 4, 5]
    noisy_v0_row = 8  # noisy V0 at position 9, block 0: no clean history at all
    assert mask[noisy_v0_row].nonzero(as_tuple=True)[0].tolist() == [0, 9, 10]
    clean_v2_row = 5  # clean V2 at position 6, block 2, K=1 window = blocks {1,2}
    assert mask[clean_v2_row].nonzero(as_tuple=True)[0].tolist() == [0, 3, 4, 5, 6, 7, 8]


def test_select_teacher_forcing_noisy_outputs_preserves_order_and_gradient():
    packed = _make_packed_video_sequence()
    clean_tokens = [torch.zeros_like(token) for token in packed.vision.tokens]
    expanded = expand_packed_sequence_for_teacher_forcing(
        packed,
        clean_vision_tokens=clean_tokens,
        block_size=1,
        history_blocks=1,
    )
    assert expanded.teacher_forcing is not None
    output = torch.arange(expanded.sequence_length * 2, dtype=torch.float32).reshape(expanded.sequence_length, 2)
    output.requires_grad_()

    noisy = select_teacher_forcing_noisy_outputs(output, expanded.teacher_forcing.layout)

    torch.testing.assert_close(noisy, output[expanded.teacher_forcing.layout.noisy_output_indexes])
    noisy.sum().backward()
    assert output.grad is not None
    selected = torch.zeros(expanded.sequence_length, dtype=torch.bool)
    selected[expanded.teacher_forcing.layout.noisy_output_indexes] = True
    assert torch.equal(output.grad[selected], torch.ones_like(output.grad[selected]))
    assert torch.equal(output.grad[~selected], torch.zeros_like(output.grad[~selected]))


def _make_packed_video_action_sequence() -> PackedSequence:
    """Two packed samples, each ``[UND | vision | action]`` with offset-0 alignment.

    Sample 0: UND=2, vision T=3 (1x1), action A=5 (cf=2 -> latent frames [0,1,1,2,2]).
    Sample 1: UND=1, vision T=2 (1x2), action A=3 (cf=2 -> latent frames [0,1,1]).
    """

    noisy_vision_0 = torch.tensor([[[[[10.0]], [[11.0]], [[12.0]]]]])
    noisy_vision_1 = torch.tensor([[[[[20.0, 21.0]], [[22.0, 23.0]]]]])
    noisy_action_0 = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    noisy_action_1 = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    vision = ModalityData(
        sequence_indexes=torch.tensor([2, 3, 4, 11, 12, 13, 14]),
        timesteps=torch.tensor([0.4, 0.4, 0.7, 0.7, 0.7, 0.7]),
        mse_loss_indexes=torch.tensor([3, 4, 11, 12, 13, 14]),
        spans=[
            ModalitySpan(2, 1, 0, 0, 1, (1, 1, 1)),
            ModalitySpan(3, 1, 0, 1, 1, (1, 1, 1)),
            ModalitySpan(4, 1, 0, 2, 1, (1, 1, 1)),
            ModalitySpan(11, 2, 1, 0, 2, (1, 1, 2)),
            ModalitySpan(13, 2, 1, 2, 2, (1, 1, 2)),
        ],
        token_shapes=[(3, 1, 1), (2, 1, 2)],
        tokens=[noisy_vision_0, noisy_vision_1],
        condition_mask=[torch.tensor([[[1.0]], [[0.0]], [[0.0]]]), torch.zeros(2, 1, 1)],
        noisy_frame_indexes=[torch.tensor([1, 2]), torch.tensor([0, 1])],
    )
    action = ModalityData(
        sequence_indexes=torch.tensor([5, 6, 7, 8, 9, 15, 16, 17]),
        timesteps=torch.tensor([0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7]),
        mse_loss_indexes=torch.tensor([6, 7, 8, 9, 15, 16, 17]),
        spans=[
            ModalitySpan(5, 1, 0, 0, 1, (1,)),
            ModalitySpan(6, 1, 0, 1, 1, (1,)),
            ModalitySpan(7, 1, 0, 2, 1, (1,)),
            ModalitySpan(8, 1, 0, 3, 1, (1,)),
            ModalitySpan(9, 1, 0, 4, 1, (1,)),
            ModalitySpan(15, 1, 1, 0, 1, (1,)),
            ModalitySpan(16, 1, 1, 1, 1, (1,)),
            ModalitySpan(17, 1, 1, 2, 1, (1,)),
        ],
        token_shapes=[(5,), (3,)],
        tokens=[noisy_action_0, noisy_action_1],
        condition_mask=[torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]]), torch.zeros(3, 1)],
        noisy_frame_indexes=[torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 2])],
    )
    position_ids = torch.zeros(3, 18, dtype=torch.long)
    # Temporal axis: per sample the first action step shares the first vision
    # latent frame's coordinate (action_start_frame_offset=0).
    position_ids[0] = torch.tensor([5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 0, 3, 3, 4, 4, 3, 4, 5])
    return PackedSequence(
        sample_lens=[10, 8],
        split_lens=[2, 8, 1, 7],
        attn_modes=["causal", "full", "causal", "full"],
        is_image_batch=False,
        uses_single_timestep=True,
        sequence_length=18,
        text_ids=torch.tensor([101, 102, 103]),
        text_indexes=torch.tensor([0, 1, 10]),
        position_ids=position_ids,
        vision=vision,
        action=action,
    )


def test_expand_packed_sequence_interleaves_vision_and_action_per_block():
    packed = _make_packed_video_action_sequence()
    clean_vision = [torch.full_like(token, -1.0) for token in packed.vision.tokens]
    clean_action = [torch.full_like(token, -2.0) for token in packed.action.tokens]

    expanded = expand_packed_sequence_for_teacher_forcing(
        packed,
        clean_vision_tokens=clean_vision,
        block_size=2,
        history_blocks=1,
        clean_action_tokens=clean_action,
        temporal_compression_factor=2,
    )

    assert expanded.teacher_forcing is not None
    layout = expanded.teacher_forcing.layout
    # Sample 0 (S=2): vision blocks [0 | 1 1]; action blocks [0,1,1,1,1].
    # GEN order: [V0 A0 | V1 V2 A1 A2 A3 A4] -> sources [2,5,3,4,6,7,8,9].
    # Sample 1 (S=2): vision blocks [0 | 1]; action blocks [0,1,1].
    # GEN order: [V0a V0b A0 | V1a V1b A1 A2] -> sources [11,12,15,13,14,16,17].
    assert layout.original_sample_lens == (10, 8)
    assert layout.sample_lens == (18, 15)
    assert layout.split_lens == (2, 16, 1, 14)
    expected_gen_source_0 = [2, 5, 3, 4, 6, 7, 8, 9]
    expected_gen_source_1 = [11, 12, 15, 13, 14, 16, 17]
    assert layout.source_sequence_indexes.tolist() == (
        [0, 1] + expected_gen_source_0 * 2 + [10] + expected_gen_source_1 * 2
    )
    assert layout.clean_token_indexes.tolist() == [2, 4, 5, 19, 20, 22, 23]
    assert layout.clean_action_token_indexes.tolist() == [3, 6, 7, 8, 9, 21, 24, 25]
    # Per sample: vision (frame order) then action (step order).
    assert layout.noisy_output_indexes.tolist() == [10, 12, 13, 11, 14, 15, 16, 17] + [26, 27, 29, 30, 28, 31, 32]
    assert layout.noisy_action_output_indexes.tolist() == [11, 14, 15, 16, 17, 28, 31, 32]

    assert expanded.text_indexes.tolist() == [0, 1, 18]
    assert expanded.vision.sequence_indexes.tolist() == [10, 12, 13, 26, 27, 29, 30]
    assert expanded.vision.mse_loss_indexes.tolist() == [12, 13, 26, 27, 29, 30]
    assert expanded.action is not None
    assert expanded.action.sequence_indexes.tolist() == [11, 14, 15, 16, 17, 28, 31, 32]
    assert expanded.action.mse_loss_indexes.tolist() == [14, 15, 16, 17, 28, 31, 32]
    assert [span.sequence_start for span in expanded.vision.spans] == [10, 12, 13, 26, 29]
    assert [span.sequence_start for span in expanded.action.spans] == [11, 14, 15, 16, 17, 28, 31, 32]
    assert expanded.action.tokens == packed.action.tokens
    assert expanded.action.condition_mask == packed.action.condition_mask
    assert expanded.teacher_forcing.clean_action_tokens == clean_action

    # Clean/noisy slots of each modality share mRoPE positions.
    assert torch.equal(
        expanded.position_ids[:, layout.clean_token_indexes],
        expanded.position_ids[:, torch.tensor([10, 12, 13, 26, 27, 29, 30])],
    )
    assert torch.equal(
        expanded.position_ids[:, layout.clean_action_token_indexes],
        expanded.position_ids[:, layout.noisy_action_output_indexes],
    )

    # Cross-sample isolation still holds on the interleaved layout.
    mask = build_dense_teacher_forcing_gen_mask(layout)
    query_sample_ids = layout.sample_ids[layout.gen_query_indexes]
    assert not mask[query_sample_ids == 0][:, layout.sample_ids == 1].any()
    assert not mask[query_sample_ids == 1][:, layout.sample_ids == 0].any()


def test_expand_packed_sequence_rejects_nonzero_action_start_frame_offset():
    packed = _make_packed_video_action_sequence()
    packed.position_ids[0, 5] = 8  # sample 0's first action step drifts off vision frame 0

    with pytest.raises(ValueError, match="action_start_frame_offset"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=[torch.zeros_like(token) for token in packed.vision.tokens],
            block_size=1,
            history_blocks=1,
            clean_action_tokens=[torch.zeros_like(token) for token in packed.action.tokens],
            temporal_compression_factor=2,
        )


def test_expand_packed_sequence_requires_clean_action_tokens_with_action_data():
    packed = _make_packed_video_action_sequence()

    with pytest.raises(ValueError, match="clean_action_tokens is required"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=[torch.zeros_like(token) for token in packed.vision.tokens],
            block_size=1,
            history_blocks=1,
            temporal_compression_factor=2,
        )


def test_expand_packed_sequence_rejects_clean_action_tokens_without_action_data():
    packed = _make_packed_video_sequence()

    with pytest.raises(ValueError, match="no action data"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=[torch.zeros_like(token) for token in packed.vision.tokens],
            block_size=1,
            history_blocks=1,
            clean_action_tokens=[torch.zeros(2, 2)],
        )


def test_expand_packed_sequence_rejects_action_steps_beyond_video():
    packed = _make_packed_video_action_sequence()

    with pytest.raises(ValueError, match="beyond the video"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=[torch.zeros_like(token) for token in packed.vision.tokens],
            block_size=1,
            history_blocks=1,
            clean_action_tokens=[torch.zeros_like(token) for token in packed.action.tokens],
            temporal_compression_factor=1,  # A=5 steps then map to latent frames 0..4 > T-1=2
        )


def _visible_key_indexes(group) -> list[int]:
    keys: list[int] = []
    for start, end in group.kv_slices:
        keys.extend(range(start, end))
    return keys


def test_block_attention_groups_match_dense_mask_visible_keys():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(5, 1, 2), (3, 2, 1)],
        block_size=2,
        history_blocks=1,
        action_token_counts=[7, 4],
        temporal_compression_factor=2,
    )
    mask = build_dense_teacher_forcing_gen_mask(layout)
    groups = build_teacher_forcing_block_attention_groups(layout)

    covered = [False] * mask.shape[0]
    for group in groups:
        visible = _visible_key_indexes(group)
        for query_row in range(group.query_start, group.query_end):
            expected = torch.nonzero(mask[query_row], as_tuple=True)[0].tolist()
            assert visible == expected
            assert not covered[query_row]
            covered[query_row] = True
    assert all(covered)


def test_block_attention_groups_hide_current_clean_from_noisy_queries():
    layout = build_teacher_forcing_layout(
        und_token_counts=[2],
        vision_token_shapes=[(5, 1, 1)],
        block_size=2,
        history_blocks=2,
    )
    groups = build_teacher_forcing_block_attention_groups(layout)
    gen_len = layout.split_lens[1] // 2
    und_count = layout.split_lens[0]
    clean_start = und_count
    noisy_start = clean_start + gen_len
    num_blocks = len(layout.block_token_spans[0])

    for block_id, (gen_start, gen_end) in enumerate(layout.block_token_spans[0]):
        noisy_group = groups[num_blocks + block_id]
        visible = set(_visible_key_indexes(noisy_group))
        current_clean = set(range(clean_start + gen_start, clean_start + gen_end))
        current_noisy = set(range(noisy_start + gen_start, noisy_start + gen_end))
        assert noisy_group.query_start == gen_len + gen_start
        assert current_clean.isdisjoint(visible)
        assert current_noisy <= visible


def test_as_frame_timesteps_converts_vector_once():
    scalar = as_frame_timesteps(0.25, 4)
    assert scalar == [0.25, 0.25, 0.25, 0.25]
    per_frame = as_frame_timesteps(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]), 4)
    assert per_frame == pytest.approx([0.1, 0.2, 0.3, 0.4])
    broadcast = as_frame_timesteps(torch.tensor([0.7]), 3)
    assert broadcast == pytest.approx([0.7, 0.7, 0.7])
    assert as_frame_timesteps(1.0, 0) == []


def test_as_frame_timesteps_rejects_short_vector():
    with pytest.raises(ValueError, match="per-frame timestep"):
        as_frame_timesteps(torch.tensor([0.1, 0.2]), 3)


def test_remap_indexes_matches_python_dict_on_interleaved_layout():
    layout = build_teacher_forcing_layout(
        und_token_counts=[4, 3],
        vision_token_shapes=[(6, 2, 2), (5, 2, 2)],
        block_size=2,
        history_blocks=2,
        action_token_counts=[8, 6],
        temporal_compression_factor=2,
    )
    source_indexes = torch.arange(layout.original_sample_lens[0] + layout.original_sample_lens[1], dtype=torch.long)
    for stream in (TeacherForcingStream.UND, TeacherForcingStream.NOISY):
        table = _build_source_to_stream_index(layout, stream)
        stream_indexes = torch.nonzero(layout.stream_ids == int(stream), as_tuple=True)[0]
        expected = {
            int(layout.source_sequence_indexes[new_index]): int(new_index) for new_index in stream_indexes.tolist()
        }
        present = torch.tensor(sorted(expected), dtype=torch.long)
        remapped = _remap_indexes(present, table, f"{stream.name}.sequence_indexes")
        assert remapped is not None
        assert remapped.tolist() == [expected[int(index)] for index in present.tolist()]
        missing = source_indexes[torch.isin(source_indexes, present, invert=True)]
        if missing.numel() == 0:
            continue
        with pytest.raises(ValueError, match="outside the supported source stream"):
            _remap_indexes(missing[:1], table, f"{stream.name}.sequence_indexes")
