# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.causal_teacher_forcing import (
    expand_teacher_forcing_training_sequence,
    validate_teacher_forcing_config,
)
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def _config(**overrides):
    values = dict(
        causal_training_strategy="teacher_forcing",
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        video_temporal_causal=False,
        teacher_forcing_block_size_min=2,
        teacher_forcing_block_size_max=2,
        teacher_forcing_history_blocks_min=3,
        teacher_forcing_history_blocks_max=3,
        teacher_forcing_dense_mode="global",
        joint_attn_implementation="teacher_forcing",
        parallelism=SimpleNamespace(context_parallel_shard_degree=1),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _packed_noisy_video() -> PackedSequence:
    noisy = torch.tensor([[[[[10.0]], [[20.0]], [[30.0]]]]])
    return PackedSequence(
        sample_lens=[4],
        split_lens=[1, 3],
        attn_modes=["causal", "full"],
        sequence_length=4,
        text_ids=torch.tensor([101]),
        text_indexes=torch.tensor([0]),
        position_ids=torch.arange(12).reshape(3, 4),
        vision=ModalityData(
            sequence_indexes=torch.tensor([1, 2, 3]),
            timesteps=torch.tensor([0.5, 0.5, 0.5]),
            mse_loss_indexes=torch.tensor([1, 2, 3]),
            token_shapes=[(3, 1, 1)],
            tokens=[noisy],
            condition_mask=[torch.zeros(3, 1, 1)],
            noisy_frame_indexes=[torch.tensor([0, 1, 2])],
        ),
    )


def _packed_noisy_video_action() -> PackedSequence:
    """One sample: [UND(1) | vision T=3 (1x1) | action A=5], offset-0 alignment."""

    noisy_vision = torch.tensor([[[[[10.0]], [[20.0]], [[30.0]]]]])
    noisy_action = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    position_ids = torch.zeros(3, 9, dtype=torch.long)
    # Temporal axis: vision latent frames [1,2,3]; action steps share the vision
    # start coordinate (action_start_frame_offset=0).
    position_ids[0] = torch.tensor([0, 1, 2, 3, 1, 2, 3, 4, 5])
    return PackedSequence(
        sample_lens=[9],
        split_lens=[1, 8],
        attn_modes=["causal", "full"],
        sequence_length=9,
        text_ids=torch.tensor([101]),
        text_indexes=torch.tensor([0]),
        position_ids=position_ids,
        vision=ModalityData(
            sequence_indexes=torch.tensor([1, 2, 3]),
            timesteps=torch.tensor([0.5, 0.5, 0.5]),
            mse_loss_indexes=torch.tensor([1, 2, 3]),
            token_shapes=[(3, 1, 1)],
            tokens=[noisy_vision],
            condition_mask=[torch.zeros(3, 1, 1)],
            noisy_frame_indexes=[torch.tensor([0, 1, 2])],
        ),
        action=ModalityData(
            sequence_indexes=torch.tensor([4, 5, 6, 7, 8]),
            timesteps=torch.tensor([0.5] * 5),
            mse_loss_indexes=torch.tensor([4, 5, 6, 7, 8]),
            token_shapes=[(5,)],
            tokens=[noisy_action],
            condition_mask=[torch.zeros(5, 1)],
            noisy_frame_indexes=[torch.tensor([0, 1, 2, 3, 4])],
        ),
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"causal_training_strategy": "none"}, "teacher_forcing"),
        ({"vision_gen": False}, "vision_gen"),
        ({"sound_gen": True}, "sound_gen"),
        ({"video_temporal_causal": True}, "video_temporal_causal"),
        ({"teacher_forcing_block_size_min": 0}, "block_size"),
        ({"teacher_forcing_block_size_min": 3, "teacher_forcing_block_size_max": 2}, "block_size"),
        ({"teacher_forcing_history_blocks_min": 0}, "history_blocks"),
        ({"teacher_forcing_history_blocks_min": 4, "teacher_forcing_history_blocks_max": 3}, "history_blocks"),
        ({"teacher_forcing_dense_mode": "invalid"}, "dense_mode"),
        ({"parallelism": SimpleNamespace(context_parallel_shard_degree=2)}, "context parallel"),
        ({"joint_attn_implementation": "two_way"}, "joint_attn_implementation"),
    ],
)
def test_validate_teacher_forcing_config_rejects_unsupported_settings(overrides, error: str):
    with pytest.raises(ValueError, match=error):
        validate_teacher_forcing_config(_config(**overrides))


def test_expand_teacher_forcing_training_sequence_uses_configured_batch_shared_geometry():
    packed = _packed_noisy_video()
    clean = [torch.tensor([[[[[1.0]], [[2.0]], [[3.0]]]]])]

    expanded = expand_teacher_forcing_training_sequence(
        packed,
        clean_vision_tokens=clean,
        config=_config(),
    )

    assert expanded.teacher_forcing is not None
    assert expanded.teacher_forcing.layout.geometry.block_sizes == (2,)
    assert expanded.teacher_forcing.layout.geometry.history_blocks == (3,)
    assert expanded.teacher_forcing.clean_vision_tokens == clean
    assert expanded.vision is not None
    assert expanded.vision.tokens == packed.vision.tokens
    assert expanded.vision.tokens[0].flatten().tolist() == [10.0, 20.0, 30.0]


def test_expand_teacher_forcing_training_sequence_rejects_images():
    packed = _packed_noisy_video()
    packed.is_image_batch = True

    with pytest.raises(ValueError, match="video"):
        expand_teacher_forcing_training_sequence(
            packed,
            clean_vision_tokens=[torch.zeros(1, 1, 3, 1, 1)],
            config=_config(),
        )


def test_expand_teacher_forcing_training_sequence_supports_vision_action():
    packed = _packed_noisy_video_action()
    clean_vision = [torch.tensor([[[[[1.0]], [[2.0]], [[3.0]]]]])]
    clean_action = [torch.zeros(5, 2)]

    expanded = expand_teacher_forcing_training_sequence(
        packed,
        clean_vision_tokens=clean_vision,
        config=_config(action_gen=True),
        clean_action_tokens=clean_action,
        temporal_compression_factor=2,
    )

    assert expanded.teacher_forcing is not None
    layout = expanded.teacher_forcing.layout
    # S=2: vision frame blocks [0 | 1 1]; action latent frames ceil(j/2)=[0,1,1,2,2]
    # give action blocks [0,1,1,1,1]. Interleaved GEN order per stream:
    # [V0 A0 | V1 V2 A1 A2 A3 A4].
    assert layout.block_ids.tolist() == [-1] + [0, 0, 1, 1, 1, 1, 1, 1] * 2
    assert layout.source_sequence_indexes.tolist() == [0] + [1, 4, 2, 3, 5, 6, 7, 8] * 2
    assert layout.clean_token_indexes.tolist() == [1, 3, 4]
    assert layout.clean_action_token_indexes.tolist() == [2, 5, 6, 7, 8]
    assert layout.noisy_output_indexes.tolist() == [9, 11, 12, 10, 13, 14, 15, 16]
    assert layout.noisy_action_output_indexes.tolist() == [10, 13, 14, 15, 16]
    assert expanded.vision is not None and expanded.action is not None
    assert expanded.vision.sequence_indexes.tolist() == [9, 11, 12]
    assert expanded.action.sequence_indexes.tolist() == [10, 13, 14, 15, 16]
    assert expanded.action.mse_loss_indexes.tolist() == [10, 13, 14, 15, 16]
    assert expanded.teacher_forcing.clean_action_tokens == clean_action
    assert torch.equal(
        expanded.position_ids[:, layout.clean_action_token_indexes],
        expanded.position_ids[:, layout.noisy_action_output_indexes],
    )


def test_expand_teacher_forcing_training_sequence_rejects_nonzero_action_offset():
    packed = _packed_noisy_video_action()
    packed.position_ids[0, 4] = 2  # first action step no longer aligned with vision frame 0

    with pytest.raises(ValueError, match="action_start_frame_offset"):
        expand_teacher_forcing_training_sequence(
            packed,
            clean_vision_tokens=[torch.zeros(1, 1, 3, 1, 1)],
            config=_config(action_gen=True),
            clean_action_tokens=[torch.zeros(5, 2)],
            temporal_compression_factor=2,
        )


def test_post_noise_packing_hook_supports_action_payloads():
    packed = _packed_noisy_video_action()
    assert packed.vision is not None and packed.action is not None
    packed.vision.tokens = [token.to(torch.bfloat16) for token in packed.vision.tokens]
    packed.action.tokens = [token.to(torch.bfloat16) for token in packed.action.tokens]
    clean_x0_vision = torch.tensor([[[[[1.0]], [[2.0]], [[3.0]]]]], dtype=torch.float32)
    clean_x0_action = torch.zeros(5, 2, dtype=torch.float32)
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[clean_x0_vision],
        x0_tokens_action=[clean_x0_action],
    )
    model = SimpleNamespace(
        config=_config(action_gen=True),
        precision=torch.bfloat16,
        tokenizer_vision_gen=SimpleNamespace(temporal_compression_factor=2),
    )

    expanded = OmniMoTCausalModel.post_noise_packing_hook(model, packed, gen_data_clean)

    assert expanded.teacher_forcing is not None
    assert expanded.teacher_forcing.clean_action_tokens is not None
    assert expanded.teacher_forcing.clean_action_tokens[0].dtype == torch.bfloat16
    assert gen_data_clean.x0_tokens_action[0] is clean_x0_action
    assert gen_data_clean.x0_tokens_action[0].dtype == torch.float32


def test_post_noise_packing_hook_requires_clean_action_tokens():
    packed = _packed_noisy_video_action()
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, 3, 1, 1)],
    )
    model = SimpleNamespace(
        config=_config(action_gen=True),
        precision=torch.bfloat16,
        tokenizer_vision_gen=SimpleNamespace(temporal_compression_factor=2),
    )

    with pytest.raises(ValueError, match="clean action tokens"):
        OmniMoTCausalModel.post_noise_packing_hook(model, packed, gen_data_clean)


def test_post_noise_packing_hook_casts_clean_model_input_without_mutating_fp32_x0():
    packed = _packed_noisy_video()
    assert packed.vision is not None
    packed.vision.tokens = [token.to(torch.bfloat16) for token in packed.vision.tokens]
    clean_x0 = torch.tensor([[[[[1.0]], [[2.0]], [[3.0]]]]], dtype=torch.float32)
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[clean_x0],
    )
    model = SimpleNamespace(config=_config(), precision=torch.bfloat16)

    expanded = OmniMoTCausalModel.post_noise_packing_hook(model, packed, gen_data_clean)

    assert expanded.teacher_forcing is not None
    assert expanded.teacher_forcing.clean_vision_tokens[0].dtype == torch.bfloat16
    assert gen_data_clean.x0_tokens_vision[0] is clean_x0
    assert gen_data_clean.x0_tokens_vision[0].dtype == torch.float32
