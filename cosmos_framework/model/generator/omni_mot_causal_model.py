# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scheme-B single-forward causal teacher-forcing model."""

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.model.generator.causal_teacher_forcing import (
    expand_teacher_forcing_training_sequence,
    validate_teacher_forcing_config,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class OmniMoTCausalModel(OmniMoTModel):
    """Vision (+ optional action) causal model using clean/noisy streams in one forward."""

    def __init__(self, config: OmniMoTModelConfig):
        validate_teacher_forcing_config(config)
        super().__init__(config)

    def post_noise_packing_hook(
        self,
        packed_sequence: PackedSequence,
        gen_data_clean: GenerationDataClean,
    ) -> PackedSequence:
        """Attach clean ``x0`` after the ordinary path has installed noisy ``xt``."""

        if gen_data_clean.x0_tokens_vision is None:
            raise ValueError("teacher-forcing causal training requires clean vision tokens")
        clean_vision_tokens = [token.to(dtype=self.precision) for token in gen_data_clean.x0_tokens_vision]

        clean_action_tokens = None
        temporal_compression_factor = None
        if packed_sequence.action is not None:
            if gen_data_clean.x0_tokens_action is None:
                raise ValueError("teacher-forcing causal training requires clean action tokens when action is packed")
            # Match each clean payload's dtype to its noisy counterpart so the
            # expansion's clean/noisy payload checks hold.
            clean_action_tokens = [
                token.to(dtype=noisy_token.dtype)
                for token, noisy_token in zip(gen_data_clean.x0_tokens_action, packed_sequence.action.tokens)
            ]
            assert self.tokenizer_vision_gen is not None
            temporal_compression_factor = self.tokenizer_vision_gen.temporal_compression_factor

        return expand_teacher_forcing_training_sequence(
            packed_sequence,
            clean_vision_tokens=clean_vision_tokens,
            config=self.config,
            clean_action_tokens=clean_action_tokens,
            temporal_compression_factor=temporal_compression_factor,
        )
