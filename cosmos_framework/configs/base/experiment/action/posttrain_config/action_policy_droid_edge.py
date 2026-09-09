# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_droid_edge`` — Cosmos3-Edge DROID action-policy SFT recipe.

Edge-tier sibling of ``action_policy_droid_nano``: same DROID dataflow
(``DROIDLeRobotDataset``, ``joint_pos`` 8D + ``use_state``, raw/un-normalized,
    concat_view 256p, whole-episode ``chunk_length=-1``, JSON action prompts), same optimizer /
scheduler / trainer / checkpoint blocks, but the model baseline is
``EDGE_MODEL_CONFIG`` (Nemotron-2B-Dense-VL dense backbone) instead of
``NANO_MODEL_CONFIG``. ``EDGE_MODEL_CONFIG`` already ships ``action_gen=True``
and the video-style loss scales (``loss_scale=10.0``, ``image_loss_scale=None``),
so only the action-recipe deltas are applied here. ``DROID_ROOT`` is the
versioned parent (e.g. ``.../droid_plus_lerobot_640x360_20260412``);
``use_success_only=True`` keeps the success split.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_droid_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_droid_edge_model_config() -> dict:
    """DROID model config on the Edge baseline: selective activation
    checkpointing, fresh diffusion-expert init. Keep
    ``encode_exact_durations=[17, 61, 73]`` as VAE warmup shapes; whole-episode
    4N+1 lengths not in the list fall back to eager exact encode."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)  # action_gen=True, max_action_dim=64
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    # Edge baseline already sets rectified_flow loss_scale=10.0 / image_loss_scale=None.
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    # Action SFT does not keep a second fp32 net_ema; live weights are the checkpoint.
    cfg["ema"]["enabled"] = False
    cfg["resolution"] = "256"
    # Sample block size in [1, 4] and history window in [1, 32] each forward.
    cfg["teacher_forcing_block_size_min"] = 1
    cfg["teacher_forcing_block_size_max"] = 4
    cfg["teacher_forcing_history_blocks_min"] = 1
    cfg["teacher_forcing_history_blocks_max"] = 32
    return cfg


action_policy_droid_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_droid_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_policy_droid_edge_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads. Edge extra vs nano:
            # k_norm_und_for_gen (und-K norm trains with the gen pathway, as in vision_sft_edge).
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "k_norm_und_for_gen",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,  # DROID nano reference (gbs 8192); local TOML lowers this
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML)
            f_max=[0.4],  # matches the DROID nano recipe
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke (real / local run sets via TOML)
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,  # real run sets via TOML
            logging_iter=1,
            max_iter=100,  # smoke
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema (EMA warm-starts from net, see dcp.py) and the action
            # heads, so they init fresh from the base (the base action heads are
            # not DROID-trained).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_droid",
            # Token-pack whole episodes. Mutually exclusive with a count cap;
            # a TOML max_samples_per_batch override must null this field.
            max_samples_per_batch=None,
            # Pre-expansion token cap (UND+vis+action). Index drop uses
            # max_pre_tf_tokens=48000 (~2257 frames at 256p with extra=500).
            # Keep packing lookahead tiny so a near-full batch does not
            # buffer extra decoded whole-episode videos (default is 10).
            max_sequence_length=48000,
            lookahead_limit=1,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=1,  # 1 queued decoded episode per worker; 2 held ~1 TiB host RSS
                sampler=None,
                # Shuffling is handled by the dataset (iterable_shuffle=True below):
                # ActionIterableShuffleDataset streams rank x worker-sharded, episode-order-
                # shuffled, sequential-within-episode.
                datasets=dict(
                    droid=dict(
                        ratio=1,
                        dataset=L(get_action_droid_sft_dataset)(
                            # Versioned DROID LeRobot parent (basename must be a LEROBOT_ROOTS
                            # key). use_success_only=True keeps success/. Example:
                            #   DROID_ROOT=.../droid_plus_lerobot_640x360_20260412
                            root="${oc.env:DROID_ROOT}",
                            fps=15.0,
                            # -1 = whole-episode: one sample per episode, all frames from 0,
                            # no max_episode_blocks cap (4N+1 tail padding only).
                            chunk_length=-1,
                            max_episode_blocks=-1,
                            # Drop (do not truncate) episodes whose 4N+1 padded
                            # length would exceed the 8-card 256p HBM budget
                            # (~2257 frames / 48000 pre-TF tokens, extra=500).
                            # Index-only; workers never decode the dropped videos.
                            max_pre_tf_tokens=48000,
                            action_space="joint_pos",
                            # Policy-only task mode. "joint" would randomly pick
                            # forward_dynamics/inverse_dynamics/policy per sample (multi-task),
                            # which dilutes each per-task loss by ~1/3.
                            mode="policy",
                            use_state=True,
                            iterable_shuffle=True,  # rank x worker episode-shuffle stream
                            episode_shuffle_seed=42,
                            # SR boost: random crop+rescale + ColorJitter in _compose_multi_view.
                            # Off for whole-episode: ColorJitter/interpolate copies dominate host RSS.
                            use_image_augmentation=False,
                            # keep_ranges filter. Off by default (no JSON required). When True in
                            # whole-episode mode, each original episode's valid [start,end) segments
                            # are concatenated into one sample (empty / <2-frame dropped at index
                            # build). Windowed mode still keeps per-range sliding windows.
                            use_filter_dict=False,
                            filter_dict_path=None,
                            action_normalization=None,
                            viewpoint="concat_view",  # wrist (top) + L/R shoulder 1/2 (bottom) → 256p 4:3
                            resolution="256",  # 640x360 data @ 256p (320x256)
                            # Close torchcodec/FFmpeg after each episode. Packed-mp4 AV1
                            # decoder state is large; a 64-slot LRU does not help unique
                            # whole-episode streams and dominates host RSS.
                            video_decoder_cache_size=0,
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            format_prompt_as_json=True,
                            use_success_only=True,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_droid_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
