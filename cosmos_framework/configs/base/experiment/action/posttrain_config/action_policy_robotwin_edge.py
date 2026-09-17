# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_robotwin_edge`` — Cosmos3-Edge RoboTwin action-policy SFT recipe.

RoboTwin sibling of ``action_policy_libero_edge``: same Edge model baseline
(``EDGE_MODEL_CONFIG``, Nemotron-2B-Dense-VL backbone) and the same optimizer /
scheduler / trainer / checkpoint blocks, but feeds
``RoboTwinLeRobotDataset`` (dual-arm ALOHA, 14D ``joint_pos`` + ``use_state``,
raw/un-normalized actions, DROID-style concat_view: overhead ``cam_high`` on
top, two wrist cameras below) through ``get_action_robotwin_sft_dataset``.
``ROBOTWIN_ROOT`` points at a single RoboTwin-LeRobot-v3.0 task dataset dir
(e.g. ``.../RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50``).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robotwin_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_robotwin_edge_model_config() -> dict:
    """RoboTwin model config on the Edge baseline: selective activation
    checkpointing, fresh diffusion-expert init. Keep
    ``encode_exact_durations=[17, 61, 73]`` to match the Cosmos3 base;
    whole-episode 4N+1 lengths not in the list fall back to eager exact encode."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)  # action_gen=True, max_action_dim=64
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    # Edge baseline already sets rectified_flow loss_scale=10.0 / image_loss_scale=None.
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    return cfg


action_policy_robotwin_edge = LazyDict(
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
            name="action_policy_robotwin_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_policy_robotwin_edge_model_config(),
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
            lr=5.0e-05,
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
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke (real run sets via TOML)
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
            # not RoboTwin-trained).
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
            dataset_name="action_robotwin",
            max_samples_per_batch=1,  # one unlimited whole-episode per micro-batch; override via TOML
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=8,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                # Shuffling is handled by the dataset (iterable_shuffle=True below):
                # ActionIterableShuffleDataset streams rank x worker-sharded, episode-order-
                # shuffled, sequential-within-episode.
                datasets=dict(
                    robotwin=dict(
                        ratio=1,
                        dataset=L(get_action_robotwin_sft_dataset)(
                            # Single RoboTwin-LeRobot-v3.0 task dataset dir (contains meta/info.json), e.g.
                            #   ROBOTWIN_ROOT=.../RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50
                            root="${oc.env:ROBOTWIN_ROOT}",
                            fps=30.0,  # native 30 FPS grid (no temporal subsampling)
                            # -1 = whole-episode: one sample per episode, all frames/actions
                            # from frame 0, no length cap (4N+1 tail padding only).
                            chunk_length=-1,
                            max_episode_blocks=-1,
                            action_space="joint_pos",  # 14D dual-arm joints+grippers (RoboTwin has no EE pose)
                            mode="policy",
                            use_state=True,  # prepend the 14D initial joint state
                            action_normalization=None,  # raw joint values, un-normalized
                            viewpoint="concat_view",  # cam_high top + L/R wrist bottom -> 720x640
                            use_image_augmentation=False,  # sim data; enable via overrides if desired
                            split="train",
                            split_val_ratio=0.03,
                            iterable_shuffle=True,  # rank x worker episode-shuffle stream
                            episode_shuffle_seed=42,
                            resolution=None,  # auto-detect tier from 720x640 composite
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            format_prompt_as_json=True,  # structured JSON prompts (set False for plain-text)
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
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


for _item in [action_policy_robotwin_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
