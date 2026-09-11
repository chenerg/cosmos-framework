---
name: cosmos3-post-training
description: >
  Guide users through Cosmos3 supervised fine-tuning (SFT) post-training:
  preparing the example dataset and Wan2.2 VAE, converting the base
  checkpoint to DCP, launching distributed training (paired launch shell
  recommended, raw `torchrun` as an alternative), running T2V/I2V/V2V
  inference with the trained DCP checkpoint, and optionally exporting it to
  Hugging Face safetensors. Also covers action-policy SFT (DROID / LIBERO /
  RoboTwin): recipe TOMLs, PackingDataLoader batch sizing
  (`max_samples_per_batch` vs `max_sequence_length`), whole-episode vs
  windowed fetching, and causal teacher forcing. Use when the user asks how
  to post-train Cosmos3, fine-tune on a custom video dataset, train a robot
  action policy, export a trained checkpoint, or invoke one of the recipe
  launch shells (`launch_sft_vision_nano.sh`, `launch_sft_action_policy_droid_nano.sh`,
  `launch_sft_action_policy_libero_10_nano.sh`, `launch_sft_llava_ov.sh`,
  `launch_sft_videophy2_nano.sh`, plus the `_super` LoRA variant)
  — or any question about `cu130-train` / `cu128-train`,
  `convert_model_to_dcp` / `export_model` / `train`,
  `max_samples_per_batch` / `max_sequence_length`,
  or SFT output paths. For dataset captioning / JSONL assembly, see
  `docs/dataset_jsonl.md`. For action dataset internals, see
  `cosmos3-action-dataset`.
---

# Cosmos3 Post-Training (SFT)

## When to use this skill

- User wants to fine-tune Cosmos3-Nano (or Cosmos3-Super via LoRA) on the example Bridge video dataset or a custom video dataset (SFT)
- User wants to SFT a robot action policy (DROID / LIBERO / RoboTwin), including whole-episode or windowed fetching and causal teacher forcing
- User asks which fields in a recipe TOML to override (`[model.parallelism].data_parallel_shard_degree`, `[dataloader_train].max_samples_per_batch`, `[dataloader_train].max_sequence_length`, `[optimizer].lr`, `[trainer].max_iter`, `[checkpoint].load_path`, ...) or which experiment SKU to pick
- User wants to convert a base Hugging Face checkpoint to DCP, or convert a trained DCP back to safetensors
- For installation, `--group=cu130-train` / `cu128-train`, or LD_LIBRARY_PATH issues, hand off to **cosmos3-setup**
- For inference parameters, parallelism presets, or online serving, hand off to **cosmos3-inference**
- For adding / debugging an action dataset class, hand off to **cosmos3-action-dataset**
- For raw-video captioning or assembling a SFT JSONL, see `docs/dataset_jsonl.md` (the captioning flow has moved out of `docs/training.md`)

## Path convention

All paths below are relative to the cosmos3 package root (`../../../` from this skill file). All `uv run` / `python` / `torchrun` / `bash` commands should also be run from there.

## Where to find answers

The canonical reference is `docs/training.md`. Use this table to route questions:

| User question                                                               | Go to                                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Full step-by-step SFT workflow                                              | `docs/training.md`                                                              |
| Which install group? (`cu130-train` vs `cu128-train`)                       | `docs/setup.md` § CUDA Variants                                                 |
| Which recipes exist? (Vision SFT / Reasoner / action policy)                | `docs/training.md` § Step 1; action: `docs/action_policy_droid_posttrain.md`, `docs/action_policy_libero_posttrain.md` |
| How do I size the training batch (count vs tokens)?                         | this skill § PackingDataLoader batch sizing                                     |
| How do I download the example dataset / Wan VAE?                            | `docs/training.md` § Step 1 - Prepare data and config                           |
| How do I convert a base HF checkpoint to DCP?                               | `docs/training.md` § Step 2 — Prepare checkpoint                                |
| How do I launch training (paired shell, recommended)?                       | `docs/training.md` § Step 3 → Generator Post-Training → Option A                |
| How do I launch training with raw `torchrun`?                               | `docs/training.md` § Step 3 → Generator Post-Training → Option B                |
| How do I override `DATASET_PATH` / `BASE_CHECKPOINT_PATH` / `WAN_VAE_PATH`? | `docs/training.md` § Step 3 → Generator Post-Training → Overriding the defaults |
| Which TOML keys are commonly tuned?                                         | `docs/training.md` § Config                                                     |
| LoRA knobs (`lora_enabled` / `lora_rank` / `lora_alpha`)                    | `docs/training.md` § Config — `[model]` block (VFM only)                        |
| How do I validate the config without actually training?                     | `cosmos_framework/scripts/train.py` `--dryrun` flag (not in docs)               |
| How do I export the trained DCP back to safetensors?                        | `docs/training.md` § Export checkpoint to Hugging Face safetensors              |
| How do I run inference with the trained checkpoint?                         | `cosmos3-inference` skill (point at `$RUN_DIR/checkpoints/iter_<N>`)            |
| Where do training artifacts land?                                           | `docs/training.md` § Outputs                                                    |
| How do I caption raw videos / build a SFT JSONL?                            | `docs/dataset_jsonl.md`                                                         |

## Workflow at a glance

1. **Setup** — install the training extras: `uv sync --all-extras --group=cu130-train` (or `cu128-train` on older drivers), then `source .venv/bin/activate && export LD_LIBRARY_PATH=`.
2. **Step 1 - Prepare data and config** — for the recipe you're running, download the HF dataset to `examples/data/<dataset>/` and the Wan2.2 VAE to `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth` via `uvx hf@latest download …`. The Reasoner recipe streams its dataset from HF Hub at startup — no Step 1 download needed.
3. **Step 2 — Prepare checkpoint** — set `BASE_CHECKPOINT_NAME` (`Cosmos3-Nano` or `Cosmos3-Super`, matching the recipe) and run `python -m cosmos_framework.scripts.convert_model_to_dcp -o examples/checkpoints/$BASE_CHECKPOINT_NAME --checkpoint-path $BASE_CHECKPOINT_NAME`. Skip for the Reasoner recipe (the Qwen3-VL backbone is fetched from HF Hub at startup).
4. **Step 3 — Run training (Option A, recommended)** — from the repo root, `bash examples/launch_sft_<recipe>.sh` (e.g. `launch_sft_vision_nano.sh`). The launcher resolves `DATASET_PATH`, `BASE_CHECKPOINT_PATH`, `WAN_VAE_PATH` from the default `examples/` locations populated by Steps 1+2; export any of them in the shell first to override.
5. **Step 3 — Run training (Option B, raw `torchrun`)** — export the env vars yourself, then `IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml`. Unlike Option A, raw `torchrun` does NOT auto-resolve the env-var trio from `examples/` — they must come from the shell, or you must hand-edit the TOML to inline literal paths.
6. **Outputs** — `$RUN_DIR = $IMAGINAIRE_OUTPUT_ROOT/<job.project>/<job.group>/<job.name>`. DCP checkpoints land under `$RUN_DIR/checkpoints/iter_<N>/`; the latest iter name is in `$RUN_DIR/checkpoints/latest_checkpoint.txt`. `$RUN_DIR/config.yaml` next to the checkpoints is what inference consumes.
7. **Inference** — point `cosmos_framework.scripts.inference` at `$RUN_DIR/checkpoints/iter_<N>` together with `--config-file $RUN_DIR/config.yaml` (see `cosmos3-inference` skill for presets / input formats).
8. **Export (optional)** — `python -m cosmos_framework.scripts.export_model --checkpoint-path $RUN_DIR/checkpoints/$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt) --config-file $RUN_DIR/config.yaml -o $RUN_DIR/model` writes a portable HF safetensors checkpoint to `$RUN_DIR/model`.

## Things not obvious from the docs

- **Training extras are a separate group**: SFT requires the `cu130-train` / `cu128-train` install group, not the inference-only `cu130` / `cu128`. Re-running `uv sync` with the wrong group silently leaves training deps uninstalled.
- **Recipe = paired `examples/launch_sft_<r>.sh` + `examples/toml/sft_config/<r>.toml`**: the `.sh` declares `TOML_FILE` directly (full repo-relative path) plus `: "${DATASET_PATH:=…}"` / `: "${BASE_CHECKPOINT_PATH:=…}"` defaults that line up with where Steps 1+2 land, then sources `examples/_sft_launcher_common.sh`. `export`ed values in the user's shell win over the defaults. The helper forwards into `cosmos_framework.scripts.train --sft-toml=$TOML_FILE`, with any `TAIL_OVERRIDES` bash-array entries appended after `--` as Hydra-style `key.path=value` overrides (applied last on top of the pydantic-validated TOML schema in `cosmos_framework/configs/toml_config/sft_config.py`). `MASTER_PORT` defaults to `50012` in the helper; set it in the launcher (or `export`) only if you need to co-launch multiple jobs on one node.
- **Option A (paired launch shell) vs Option B (raw `torchrun`)**: Option A resolves the env-var trio (`DATASET_PATH` / `BASE_CHECKPOINT_PATH` / `WAN_VAE_PATH`) from `examples/` defaults so unset env runs out of the box. Option B requires you to either export those env vars yourself or hand-edit the TOML to inline the paths; the TOMLs use `${ENV:DATASET_PATH}` interpolation that's resolved at TOML load time.
- **`IMAGINAIRE_OUTPUT_ROOT` controls the entire output tree**: setting `IMAGINAIRE_OUTPUT_ROOT=outputs/train` makes everything land under `outputs/train/<job.project>/<job.group>/<job.name>/` (logs, `config.yaml`, `checkpoints/iter_<N>`, callback outputs). Unset, training falls back to `/tmp/imaginaire4-output/...`.
- **W&B is disabled by default**: every recipe TOML sets `[job].wandb_mode = "disabled"`. To log to W&B, flip it to `"online"` in the TOML and export `WANDB_API_KEY` before launching.
- **Inference uses the DCP checkpoint directly**: the standard flow points `cosmos_framework.scripts.inference` at `$RUN_DIR/checkpoints/iter_<N>` together with `--config-file $RUN_DIR/config.yaml`. The Hugging Face safetensors export (`$RUN_DIR/model`) is optional — only needed if you want a portable single-file checkpoint.
- **Parallelism degree must match topology**: in the TOML `[model.parallelism]` block, `data_parallel_shard_degree × data_parallel_replicate_degree × context_parallel_shard_degree` must equal `WORLD_SIZE`. `-1` autoselects `data_parallel_shard_degree` from torchrun world size. Mismatch → FSDP init failure.
- **`--dryrun`**: `cosmos_framework.scripts.train` accepts `--dryrun` to validate the config end-to-end without launching training. Use it whenever iterating on TOML keys or Hydra overrides.

## Action-policy SFT

Robot policy SFT is still `[job].task = "vfm"` + `python -m cosmos_framework.scripts.train --sft-toml=...`. Dataset class internals (window vs whole-episode fetch, concat_view, keep_ranges) live in **cosmos3-action-dataset**. Reproduction write-ups: [`docs/action_policy_droid_posttrain.md`](../../../docs/action_policy_droid_posttrain.md), [`docs/action_policy_libero_posttrain.md`](../../../docs/action_policy_libero_posttrain.md). Cookbook launchers (when `examples/` is incomplete): `cosmos/cookbooks/cosmos3/generator/action/finetune/`.

### Recipes

| Recipe | Experiment | Launcher / TOML | Fetch | Batch default |
| --- | --- | --- | --- | --- |
| DROID Nano (paper) | `action_policy_droid_nano` | `examples/launch_sft_action_policy_droid_nano.sh` + `examples/toml/sft_config/action_policy_droid_nano.toml` | window `chunk_length=32` | `max_samples_per_batch=32` (TOML; recipe 128) |
| DROID Edge whole-episode | `action_policy_droid_edge` | cookbook `launch_sft_action_policy_droid_edge_local.sh` + `toml/sft_config/action_policy_droid_edge_local.toml` | `chunk_length=-1` | `max_samples_per_batch=1` |
| LIBERO-10 / all Nano | `action_policy_libero_nano` / `_all_nano` | `examples/launch_sft_action_policy_libero_*.sh` | window `chunk_length=16` | recipe `max_samples_per_batch=128` |
| LIBERO / RoboTwin Edge | `action_policy_libero_edge` / `action_policy_robotwin_edge` | cookbook `*_edge_local.sh` | `chunk_length=-1` | `max_samples_per_batch=1` |

`DROID_ROOT` is the **versioned parent** that contains `success/` (basename must match a `LEROBOT_ROOTS` key, e.g. `droid_plus_lerobot_640x360_20260412`), not `.../success` itself. `LIBERO_ROOT` is the suite dir (`.../libero_10`) or the four-suite parent for the all recipe.

Shared recipe knobs: `action_space=joint_pos` (DROID 8D) or `frame_wise_relative` (LIBERO), `use_state=True`, `mode=policy` (not `joint` multi-task), `format_prompt_as_json=True`, `viewpoint=concat_view`, `resolution=480`. Policy JSON omits clip `duration` and `actions[].time` so train/infer stay causal-safe; FD/ID keep the timeline. `teacher_forcing_dense_mode` selects the GEN kernel: `tnd` (default; packed varlen TND), `per_sample` (masked SDPA per packed sample), `global` (one dense masked SDPA; HBM-heavy on long packs).

### PackingDataLoader batch sizing

VFM action (and vision) recipes size each micro-batch with **exactly one** of `[dataloader_train].max_samples_per_batch` or `[dataloader_train].max_sequence_length`. The other must be null (`PackingDataLoader` asserts XOR).

| Knob | Caps | Behavior |
| --- | --- | --- |
| `max_samples_per_batch` | sample **count** per packed micro-batch | stop when the count is reached |
| `max_sequence_length` | **pre-expansion** tokens (`UND + vis + action`) | stop before adding a sample that would exceed; a sample already `>=` the cap is **discarded**, not truncated |

On VFM these TOML keys land on `PackingDataLoader` itself. (VLM remaps them to `dataloader_train.batcher.max_batch_size` / `max_tokens` — action recipes are VFM.)

`max_sequence_length` is **not** the teacher-forcing attention length. After Scheme-B expansion, packed ≈ `UND + 2×(vis+action)`. Size the TOML cap against the **pre-TF** count.

Global batch with a count cap: `max_samples_per_batch × WORLD_SIZE × grad_accum_iter`. Token packing makes the per-step sample count variable.

**Windowed** (`chunk_length` 16 or 32): pick either knob. The Nano DROID / LIBERO references use the count cap (short RGB windows, packing many samples is cheap on host RAM). Vision SFT typically omits the count cap and uses a token budget.

**Whole-episode** (`chunk_length=-1`): set `max_samples_per_batch=1` (and `max_sequence_length=None`) so each step is one full episode. Do not switch to a token cap and pack several decoded episodes — host RSS holds the uint8 videos (480p concat ~0.3–0.6 GiB each) × ranks × workers × prefetch. On this ~1.5 TiB node that OOM-killed `pt_data_worker` even with `prefetch_factor=1`.

Packing-time discard (`Discarding oversized sample` in the log) still **decodes** the video first. To skip episodes that would OOM **without opening the file**, set dataset `max_pre_tf_tokens` (converted to a 4N+1 frame cap via concat resolution) or `max_episode_length_frames`. That is an index drop, not truncation. `max_episode_blocks` **truncates** to `1+4N` frames (`-1` = no truncate).

DROID concat 480p (`K=391` vis tokens / latent, packing `extra=500`): `max_pre_tf_tokens=48000` → **477 frames**. 8-card Edge selective AC: 120 blocks (~481 frames / ~9.6e4 post-TF packed) trains; 128 blocks / unbounded long episodes OOM. Sweep numbers: [`docs/max_episode_blocks_npu0_3_dp2_dp4_iter_time.md`](../../../docs/max_episode_blocks_npu0_3_dp2_dp4_iter_time.md).

Whole-episode host-RAM extras (recipe Python / Hydra tail, not TOML schema): `dataloader_train.lookahead_limit` (Edge recipe `1`; default `10` caches extra decoded videos) and `dataloader_train.dataloader.prefetch_factor` (`2` in the Edge recipe; `1` if RSS is tight). `num_workers=4` is the 8-card 480p floor that stops data-wait; `ActionIterableShuffleDataset` wraps when `ranks × workers >` episode count.

Optional VAE pre-encode on `PackingDataLoader` (Hydra tail; default off). After packing, the rank process encodes uint8 video with the model's frozen `tokenizer_vision_gen` and yields `video_latents` so `get_data_and_condition` skips encode. Trainer calls `attach_vision_tokenizer` after `model.on_train_start`.

| Knob | Default | Meaning |
| --- | --- | --- |
| `dataloader_train.encode_vision_latents` | `false` | Encode packed videos before yield; drop pixels unless `keep_video_pixels` |
| `dataloader_train.encoded_prefetch_depth` | `0` | `0` = encode in `next()`. `>=1` pre-fills that many encoded batches on a producer thread so training can overlap the next encode |
| `dataloader_train.keep_video_pixels` | `false` | Keep uint8 `video` after encode (viz only) |

Do **not** construct a second VAE. On 64 GiB 910B3, `encoded_prefetch_depth=1` can OOM if next-batch pixels overlap backward; fall back to `0`. Sync encode moves `timer/encoding` into `timer/dataloader_train` and does not hide wall time. Probe: `run_bench_20step_tf_vae_pack.py`.

### Causal teacher forcing

Scheme-B causal training needs **both**:

1. TOML `[model] causal_training_strategy = "teacher_forcing"` (plus `teacher_forcing_block_size_{min,max}`, `teacher_forcing_history_blocks_{min,max}`, `teacher_forcing_dense_mode` = `tnd` | `per_sample` | `global`)
2. Hydra model group `model=mot_causal_fsdp` (or `mot_causal_ddp`). Recipe defaults are `mot_fsdp`; TOML alone is not enough. The local DROID Edge recipe defaults to `tnd`. Switch without editing the TOML: `EXTRA_TAIL_OVERRIDES="model.config.teacher_forcing_dense_mode=per_sample"`.

The local Edge launcher already appends `model=mot_causal_fsdp`. S/K sample Uniform unless min=max is pinned.

### This Ascend node (Edge whole-episode)

Use the conda env in repo-root [`AGENTS.md`](../../../AGENTS.md) (no `uv`; keep CANN `LD_LIBRARY_PATH`; multi-card `HCCL_OP_EXPANSION_MODE=AIV`). Local paths and the launcher:

```bash
# from cosmos/cookbooks/cosmos3/generator/action/finetune/
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  HCCL_OP_EXPANSION_MODE=AIV \
  bash launch_sft_action_policy_droid_edge_local.sh
```

Override packing / length without editing the TOML (`EXTRA_TAIL_OVERRIDES`, space-separated):

```bash
# count-capped whole episode (recommended)
EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=1"

# token-capped packing (windowed / vision). Mutually exclusive with the count cap.
EXTRA_TAIL_OVERRIDES="dataloader_train.max_sequence_length=48000 dataloader_train.max_samples_per_batch=null"

# drop long episodes at index (no decode); do not use max_episode_blocks to "skip"
EXTRA_TAIL_OVERRIDES="dataloader_train.dataloader.datasets.droid.dataset.max_pre_tf_tokens=48000"
```

Local weights: `DROID_ROOT=/data5T/Embodied-AI/datasets/droid_plus_lerobot_640x360_20260412`（symlink → `Cosmos3-DROID`；`info.json` **500** episodes / 146133 frames；`split_val_ratio=0.03` `split_seed=42` → train 485，whole-episode 丢掉 &lt;33 帧后 **484**。不要再用 9 月初扫测的 193）。Edge DCP + processor and Wan VAE under `/data5T/Embodied-AI/ckpts/` (see `AGENTS.md`).

8-card / teacher-forcing probes: one directory per run under [`experiments/`](../../../experiments/README.md). Do not leave metrics only in `bench_max_episode_blocks/` or a loose `docs/*.md`.

## Related skills

| Skill                                  | When to use                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| `../cosmos3-setup/SKILL.md`            | Initial install, CUDA variant selection, container/`LD_LIBRARY_PATH` setup   |
| `../cosmos3-inference/SKILL.md`        | Inference parameters, parallelism presets, input JSON format, online serving |
| `../cosmos3-action-dataset/SKILL.md`   | Action dataset classes, window vs whole-episode fetch, keep_ranges, padding  |
| `../cosmos3-codebase-nav/SKILL.md`     | Locating configs, scripts, and defaults inside the package                   |
| `../cosmos3-env-troubleshoot/SKILL.md` | Debugging environment / runtime errors during training                       |
