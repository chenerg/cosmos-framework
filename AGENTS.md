# AGENTS.md — Cosmos-Framework

Read this file first — it is the canonical map for navigating the Cosmos repository and stays up to date.

**Cosmos** is a framework for training and serving world foundation models. Everything lives in a single top-level `cosmos_framework/` Python package:

- **Training infrastructure** — top-level subpackages under `cosmos_framework/` (data, model, trainer, callbacks, checkpoint, …).
- **Inference infrastructure** — `cosmos_framework/inference/` (Diffusers / Transformers / vLLM-friendly inference core, online serving via Ray + Gradio).
- **Backend packages** — `packages/{diffusers,transformers,vllm}-cosmos3/` provide library-style shims that load Cosmos3 checkpoints into the respective ecosystems.
- **Entry-point scripts** — `cosmos_framework/scripts/` (`train.py`, `inference.py`, `export_model.py`, …) invoked as `python -m cosmos_framework.scripts.<name>`. Primary training entry point: `cosmos_framework.scripts.train` driven by a structured, pydantic-validated TOML interface (`--sft-toml=<recipe-toml>`); the schema lives at [`cosmos_framework/configs/toml_config/sft_config.py`](./cosmos_framework/configs/toml_config/sft_config.py) and the canonical recipe pattern is documented in [`examples/README.md`](./examples/README.md).

> All paths below are relative to the repository root (the directory containing `pyproject.toml`, the `cosmos_framework/` Python package, and `packages/`).

**Training-script reference:** `cosmos/` is a symlink to the sibling checkout [`../cosmos`](../cosmos) (NVIDIA Cosmos cookbooks). When a launch recipe is missing or incomplete under `examples/` / `docs/` here, look at the matching cookbook in [`cosmos/cookbooks/cosmos3/`](./cosmos/cookbooks/cosmos3/) — especially [`generator/action/finetune/`](./cosmos/cookbooks/cosmos3/generator/action/finetune/) for DROID / LIBERO / RoboTwin policy SFT. Do not copy those scripts blindly; adapt paths and the conda env from this file.

## Local Machine Resources (this dev server)

Datasets and model weights already exist locally under `/mi/data2T/Embodied-AI` — reference them directly instead of re-downloading:

| Resource | Local path |
| ------------------------------------ | ----------------------------------------------------------------------- |
| SFT dataset (BridgeData2 subset) | `/mi/data2T/Embodied-AI/datasets/BridgeData2-Subset-Synthetic-Captions` |
| Cosmos3-Edge HF checkpoint/processor | `/mi/data2T/Embodied-AI/ckpts/Cosmos/Cosmos3-Edge` |
| Cosmos3-Edge DCP checkpoint | `/mi/data2T/Embodied-AI/ckpts/Cosmos/Cosmos3-Edge-DCP` |
| Cosmos3-Nano HF checkpoint | `/mi/data2T/Embodied-AI/ckpts/Cosmos/Cosmos3-Nano` |
| Wan2.2 VAE | `/mi/data2T/Embodied-AI/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |

More checkpoints (lerobot, openpi-assets, VGGT-Omega, …) live in `/mi/data2T/Embodied-AI/ckpts/`; more datasets (libero, Robotwin, …) in `/mi/data2T/Embodied-AI/datasets/`.

## Commands

Do NOT use `uv` on this dev server. Run everything (lint, type-check, tests, scripts) with the conda env at `/workspace/chenzhi/miniconda3/envs/cosmos-framework` (invoke its binaries directly, e.g. `ENV=/workspace/chenzhi/miniconda3/envs/cosmos-framework; $ENV/bin/python ...`). Python 3.11 + `torch_npu==2.10.0.post4` matches host CANN 9.1.0; keep `LD_LIBRARY_PATH` from CANN `set_env.sh`.

| Task                   | Command                                                     |
| ---------------------- | ----------------------------------------------------------- |
| Lint                   | `$ENV/bin/ruff check .`                                     |
| Format check           | `$ENV/bin/ruff format --check .`                            |
| Auto-fix lint + format | `$ENV/bin/ruff check --fix . && $ENV/bin/ruff format .`     |
| Type-check             | `$ENV/bin/pyrefly check`                                    |
| Test (all)             | `$ENV/bin/pytest`                                           |
| Test (single file)     | `$ENV/bin/pytest --capture=no <path>`                       |

Config files: `.ruff.toml` (ruff), `pyrefly.toml` (pyrefly), `.pytest.toml` (pytest), `conftest.py` (pytest fixtures).

A `justfile` is provided at the root with longer recipes (`just install`, `just lint`, `just test`, `just docker-cu130`).

## Rules

- Always cite code as markdown links with the line in the URL fragment, e.g. `[optimizer.py:273](cosmos_framework/utils/generator/optimizer.py#L273)`. Do not put the line number only in the link label.
- When unsure, point the user to the closest doc rather than guessing.
- Keep this file short. Link out to skills and docs for detail — this file is included in every prompt.
- Inference code belongs under `cosmos_framework/inference/`; training infrastructure belongs under the other `cosmos_framework/` subpackages. Don't blur the two — if you find yourself adding training-time imports inside `cosmos_framework/inference/` (or vice versa), reconsider.

## Key File Locations

### Training (`cosmos_framework/`)

| What                                                 | Where                                           |
| ---------------------------------------------------- | ----------------------------------------------- |
| Algorithms (losses, RL, reward)                      | `cosmos_framework/algorithm/{loss,reward,rl}`   |
| Training loop                                        | `cosmos_framework/trainer/`                     |
| Models + parallelism                                 | `cosmos_framework/model/`                       |
| Datasets / data loading                              | `cosmos_framework/data/`                        |
| Checkpoint I/O                                       | `cosmos_framework/checkpoint/`                  |
| Callbacks (logging, eval)                            | `cosmos_framework/callbacks/`                   |
| RL workers (rollout, reward, reference, simulations) | `cosmos_framework/workers/`                     |
| Controller / orchestrator                            | `cosmos_framework/controller/`                  |
| Launchers (Slurm, torchrun, k8s)                     | `cosmos_framework/launcher/`                    |
| Evaluation harness                                   | `cosmos_framework/evaluation/`                  |
| CLI tools                                            | `cosmos_framework/tools/`, `tools/` (repo root) |

For a per-subpackage tour with descriptions, see [`docs/code_structure.md`](./docs/code_structure.md).

### Inference (`cosmos_framework/inference/`)

| What                     | Where                                                                            |
| ------------------------ | -------------------------------------------------------------------------------- |
| CLI entry point          | `cosmos_framework/scripts/inference.py`                                          |
| Args / param definitions | `cosmos_framework/inference/args.py`                                             |
| Per-modality defaults    | `cosmos_framework/inference/defaults/<mode>/sample_args.json`                    |
| Model / inference core   | `cosmos_framework/inference/model.py`, `cosmos_framework/inference/inference.py` |
| Ray serving              | `cosmos_framework/inference/ray/`                                                |
| Backend packages         | `packages/{diffusers,transformers,vllm}-cosmos3/`                                |
| Example inputs           | `inputs/omni/*.json`, `inputs/reasoner/*.json`                                   |

## Documentation

| Doc                                                | What it covers                                                    |
| -------------------------------------------------- | ----------------------------------------------------------------- |
| [docs/setup.md](./docs/setup.md)                   | Install, NGC base image, CUDA variants, base-checkpoint download. |
| [docs/code_structure.md](./docs/code_structure.md) | Repo layout and per-subpackage tour of `cosmos_framework/`.       |
| [docs/training.md](./docs/training.md)             | Single- and multi-node launches, parallelism, mixed precision.    |
| [docs/inference.md](./docs/inference.md)           | Sample arguments, parallelism, schemas, troubleshooting.          |
| [docs/faq.md](./docs/faq.md)                       | Troubleshooting (OOM, NCCL, slow training) + env vars.            |
| [experiments/](./experiments/README.md)            | One directory per local training probe (time / loss / HBM / RSS). |

Agent skills (codebase navigation, env troubleshooting, inference, post-training, setup) live in [`.agents/skills/`](./.agents/skills) and [`.claude/skills/`](./.claude/skills).

## Common Tasks

### Training

| Task                     | Command                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Single-GPU train (smoke) | `python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml`                                                           |
| Multi-GPU train          | `IMAGINAIRE_OUTPUT_ROOT=outputs/train torchrun --nproc-per-node=8 -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml` |
| Resume from checkpoint   | Re-run the same `train --sft-toml=<recipe>.toml` against the same `IMAGINAIRE_OUTPUT_ROOT` (auto-resume from latest DCP).                              |
| Export DCP → HF          | `python -m cosmos_framework.scripts.export_model --src <dcp> --dst <hf>`                                                                               |
| Run a config sweep       | `just run python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml -- key.path=value ...`                            |
| Cookbook launch scripts  | Sibling repo via `cosmos/` → `../cosmos`: `cosmos/cookbooks/cosmos3/{generator,reasoner}/**/finetune/launch_sft_*.sh`                                 |

### Inference

| Task                    | Command                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Single-GPU inference    | `python -m cosmos_framework.scripts.inference -i inputs/omni/t2v.json -o outputs/ --checkpoint-path Cosmos3-Nano`                 |
| Multi-GPU inference     | `torchrun --nproc-per-node=4 -m cosmos_framework.scripts.inference --parallelism-preset=latency -i ... -o outputs/ ...`           |
| Start online Ray server | `python -m cosmos_framework.inference.ray.serve --parallelism-preset=latency -o outputs/ray_serve --checkpoint-path Cosmos3-Nano` |
| Launch Gradio UI        | `python -m cosmos_framework.inference.ray.gradio --port=8080`                                                                     |
| See all CLI flags       | `python -m cosmos_framework.scripts.inference --help`                                                                             |

## Gotchas

- **`LD_LIBRARY_PATH` — depends on the environment, do NOT blindly clear it**:
  - **NGC / PyTorch CUDA containers**: run `export LD_LIBRARY_PATH=''` before any `python` call or you'll hit a `torch._C` import error. See [`docs/setup.md`](./docs/setup.md#pytorch-import-issue).
  - **This dev server (Ascend NPU, aarch64)**: keep `LD_LIBRARY_PATH` as set by CANN's `set_env.sh` (points at `/usr/local/Ascend/ascend-toolkit/latest/lib64`). Clearing it breaks every `import torch`: torch auto-loads the `torch_npu` backend via the `torch.backends` entry point, and `torch_npu._C` needs `libhccl.so` from that path. For NPU-free work (e.g. data-only scripts), bypass with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` instead of touching `LD_LIBRARY_PATH`.
- **HF downloads on this dev server**: always pull from **hf-mirror, with the local proxy unset**, and disable Xet:
  ```
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_XET=1
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  ```
  then `hf download` / `snapshot_download`. `huggingface.co` times out without the proxy (`127.0.0.1:17890`); the proxy + Xet path stalls and can leave sparse, corrupt files that size-only resume checks treat as complete. hf-mirror + plain HTTP is the working combination (speed is similar with/without proxy; official Hub is unreachable direct). Verify with `du -sh` (real blocks), not `ls -lh` (apparent size).
- **Reproducibility**: always pass `--seed <int>`. Without it a random seed is used each run.
- **JSON paths**: relative paths inside input JSON files resolve relative to the JSON file's directory, not the working directory.
- **Resume**: re-running the same inference command skips already-generated outputs automatically.
- **Separation of concerns**: keep training-time imports out of `cosmos_framework/inference/`, and keep heavyweight inference-only deps (vLLM, Ray Serve, Gradio) gated behind optional extras so plain training installs stay slim.
