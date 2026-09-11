# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import math
import queue
import threading
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Union

import numpy as np
import torch
import webdataset
from torch.utils.data.dataloader import default_collate

from cosmos_framework.model.generator.tokenizers.uniae.frame_math import (
    get_uniae_chunk_frames,
    get_uniae_latent_num_frames,
    normalize_uniae_chunk_frames,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.data_utils import read_positive_int_metadata
from cosmos_framework.utils.lazy_config import instantiate

_TIMING_KEYS = {"_sample_time", "_aug_time", "_pre_aug_time", "_aug_step_times"}
_BATCH_TIMING_KEYS = {
    "_worker_batch_time",
    "_worker_aug_time",
    "_worker_io_time",
    "_worker_aug_step_times",
    "_worker_id",
}

_VISION_LATENT_KEY = {"video": "video_latents", "images": "image_latents"}
_VISION_PIXEL_SHAPE_KEY = {"video": "video_pixel_shapes", "images": "image_pixel_shapes"}
_ENCODED_QUEUE_SENTINEL = object()


def _infer_tokenizer_device(tokenizer: Any) -> torch.device:
    """Device of the frozen vision tokenizer weights.

    Prefer ``parameters().device`` over ``WanVAE.device``. The latter is often
    the unindexed ``npu``/``cuda`` enum, which becomes ``npu:0`` and mismatches
    rank-local weights on ``npu:LOCAL_RANK``.
    """
    inner = getattr(tokenizer, "model", None)
    model = getattr(inner, "model", inner)
    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            return next(params()).device
        except StopIteration:
            pass
    device = getattr(inner, "device", None)
    if device is None:
        device = getattr(tokenizer, "device", None)
    if device is None:
        device = torch.device("cpu")
    return torch.device(device) if not isinstance(device, torch.device) else device


def _set_current_torch_device(device: torch.device) -> None:
    if device.type == "cpu":
        return
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)


def _maybe_encode_side_stream(device: torch.device):
    """VAE-encode stream so encode kernels do not occupy the default HCCL stream.

    Returns ``(stream, stream_context)``. CPU and missing stream APIs return
    ``(None, None)`` so tests and CPU dataloaders keep working.
    """
    if device.type == "cpu":
        return None, None
    if device.type == "npu" and hasattr(torch, "npu"):
        npu = torch.npu
        if (
            callable(getattr(npu, "is_available", None))
            and npu.is_available()
            and hasattr(npu, "Stream")
            and hasattr(npu, "stream")
        ):
            return npu.Stream(device=device), npu.stream
        return None, None
    if device.type == "cuda" and torch.cuda.is_available() and hasattr(torch.cuda, "Stream"):
        return torch.cuda.Stream(device=device), torch.cuda.stream
    return None, None


def _latents_to_cpu(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_latents_to_cpu(item) for item in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    return obj


def _as_bcthw(item: torch.Tensor) -> torch.Tensor:
    if item.ndim == 4:
        return item.unsqueeze(0)
    if item.ndim == 5:
        return item
    raise ValueError(f"Vision tensor must be [C,T,H,W] or [B,C,T,H,W], got shape {tuple(item.shape)}")


def _unwrap_vision_sample(item: Any) -> list[torch.Tensor]:
    if isinstance(item, torch.Tensor):
        return [item]
    if isinstance(item, (list, tuple)):
        tensors = [elem for elem in item if elem is not None]
        if not tensors:
            raise ValueError("Vision sample is empty")
        return tensors
    raise TypeError(f"Unsupported vision sample type: {type(item)}")


def encode_packed_vision_batch(
    batch: dict,
    tokenizer: Any,
    *,
    keep_video_pixels: bool = False,
    device: torch.device | None = None,
) -> dict:
    """VAE-encode packed ``video`` / ``images`` in-place and optionally drop pixels.

    Writes ``video_latents`` (or ``image_latents``), ``video_pixel_shapes``, and
    ``vae_latents_ready=True``. Latents are moved to CPU so a prefetch queue does
    not pin extra device memory across training steps.
    """
    media_key = "images" if "images" in batch else "video" if "video" in batch else None
    if media_key is None:
        return batch

    items = batch[media_key]
    if not isinstance(items, list):
        items = [items]

    encode_device = device if device is not None else _infer_tokenizer_device(tokenizer)
    _set_current_torch_device(encode_device)
    encode_stream, encode_stream_ctx = _maybe_encode_side_stream(encode_device)
    per_camera = "enable_per_camera_vae_encoding" in batch
    sample_n_views = None
    frames_per_view = None
    if per_camera:
        sample_n_views = read_positive_int_metadata(batch, "sample_n_views", expected_count=len(items))
        frames_per_view = read_positive_int_metadata(
            batch,
            "num_video_frames_per_view",
            expected_count=len(items),
        )
        if sample_n_views is None or frames_per_view is None:
            raise ValueError(
                "sample_n_views and num_video_frames_per_view are required when per-camera VAE encoding is enabled."
            )

    latents: list[Any] = []
    shapes: list[tuple[int, int, int]] = []

    def _encode_clip(video: torch.Tensor) -> torch.Tensor:
        video_5d = _as_bcthw(video)
        num_pixel_frames = int(video_5d.shape[2])
        height = int(video_5d.shape[-2])
        width = int(video_5d.shape[-1])
        shapes.append((num_pixel_frames, height, width))
        if video_5d.dtype == torch.uint8:
            video_5d = video_5d.to(device=encode_device, dtype=torch.float32) / 127.5 - 1.0
        else:
            video_5d = video_5d.to(device=encode_device, dtype=torch.float32)
        encoded = tokenizer.encode(video_5d)
        return encoded.contiguous().float()

    encode_cm = encode_stream_ctx(encode_stream) if encode_stream is not None else contextlib.nullcontext()
    with torch.no_grad(), encode_cm:
        for sample_idx, item in enumerate(items):
            clips = _unwrap_vision_sample(item)
            encoded_clips: list[torch.Tensor] = []
            n_views = 1 if sample_n_views is None else int(sample_n_views[sample_idx])
            view_frames = None if frames_per_view is None else int(frames_per_view[sample_idx])
            for clip in clips:
                if n_views > 1:
                    if view_frames is None:
                        raise ValueError("num_video_frames_per_view is required when sample_n_views > 1")
                    video_5d = _as_bcthw(clip)
                    expected = n_views * view_frames
                    actual = int(video_5d.shape[2])
                    if actual != expected:
                        raise ValueError(
                            "Multiview vision length must equal sample_n_views * num_video_frames_per_view: "
                            f"got T={actual}, sample_n_views={n_views}, num_video_frames_per_view={view_frames}."
                        )
                    view_latents = []
                    for view_idx in range(n_views):
                        view_clip = video_5d.narrow(2, view_idx * view_frames, view_frames)
                        view_latents.append(_encode_clip(view_clip))
                    encoded_clips.append(torch.cat(view_latents, dim=2))
                else:
                    encoded_clips.append(_encode_clip(clip))
            if isinstance(item, list) and len(item) > 1:
                latents.append(encoded_clips)
            else:
                latents.append(encoded_clips[0])
    if encode_stream is not None:
        encode_stream.synchronize()
    latents = _latents_to_cpu(latents)

    batch[_VISION_LATENT_KEY[media_key]] = latents
    batch[_VISION_PIXEL_SHAPE_KEY[media_key]] = shapes
    batch["vae_latents_ready"] = True
    if not keep_video_pixels:
        del batch[media_key]
    return batch


def custom_collate_fn(batch):
    """
    Collate function that works like default_collate for all keys other than "text_token_ids", "images", and "video".
    For "text_token_ids", "images", and "video" it simply returns them in a list, instead of stacking them as a tensor.
    """
    list_collate_keys = {
        "text_token_ids",
        "images",
        "video",
        "action",
        "action_raw",
        "domain_id",
        "sequence_plan",
        "sound",
        "raw_action_dim",
        "image_size",
        "action_processing_record",
    }

    # Data keys where a per-sample value of ``None`` is a meaningful signal
    # (e.g. audio extraction failed for that sample → ``sound=None`` paired
    # with ``plan.has_sound=False``, or a video-only sample in an action
    # recipe → ``action=None`` paired with ``plan.has_action=False``).
    # These keys must be kept as a list with ``None`` placeholders so the
    # model can align per-sample data 1:1 with per-sample plans.  Dropping
    # the entire key on any None would leave the remaining tensors
    # mis-aligned with the plans whose flags were set BEFORE collation.
    sparse_data_keys = {"sound", "action", "action_raw", "domain_id", "raw_action_dim"}

    # Handle the case where the batch is already a dictionary (e.g. column-wise batching)
    if isinstance(batch, dict):
        return {key: (value if key in list_collate_keys else default_collate(value)) for key, value in batch.items()}

    # Handle standard list of samples
    elem = batch[0]
    if isinstance(elem, dict):
        # Some Action datasets add optional metadata keys (for example
        # ``additional_view_description`` for concat-view captions) only for a
        # subset of samples.  PyTorch can batch such samples together when
        # DataLoader batch_size > 1; collating only elem's keys and indexing
        # every sample by that key turns the optional field into a fatal
        # KeyError.  Use the union of keys and skip optional keys that are not
        # present in every sample.  Required training keys still fail loudly via
        # downstream assertions if actually missing.
        result = {}
        keys = set().union(*(d.keys() for d in batch))
        for key in keys:
            if key in _TIMING_KEYS:
                continue
            values = [d.get(key) for d in batch]
            if key == "action_processing_record":
                result[key] = values
                continue
            if any(value is None for value in values):
                # Sparse data keys keep their None placeholders to preserve
                # 1:1 alignment with sequence_plan.  Other (optional metadata)
                # keys not present in every sample are dropped.
                if key in sparse_data_keys:
                    result[key] = values
                continue
            if key in list_collate_keys:
                result[key] = values
            else:
                result[key] = default_collate(values)
        result.update(_aggregate_worker_timing(batch))
        return result
    else:
        return default_collate(batch)


def sample_has_action(sample: dict) -> bool:
    """Return whether a split packing sample carries action tokens.

    After :meth:`JointDataLoader._get_next_sample`, ``sequence_plan`` is a
    bare ``SequencePlan`` (not a list). ``action`` is a single-element list
    because it is in ``_MULTI_ITEM_KEYS``. Video-only samples either omit
    ``action`` or store ``None``.
    """
    plan = sample.get("sequence_plan")
    if plan is not None and hasattr(plan, "has_action"):
        return bool(plan.has_action)
    action = sample.get("action")
    if action is None:
        return False
    if isinstance(action, list):
        return any(item is not None for item in action)
    return True


def _aggregate_worker_timing(samples: list[dict]) -> dict:
    """Extract per-sample timing keys, aggregate into per-batch scalars."""
    info: dict[str, float | int] = {}
    if "_sample_time" in samples[0]:
        info["_worker_batch_time"] = sum(s.get("_sample_time", 0.0) for s in samples)
    if "_aug_time" in samples[0]:
        aug_total = sum(s.get("_aug_time", 0.0) for s in samples)
        info["_worker_aug_time"] = aug_total
        if "_worker_batch_time" in info:
            info["_worker_io_time"] = info["_worker_batch_time"] - aug_total
    if "_aug_step_times" in samples[0]:
        agg: dict[str, float] = {}
        for s in samples:
            for step_name, t in s.get("_aug_step_times", {}).items():
                agg[step_name] = agg.get(step_name, 0.0) + t
        info["_worker_aug_step_times"] = agg
    worker_info = torch.utils.data.get_worker_info()
    info["_worker_id"] = worker_info.id if worker_info is not None else 0
    return info


@dataclass
class _PackingMetrics:
    """Per-batch packing statistics collected during the packing loop.

    Also serves as the single source of truth for packing-related metric names
    via ``STATS_SPEC``, which the dataloading monitor callback consumes to
    drive accumulation and logging.
    """

    current_sequence_length: int = 0
    num_samples: int = 0
    dropped_count: int = 0
    from_buffer: int = 0
    from_workers: int = 0

    STATS_SPEC: ClassVar[list[tuple[str, str, str]]] = [
        # (batch_key, wandb_suffix, aggregation_type)
        ("_num_tokens", "token_fraction", "scalar"),
        ("_num_samples", "samples_per_batch", "list"),
        ("_from_buffer", "from_buffer", "list"),
        ("_from_workers", "from_workers", "list"),
        ("_buffer_size", "buffer_size", "list"),
        ("_dropped_count", "dropped", "scalar"),
    ]

    def attach_to(self, output_batch: dict, buffer_size: int) -> None:
        """Write packing statistics into the output batch dict."""
        output_batch["_num_tokens"] = self.current_sequence_length
        output_batch["_num_samples"] = self.num_samples
        output_batch["_from_buffer"] = self.from_buffer
        output_batch["_from_workers"] = self.from_workers
        output_batch["_buffer_size"] = buffer_size
        output_batch["_dropped_count"] = self.dropped_count


class JointDataLoader(webdataset.WebLoader):
    r"""
    A joint dataloader that supports loading both images and videos.
    """

    _DEFAULT_LOOKAHEAD_LIMIT: ClassVar[int] = 10

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None,
        max_samples_per_batch: int | None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        prewarm: bool = True,
        default_lookahead_limit: int = _DEFAULT_LOOKAHEAD_LIMIT,
        lookahead_limits: Dict[str, int] | None = None,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        recycle_inner_iterator: bool = False,
    ):
        """
        Initialize the JointDataLoader with multiple datasets.

        The effective mini-batch size can be controlled with either max_sequence_length or
        max_samples_per_batch. To use max_sequence_length, max_samples_per_batch needs to be None.
        Vice versa, to use max_samples_per_batch, max_sequence_length needs to be None.
        max_sequence_length and max_samples_per_batch cannot both be None simultaneously.

        Args:
            dataloaders: key - dataset_name; value - {"dataloader": dataloader, "ratio": data_ratio}
            tokenizer_spatial_compression_factor: The spatial compression factor of the tokenizer.
            tokenizer_temporal_compression_factor: The temporal compression factor of the tokenizer.
            patch_spatial: Spatial pathification factor.
            max_samples_per_batch: Max number of samples per packed batch (alternative to max_sequence_length).
            sound_latent_fps: Sound tokenizer latent rate in Hz (e.g. 25). If 0, sound tokens are not counted.
            audio_sample_rate: Audio sample rate in Hz (e.g. 48000). Used with sound_latent_fps to estimate
                sound token count.
            default_lookahead_limit: Packing-loop look-ahead fallback for dataloaders not in
                ``lookahead_limits``.
            lookahead_limits: Optional ``{dataset_name: int}`` per-dataloader override.
            uniae_chunk_frames: Optional UniAE full chunk size, or resolution-keyed chunk sizes.
            uniae_pad_frames: Optional UniAE boundary padding frames per chunk.

        Example:
            joint_loader = IterativeJointDataLoader(
                dataloaders{
                    "image_data": {
                        "dataloader": webdataset.WebLoader(...),
                        "ratio": 4,
                    },
                    "video_data": {
                        "dataloader": torch.utils.data.DataLoader(...),
                        "ratio": 1,
                    },
                }
            )
        """
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []
        self.lookahead_limits: list[int] = []
        self.tokenizer_spatial_compression_factor = tokenizer_spatial_compression_factor
        self.tokenizer_temporal_compression_factor = tokenizer_temporal_compression_factor
        self.patch_spatial = patch_spatial
        self.max_sequence_length = max_sequence_length
        self.max_samples_per_batch = max_samples_per_batch
        self.sound_latent_fps = sound_latent_fps
        self.audio_sample_rate = audio_sample_rate
        self.default_lookahead_limit = int(default_lookahead_limit)
        self.uniae_pad_frames = int(uniae_pad_frames) if uniae_pad_frames is not None else None
        self.uniae_chunk_frames = self._normalize_uniae_chunk_frames(uniae_chunk_frames)
        self.recycle_inner_iterator = bool(recycle_inner_iterator)

        assert (self.max_sequence_length is None) != (self.max_samples_per_batch is None), (
            "Exactly one of max_sequence_length or max_samples_per_batch must be None, but not both."
        )

        _lookahead_overrides: Dict[str, int] = dict(lookahead_limits) if lookahead_limits else {}
        unknown = set(_lookahead_overrides) - set(dataloaders)
        assert not unknown, f"lookahead_limits references unknown dataloaders {unknown}; valid: {sorted(dataloaders)}"

        for dataset_name, dataloader_data in dataloaders.items():
            if dataloader_data is None:
                continue
            assert set(dataloader_data.keys()) == {"dataloader", "ratio"}, f"Invalid config: {dataloader_data}"
            if dataloader_data["ratio"] <= 0:
                continue
            self.dataset_name_list.append(dataset_name)
            self.dataloader_list.append(instantiate(dataloader_data["dataloader"], collate_fn=custom_collate_fn))
            self.data_ratios.append(dataloader_data["ratio"])
            self.lookahead_limits.append(int(_lookahead_overrides.get(dataset_name, self.default_lookahead_limit)))

        self.global_id = 0
        self.ratio_sum = sum(self.data_ratios)

        total = self.ratio_sum if self.ratio_sum > 0 else 1.0
        lines = [f"JointDataLoader: {len(self.dataset_name_list)} streams"]
        for name, ratio in zip(self.dataset_name_list, self.data_ratios):
            lines.append(f"  {name}: ratio={ratio:.4g} ({ratio / total:.1%})")
        log.info("\n".join(lines))

        self.data_len = 0
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        self.buffers = [deque() for _ in range(len(self.dataloader_list))]
        for data in self.dataloader_list:
            self.data_len += len(data)

        # Pre-warm all dataloaders: force worker process spawning and first
        # batch loading so that slow dataset initialisation (e.g. action
        # datasets with spawn workers) happens here rather than mid-training
        # where it would cause NCCL collective timeouts.
        if prewarm:
            self._prewarm_dataloaders()
        else:
            log.info(
                "JointDataLoader: prewarm DISABLED (debug mode); first iteration may incur per-stream cold-load cost"
            )

    def _normalize_uniae_chunk_frames(
        self, uniae_chunk_frames: int | Mapping[str, int] | None
    ) -> int | dict[str, int] | None:
        return normalize_uniae_chunk_frames(
            uniae_chunk_frames,
            pad_frames=self.uniae_pad_frames,
            temporal_compression_factor=self.tokenizer_temporal_compression_factor,
            temporal_divisibility_name="tokenizer_temporal_compression_factor",
        )

    def _get_uniae_chunk_frames(self, spatial_shape: tuple[int, int]) -> int:
        assert self.uniae_chunk_frames is not None
        return get_uniae_chunk_frames(self.uniae_chunk_frames, spatial_shape=spatial_shape)

    def _compute_vision_latent_t_shape(self, T: int, H: int, W: int) -> int:
        if T < 1:
            raise ValueError(f"Vision media must contain at least one frame, got {T}.")
        if T == 1 or self.uniae_chunk_frames is None:
            return 1 + (T - 1) // self.tokenizer_temporal_compression_factor

        assert self.uniae_pad_frames is not None
        return get_uniae_latent_num_frames(
            T,
            self.uniae_chunk_frames,
            pad_frames=self.uniae_pad_frames,
            temporal_compression_factor=self.tokenizer_temporal_compression_factor,
            spatial_shape=(H, W),
        )

    def _prewarm_dataloaders(self) -> None:
        """Force all dataloader iterators to spawn workers and produce one batch.

        The first ``next()`` call on an ``InfiniteDataLoader`` iterator triggers
        ``DataLoader.__iter__()`` which spawns worker processes.  For action
        dataloaders using ``multiprocessing_context='spawn'``, each worker must
        fully initialise heavy datasets (BridgeOrigLeRobotDataset, embodiment_a, etc.)
        from scratch.  If this happens lazily during training, the resulting
        delay (potentially minutes) causes NCCL collective timeouts when faster
        ranks enter the forward pass while slower ranks are still loading data.

        By pulling one batch from every dataloader here — before any training
        iteration — we ensure all workers are alive and warmed up.  The fetched
        samples are pushed into the per-dataloader buffer so they are consumed
        normally by the first iteration that selects that dataloader.

        A ``dist.barrier()`` at the end synchronises all ranks so that training
        only begins once every rank has finished pre-warming.
        """
        import time

        for i, (name, dl_iter) in enumerate(zip(self.dataset_name_list, self.dataloaders)):
            t0 = time.monotonic()
            try:
                batch = next(dl_iter)
            except StopIteration:
                log.warning(f"Pre-warm: dataloader {name!r} is empty, skipping")
                continue
            elapsed = time.monotonic() - t0

            # Split the collated batch into individual samples and push them
            # into the buffer — identical to the splitting logic in
            # _get_next_sample — so the samples are not wasted.
            is_image_batch = "images" in batch
            input_images_or_videos = batch["images" if is_image_batch else "video"]
            batch_size = len(input_images_or_videos)

            for j in range(batch_size):
                sample = {}
                for k, v in batch.items():
                    if k in _BATCH_TIMING_KEYS:
                        sample[k] = v
                    elif isinstance(v, list) and k in self._MULTI_ITEM_KEYS:
                        elem = v[j]
                        if isinstance(elem, list):
                            sample[k] = elem
                        else:
                            sample[k] = v[j : j + 1]
                    elif isinstance(v, list):
                        sample[k] = v[j]
                    elif isinstance(v, torch.Tensor) and v.dim() > 0:
                        sample[k] = v[j : j + 1]
                    else:
                        sample[k] = v[j : j + 1]
                self.buffers[i].append(sample)

            log.info(
                f"Pre-warm: dataloader {name!r} ready — {batch_size} samples buffered in {elapsed:.1f}s",
                rank0_only=False,
            )

        # Synchronise so training only starts once every rank is warmed up.
        if torch.distributed.is_initialized():
            log.info("Pre-warm: waiting at barrier for all ranks …")
            torch.distributed.barrier()
            log.info("Pre-warm: all ranks ready")

    def _compute_num_tokens_per_sample(self, data_batch: dict) -> int:
        """
        This function computes the number of tokens per sample in the data batch.
        This includes text + vision generation tokens + action tokens.

        Args:
            data_batch (dict): The data batch containing the text tokens.

        Returns:
            int: The number of tokens per sample.
        """

        # The token sequence we have is
        # <text tokens> <eos> <vision_start> <image tokens> <vision_end> [<action tokens>]
        # The spatial dimension of image tokens is compressed by
        # vae spatial downsampling factor + pathification
        # The temporal dimension of image tokens is compressed by
        # vae temporal downsampling factor
        # Action tokens have 1 token per time step (no spatial dimension)

        has_text_tokens = "text_token_ids" in data_batch
        num_tokens = 0
        if has_text_tokens:
            text_token_ids = data_batch["text_token_ids"]
            if isinstance(text_token_ids, list):
                num_text_tokens = text_token_ids[0].shape[0]
            else:
                num_text_tokens = text_token_ids.shape[1]
            num_tokens += num_text_tokens + 1

        # Vision part
        is_image_batch = "images" in data_batch
        input_images_or_videos = data_batch["images" if is_image_batch else "video"]
        if "enable_per_camera_vae_encoding" in data_batch:
            sample_n_views_values = read_positive_int_metadata(data_batch, "sample_n_views", expected_count=1)
            frames_per_view_values = read_positive_int_metadata(
                data_batch,
                "num_video_frames_per_view",
                expected_count=1,
            )
            if sample_n_views_values is None or frames_per_view_values is None:
                raise ValueError(
                    "sample_n_views and num_video_frames_per_view are required when per-camera VAE encoding is enabled."
                )
            sample_n_views = sample_n_views_values[0]
            frames_per_view = frames_per_view_values[0]
        else:
            sample_n_views = None
            frames_per_view = None

        # iterate over all the media in the batch
        for media in input_images_or_videos if isinstance(input_images_or_videos, list) else [input_images_or_videos]:
            if is_image_batch:
                _, H, W = media.shape
                T = 1
            else:
                _, T, H, W = media.shape

            latent_h_shape = H // self.tokenizer_spatial_compression_factor
            latent_w_shape = W // self.tokenizer_spatial_compression_factor
            patch_h_shape = math.ceil(latent_h_shape / self.patch_spatial)
            patch_w_shape = math.ceil(latent_w_shape / self.patch_spatial)
            if sample_n_views is not None and frames_per_view is not None and not is_image_batch:
                # The multiview dataset resizes every camera to the same H/W and
                # selects the same synchronized frame count before concatenating
                # the camera-major clips along T.
                expected_frames = sample_n_views * frames_per_view
                if T != expected_frames:
                    raise ValueError(
                        "Multiview vision length must equal sample_n_views * num_video_frames_per_view: "
                        f"got T={T}, sample_n_views={sample_n_views}, num_video_frames_per_view={frames_per_view}."
                    )
                latent_t_shape = sample_n_views * self._compute_vision_latent_t_shape(frames_per_view, H, W)
            else:
                latent_t_shape = self._compute_vision_latent_t_shape(T, H, W)

            num_vision_tokens = patch_h_shape * patch_w_shape * latent_t_shape
            if has_text_tokens:
                num_vision_tokens += 2
            num_tokens += num_vision_tokens

        # Action part: each action time step is 1 token.
        # Action tensor shape is (T_action, D) per sample; stored as a single-element list.
        if "action" in data_batch:
            list_of_actions = data_batch["action"]
            for action in list_of_actions:
                # skip None actions
                if action is None:
                    continue
                num_action_tokens = action.shape[0]
                num_tokens += num_action_tokens

        # Sound part — estimate sound tokens from audio waveform length
        if self.sound_latent_fps > 0 and "sound" in data_batch:
            sound_data = data_batch["sound"]
            if isinstance(sound_data, list) and len(sound_data) > 0:
                first_sound = sound_data[0]
                # Unwrap nested list if needed
                if isinstance(first_sound, list):
                    first_sound = first_sound[0]
                if first_sound is not None and isinstance(first_sound, torch.Tensor):
                    num_audio_samples = first_sound.shape[-1]
                    audio_duration = num_audio_samples / self.audio_sample_rate
                    num_sound_tokens = int(audio_duration * self.sound_latent_fps)
                    num_tokens += num_sound_tokens

        return num_tokens

    # Keys whose value per sample is a list of tensors to be flattened into one list in the batch
    _FLATTEN_LIST_KEYS = {"image_size"}

    def _update_output_batch(self, output_batch: dict, output: dict):
        for key, value in output.items():
            if key in _BATCH_TIMING_KEYS:
                if key not in output_batch:
                    output_batch[key] = value
            elif key in self._FLATTEN_LIST_KEYS and isinstance(value, list):
                if key not in output_batch:
                    output_batch[key] = value
                else:
                    output_batch[key].extend(value)
            elif key not in output_batch:
                output_batch[key] = [value]
            else:
                output_batch[key].append(value)

    def __len__(self) -> int:
        return self.data_len

    # Keys where each sample may hold multiple tensors (e.g. multiple video
    # clips in a packed sequence).  Kept as single-element lists per sample
    # via v[i:i+1] so that _update_output_batch yields list[list[Tensor]].
    _MULTI_ITEM_KEYS = {"text_token_ids", "images", "video", "action", "action_raw", "sound"}

    def _get_next_sample(self, index_id: int) -> dict:
        """Pop the next single-sample dict from the buffer for the given dataloader.

        If the buffer is empty, fetches the next collated batch from the inner
        dataloader and splits it into individual samples.

        Splitting rules:
            - Multi-item list values (keys in ``_MULTI_ITEM_KEYS``): sliced
              via ``v[i:i+1]`` to yield a single-element list ``[tensor]``.
              A packed sequence can contain multiple items per key.
            - Per-sequence metadata list values (all other list keys, e.g.
              ``sequence_plan``, ``domain_id``): direct-indexed via ``v[i]``
              to yield the bare element.
            - Tensor values ``(B, ...)``: sliced to ``(1, ...)`` via
              ``v[i : i + 1]`` to preserve the batch dimension.

        After ``_update_output_batch`` accumulates samples, the packed output
        batch has the following shapes:
            - Multi-item keys (``text_token_ids``, ``video``, ``images``,
              ``action_raw``, ``action``): ``list[list[Tensor]]`` — each inner
              list has one element from one sub-sample.
            - Per-sequence metadata keys (``sequence_plan``, ``domain_id``,
              ``dataset_name``, etc.): ``list[element]`` — flat list.
            - Tensor-origin keys: ``list[Tensor(1, ...)]``.

        Args:
            index_id: Index of the dataloader to fetch from.

        Returns:
            A single-sample dictionary.
        """
        buffer = self.buffers[index_id]
        if not buffer:
            try:
                batch = next(self.dataloaders[index_id])
            except StopIteration:
                if not self.recycle_inner_iterator:
                    raise
                name = self.dataset_name_list[index_id]
                log.info(
                    f"JointDataLoader: recycling exhausted iterator for {name!r}",
                    rank0_only=False,
                )
                self.dataloaders[index_id] = iter(self.dataloader_list[index_id])
                batch = next(self.dataloaders[index_id])

            is_image_batch = "images" in batch
            input_images_or_videos = batch["images" if is_image_batch else "video"]
            batch_size = len(input_images_or_videos)

            for i in range(batch_size):
                sample = {}
                for k, v in batch.items():
                    if k in _BATCH_TIMING_KEYS:
                        sample[k] = v
                    elif isinstance(v, list) and k in self._MULTI_ITEM_KEYS:
                        # For multi-item keys (images, video, etc.), the collated
                        # value is a list with one element per sample.  If the element
                        # is itself a list (e.g. image editing: [src, tgt]), use v[i]
                        # directly to avoid wrapping it in a redundant single-element
                        # list.  Otherwise keep the v[i:i+1] slice so that
                        # _update_output_batch produces list[list[Tensor]].
                        elem = v[i]
                        if isinstance(elem, list):
                            sample[k] = elem
                        else:
                            sample[k] = v[i : i + 1]
                    elif isinstance(v, list):
                        sample[k] = v[i]
                    else:
                        sample[k] = v[i : i + 1]
                buffer.append(sample)

        return buffer.popleft()

    def set_start_iteration(self, iteration: int):
        self.global_id = iteration

    def __iter__(self):
        raise NotImplementedError("__iter__ function is not implemented yet")


class IterativeJointDataLoader(JointDataLoader):
    r"""
    An iterative joint dataloader that supports loading multiple modalities.

    The behavior depends on the ``seed`` parameter:

    - **seed is not None** (Default):
      The modality is randomly selected at each iteration based on the probability distribution
      derived from the ratios. The random state is seeded with ``seed + global_id``, ensuring
      that all ranks select the same modality at the same iteration (assuming synchronized global_id).
      This prevents load imbalance due to mixed resolutions across ranks.

    - **seed is None**:
      The modality selection follows a deterministic round-robin pattern based on the ratios.
      For example, with 2 modalities (image and video) and ratio 2:1:
        - Iterations 0, 1: all ranks process images
        - Iteration 2: all ranks process videos
        - ... and so on.
      This also ensures all ranks process the same modality at the same iteration.
    """

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        seed: int | None = 42,
        prewarm: bool = True,
        default_lookahead_limit: int = JointDataLoader._DEFAULT_LOOKAHEAD_LIMIT,
        lookahead_limits: Dict[str, int] | None = None,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
    ):
        super().__init__(
            dataloaders,
            tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor,
            patch_spatial,
            max_sequence_length,
            max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            prewarm=prewarm,
            default_lookahead_limit=default_lookahead_limit,
            lookahead_limits=lookahead_limits,
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
        )
        self.seed = seed
        # Calculate probabilities for random sampling
        total_ratio = sum(self.data_ratios)
        self.data_probs = np.array([ratio / total_ratio for ratio in self.data_ratios])

    def __iter__(self):
        while True:
            if self.seed is not None:
                rng = np.random.RandomState(self.seed + self.global_id)
                index_id = rng.choice(len(self.dataloader_list), p=self.data_probs)
            else:
                data_id = self.global_id % self.ratio_sum
                index_id = self._get_dataloader_index(data_id)

            metrics = _PackingMetrics()
            output_batch = dict()
            skipped_samples = deque()
            lookahead_limit = self.lookahead_limits[index_id]
            lookahead_count = 0
            pack_has_action: bool | None = None

            while True:
                # Check max samples limit first
                if self.max_samples_per_batch is not None and metrics.num_samples >= self.max_samples_per_batch:
                    break

                # If we have started packing and tried lookahead_limit times to find a fitting sample but failed, stop.
                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                had_buffer = len(self.buffers[index_id]) > 0
                try:
                    output = self._get_next_sample(index_id)
                except StopIteration:
                    break  # No more data in this dataloader

                if had_buffer:
                    metrics.from_buffer += 1
                else:
                    metrics.from_workers += 1

                sample_action = sample_has_action(output)
                if pack_has_action is None:
                    pack_has_action = sample_action
                elif sample_action != pack_has_action:
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                num_tokens_in_current_sample = self._compute_num_tokens_per_sample(output)

                if (
                    self.max_sequence_length is not None
                    and metrics.current_sequence_length + num_tokens_in_current_sample >= self.max_sequence_length
                ):
                    if len(output_batch) == 0:
                        # This case happens when current_sequence_length = 0 and num_tokens_in_current_sample > self.max_sequence_length
                        # In this case, we should simply discard the current sample and get the next sample.
                        log.info(
                            f"Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        metrics.dropped_count += 1
                        continue

                    # current_sequence_length > 0 and selected sample is too large to fit in the remaining space.
                    # Instead of stopping immediately (creating large padding), we buffer this large sample
                    # and try to find a smaller one that fits in the remaining space.
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                metrics.current_sequence_length += num_tokens_in_current_sample
                metrics.num_samples += 1
                output["dataset_name"] = self.dataset_name_list[index_id]
                self._update_output_batch(output_batch, output)

            # Add back skipped samples to the buffer for the next batch.
            # appendleft puts item at HEAD. So we insert S3, then S2, then S1.
            for sample in reversed(skipped_samples):
                self.buffers[index_id].appendleft(sample)

            if len(output_batch) == 0:
                return

            metrics.attach_to(output_batch, buffer_size=len(self.buffers[index_id]))
            self.global_id += 1
            yield output_batch

    def _get_dataloader_index(self, data_id):
        """Maps global id to the corresponding dataloader index based on ratio."""
        for i, r in enumerate(self.data_ratios):
            if data_id < r:
                return i
            data_id -= r
        raise ValueError("Invalid data_id")


class RankPartitionedDataLoader:
    """Assigns each rank to exactly one dataset based on ratios.

    For N GPUs with datasets having ratios r_1:r_2:...:r_k, the first
    N * r_1 / sum(r) ranks are assigned dataset 1, the next N * r_2 / sum(r)
    ranks are assigned dataset 2, etc.  Each rank instantiates a single
    PyTorch DataLoader for its assigned dataset.

    The sharding information (``shard_world_size`` and ``shard_rank``) is set
    on each dataset so that it shards data only across ranks that share the
    same dataset, rather than across the full world.

    Example:
        With 128 GPUs and datasets ``{"video": {"dataset": ..., "ratio": 3},
        "image": {"dataset": ..., "ratio": 1}}``:

        - Ranks   0-95  -> video  (shard_world_size=96, shard_rank=0..95)
        - Ranks  96-127 -> image  (shard_world_size=32, shard_rank=0..31)
    """

    def __init__(
        self,
        datasets: dict[str, dict[str, Any]],
        **dataloader_kwargs: Any,
    ):
        """
        Args:
            datasets: Mapping of dataset name to config dict with keys:

                - ``"dataset"`` (required): a lazy config or dataset instance.
                - ``"ratio"`` (required): positive int weight.
                - ``"dataloader_kwargs"`` (optional): dict of keyword arguments
                  that override the top-level ``**dataloader_kwargs`` for this
                  dataset only (e.g. different ``num_workers`` or ``batch_size``).

            **dataloader_kwargs: Default kwargs forwarded to
                ``torch.utils.data.DataLoader``. ``collate_fn`` defaults to
                ``custom_collate_fn`` if not given.
        """
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        log.info(f"RankPartitionedDataLoader: world_size: {world_size} and rank: {rank}", rank0_only=False)

        _VALID_KEYS = {"dataset", "ratio", "dataloader_kwargs"}
        names: list[str] = []
        dataset_configs: list[Any] = []
        ratios: list[int] = []
        per_dataset_kwargs: list[dict[str, Any]] = []
        for name, cfg in datasets.items():
            extra = set(cfg.keys()) - _VALID_KEYS
            assert not extra, f"Dataset {name!r}: unexpected keys {extra}. Allowed: {_VALID_KEYS}"
            if cfg["ratio"] <= 0:
                log.warning(
                    f"RankPartitionedDataLoader: Skipping dataset {name} with ratio {cfg['ratio']}", rank0_only=False
                )
                continue
            names.append(name)
            dataset_configs.append(cfg["dataset"])
            ratios.append(cfg["ratio"])
            per_dataset_kwargs.append(cfg.get("dataloader_kwargs", {}))

        assert len(names) > 0, "No datasets with positive ratios provided."
        assert world_size >= len(names), (
            f"world_size ({world_size}) must be >= number of datasets ({len(names)}) "
            f"so each dataset gets at least one rank."
        )

        total_ratio = sum(ratios)
        ideal = [r / total_ratio * world_size for r in ratios]
        allocations = [max(1, int(q)) for q in ideal]
        remaining = world_size - sum(allocations)
        if remaining > 0:
            remainders = sorted(range(len(ratios)), key=lambda i: ideal[i] - allocations[i], reverse=True)
            for j in range(remaining):
                allocations[remainders[j]] += 1
        elif remaining < 0:
            deficit = -remaining
            while deficit > 0:
                best = max(
                    (i for i in range(len(allocations)) if allocations[i] > 1),
                    key=lambda i: (allocations[i] - ideal[i], allocations[i]),
                )
                allocations[best] -= 1
                deficit -= 1

        expected_ratios = [r / total_ratio for r in ratios]
        actual_ratios = [a / world_size for a in allocations]
        lines = [f"RankPartitionedDataLoader allocation ({world_size} GPUs):"]
        start = 0
        for i, (name, alloc) in enumerate(zip(names, allocations)):
            end = start + alloc - 1
            lines.append(
                f"  {name} (ratio {ratios[i]}): ranks {start}-{end} ({alloc} GPUs) "
                f"| expected {expected_ratios[i]:.2%}, actual {actual_ratios[i]:.2%}"
            )
            start += alloc
        log.info("\n".join(lines), rank0_only=False)

        cumulative = 0
        my_dataset_idx = -1
        for i, alloc in enumerate(allocations):
            if rank < cumulative + alloc:
                my_dataset_idx = i
                break
            cumulative += alloc
        assert my_dataset_idx >= 0

        shard_rank = rank - cumulative
        shard_world_size = allocations[my_dataset_idx]

        dataset: Any = instantiate(dataset_configs[my_dataset_idx])
        dataset.shard_world_size = shard_world_size
        dataset.shard_rank = shard_rank
        dataset.shard_id = my_dataset_idx

        merged_kwargs = {**dataloader_kwargs, **per_dataset_kwargs[my_dataset_idx]}
        merged_kwargs.setdefault("collate_fn", custom_collate_fn)
        self.dataloader = torch.utils.data.DataLoader(dataset, **merged_kwargs)
        self.dataset_name = names[my_dataset_idx]
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)


class PackingDataLoader(JointDataLoader):
    """Packs multiple samples from a single dataloader into token-budget-constrained batches.

    Unlike the other ``JointDataLoader`` subclasses which manage multiple
    dataloaders with configurable ratios, this class wraps a single dataloader
    and greedily packs consecutive samples until the token budget
    (``max_sequence_length``) or sample count limit (``max_samples_per_batch``)
    is reached.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader | webdataset.WebLoader,
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        dataset_name: str = "default",
        lookahead_limit: int = JointDataLoader._DEFAULT_LOOKAHEAD_LIMIT,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        recycle_inner_iterator: bool = False,
        prewarm: bool = True,
        encode_vision_latents: bool = False,
        encoded_prefetch_depth: int = 0,
        keep_video_pixels: bool = False,
    ):
        """
        Args:
            dataloader: A single dataloader (or lazy config) to draw samples from.
            tokenizer_spatial_compression_factor: Spatial compression factor of the tokenizer.
            tokenizer_temporal_compression_factor: Temporal compression factor of the tokenizer.
            patch_spatial: Spatial patchification factor.
            max_sequence_length: Max total tokens per packed batch. Mutually exclusive with
                ``max_samples_per_batch``.
            max_samples_per_batch: Max number of samples per packed batch. Mutually exclusive
                with ``max_sequence_length``.
            sound_latent_fps: Sound tokenizer latent rate in Hz. If 0, sound tokens are not counted.
            audio_sample_rate: Audio sample rate in Hz.
            dataset_name: Name tag attached to every sample in the output batch.
            lookahead_limit: Packing-loop look-ahead for the wrapped dataloader.
            uniae_chunk_frames: Optional UniAE full chunk size, or resolution-keyed chunk sizes.
            uniae_pad_frames: Optional UniAE boundary padding frames per chunk.
            encode_vision_latents: When True, VAE-encode packed videos before yield. Requires
                :meth:`attach_vision_tokenizer` after the model tokenizer is constructed.
            encoded_prefetch_depth: Encoded-batch queue depth. 0 = encode in ``next()``.
                ``>=1`` fills that many batches before the first yield and keeps the queue
                full on a producer thread so training can overlap the next encode.
            keep_video_pixels: Keep uint8 ``video``/``images`` after encode (viz only).
        """
        wrapped = {dataset_name: {"dataloader": dataloader, "ratio": 1}}
        super().__init__(
            dataloaders=wrapped,
            tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
            patch_spatial=patch_spatial,
            max_sequence_length=max_sequence_length,
            max_samples_per_batch=max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            lookahead_limits={dataset_name: int(lookahead_limit)},
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
            recycle_inner_iterator=recycle_inner_iterator,
            prewarm=prewarm,
        )
        if encoded_prefetch_depth < 0:
            raise ValueError(f"encoded_prefetch_depth must be >= 0, got {encoded_prefetch_depth}")
        self.encode_vision_latents = bool(encode_vision_latents)
        self.encoded_prefetch_depth = int(encoded_prefetch_depth) if self.encode_vision_latents else 0
        self.keep_video_pixels = bool(keep_video_pixels)
        self._vision_tokenizer: Any = None
        self._vision_tokenizer_device: torch.device | None = None

    def attach_vision_tokenizer(self, tokenizer: Any) -> None:
        """Bind the model's frozen vision tokenizer for packed-batch encode."""
        if tokenizer is None:
            raise RuntimeError("attach_vision_tokenizer received tokenizer=None")
        self._vision_tokenizer = tokenizer
        self._vision_tokenizer_device = _infer_tokenizer_device(tokenizer)
        log.info(
            "PackingDataLoader: attached vision tokenizer "
            f"{type(tokenizer).__name__} on {self._vision_tokenizer_device} "
            f"(prefetch_depth={self.encoded_prefetch_depth}, keep_pixels={self.keep_video_pixels})",
            rank0_only=False,
        )

    def _encode_packed_batch(self, batch: dict) -> dict:
        if self._vision_tokenizer is None:
            raise RuntimeError(
                "PackingDataLoader.encode_vision_latents=True requires attach_vision_tokenizer "
                "before iteration (trainer.train attaches model.tokenizer_vision_gen)."
            )
        return encode_packed_vision_batch(
            batch,
            self._vision_tokenizer,
            keep_video_pixels=self.keep_video_pixels,
            device=self._vision_tokenizer_device,
        )

    def _iter_packed(self) -> Iterator[dict]:
        inner = self.dataloader_list[0]
        ds_name = getattr(inner, "dataset_name", self.dataset_name_list[0])

        while True:
            current_sequence_length = 0
            num_samples = 0
            output_batch: dict = {}

            skipped_samples: deque = deque()
            # PackingDataLoader wraps a single dataloader, so lookahead_limits has one entry.
            lookahead_limit = self.lookahead_limits[0]
            lookahead_count = 0
            # Teacher-forcing expansion requires a homogeneous V+A or V-only
            # pack. Keep the first sample's action layout and skip mismatches.
            pack_has_action: bool | None = None

            while True:
                if self.max_samples_per_batch is not None and num_samples >= self.max_samples_per_batch:
                    break

                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                try:
                    output = self._get_next_sample(0)
                except StopIteration:
                    break

                sample_action = sample_has_action(output)
                if pack_has_action is None:
                    pack_has_action = sample_action
                elif sample_action != pack_has_action:
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                num_tokens_in_current_sample = self._compute_num_tokens_per_sample(output)

                if (
                    self.max_sequence_length is not None
                    and current_sequence_length + num_tokens_in_current_sample >= self.max_sequence_length
                ):
                    if len(output_batch) == 0:
                        # This case happens when current_sequence_length = 0 and num_tokens_in_current_sample > self.max_sequence_length
                        # In this case, we should simply discard the current sample and get the next sample.
                        log.error(
                            f"PackingDataLoader: Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        continue

                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                current_sequence_length += num_tokens_in_current_sample
                num_samples += 1
                # Allows the dataset name to be overridden by the sample itself.
                output["dataset_name"] = output.get("dataset_name", ds_name)
                self._update_output_batch(output_batch, output)

            for sample in reversed(skipped_samples):
                self.buffers[0].appendleft(sample)

            if len(output_batch) == 0:
                return

            self.global_id += 1
            yield output_batch

    def _iter_encoded_prefetch(self, packed_iter: Iterator[dict]) -> Iterator[dict]:
        encoded_queue: queue.Queue = queue.Queue(maxsize=self.encoded_prefetch_depth)
        stop = threading.Event()

        def _producer() -> None:
            try:
                device = self._vision_tokenizer_device or _infer_tokenizer_device(self._vision_tokenizer)
                self._vision_tokenizer_device = device
                _set_current_torch_device(device)
                for packed in packed_iter:
                    if stop.is_set():
                        break
                    encoded_queue.put(self._encode_packed_batch(packed))
                encoded_queue.put(_ENCODED_QUEUE_SENTINEL)
            except Exception as exc:  # noqa: BLE001 — surface any producer failure to next()
                encoded_queue.put(exc)

        producer = threading.Thread(target=_producer, name="packing-vae-encode", daemon=True)
        producer.start()
        try:
            while True:
                item = encoded_queue.get()
                if item is _ENCODED_QUEUE_SENTINEL:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()

    def __iter__(self):
        packed_iter = self._iter_packed()
        if not self.encode_vision_latents:
            yield from packed_iter
            return
        if self.encoded_prefetch_depth <= 0:
            for packed in packed_iter:
                yield self._encode_packed_batch(packed)
            return
        yield from self._iter_encoded_prefetch(packed_iter)


class RandomJointDataLoader(JointDataLoader):
    r"""
    A random joint dataloader that supports loading multiple modalities with stochastic sampling.

    In this dataloader, the modality is randomly selected at each iteration based on the
    probability distribution derived from the ratios. Each rank independently samples a
    modality, so different ranks may process different modalities at the same iteration.

    For example, with 2 modalities (image and video) and ratio 2:1:
        - Each iteration has 66.7% probability of selecting images
        - Each iteration has 33.3% probability of selecting videos
        - The selection is independent across iterations and ranks

    Note: Unlike IterativeJointDataLoader, this does not guarantee synchronized modality
    selection across ranks.
    """

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        default_lookahead_limit: int = JointDataLoader._DEFAULT_LOOKAHEAD_LIMIT,
        lookahead_limits: Dict[str, int] | None = None,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
    ):
        super().__init__(
            dataloaders,
            tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor,
            patch_spatial,
            max_sequence_length,
            max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            default_lookahead_limit=default_lookahead_limit,
            lookahead_limits=lookahead_limits,
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
        )

        # Convert data ratios to probabilities
        self.data_ratios = np.array([ratio / sum(self.data_ratios) for ratio in self.data_ratios])

    def __iter__(self):
        while True:
            index_id = np.random.choice(len(self.dataloader_list), p=self.data_ratios)

            metrics = _PackingMetrics()
            output_batch = dict()
            skipped_samples = deque()
            lookahead_limit = self.lookahead_limits[index_id]
            lookahead_count = 0
            pack_has_action: bool | None = None

            while True:
                # Check max samples limit first
                if self.max_samples_per_batch is not None and metrics.num_samples >= self.max_samples_per_batch:
                    break

                # If we have started packing and tried lookahead_limit times to find a fitting sample but failed, stop.
                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                had_buffer = len(self.buffers[index_id]) > 0
                try:
                    output = self._get_next_sample(index_id)
                except StopIteration:
                    break  # No more data in this dataloader

                if had_buffer:
                    metrics.from_buffer += 1
                else:
                    metrics.from_workers += 1

                sample_action = sample_has_action(output)
                if pack_has_action is None:
                    pack_has_action = sample_action
                elif sample_action != pack_has_action:
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                num_tokens_in_current_sample = self._compute_num_tokens_per_sample(output)

                if (
                    self.max_sequence_length is not None
                    and metrics.current_sequence_length + num_tokens_in_current_sample >= self.max_sequence_length
                ):
                    if len(output_batch) == 0:
                        # This case happens when current_sequence_length = 0 and num_tokens_in_current_sample > self.max_sequence_length
                        # In this case, we should simply discard the current sample and get the next sample.
                        log.info(
                            f"Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        metrics.dropped_count += 1
                        continue

                    # current_sequence_length > 0 and selected sample is too large to fit in the remaining space.
                    # Instead of stopping immediately (creating large padding), we buffer this large sample
                    # and try to find a smaller one that fits in the remaining space.
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                metrics.current_sequence_length += num_tokens_in_current_sample
                metrics.num_samples += 1
                output["dataset_name"] = self.dataset_name_list[index_id]
                self._update_output_batch(output_batch, output)

            # Add back skipped samples to the buffer for the next batch.
            # appendleft puts item at HEAD. So we insert S3, then S2, then S1.
            for sample in reversed(skipped_samples):
                self.buffers[index_id].appendleft(sample)

            if len(output_batch) == 0:
                return

            metrics.attach_to(output_batch, buffer_size=len(self.buffers[index_id]))
            yield output_batch
