# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for PackingDataLoader VAE pre-encode / prefetch."""

from __future__ import annotations

import torch

from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    custom_collate_fn,
    encode_packed_vision_batch,
)
from cosmos_framework.model.generator.omni_mot_model import _pixel_t_h_w


class _FakeTokenizer:
    spatial_compression_factor = 16
    temporal_compression_factor = 4
    latent_ch = 48
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.encode_calls = 0
        self.seen_shapes: list[tuple[int, ...]] = []

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        assert video.ndim == 5
        assert video.dtype == torch.float32
        self.encode_calls += 1
        self.seen_shapes.append(tuple(video.shape))
        _, _, t, h, w = video.shape
        t_lat = 1 + (t - 1) // 4
        return torch.zeros(video.shape[0], 48, t_lat, h // 16, w // 16, dtype=torch.float32)


class _FakeVideoDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 4, t: int = 9, h: int = 32, w: int = 32) -> None:
        self.n = n
        self.t = t
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        del index
        return {
            "video": torch.randint(0, 256, (3, self.t, self.h, self.w), dtype=torch.uint8),
            "text_token_ids": torch.arange(8, dtype=torch.long),
        }


def _make_packer(
    *,
    n: int = 4,
    encode: bool = True,
    prefetch: int = 0,
    keep_pixels: bool = False,
) -> tuple[PackingDataLoader, _FakeTokenizer]:
    inner = torch.utils.data.DataLoader(
        _FakeVideoDataset(n=n),
        batch_size=1,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    packer = PackingDataLoader(
        dataloader=inner,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=1,
        encode_vision_latents=encode,
        encoded_prefetch_depth=prefetch,
        keep_video_pixels=keep_pixels,
        prewarm=False,
    )
    tokenizer = _FakeTokenizer()
    if encode:
        packer.attach_vision_tokenizer(tokenizer)
    return packer, tokenizer


def test_pixel_t_h_w_from_tensor_and_shape() -> None:
    video_4d = torch.zeros(3, 9, 32, 48)
    video_5d = torch.zeros(1, 3, 9, 32, 48)
    assert _pixel_t_h_w(video_4d, None) == (9, 32, 48)
    assert _pixel_t_h_w(video_5d, None) == (9, 32, 48)
    assert _pixel_t_h_w(None, (9, 32, 48)) == (9, 32, 48)


def test_encode_packed_vision_batch_drops_pixels_and_writes_latents() -> None:
    video = torch.randint(0, 256, (3, 9, 32, 32), dtype=torch.uint8)
    batch = {"video": [[video], [video.clone()]]}
    tokenizer = _FakeTokenizer()
    out = encode_packed_vision_batch(batch, tokenizer, keep_video_pixels=False, device=torch.device("cpu"))
    assert out["vae_latents_ready"] is True
    assert "video" not in out
    assert len(out["video_latents"]) == 2
    assert out["video_pixel_shapes"] == [(9, 32, 32), (9, 32, 32)]
    latent = out["video_latents"][0]
    assert latent.shape == (1, 48, 3, 2, 2)
    assert tokenizer.encode_calls == 2


def test_packing_dataloader_sync_encode_yields_latents() -> None:
    packer, tokenizer = _make_packer(encode=True, prefetch=0)
    batch = next(iter(packer))
    assert batch["vae_latents_ready"] is True
    assert "video" not in batch
    assert "video_latents" in batch
    assert batch["video_pixel_shapes"] == [(9, 32, 32)]
    assert tokenizer.encode_calls == 1
    latent = batch["video_latents"][0]
    assert latent.shape == (1, 48, 3, 2, 2)


def test_packing_dataloader_encode_requires_attach() -> None:
    packer, _ = _make_packer(encode=True, prefetch=0)
    packer._vision_tokenizer = None
    try:
        next(iter(packer))
    except RuntimeError as exc:
        assert "attach_vision_tokenizer" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when tokenizer is missing")


def test_packing_dataloader_prefetch_yields_two_encoded_batches() -> None:
    packer, tokenizer = _make_packer(n=4, encode=True, prefetch=1)
    it = iter(packer)
    first = next(it)
    second = next(it)
    assert first["vae_latents_ready"] and second["vae_latents_ready"]
    assert "video" not in first and "video" not in second
    assert first["video_latents"][0].shape == (1, 48, 3, 2, 2)
    assert tokenizer.encode_calls >= 2


def test_packing_dataloader_disabled_encode_keeps_uint8_video() -> None:
    packer, tokenizer = _make_packer(encode=False)
    batch = next(iter(packer))
    assert "video" in batch
    assert "video_latents" not in batch
    assert tokenizer.encode_calls == 0
