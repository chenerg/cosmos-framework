# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Whole-episode (``chunk_length=-1``) fetching tests.

Unit tests cover the pure index math of
``BaseActionLeRobotDataset._whole_episode_rows`` (FPS stride subsampling,
``max_episode_blocks`` cap, ``4N+1`` round-up with tail-row padding).

The real-data tests validate original episode sizes against fetched tensor
dimensions on the locally available RoboTwin / DROID LeRobot datasets; they
are skipped when those paths are absent.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    BaseActionLeRobotDataset,
    max_frames_for_pre_tf_sequence_length,
    round_up_4n1,
)


def _first_existing(*paths: str) -> str:
    for path in paths:
        if os.path.isdir(path):
            return path
    return paths[-1]


ROBOTWIN_ROOT = "/mi/data2T/Embodied-AI/datasets/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50"
DROID_ROOT = _first_existing(
    "/data5T/Embodied-AI/datasets/droid_plus_lerobot_640x360_20260412",
    "/mi/data2T/Embodied-AI/datasets/droid_plus_lerobot_640x360_20260412",
)


@pytest.fixture(autouse=True)
def _hf_offline_env(monkeypatch):
    # BaseActionLeRobotDataset.__init__ sets HF_HUB_OFFLINE when unset; pre-set
    # it via monkeypatch so conftest's env-modification guard stays clean.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------------------
# _whole_episode_rows unit tests (no real data)
# ---------------------------------------------------------------------------


def _make_base(fps: float, max_episode_blocks: int) -> BaseActionLeRobotDataset:
    return BaseActionLeRobotDataset(
        fps=fps,
        chunk_length=-1,
        split_seed=42,
        split_val_ratio=0.0,
        split="full",
        mode="policy",
        embodiment_type="droid_lerobot",
        viewpoint="concat_view",
        pose_convention="backward_framewise",
        max_episode_blocks=max_episode_blocks,
    )


def _fake_ds(episode_lengths: list[int], native_fps: float) -> Any:
    episodes = []
    start = 0
    for length in episode_lengths:
        episodes.append({"dataset_from_index": start, "dataset_to_index": start + length})
        start += length
    return SimpleNamespace(meta=SimpleNamespace(fps=native_fps, episodes=episodes))


def _pre_tf_tokens(frames: int, k: int, extra: int = 500) -> int:
    t_lat = 1 + (frames - 1) // 4
    return extra + t_lat * k + frames


def test_max_frames_for_pre_tf_sequence_length_matches_480p_budget():
    # extra=500 slack: 477 frames stay under 48000, next 4N+1 (481) goes over.
    k = 391
    cap = max_frames_for_pre_tf_sequence_length(48000, spatial_tokens_per_latent=k)
    assert cap == 477
    assert _pre_tf_tokens(cap, k) < 48000
    assert _pre_tf_tokens(cap + 4, k) >= 48000
    assert round_up_4n1(481) == 481
    assert round_up_4n1(482) == 485
    # 256p has far more frame headroom at the same token budget.
    assert max_frames_for_pre_tf_sequence_length(48000, spatial_tokens_per_latent=80) > 1000


def test_padded_whole_episode_frames_stride_1():
    base = _make_base(fps=15.0, max_episode_blocks=-1)
    assert base._padded_whole_episode_frames(481, native_fps=15.0) == 481
    assert base._padded_whole_episode_frames(482, native_fps=15.0) == 485
    assert base._padded_whole_episode_frames(1, native_fps=15.0) == 1


def test_rows_exact_4n1_no_padding():
    base = _make_base(fps=15.0, max_episode_blocks=-1)
    ds = _fake_ds([101], native_fps=15.0)
    obs_rows, act_rows, n = base._whole_episode_rows(ds, 0)
    assert n == 101
    assert obs_rows == list(range(101))
    assert act_rows == list(range(100))


def test_rows_round_up_with_tail_padding():
    base = _make_base(fps=15.0, max_episode_blocks=-1)
    ds = _fake_ds([100], native_fps=15.0)
    obs_rows, act_rows, n = base._whole_episode_rows(ds, 0)
    assert n == 100
    assert len(obs_rows) == 101  # (100-1+3)//4*4 + 1
    assert obs_rows[:100] == list(range(100))
    assert obs_rows[100] == 99  # tail-row padding repeats the last real row
    assert act_rows == obs_rows[:100]


def test_rows_cap_applies():
    base = _make_base(fps=15.0, max_episode_blocks=4)  # cap = 1 + 4*4 = 17 frames
    ds = _fake_ds([100], native_fps=15.0)
    obs_rows, act_rows, n = base._whole_episode_rows(ds, 0)
    assert n == 17
    assert obs_rows == list(range(17))  # exactly 4N+1, no padding
    assert act_rows == list(range(16))


def test_rows_cap_shorter_episode_pads_up():
    base = _make_base(fps=15.0, max_episode_blocks=30)  # cap 121 > episode length
    ds = _fake_ds([10], native_fps=15.0)
    obs_rows, _act_rows, n = base._whole_episode_rows(ds, 0)
    assert n == 10
    assert len(obs_rows) == 13  # (10-1+3)//4*4 + 1
    assert obs_rows[10:] == [9, 9, 9]


def test_rows_fps_stride_subsampling():
    base = _make_base(fps=15.0, max_episode_blocks=-1)
    ds = _fake_ds([100], native_fps=30.0)  # stride 2
    obs_rows, act_rows, n = base._whole_episode_rows(ds, 0)
    assert n == 50
    assert obs_rows[:50] == list(range(0, 100, 2))
    assert len(obs_rows) == 53  # (50-1+3)//4*4 + 1
    assert obs_rows[50:] == [98, 98, 98]
    assert act_rows == obs_rows[:52]


def test_rows_second_episode_offsets():
    base = _make_base(fps=15.0, max_episode_blocks=-1)
    ds = _fake_ds([40, 9], native_fps=15.0)
    obs_rows, _act_rows, n = base._whole_episode_rows(ds, 1)
    assert n == 9
    assert obs_rows == list(range(40, 49))  # 9 = 4N+1, no padding


def test_rows_too_short_raises():
    base = _make_base(fps=15.0, max_episode_blocks=-1)
    ds = _fake_ds([1], native_fps=15.0)
    with pytest.raises(ValueError, match="at least 2"):
        base._whole_episode_rows(ds, 0)


def test_rows_non_integer_fps_ratio_raises():
    base = _make_base(fps=12.0, max_episode_blocks=-1)
    ds = _fake_ds([100], native_fps=30.0)  # 30/12 = 2.5
    with pytest.raises(AssertionError, match="integer native/target FPS"):
        base._whole_episode_rows(ds, 0)


def test_split_contiguous_row_runs_idle_gap():
    # Concat [2,5)+[8,12) at stride 1 from start=0 → two decode ranges, not min..max.
    rows = [2, 3, 4, 8, 9, 10, 11]
    runs = BaseActionLeRobotDataset._split_contiguous_row_runs(rows, stride=1)
    assert runs == [[2, 3, 4], [8, 9, 10, 11]]


def test_split_contiguous_row_runs_stride_and_single_run():
    rows = list(range(0, 20, 2))
    assert BaseActionLeRobotDataset._split_contiguous_row_runs(rows, stride=2) == [rows]
    assert BaseActionLeRobotDataset._split_contiguous_row_runs([], stride=1) == []


def _packed_episode(
    *,
    dataset_from: int,
    length: int,
    from_ts: float,
    to_ts: float | None = None,
    vid: str = "observation.image.wrist_image_left",
) -> dict[str, Any]:
    ep = {
        "dataset_from_index": dataset_from,
        "dataset_to_index": dataset_from + length,
        "length": length,
        f"videos/{vid}/from_timestamp": from_ts,
    }
    if to_ts is not None:
        ep[f"videos/{vid}/to_timestamp"] = to_ts
    return ep


def test_episode_mp4_slice_uses_to_timestamp():
    vid = "observation.image.exterior_image_1_left"
    # Matches Cosmos3-DROID episode 1: starts at 31.4s inside file-000 (15 fps).
    ep = _packed_episode(dataset_from=471, length=457, from_ts=31.4, to_ts=61.86666666666666, vid=vid)
    start, stop = BaseActionLeRobotDataset._episode_mp4_slice(ep, vid, native_fps=15.0)
    assert start == 471
    assert stop == 928


def test_episode_file_frame_indices_offset_into_packed_mp4():
    vid = "observation.image.exterior_image_1_left"
    ep = _packed_episode(dataset_from=471, length=457, from_ts=31.4, to_ts=61.86666666666666, vid=vid)
    # Native rows 471,473,475 → file frames 471,473,475 (stride 2 inside the episode).
    rows = [471, 473, 475]
    assert BaseActionLeRobotDataset._episode_file_frame_indices(ep, vid, rows, 15.0) == [471, 473, 475]


def test_episode_file_frame_indices_keep_ranges_stay_inside_slice():
    vid = "cam"
    ep = _packed_episode(dataset_from=100, length=20, from_ts=10.0, to_ts=10.0 + 20 / 15.0, vid=vid)
    # keep_ranges [2,5)+[8,12) at start=100 → rows 102,103,104,108,109,110,111
    rows = [102, 103, 104, 108, 109, 110, 111]
    indices = BaseActionLeRobotDataset._episode_file_frame_indices(ep, vid, rows, 15.0)
    assert indices == [152, 153, 154, 158, 159, 160, 161]
    file_start, file_stop = BaseActionLeRobotDataset._episode_mp4_slice(ep, vid, 15.0)
    assert file_start == 150
    assert all(file_start <= i < file_stop for i in indices)


def test_episode_file_frame_indices_rejects_packed_file_span():
    vid = "cam"
    # Episode occupies frames [150, 170) of a packed mp4 that is much longer.
    ep = _packed_episode(dataset_from=100, length=20, from_ts=10.0, to_ts=10.0 + 20 / 15.0, vid=vid)
    # Bug-shaped request: hf rows covering many packed episodes (would be ~whole file).
    rows = list(range(0, 13769))
    with pytest.raises(ValueError, match="outside episode mp4 slice"):
        BaseActionLeRobotDataset._episode_file_frame_indices(ep, vid, rows, 15.0)


def test_repeat_last_time_pads_dim0():
    t = torch.arange(3).view(3, 1)
    out = BaseActionLeRobotDataset._repeat_last_time(t, extra=2)
    assert out.tolist() == [[0], [1], [2], [2], [2]]
    assert BaseActionLeRobotDataset._repeat_last_time(t, extra=0) is t


def test_convert_video_uint8_permutes_only():
    base = _make_base(fps=15.0, max_episode_blocks=-1)
    video = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(2, 3, 4, 5)
    out = base._convert_video(video)
    assert out.dtype == torch.uint8
    assert out.shape == (3, 2, 4, 5)
    assert torch.equal(out, video.permute(1, 0, 2, 3))


def test_droid_compose_multi_view_uint8_pops_cameras():
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    obj = DROIDLeRobotDataset.__new__(DROIDLeRobotDataset)
    obj._image_features = {"wrist": "w", "left": "l", "right": "r"}
    obj._use_image_augmentation = False
    obj._image_augmentor = None
    sample = {
        "w": torch.arange(2 * 3 * 8 * 8, dtype=torch.uint8).reshape(2, 3, 8, 8),
        "l": torch.full((2, 3, 8, 8), 40, dtype=torch.uint8),
        "r": torch.full((2, 3, 8, 8), 200, dtype=torch.uint8),
    }
    out = DROIDLeRobotDataset._compose_multi_view(obj, sample)
    assert out.dtype == torch.uint8
    assert out.shape == (2, 3, 12, 8)
    assert "w" not in sample and "l" not in sample and "r" not in sample


def _droid_uninitialized() -> Any:
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    obj = DROIDLeRobotDataset.__new__(DROIDLeRobotDataset)
    obj._fps = 15.0
    obj._max_episode_blocks = -1
    obj._keep_ranges_by_episode = {}
    return obj


def test_droid_concat_keep_ranges_two_and_four_segments():
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    obj = _droid_uninitialized()
    fake = _fake_ds([20], native_fps=15.0)
    fake.meta.episodes[0]["episode_id"] = "ep0"
    obj._keep_ranges_by_episode = {"ep0": [[2, 5], [8, 12]]}
    obs, _act, n = DROIDLeRobotDataset._whole_episode_rows(obj, fake, 0)
    assert n == 7  # 3 + 4
    assert obs[:7] == [2, 3, 4, 8, 9, 10, 11]
    # 7 → 4N+1 = 9, last-row pad
    assert obs[7:] == [11, 11]

    obj._keep_ranges_by_episode = {"ep0": [[0, 2], [4, 6], [8, 10], [12, 14]]}
    obs4, _act4, n4 = DROIDLeRobotDataset._whole_episode_rows(obj, fake, 0)
    assert n4 == 8  # 4-seg must be kept, not dropped
    assert obs4[:8] == [0, 1, 4, 5, 8, 9, 12, 13]


def test_droid_keep_ranges_key_parse_and_clip():
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import (
        clip_droid_keep_ranges,
        droid_keep_ranges_episode_id,
        droid_keep_ranges_gcs_key,
    )

    ep = "AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023"
    gcs = droid_keep_ranges_gcs_key(ep)
    assert droid_keep_ranges_episode_id(gcs) == ep
    assert droid_keep_ranges_episode_id(ep) == ep
    assert clip_droid_keep_ranges([[-3, 5], [10, 99], [50, 50]], 20) == [[0, 5], [10, 20]]


def test_convert_keep_ranges_omits_empty_keeps_multiset():
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import droid_keep_ranges_gcs_key
    from cosmos_framework.scripts.convert_droid_keep_ranges import convert_keep_ranges

    ep_empty = "lab/success/empty"
    ep_four = "lab/success/four"
    src = {
        droid_keep_ranges_gcs_key(ep_empty): [],
        droid_keep_ranges_gcs_key(ep_four): [[0, 2], [4, 6], [8, 10], [12, 14]],
    }
    out = convert_keep_ranges(src)
    assert ep_empty not in out
    assert out[ep_four] == [[0, 2], [4, 6], [8, 10], [12, 14]]


def test_windowed_mode_warns_on_explicit_cap():
    # max_episode_blocks is ignored (with a warning) when chunk_length > 0.
    ds = BaseActionLeRobotDataset(
        fps=15.0,
        chunk_length=16,
        split_seed=42,
        split_val_ratio=0.0,
        split="full",
        mode="policy",
        embodiment_type="droid_lerobot",
        viewpoint="concat_view",
        pose_convention="backward_framewise",
        max_episode_blocks=8,
    )
    assert not ds._whole_episode


def test_libero_whole_episode_indices_round_up_and_uncapped():
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset

    ds = SimpleNamespace(_max_episode_blocks=-1)
    obs, act = LIBEROLeRobotDataset._whole_episode_obs_act_indices(ds, start=10, n=100)
    assert len(obs) == 101  # (100-1+3)//4*4 + 1
    assert obs[:100] == list(range(10, 110))
    assert obs[100] == 109
    assert act == obs[:100]


def test_libero_whole_episode_indices_cap():
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset

    ds = SimpleNamespace(_max_episode_blocks=4)  # cap = 1 + 4*4 = 17
    obs, act = LIBEROLeRobotDataset._whole_episode_obs_act_indices(ds, start=0, n=100)
    assert obs == list(range(17))
    assert act == list(range(16))


# ---------------------------------------------------------------------------
# WanVAE padding gate: pad only when AOT chunk functions are installed
# ---------------------------------------------------------------------------


def test_wanvae_should_pad_requires_aot():
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_

    with torch.device("meta"):
        vae = WanVAE_(encode_exact_durations=[17])

    # Eager mode (no _aot_chunk_fns): never pad, regardless of duration.
    for duration in (5, 17, 33, 145, 445):
        assert not vae._should_pad(duration)

    # AOT installed: pad everything except the enumerated exact durations.
    vae._aot_chunk_fns = {}
    assert vae._should_pad(33)
    assert vae._should_pad(145)
    assert not vae._should_pad(17)  # enumerated: exempt even under AOT


# ---------------------------------------------------------------------------
# Real-data validation (skipped when the local datasets are absent)
# ---------------------------------------------------------------------------


def _lerobot_meta(root: str) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id="local", root=root, revision="local")


def _flat_index_for_episode(ds: BaseActionLeRobotDataset, episode_id: int) -> int:
    # Whole-episode records have valid_len == 1, so flat index == record position.
    for flat, (_ds_idx, _start, _valid_len, eid) in enumerate(ds._episode_records):
        if eid == episode_id:
            return flat
    raise AssertionError(f"episode {episode_id} not found in records")


def _camera_hw(meta: Any, feature: str) -> tuple[int, int]:
    info = meta.info["features"][feature]
    names = info.get("names") or ["height", "width", "channel"]
    shape = info["shape"]
    return int(shape[names.index("height")]), int(shape[names.index("width")])


def _assert_tail_padding(video: torch.Tensor, action: torch.Tensor, n: int, target_t: int) -> None:
    """Padded video frames repeat the last real frame; padded action rows repeat
    the last real action.  ``action`` includes the initial-state row at index 0,
    so action row ``j`` corresponds to act-grid row ``j - 1``."""
    for frame_idx in range(n, target_t):
        assert torch.equal(video[:, frame_idx], video[:, n - 1]), f"video frame {frame_idx} is not tail padding"
    for row_idx in range(n + 1, target_t):
        assert torch.equal(action[row_idx], action[n]), f"action row {row_idx} is not tail padding"


@pytest.mark.skipif(not os.path.isdir(ROBOTWIN_ROOT), reason="local RoboTwin dataset not available")
def test_robotwin_whole_episode_real_data():
    from cosmos_framework.data.generator.action.datasets.robotwin_lerobot_dataset import RoboTwinLeRobotDataset

    meta = _lerobot_meta(ROBOTWIN_ROOT)
    lengths = [int(x) for x in meta.episodes["length"]]
    native_fps = float(meta.fps)
    cam_h, cam_w = _camera_hw(meta, "observation.images.cam_high")

    common = dict(
        root=ROBOTWIN_ROOT,
        fps=native_fps,  # stride 1
        chunk_length=-1,
        split="full",
        mode="policy",
        use_state=True,
        use_image_augmentation=False,
    )

    # Unlimited (-1): use the shortest usable episode to bound decode cost.
    ds = RoboTwinLeRobotDataset(**common, max_episode_blocks=-1)
    ep_short = min((e for e in range(len(lengths)) if lengths[e] >= 2), key=lambda e: lengths[e])
    n = lengths[ep_short]
    target_t = (n - 1 + 3) // 4 * 4 + 1
    item = ds[_flat_index_for_episode(ds, ep_short)]
    assert item["video"].shape == (3, target_t, cam_h * 3 // 2, cam_w)
    assert item["action"].shape == (target_t, 14)
    _assert_tail_padding(item["video"], item["action"], n, target_t)

    # Capped: 8 blocks = 33 frames, exercised on the longest episode.
    ds_cap = RoboTwinLeRobotDataset(**common, max_episode_blocks=8)
    ep_long = max(range(len(lengths)), key=lambda e: lengths[e])
    n_cap = min(lengths[ep_long], 33)
    target_cap = (n_cap - 1 + 3) // 4 * 4 + 1
    item_cap = ds_cap[_flat_index_for_episode(ds_cap, ep_long)]
    assert item_cap["video"].shape == (3, target_cap, cam_h * 3 // 2, cam_w)
    assert item_cap["action"].shape == (target_cap, 14)
    _assert_tail_padding(item_cap["video"], item_cap["action"], n_cap, target_cap)


@pytest.mark.skipif(not os.path.isdir(DROID_ROOT), reason="local DROID dataset not available")
def test_droid_whole_episode_real_data():
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    meta = _lerobot_meta(os.path.join(DROID_ROOT, "success"))
    # Local subsets may carry the full-dataset episodes parquet while info.json's
    # total_episodes reflects what is actually present; the framework's split
    # logic uses total_episodes, so mirror that here.
    lengths = [int(x) for x in meta.episodes["length"]][: meta.total_episodes]
    native_fps = float(meta.fps)
    wrist_h, wrist_w = _camera_hw(meta, "observation.image.wrist_image_left")

    common = dict(
        root=DROID_ROOT,
        fps=native_fps,  # stride 1
        chunk_length=-1,
        split="full",
        mode="policy",
        action_space="joint_pos",
        use_state=True,
        use_success_only=True,
        use_image_augmentation=False,
    )

    # Unlimited (-1): use the shortest usable episode (>= 33 frames — shorter
    # episodes are dropped from the index) to bound decode cost.
    ds = DROIDLeRobotDataset(**common, max_episode_blocks=-1)
    ep_short = min((e for e in range(len(lengths)) if lengths[e] >= 33), key=lambda e: lengths[e])
    n = lengths[ep_short]
    target_t = (n - 1 + 3) // 4 * 4 + 1
    item = ds[_flat_index_for_episode(ds, ep_short)]
    assert item["video"].shape == (3, target_t, wrist_h * 3 // 2, wrist_w)
    assert item["action"].shape == (target_t, 8)  # 7 joints + gripper
    _assert_tail_padding(item["video"], item["action"], n, target_t)

    # Capped: 8 blocks = 33 frames, exercised on the longest episode
    # (cheap — rows are truncated before any decode).
    ds_cap = DROIDLeRobotDataset(**common, max_episode_blocks=8)
    ep_long = max(range(len(lengths)), key=lambda e: lengths[e])
    n_cap = min(lengths[ep_long], 33)
    target_cap = (n_cap - 1 + 3) // 4 * 4 + 1
    item_cap = ds_cap[_flat_index_for_episode(ds_cap, ep_long)]
    assert item_cap["video"].shape == (3, target_cap, wrist_h * 3 // 2, wrist_w)
    assert item_cap["action"].shape == (target_cap, 8)
    _assert_tail_padding(item_cap["video"], item_cap["action"], n_cap, target_cap)


@pytest.mark.skipif(not os.path.isdir(DROID_ROOT), reason="local DROID dataset not available")
def test_droid_whole_episode_keep_ranges_concat_real_data(tmp_path):
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    target_ep = "AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023"
    empty_ep = "AUTOLab/success/2023-07-07/Fri_Jul__7_10:43:53_2023"
    filter_path = tmp_path / "keep_ranges.json"
    filter_path.write_text(json.dumps({target_ep: [[35, 446]], empty_ep: []}))

    meta = _lerobot_meta(os.path.join(DROID_ROOT, "success"))
    wrist_h, wrist_w = _camera_hw(meta, "observation.image.wrist_image_left")
    ds = DROIDLeRobotDataset(
        root=DROID_ROOT,
        fps=float(meta.fps),
        chunk_length=-1,
        split="full",
        mode="policy",
        action_space="joint_pos",
        use_state=True,
        use_success_only=True,
        use_image_augmentation=False,
        use_filter_dict=True,
        filter_dict_path=str(filter_path),
        max_episode_blocks=-1,
    )
    # empty [] and concats shorter than 33 frames dropped at index build;
    # only the 411-frame concat remains
    assert len(ds) == 1
    n, target_t = 411, 413
    item = ds[0]
    assert item["video"].shape == (3, target_t, wrist_h * 3 // 2, wrist_w)
    assert item["action"].shape == (target_t, 8)
    _assert_tail_padding(item["video"], item["action"], n, target_t)


@pytest.mark.skipif(not os.path.isdir(DROID_ROOT), reason="local DROID dataset not available")
def test_droid_keep_ranges_drops_over_max_episode_length_frames(tmp_path):
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    target_ep = "AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023"
    filter_path = tmp_path / "keep_ranges.json"
    filter_path.write_text(json.dumps({target_ep: [[35, 446]]}))  # 411 frames, pad 413

    common = dict(
        root=DROID_ROOT,
        fps=15.0,
        chunk_length=-1,
        split="full",
        mode="policy",
        action_space="joint_pos",
        use_state=True,
        use_success_only=True,
        use_image_augmentation=False,
        use_filter_dict=True,
        filter_dict_path=str(filter_path),
        max_episode_blocks=-1,
    )
    # 413 padded > 400 → dropped at index; no video decode of this episode.
    ds_drop = DROIDLeRobotDataset(**common, max_episode_length_frames=400)
    assert len(ds_drop) == 0
    # 413 <= 413 → kept.
    ds_keep = DROIDLeRobotDataset(**common, max_episode_length_frames=413)
    assert len(ds_keep) == 1


@pytest.mark.skipif(not os.path.isdir(DROID_ROOT), reason="local DROID dataset not available")
def test_droid_keep_ranges_drops_short_concat(tmp_path):
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    target_ep = "AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023"
    filter_path = tmp_path / "keep_ranges.json"
    # 20 concatenated frames < 33; must drop even though the raw episode is long.
    filter_path.write_text(json.dumps({target_ep: [[0, 20]]}))

    ds = DROIDLeRobotDataset(
        root=DROID_ROOT,
        fps=15.0,
        chunk_length=-1,
        split="full",
        mode="policy",
        action_space="joint_pos",
        use_state=True,
        use_success_only=True,
        use_image_augmentation=False,
        use_filter_dict=True,
        filter_dict_path=str(filter_path),
        max_episode_blocks=-1,
    )
    assert len(ds) == 0


def test_video_decoder_cache_rejects_bad_open_mode():
    from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
        _LRUVideoDecoderCache,
    )

    with pytest.raises(ValueError, match="open_mode"):
        _LRUVideoDecoderCache(open_mode="avio")


def test_pyav_index_decoder_roundtrip_and_close(tmp_path):
    av = pytest.importorskip("av")
    from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
        _PyAVIndexDecoder,
    )

    path = tmp_path / "toy.mp4"
    height, width, n_frames, fps = 16, 16, 8, 15
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for t in range(n_frames):
        arr = np.full((height, width, 3), t * 20, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    decoder = _PyAVIndexDecoder(str(path))
    try:
        assert decoder._stream.thread_type.name == "NONE"
        out = decoder.get_frames_at([0, 3, 7]).data
        assert out.dtype == torch.uint8
        assert out.shape == (3, 3, height, width)
    finally:
        decoder.close()
    assert decoder._container is None
    assert decoder._stream is None


def test_worker_restart_every_n_stops_iterator():
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
        ActionIterableShuffleDataset,
    )

    class _Fake:
        def get_shuffle_blocks(self):
            return [(0, 8)]

        def __getitem__(self, idx):
            return {"idx": idx}

    ds = ActionIterableShuffleDataset(_Fake(), seed=0, worker_restart_every_n=3)
    got = list(ds)
    assert [row["idx"] for row in got] == [0, 1, 2]
