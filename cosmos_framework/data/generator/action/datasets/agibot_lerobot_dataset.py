# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""AgiBot World LeRobot dataset (Cosmos3 ``BaseActionLeRobotDataset`` path).

Discovers LeRobot v3 shards under an ``Agibotworld/`` parent:

    Agibotworld/task_<id>/canonical_55d/{meta,data,videos}

or accepts a single LeRobot v3 directory (or a ``task_*`` dir that contains
``canonical_55d``). Whole-episode fetching only (``chunk_length=-1``);
windowed mode raises ``NotImplementedError``.

Cameras: head + left/right wrist (fisheye keys in ``info.json`` are ignored).
Action spaces:

- ``ee_pose`` / ``midtrain``: cookbook 29D FK layout (all three poses
  ``backward_framewise`` relative), matching ``AgiBotWorldBetaLeRobotDataset``.
- ``joint_pos``: 16D joints+grippers + 9D head EE delta.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.agibot_fk import (
    AGIBOT_WORLD_GRIPPER_TO_OPENCV_BY_WRIST,
    apply_agibot_gripper_to_opencv,
    apply_robot_base_motion_to_poses,
    compute_fk_transforms_batch,
    convert_gripper_state_to_open_fraction,
)
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Joint,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.pose_utils import convert_rotation, pose_abs_to_rel
from cosmos_framework.data.generator.action.viewpoint_utils import Viewpoint
from cosmos_framework.utils import log

DEFAULT_AGIBOT_ROOT = "/data5T/Embodied-AI/datasets/Agibotworld"
_CANONICAL_SUBDIR = "canonical_55d"
_AGIBOT_MIN_EPISODE_FRAMES = 33

_EE_POSE_SPACES = ("ee_pose", "midtrain")
_SUPPORTED_ACTION_SPACES = ("ee_pose", "midtrain", "joint_pos")

_HEAD_CAM = "observation.images.head"
_HAND_LEFT_CAM = "observation.images.hand_left"
_HAND_RIGHT_CAM = "observation.images.hand_right"

_OBS_EFFECTOR = "observation.states.effector.position"
_OBS_JOINT = "observation.states.joint.position"
_OBS_HEAD = "observation.states.head.position"
_OBS_WAIST = "observation.states.waist.position"
_OBS_ROBOT_POS = "observation.states.robot.position"
_OBS_ROBOT_QUAT = "observation.states.robot.orientation"
_ACT_JOINT = "actions.joint.position"
_ACT_EFFECTOR = "actions.effector.position"

_OBS_STATE_FEATURES = (
    _OBS_JOINT,
    _OBS_EFFECTOR,
    _OBS_HEAD,
    _OBS_WAIST,
    _OBS_ROBOT_POS,
    _OBS_ROBOT_QUAT,
)


def _is_lerobot_v3(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def resolve_agibot_shard_roots(root: str | Path) -> tuple[list[str], list[str]]:
    """Resolve ``root`` to ``(shard_paths, task_labels)``.

    Symlinks are **not** de-duplicated: ``task_350 -> task_352/`` registers
    both names so a parent can simulate multiple tasks.
    """
    root_p = Path(root)
    if _is_lerobot_v3(root_p):
        return [str(root_p)], [root_p.name]

    canonical = root_p / _CANONICAL_SUBDIR
    if _is_lerobot_v3(canonical):
        return [str(canonical)], [root_p.name]

    if not root_p.is_dir():
        raise FileNotFoundError(f"AgiBot root does not exist: {root_p}")

    shards: list[str] = []
    labels: list[str] = []
    skipped: list[str] = []
    for child in sorted(root_p.iterdir(), key=lambda p: p.name):
        if not child.name.startswith("task_"):
            continue
        cand = child / _CANONICAL_SUBDIR
        if _is_lerobot_v3(cand):
            shards.append(str(cand))
            labels.append(child.name)
        else:
            skipped.append(child.name)

    if skipped:
        log.info(
            f"AgiBotLeRobotDataset: skipped {len(skipped)} task dirs without "
            f"{_CANONICAL_SUBDIR}/meta/info.json: {skipped}"
        )
    if not shards:
        raise FileNotFoundError(
            f"No LeRobot v3 shards under {root_p}. Expected either a dataset "
            f"with meta/info.json, a task dir with {_CANONICAL_SUBDIR}/, or "
            f"task_*/{_CANONICAL_SUBDIR}/ children."
        )
    return shards, labels


def _split_task_for_caption(task: str) -> str:
    ai_caption, separator, _debug = str(task).partition("|")
    if not separator:
        return str(task).strip()
    return ai_caption.strip()


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _assemble_agibot_world_state(
    effector_pos: np.ndarray,
    joint_pos: np.ndarray,
    head_pos: np.ndarray,
    waist_pos: np.ndarray,
) -> np.ndarray:
    """20D gripper state: joints(14) + effector(2) + head(2) + waist(2)."""
    body_head = np.stack(
        [head_pos[:, 0], head_pos[:, 1], waist_pos[:, 0], waist_pos[:, 1]],
        axis=-1,
    )
    return np.concatenate([joint_pos, effector_pos, body_head], axis=-1).astype(np.float32, copy=False)


def _poses_to_9d(poses: np.ndarray) -> np.ndarray:
    """``[T,4,4]`` absolute poses → ``[T,9]`` xyz+rot6d."""
    translation = poses[:, :3, 3]
    rotation = np.asarray(
        convert_rotation(poses[:, :3, :3], input_format="matrix", output_format="rot6d"),
        dtype=np.float32,
    )
    return np.concatenate([translation, rotation], axis=-1).astype(np.float32)


def _open_fraction_per_hand(effector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert obs effector ``[T,2]`` to open fractions, one hand at a time."""
    left = convert_gripper_state_to_open_fraction(effector[:, 0:1])
    right = convert_gripper_state_to_open_fraction(effector[:, 1:2])
    return left, right


class AgiBotLeRobotDataset(BaseActionLeRobotDataset):
    """AgiBot World dual-arm dataset on the Cosmos3 LeRobot adapter.

    ``chunk_length`` must be ``-1`` (whole-episode). ``fps`` defaults to 15
    (native 30 → integer stride 2). Concat view puts the head camera on top
    and the two wrist cameras half-size below.
    """

    EMBODIMENT_TYPE: str = "agibotworld"

    def __init__(
        self,
        root: str = DEFAULT_AGIBOT_ROOT,
        fps: float = 15.0,
        chunk_length: int = -1,
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        split: str = "train",
        mode: str = "policy",
        action_space: str = "ee_pose",
        use_state: bool = True,
        action_normalization: ActionNormalization | None = None,
        tolerance_s: float = 3e-4,
        viewpoint: Viewpoint = "concat_view",
        use_image_augmentation: bool = False,
        enable_fast_init: bool = False,
        max_episode_blocks: int = -1,
        max_episode_length_frames: int | None = None,
        video_backend: str | None = None,
        video_decoder_cache_size: int = 64,
        video_decoder_open_mode: str = "fsspec",
        skip_video_loading: bool = False,
    ) -> None:
        if chunk_length != -1:
            raise NotImplementedError(
                f"AgiBotLeRobotDataset only supports whole-episode fetching "
                f"(chunk_length=-1); got chunk_length={chunk_length}."
            )
        if action_space not in _SUPPORTED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space={action_space!r}. Supported: {_SUPPORTED_ACTION_SPACES}.")
        if viewpoint not in ("concat_view", "ego_view"):
            raise ValueError(f"Unsupported viewpoint={viewpoint!r}. Use concat_view or ego_view.")

        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=self.EMBODIMENT_TYPE,
            viewpoint=viewpoint,
            pose_convention="backward_framewise",
            rotation_format="rot6d",
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            enable_fast_init=enable_fast_init,
            max_episode_blocks=max_episode_blocks,
            min_episode_length_frames=_AGIBOT_MIN_EPISODE_FRAMES,
            max_episode_length_frames=max_episode_length_frames,
            video_backend=video_backend,
            video_decoder_cache_size=video_decoder_cache_size,
            video_decoder_open_mode=video_decoder_open_mode,
            skip_video_loading=skip_video_loading,
        )
        self._action_space = "ee_pose" if action_space in _EE_POSE_SPACES else action_space
        self._use_state = use_state
        self._use_image_augmentation = use_image_augmentation
        self._image_augmentor: T.Compose | None = None
        self._delta_timestamps = {}

        self._all_shard_roots, self._shard_labels = resolve_agibot_shard_roots(root)
        self._register_sources()

    def _register_sources(self, indices: list[int] | None = None) -> None:
        """Register shards with ``task_*`` labels (not ``canonical_55d``)."""
        if indices is None:
            indices = list(range(len(self._all_shard_roots)))
        if not indices:
            return
        for i in indices:
            self._register_source(
                root=self._all_shard_roots[i],
                delta_timestamps=self._delta_timestamps,
                tolerance_s=self._tolerance_s,
                video_backend=self._video_backend,
                dataset_label=self._shard_labels[i],
            )

    def _compose_multi_view(self, sample: dict[str, Any]) -> torch.Tensor:
        """Head full-size on top, left/right wrists half-size below.

        Returns:
            Composited video in ``(T,C,3H/2,W)``.
        """
        top = sample[_HEAD_CAM]  # [T,C,H,W]
        left = sample[_HAND_LEFT_CAM]
        right = sample[_HAND_RIGHT_CAM]

        if self._use_image_augmentation:
            if self._image_augmentor is None:
                _, _, h, w = top.shape
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                        T.Resize((h, w), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            n, m = top.shape[0], top.shape[0] + left.shape[0]
            combined = self._image_augmentor(torch.cat([top, left, right], dim=0))
            del top, left, right
            top, left, right = combined[:n], combined[n:m], combined[m:]
            del combined

        _, _, h_t, w_t = top.shape
        half_h, half_w = h_t // 2, w_t // 2
        left_ds = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        del left
        right_ds = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        del right
        bottom = torch.cat([left_ds, right_ds], dim=-1)
        del left_ds, right_ds
        composite = torch.cat([top, bottom], dim=-2)
        del top, bottom
        sample.pop(_HEAD_CAM, None)
        sample.pop(_HAND_LEFT_CAM, None)
        sample.pop(_HAND_RIGHT_CAM, None)
        return composite

    def _build_action_spec(self) -> ActionSpec:
        if self._action_space == "joint_pos":
            return build_action_spec(
                Joint(n=7, prefix="left"),
                Gripper(prefix="left"),
                Joint(n=7, prefix="right"),
                Gripper(prefix="right"),
                Pos(prefix="head"),
                Rot("rot6d", prefix="head"),
            )
        return build_action_spec(
            Pos(prefix="head"),
            Rot("rot6d", prefix="head"),
            Pos(prefix="right"),
            Rot("rot6d", prefix="right"),
            Gripper(prefix="right"),
            Pos(prefix="left"),
            Rot("rot6d", prefix="left"),
            Gripper(prefix="left"),
        )

    @property
    def action_dim(self) -> int:
        return 25 if self._action_space == "joint_pos" else 29

    def _whole_episode_tabular_grids(self) -> dict[str, str]:
        grids = {key: "obs" for key in _OBS_STATE_FEATURES}
        if self._action_space == "joint_pos":
            grids[_ACT_JOINT] = "act"
            grids[_ACT_EFFECTOR] = "act"
        return grids

    def _whole_episode_camera_features(self) -> list[str]:
        if self._viewpoint == "concat_view":
            return [_HEAD_CAM, _HAND_LEFT_CAM, _HAND_RIGHT_CAM]
        return [_HEAD_CAM]

    def _fk_from_sample(self, sample: dict[str, Any]) -> dict[str, np.ndarray]:
        states = _assemble_agibot_world_state(
            effector_pos=_as_np(sample[_OBS_EFFECTOR]),
            joint_pos=_as_np(sample[_OBS_JOINT]),
            head_pos=_as_np(sample[_OBS_HEAD]),
            waist_pos=_as_np(sample[_OBS_WAIST]),
        )
        native = compute_fk_transforms_batch(states, "agibot_world_gripper")
        native = apply_robot_base_motion_to_poses(
            native,
            _as_np(sample[_OBS_ROBOT_POS]),
            _as_np(sample[_OBS_ROBOT_QUAT]),
        )
        return apply_agibot_gripper_to_opencv(native, AGIBOT_WORLD_GRIPPER_TO_OPENCV_BY_WRIST)

    def _build_ee_pose_action(self, sample: dict[str, Any], fk: dict[str, np.ndarray]) -> torch.Tensor:
        head_rel = pose_abs_to_rel(fk["head_camera"], rotation_format="rot6d", pose_convention="backward_framewise")
        right_rel = pose_abs_to_rel(fk["right_wrist"], rotation_format="rot6d", pose_convention="backward_framewise")
        left_rel = pose_abs_to_rel(fk["left_wrist"], rotation_format="rot6d", pose_convention="backward_framewise")
        left_open, right_open = _open_fraction_per_hand(_as_np(sample[_OBS_EFFECTOR]))
        rel = np.concatenate(
            [head_rel, right_rel, right_open[1:], left_rel, left_open[1:]],
            axis=-1,
        ).astype(np.float32)
        action = torch.from_numpy(rel).float()
        if self._use_state:
            abs0 = np.concatenate(
                [
                    _poses_to_9d(fk["head_camera"][:1])[0],
                    _poses_to_9d(fk["right_wrist"][:1])[0],
                    right_open[0],
                    _poses_to_9d(fk["left_wrist"][:1])[0],
                    left_open[0],
                ]
            ).astype(np.float32)
            action = torch.cat([torch.from_numpy(abs0).float().unsqueeze(0), action], dim=0)
        return action

    def _build_joint_pos_action(self, sample: dict[str, Any], fk: dict[str, np.ndarray]) -> torch.Tensor:
        joint_act = _as_np(sample[_ACT_JOINT])
        eff_act = _as_np(sample[_ACT_EFFECTOR])
        head_rel = pose_abs_to_rel(fk["head_camera"], rotation_format="rot6d", pose_convention="backward_framewise")
        rel = np.concatenate(
            [joint_act[:, 0:7], eff_act[:, 0:1], joint_act[:, 7:14], eff_act[:, 1:2], head_rel],
            axis=-1,
        ).astype(np.float32)
        action = torch.from_numpy(rel).float()
        if self._use_state:
            joint0 = _as_np(sample[_OBS_JOINT])[0]
            left_open, right_open = _open_fraction_per_hand(_as_np(sample[_OBS_EFFECTOR]))
            abs0 = np.concatenate(
                [
                    joint0[0:7],
                    left_open[0],
                    joint0[7:14],
                    right_open[0],
                    _poses_to_9d(fk["head_camera"][:1])[0],
                ]
            ).astype(np.float32)
            action = torch.cat([torch.from_numpy(abs0).float().unsqueeze(0), action], dim=0)
        return action

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)
        ai_caption = _split_task_for_caption(sample["task"])

        if self._skip_video_loading:
            video = None
        elif self._viewpoint == "concat_view":
            video = self._compose_multi_view(sample)
        else:
            video = sample[_HEAD_CAM]

        fk = self._fk_from_sample(sample)
        extras: dict[str, Any] = {
            "initial_pose": torch.from_numpy(fk["head_camera"][0].copy()).float(),
            "initial_pose_right": torch.from_numpy(fk["right_wrist"][0].copy()).float(),
            "initial_pose_left": torch.from_numpy(fk["left_wrist"][0].copy()).float(),
        }
        if self._viewpoint == "concat_view":
            extras["additional_view_description"] = (
                "The top row shows the head-mounted camera view looking down at the workspace. "
                "The bottom row contains two horizontally concatenated wrist-mounted camera views: "
                "the left hand camera on the left and the right hand camera on the right."
            )

        if self._action_space == "joint_pos":
            action = self._build_joint_pos_action(sample, fk)
        else:
            action = self._build_ee_pose_action(sample, fk)

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            **extras,
        )
