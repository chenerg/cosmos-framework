# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboTwin LeRobot dataset (dual-arm ALOHA, joint-space action policy).

Loads a single RoboTwin-LeRobot-v3.0 task dataset (e.g.
``RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50``) through the
shared :class:`BaseActionLeRobotDataset` machinery (lazy LeRobot shards,
episode-span flat index, ``get_shuffle_blocks`` streaming support).

Data layout (fixed for the v3.0 release, so no per-version registry like
DROID's):

- ``observation.state`` / ``action``: 14D joint-space vectors
  ``[left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]``,
  grippers already normalized to ``[0, 1]`` (no flipping needed).
- Three 480x640 cameras: ``cam_high`` (overhead third-person),
  ``cam_left_wrist``, ``cam_right_wrist``.
- One instruction string per episode in ``tasks.parquet``.

The multi-view composition matches ``DROIDLeRobotDataset._compose_multi_view``
(one full-size view on top, two half-size views tiled below): here the
overhead ``cam_high`` sits on top and the two wrist cameras form the bottom
row.

Action spaces: only ``joint_pos`` (14D) is supported. The DROID ``midtrain``
and ``ee_pose_delta`` spaces require end-effector cartesian poses, which
RoboTwin does not store — requesting them raises ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Joint,
    build_action_spec,
)
from cosmos_framework.data.generator.action.viewpoint_utils import Viewpoint

_STATE_FEATURE = "observation.state"  # 14D joint state
_ACTION_FEATURE = "action"  # 14D joint action
_IMAGE_FEATURES = {
    "top": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}

_EE_POSE_ACTION_SPACES = ("midtrain", "ee_pose_delta")
_SUPPORTED_ACTION_SPACES = ("joint_pos",)


class RoboTwinLeRobotDataset(BaseActionLeRobotDataset):
    """RoboTwin dual-arm ALOHA action-policy dataset.

    14D joint-space actions ``[left_joint(6), left_gripper, right_joint(6),
    right_gripper]`` at the dataset's native 30 FPS grid, with the
    DROID-style three-view composite (``cam_high`` full-size on top, the two
    wrist cameras half-size below).
    """

    EMBODIMENT_TYPE: str = "robotwin_lerobot"

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        split: str = "train",
        mode: str = "policy",
        action_space: str = "joint_pos",
        use_state: bool = False,
        action_normalization: ActionNormalization | None = None,
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "concat_view",
        use_image_augmentation: bool = False,
        enable_fast_init: bool = False,
    ) -> None:
        if action_space in _EE_POSE_ACTION_SPACES:
            raise NotImplementedError(
                f"action_space={action_space!r} requires end-effector cartesian poses, but RoboTwin "
                "stores only 14D joint-space state/action (no EE pose feature). "
                "Use action_space='joint_pos'."
            )
        if action_space not in _SUPPORTED_ACTION_SPACES:
            raise ValueError(
                f"Unsupported action_space={action_space!r}. Supported: {_SUPPORTED_ACTION_SPACES} "
                f"(EE-pose spaces {_EE_POSE_ACTION_SPACES} are unavailable for RoboTwin)."
            )
        if viewpoint not in ("concat_view", "third_person_view"):
            raise ValueError(f"Unsupported viewpoint={viewpoint!r}. Use concat_view or third_person_view.")

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
            rotation_format=None,  # joint-space actions carry no rotation block
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            enable_fast_init=enable_fast_init,
        )
        self._action_space = action_space
        self._use_state = use_state
        self._use_image_augmentation = use_image_augmentation
        self._image_augmentor: T.Compose | None = None

        # Single-dataset root (one task/embodiment dir containing meta/info.json).
        self._all_shard_roots = [root]

        observation_ts = [i * self._dt for i in range(0, self._chunk_length + 1)]
        action_ts = [i * self._dt for i in range(0, self._chunk_length)]
        self._delta_timestamps: dict[str, list[float]] = {_ACTION_FEATURE: action_ts}
        if self._use_state:
            self._delta_timestamps[_STATE_FEATURE] = observation_ts
        if self._viewpoint == "concat_view":
            for view in ("top", "left", "right"):
                self._delta_timestamps[_IMAGE_FEATURES[view]] = observation_ts
        else:  # third_person_view
            self._delta_timestamps[_IMAGE_FEATURES["top"]] = observation_ts

        # Eager source registration (mirrors DROIDLeRobotDataset):
        # ActionIterableShuffleDataset block-strides over the full flat index,
        # so every worker needs it present at construction. Metadata-only; the
        # heavy per-shard video readers stay lazy behind the LRU in _get_dataset.
        self._register_sources()

    def _compose_multi_view(self, sample: dict[str, Any]) -> torch.Tensor:
        """Compose overhead + two wrist views into a single frame.

        Layout (per frame, same geometry as ``DROIDLeRobotDataset``):
            ┌──────────────┐
            │   cam_high   │   (H, W)
            ├───────┬──────┤
            │ Lwrist│Rwrist│   (H/2, W/2) each
            └───────┴──────┘

        The two wrist cameras are downscaled by 2x so they tile to the same
        width as the overhead view. The output height is 3H/2.

        Returns:
            Composited raw video tensor in ``(T,C,3H/2,W)`` float format.
        """
        top = sample[_IMAGE_FEATURES["top"]]  # [T,C,H,W]
        left = sample[_IMAGE_FEATURES["left"]]  # [T,C,H,W]
        right = sample[_IMAGE_FEATURES["right"]]  # [T,C,H,W]

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
            top, left, right = combined[:n], combined[n:m], combined[m:]

        _, _, h_t, w_t = top.shape
        half_h, half_w = h_t // 2, w_t // 2

        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]

        return torch.cat([top, bottom], dim=-2)  # [T,C,3H/2,W]

    def _build_action_spec(self) -> ActionSpec:
        """RoboTwin dual-arm ALOHA: 14D
        ``[left_joint(6), left_gripper, right_joint(6), right_gripper]``."""
        return build_action_spec(
            Joint(n=6, prefix="left"),
            Gripper(prefix="left"),
            Joint(n=6, prefix="right"),
            Gripper(prefix="right"),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)

        # One instruction string per episode (no " | " multi-annotation).
        ai_caption = sample["task"]

        if self._skip_video_loading:
            video = None
        elif self._viewpoint == "concat_view":
            video = self._compose_multi_view(sample)
        else:
            video = sample[_IMAGE_FEATURES["top"]]  # [T,C,H,W]

        # joint_pos: keep the native 14D layout; grippers are already in [0, 1].
        action = sample[_ACTION_FEATURE][-self._chunk_length :].float()  # [chunk, 14]
        if self._use_state:
            initial_state = sample[_STATE_FEATURE][-self._chunk_length - 1].float()  # [14]
            action = torch.cat([initial_state.unsqueeze(0), action], dim=0)  # [chunk+1, 14]

        extras: dict[str, Any] = {}
        if self._viewpoint == "concat_view":
            extras["additional_view_description"] = (
                "The top row is a third-person overhead view of the scene. "
                "The bottom row contains two horizontally concatenated wrist-mounted camera views, "
                "the left arm's wrist on the left and the right arm's wrist on the right."
            )

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            **extras,
        )

    @property
    def action_dim(self) -> int:
        return 14
