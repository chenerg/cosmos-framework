# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Remap KarlP/droid ``keep_ranges_1_0_1.json`` GCS keys to LeRobot ``episode_id``.

Default conversion is the **full** dict (all ~95k trajectories). Empty ``[]``
entries are omitted; the loader also drops them at index build. Segment counts
are not capped. Ranges are **not** clipped to a particular LeRobot ``length``
unless ``--lerobot-root`` is passed (debug subset only).

Usage
-----
    python -m cosmos_framework.scripts.convert_droid_keep_ranges \\
        --input keep_ranges_1_0_1.json \\
        --output $DROID_ROOT/keep_ranges_1_0_1_whole_episode.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import (
    clip_droid_keep_ranges,
    droid_keep_ranges_episode_id,
)


def _load_lerobot_episode_lengths(root: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    lengths: dict[str, int] = {}
    for parquet in sorted(root.glob("*/meta/episodes/**/*.parquet")):
        table = pq.read_table(parquet, columns=["episode_id", "length"])
        for ep_id, length in zip(table["episode_id"].to_pylist(), table["length"].to_pylist()):
            lengths[str(ep_id)] = int(length)
    if not lengths:
        raise FileNotFoundError(f"No episodes parquet under {root}/*/meta/episodes")
    return lengths


def convert_keep_ranges(
    src: dict,
    *,
    episode_lengths: dict[str, int] | None = None,
    omit_empty: bool = True,
) -> dict[str, list[list[int]]]:
    """GCS-key (or already-short) dict -> ``{episode_id: [[s, e], ...]}``."""
    out: dict[str, list[list[int]]] = {}
    for key, ranges in src.items():
        ep_id = droid_keep_ranges_episode_id(key)
        pairs = [[int(s), int(e)] for s, e in (ranges or [])]
        if episode_lengths is not None:
            length = episode_lengths.get(ep_id)
            if length is None:
                continue
            pairs = clip_droid_keep_ranges(pairs, length)
        if omit_empty and not pairs:
            continue
        out[ep_id] = pairs
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="KarlP keep_ranges_1_0_1.json")
    parser.add_argument("--output", type=Path, required=True, help="Remapped JSON path")
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=None,
        help="Optional versioned DROID parent; intersect + clip to that tree only.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Write empty [] entries instead of omitting them.",
    )
    args = parser.parse_args(argv)

    with args.input.open() as f:
        src = json.load(f)
    lengths = _load_lerobot_episode_lengths(args.lerobot_root) if args.lerobot_root is not None else None
    out = convert_keep_ranges(src, episode_lengths=lengths, omit_empty=not args.keep_empty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {len(out)} episodes -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
