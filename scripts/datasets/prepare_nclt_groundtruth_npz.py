#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def _read_groundtruth_csv(path: Path) -> np.ndarray:
    # NCLT groundtruth rows are:
    # timestamp_us, x, y, z, roll, pitch, yaw
    arr = np.loadtxt(
        str(path),
        delimiter=",",
        usecols=(0, 1, 2, 3, 4, 5, 6),
        dtype=np.float64,
    )
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"Unexpected NCLT groundtruth shape at {path}: {arr.shape}")
    return arr


def _downsample(arr: np.ndarray, stride: int, max_samples: int) -> np.ndarray:
    out = arr
    if stride > 1:
        out = out[::stride]
    if max_samples > 0 and out.shape[0] > max_samples:
        idx = np.linspace(0, out.shape[0] - 1, num=max_samples, dtype=np.int64)
        out = out[idx]
    return out


def _prepare_one(csv_path: Path, out_path: Path, downsample_stride: int, max_samples: int) -> None:
    data = _read_groundtruth_csv(csv_path)
    data = _downsample(data, stride=downsample_stride, max_samples=max_samples)

    ts_us = data[:, 0].astype(np.float64, copy=False)
    # x_dim=6 expected by TG7 NCLT smoke config.
    x = data[:, 1:7].astype(np.float32, copy=False)
    # y_dim=2 expected by TG7 NCLT smoke config (position projection).
    y = data[:, 1:3].astype(np.float32, copy=False)

    if x.shape[0] < 2000 and "2012-04-29" in csv_path.name:
        raise ValueError(
            f"{csv_path} produced only {x.shape[0]} rows after downsampling; "
            "need at least 2000 rows for TG7 test split."
        )
    if x.shape[0] < 300 and "2012-01-22" in csv_path.name:
        raise ValueError(
            f"{csv_path} produced only {x.shape[0]} rows after downsampling; "
            "need at least ~300 rows for TG7 train/val splits."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        x=x,
        y=y,
        ts_us=ts_us,
    )
    print(f"[ok] wrote {out_path} x={x.shape} y={y.shape}")


def _default_sessions() -> List[str]:
    return ["2012-01-22", "2012-04-29"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare NCLT groundtruth CSV into TG7 loader-ready NPZ files.")
    ap.add_argument("--raw-root", type=str, default="external_data/nclt/raw/ground_truth")
    ap.add_argument("--out-root", type=str, default="external_data/nclt")
    ap.add_argument("--sessions", nargs="*", default=_default_sessions())
    ap.add_argument(
        "--downsample-stride",
        type=int,
        default=10,
        help="Temporal stride to reduce file size while preserving long trajectories.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=300000,
        help="Cap rows per session after downsampling (0 disables cap).",
    )
    args = ap.parse_args()

    if args.downsample_stride <= 0:
        raise ValueError("--downsample-stride must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    prepared = out_root / "prepared"

    for session in args.sessions:
        session_s = str(session).strip()
        csv_path = raw_root / f"groundtruth_{session_s}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {csv_path}")
        out_path = prepared / f"{session_s}.npz"
        _prepare_one(
            csv_path=csv_path,
            out_path=out_path,
            downsample_stride=int(args.downsample_stride),
            max_samples=int(args.max_samples),
        )

    print("")
    print("Done.")
    print(f"Set: export NCLT_ROOT={out_root}")
    print(
        "Test: .venv/bin/python -m bench.tasks.smoke_data "
        "--suite-yaml bench/configs/suite_tg7_datasets_smoke.yaml "
        "--task TG7_nclt_loader_smoke_v0 --seed 0"
    )


if __name__ == "__main__":
    main()
