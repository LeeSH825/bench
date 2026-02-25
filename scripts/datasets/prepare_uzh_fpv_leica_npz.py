#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

import numpy as np


def _read_leica_positions(zip_path: Path) -> np.ndarray:
    with ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        leica_name = None
        for name in names:
            if name.lower().endswith("/leica.txt") or name.lower().endswith("leica.txt"):
                leica_name = name
                break
        if leica_name is None:
            raise FileNotFoundError(f"No leica.txt found in {zip_path}; entries={names[:10]}")

        xyz: List[Tuple[float, float, float]] = []
        with zf.open(leica_name, "r") as f:
            for raw in f:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                # Leica measurement lines in this file format start with "3,"
                # and contain x,y,z around columns 10,11,12.
                if len(parts) < 13 or parts[0] != "3":
                    continue
                try:
                    x = float(parts[10])
                    y = float(parts[11])
                    z = float(parts[12])
                except Exception:
                    continue
                xyz.append((x, y, z))

    if len(xyz) < 20:
        raise RuntimeError(f"Too few Leica points parsed from {zip_path}: {len(xyz)}")
    return np.asarray(xyz, dtype=np.float64)


def _resample_positions(pos: np.ndarray, target_steps: int) -> np.ndarray:
    n = int(pos.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 positions to resample, got {n}")
    if target_steps <= 1:
        raise ValueError(f"target_steps must be >1, got {target_steps}")

    t_src = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
    t_dst = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float64)
    out = np.zeros((target_steps, 3), dtype=np.float64)
    for j in range(3):
        out[:, j] = np.interp(t_dst, t_src, pos[:, j])
    return out


def _make_state_obs(pos: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    x = np.concatenate([pos, vel, acc], axis=1).astype(np.float32, copy=False)  # [T,9]
    y = acc.astype(np.float32, copy=False)  # [T,3]
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare UZH-FPV raw Leica zip into TG7 loader-ready NPZ.")
    ap.add_argument("--zip-path", type=str, required=True, help="Path to raw indoor_forward_*.zip")
    ap.add_argument("--out-root", type=str, default="external_data/uzh_fpv")
    ap.add_argument("--target-steps", type=int, default=3020, help="Target timeline length for TG7 protocol.")
    ap.add_argument("--dt", type=float, default=0.01, help="Sampling interval for derived velocity/acceleration.")
    ap.add_argument(
        "--session-name",
        type=str,
        default="6th_indoor_forward_facing",
        help="Output prepared NPZ name (expected by TG7 loader default).",
    )
    args = ap.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    if args.target_steps <= 1:
        raise ValueError("--target-steps must be > 1")
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")

    pos_raw = _read_leica_positions(zip_path)
    pos = _resample_positions(pos_raw, target_steps=int(args.target_steps))
    x, y = _make_state_obs(pos, dt=float(args.dt))

    out_root = Path(args.out_root).expanduser().resolve()
    out_path = out_root / "prepared" / f"{args.session_name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        x=x,
        y=y,
        source_zip=str(zip_path),
    )

    print(f"[ok] wrote {out_path} x={x.shape} y={y.shape}")
    print("")
    print("Done.")
    print(f"Set: export UZH_FPV_ROOT={out_root}")
    print(
        "Test: .venv/bin/python -m bench.tasks.smoke_data "
        "--suite-yaml bench/configs/suite_tg7_datasets_smoke.yaml "
        "--task TG7_uzh_fpv_loader_smoke_v0 --seed 0"
    )


if __name__ == "__main__":
    main()
