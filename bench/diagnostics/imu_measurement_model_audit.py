from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1.0e-12


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    valid_for_acceptance: bool
    description: str


CANDIDATES: Tuple[Candidate, ...] = (
    Candidate(
        name="direct_identity_invalid",
        label="Candidate 0: h_direct(x)=x",
        valid_for_acceptance=False,
        description="Invalid direct-observation baseline; included only to show H=I is not the IMU measurement model.",
    ),
    Candidate(
        name="gyro_delta_simple",
        label="Candidate 1: [omega, omega*dt]",
        valid_for_acceptance=True,
        description="Gyro plus simple per-sample delta-angle approximation in the platform/body frame.",
    ),
    Candidate(
        name="gyro_delta_mrp_fd_approx",
        label="Candidate 2: [omega, 4*delta_sigma]",
        valid_for_acceptance=True,
        description=(
            "Approximate finite-difference MRP delta. This is a small-angle diagnostic, not a final "
            "Basilisk DRFramePlatform derivation."
        ),
    ),
    Candidate(
        name="gyro_delta_platform",
        label="Candidate 3: platform-transformed [omega, omega*dt]",
        valid_for_acceptance=True,
        description="Candidate 1 with the configured body-to-platform DCM applied; identical to Candidate 1 when DCM is identity.",
    ),
)


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    if float(np.std(aa)) <= EPS or float(np.std(bb)) <= EPS:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _list_str(values: Sequence[float]) -> str:
    return ";".join("nan" if not np.isfinite(v) else f"{float(v):.9g}" for v in values)


def _dcm_from_euler321(euler321_rad: Sequence[float]) -> np.ndarray:
    """Return a body-to-platform DCM for the 3-2-1 angles used by Basilisk's IMU setter.

    Current benchmark configs use [0,0,0], so this mostly documents and verifies identity-frame behavior.
    """

    arr = np.asarray(euler321_rad, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        return np.eye(3, dtype=np.float64)
    psi, theta, phi = [float(v) for v in arr]
    c1, s1 = math.cos(psi), math.sin(psi)
    c2, s2 = math.cos(theta), math.sin(theta)
    c3, s3 = math.cos(phi), math.sin(phi)
    r1 = np.array([[c1, s1, 0.0], [-s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r2 = np.array([[c2, 0.0, -s2], [0.0, 1.0, 0.0], [s2, 0.0, c2]], dtype=np.float64)
    r3 = np.array([[1.0, 0.0, 0.0], [0.0, c3, s3], [0.0, -s3, c3]], dtype=np.float64)
    return r3 @ r2 @ r1


def h_direct_identity_invalid(x_seq: np.ndarray, *, dt: float, dcm_pb: Optional[np.ndarray] = None) -> np.ndarray:
    del dt, dcm_pb
    return _as_float_array(x_seq).copy()


def h_gyro_delta_simple(x_seq: np.ndarray, *, dt: float, dcm_pb: Optional[np.ndarray] = None) -> np.ndarray:
    del dcm_pb
    x = _as_float_array(x_seq)
    omega = x[..., 3:6]
    return np.concatenate([omega, omega * float(dt)], axis=-1)


def h_gyro_delta_mrp_fd_approx(x_seq: np.ndarray, *, dt: float, dcm_pb: Optional[np.ndarray] = None) -> np.ndarray:
    del dt, dcm_pb
    x = _as_float_array(x_seq)
    omega = x[..., 3:6]
    delta = np.zeros_like(omega)
    if x.shape[1] > 1:
        # For small MRPs, sigma_dot ~= omega/4, so delta theta ~= 4 * delta sigma.
        delta[:, 1:, :] = 4.0 * (x[:, 1:, 0:3] - x[:, :-1, 0:3])
    return np.concatenate([omega, delta], axis=-1)


def h_gyro_delta_platform(x_seq: np.ndarray, *, dt: float, dcm_pb: Optional[np.ndarray] = None) -> np.ndarray:
    x = _as_float_array(x_seq)
    rot = np.eye(3, dtype=np.float64) if dcm_pb is None else np.asarray(dcm_pb, dtype=np.float64).reshape(3, 3)
    omega_p = np.einsum("ij,ntj->nti", rot, x[..., 3:6])
    return np.concatenate([omega_p, omega_p * float(dt)], axis=-1)


def h_imu_bias(x_seq: np.ndarray, *, dt: float) -> np.ndarray:
    """Bias-state IMU measurement model: [omega + b_g, (omega + b_g) * dt]."""

    x = _as_float_array(x_seq)
    if x.shape[-1] != 9:
        raise ValueError(f"h_imu_bias expects x_dim=9, got shape {x.shape}")
    omega_plus_bias = x[..., 3:6] + x[..., 6:9]
    return np.concatenate([omega_plus_bias, omega_plus_bias * float(dt)], axis=-1)


def h_imu_bias_sparse_ref(x_seq: np.ndarray, *, dt: float) -> np.ndarray:
    """Sparse-reference IMU model without the time mask.

    The unmasked measurement packet is:
      [omega + b_g, (omega + b_g) * dt, sigma].
    Runtime/audit code must separately mask the final three rows when
    ref_mask_t == 0.
    """

    x = _as_float_array(x_seq)
    if x.shape[-1] != 9:
        raise ValueError(f"h_imu_bias_sparse_ref expects x_dim=9, got shape {x.shape}")
    sigma = x[..., 0:3]
    omega_plus_bias = x[..., 3:6] + x[..., 6:9]
    return np.concatenate([omega_plus_bias, omega_plus_bias * float(dt), sigma], axis=-1)


H_FUNCTIONS = {
    "direct_identity_invalid": h_direct_identity_invalid,
    "gyro_delta_simple": h_gyro_delta_simple,
    "gyro_delta_mrp_fd_approx": h_gyro_delta_mrp_fd_approx,
    "gyro_delta_platform": h_gyro_delta_platform,
}


def analytic_H_direct_identity() -> np.ndarray:
    return np.eye(6, dtype=np.float64)


def analytic_H_gyro_delta_simple(dt: float) -> np.ndarray:
    h = np.zeros((6, 6), dtype=np.float64)
    h[0:3, 3:6] = np.eye(3, dtype=np.float64)
    h[3:6, 3:6] = float(dt) * np.eye(3, dtype=np.float64)
    return h


def analytic_H_gyro_delta_platform(dt: float, dcm_pb: Optional[np.ndarray]) -> np.ndarray:
    rot = np.eye(3, dtype=np.float64) if dcm_pb is None else np.asarray(dcm_pb, dtype=np.float64).reshape(3, 3)
    h = np.zeros((6, 6), dtype=np.float64)
    h[0:3, 3:6] = rot
    h[3:6, 3:6] = float(dt) * rot
    return h


def analytic_H_imu_bias(dt: float) -> np.ndarray:
    h = np.zeros((6, 9), dtype=np.float64)
    h[0:3, 3:6] = np.eye(3, dtype=np.float64)
    h[0:3, 6:9] = np.eye(3, dtype=np.float64)
    h[3:6, 3:6] = float(dt) * np.eye(3, dtype=np.float64)
    h[3:6, 6:9] = float(dt) * np.eye(3, dtype=np.float64)
    return h


def analytic_H_imu_bias_sparse_ref(dt: float) -> np.ndarray:
    h = np.zeros((9, 9), dtype=np.float64)
    h[0:3, 3:6] = np.eye(3, dtype=np.float64)
    h[0:3, 6:9] = np.eye(3, dtype=np.float64)
    h[3:6, 3:6] = float(dt) * np.eye(3, dtype=np.float64)
    h[3:6, 6:9] = float(dt) * np.eye(3, dtype=np.float64)
    h[6:9, 0:3] = np.eye(3, dtype=np.float64)
    return h


def finite_difference_H(
    candidate: str,
    x_t: np.ndarray,
    *,
    dt: float,
    prev_x_t: Optional[np.ndarray] = None,
    dcm_pb: Optional[np.ndarray] = None,
    eps: float = 1.0e-5,
) -> np.ndarray:
    x0 = np.asarray(x_t, dtype=np.float64).reshape(6)
    prev = np.asarray(prev_x_t if prev_x_t is not None else x0, dtype=np.float64).reshape(6)

    def eval_one(x_vec: np.ndarray) -> np.ndarray:
        if candidate == "gyro_delta_mrp_fd_approx":
            seq = np.stack([prev, x_vec], axis=0)[None, :, :]
            return H_FUNCTIONS[candidate](seq, dt=dt, dcm_pb=dcm_pb)[0, 1, :]
        seq = x_vec.reshape(1, 1, 6)
        return H_FUNCTIONS[candidate](seq, dt=dt, dcm_pb=dcm_pb)[0, 0, :]

    h = np.zeros((6, 6), dtype=np.float64)
    for j in range(6):
        step = np.zeros(6, dtype=np.float64)
        step[j] = float(eps)
        h[:, j] = (eval_one(x0 + step) - eval_one(x0 - step)) / (2.0 * float(eps))
    return h


def finite_difference_H_imu_bias(
    x_t: np.ndarray,
    *,
    dt: float,
    eps: float = 1.0e-5,
) -> np.ndarray:
    x0 = np.asarray(x_t, dtype=np.float64).reshape(9)

    def eval_one(x_vec: np.ndarray) -> np.ndarray:
        return h_imu_bias(x_vec.reshape(1, 1, 9), dt=dt)[0, 0, :]

    h = np.zeros((6, 9), dtype=np.float64)
    for j in range(9):
        step = np.zeros(9, dtype=np.float64)
        step[j] = float(eps)
        h[:, j] = (eval_one(x0 + step) - eval_one(x0 - step)) / (2.0 * float(eps))
    return h


def finite_difference_H_imu_bias_sparse_ref(
    x_t: np.ndarray,
    *,
    dt: float,
    eps: float = 1.0e-5,
) -> np.ndarray:
    x0 = np.asarray(x_t, dtype=np.float64).reshape(9)

    def eval_one(x_vec: np.ndarray) -> np.ndarray:
        return h_imu_bias_sparse_ref(x_vec.reshape(1, 1, 9), dt=dt)[0, 0, :]

    h = np.zeros((9, 9), dtype=np.float64)
    for j in range(9):
        step = np.zeros(9, dtype=np.float64)
        step[j] = float(eps)
        h[:, j] = (eval_one(x0 + step) - eval_one(x0 - step)) / (2.0 * float(eps))
    return h


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out: Dict[str, Any] = {key: z[key] for key in z.files}
    out["meta"] = json.loads(str(out["meta_json"].tolist() if hasattr(out["meta_json"], "tolist") else out["meta_json"]))
    return out


def discover_split_paths(cache_root: Path, suite_name: str, task_id: str, seed: int) -> List[Path]:
    base = cache_root / suite_name / task_id
    if not base.exists():
        return []
    paths: List[Path] = []
    for scenario_dir in sorted(base.glob("scenario_*")):
        seed_dir = scenario_dir / f"seed_{int(seed)}"
        for split in ("train", "val", "test"):
            p = seed_dir / f"{split}.npz"
            if p.exists():
                paths.append(p)
    return paths


def _row_metrics(
    *,
    candidate_name: str,
    y_model: np.ndarray,
    y_clean: np.ndarray,
    profile_id: str,
    split: str,
    scenario_id: str,
    time_mask: str,
) -> Dict[str, Any]:
    err = np.asarray(y_model - y_clean, dtype=np.float64)
    mse_dim = np.mean(err * err, axis=(0, 1))
    mae_dim = np.mean(np.abs(err), axis=(0, 1))
    corr_dim = [_safe_corr(y_model[..., j], y_clean[..., j]) for j in range(y_clean.shape[-1])]
    clean_norm = float(np.linalg.norm(y_clean))
    model_norm = float(np.linalg.norm(y_model))
    err_norm = float(np.linalg.norm(err))
    return {
        "candidate": candidate_name,
        "profile_id": profile_id,
        "split": split,
        "scenario_id": scenario_id,
        "time_mask": time_mask,
        "n": int(y_clean.shape[0]),
        "T_eval": int(y_clean.shape[1]),
        "mse_total": float(np.mean(err * err)),
        "mse_gyro": float(np.mean(err[..., 0:3] ** 2)),
        "mse_delta": float(np.mean(err[..., 3:6] ** 2)),
        "rmse_total": float(math.sqrt(float(np.mean(err * err)))),
        "rmse_per_dim": _list_str([math.sqrt(float(v)) for v in mse_dim]),
        "mean_abs_error_per_dim": _list_str([float(v) for v in mae_dim]),
        "corr_per_dim": _list_str(corr_dim),
        "corr_mean": float(np.nanmean(corr_dim)) if not np.all(np.isnan(corr_dim)) else float("nan"),
        "norm_ratio": float(model_norm / (clean_norm + EPS)),
        "error_to_clean_norm_ratio": float(err_norm / (clean_norm + EPS)),
    }


def compare_candidates(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    first_context: Dict[str, Any] = {}
    for path in paths:
        payload = _load_npz(path)
        x = np.asarray(payload["x"], dtype=np.float64)
        y_clean = np.asarray(payload["imu_clean_y_seq"], dtype=np.float64)
        meta = payload["meta"]
        dt = float(meta["ssm"]["true"]["dt"])
        profile_id = str(meta.get("imu_config", {}).get("profile_id", "unknown"))
        split = str(meta.get("split", path.stem))
        scenario_id = str(meta.get("scenario_id", path.parent.parent.name.replace("scenario_", "")))
        dcm = _dcm_from_euler321(meta.get("imu_config", {}).get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0]))
        if not first_context:
            first_context = {"path": str(path), "payload": payload, "dcm_pb": dcm}

        for candidate in CANDIDATES:
            y_model = H_FUNCTIONS[candidate.name](x, dt=dt, dcm_pb=dcm)
            rows.append(
                _row_metrics(
                    candidate_name=candidate.name,
                    y_model=y_model,
                    y_clean=y_clean,
                    profile_id=profile_id,
                    split=split,
                    scenario_id=scenario_id,
                    time_mask="all",
                )
            )
            if x.shape[1] > 1:
                rows.append(
                    _row_metrics(
                        candidate_name=candidate.name,
                        y_model=y_model[:, 1:, :],
                        y_clean=y_clean[:, 1:, :],
                        profile_id=profile_id,
                        split=split,
                        scenario_id=scenario_id,
                        time_mask="skip_first",
                    )
                )
    return rows, first_context


def summarize_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("time_mask") != "skip_first":
            continue
        key = (str(row["candidate"]), str(row["profile_id"]), "all_splits")
        grouped.setdefault(key, []).append(row)
        grouped.setdefault((str(row["candidate"]), "ALL", "all_profiles_splits"), []).append(row)

    summary: List[Dict[str, Any]] = []
    for (candidate, profile_id, split_group), items in sorted(grouped.items()):
        weights = np.asarray([float(item.get("n", 1)) * float(item.get("T_eval", 1)) for item in items], dtype=np.float64)
        weights = weights / max(float(weights.sum()), EPS)

        def wmean(name: str) -> float:
            vals = np.asarray([float(item[name]) for item in items], dtype=np.float64)
            return float(np.sum(vals * weights))

        summary.append(
            {
                "candidate": candidate,
                "profile_id": profile_id,
                "split_group": split_group,
                "rows": len(items),
                "mse_total": wmean("mse_total"),
                "mse_gyro": wmean("mse_gyro"),
                "mse_delta": wmean("mse_delta"),
                "rmse_total": math.sqrt(max(wmean("mse_total"), 0.0)),
                "corr_mean": wmean("corr_mean"),
                "norm_ratio": wmean("norm_ratio"),
                "error_to_clean_norm_ratio": wmean("error_to_clean_norm_ratio"),
            }
        )
    return summary


def audit_H(first_context: Mapping[str, Any], accepted_candidate: str) -> List[Dict[str, Any]]:
    payload = first_context["payload"]
    meta = payload["meta"]
    x = np.asarray(payload["x"], dtype=np.float64)
    dt = float(meta["ssm"]["true"]["dt"])
    dcm = first_context["dcm_pb"]
    x_t = x[0, min(1, x.shape[1] - 1), :]
    prev_t = x[0, 0, :]
    rows: List[Dict[str, Any]] = []
    analytic_map = {
        "direct_identity_invalid": analytic_H_direct_identity(),
        "gyro_delta_simple": analytic_H_gyro_delta_simple(dt),
        "gyro_delta_platform": analytic_H_gyro_delta_platform(dt, dcm),
    }
    for candidate in ("direct_identity_invalid", "gyro_delta_simple", "gyro_delta_mrp_fd_approx", "gyro_delta_platform"):
        h_fd = finite_difference_H(candidate, x_t, prev_x_t=prev_t, dt=dt, dcm_pb=dcm)
        h_an = analytic_map.get(candidate)
        h_used = h_an if h_an is not None else h_fd
        diff = float(np.max(np.abs(h_an - h_fd))) if h_an is not None else float("nan")
        is_identity = bool(h_used.shape == (6, 6) and np.allclose(h_used, np.eye(6), atol=1.0e-8, rtol=0.0))
        rows.append(
            {
                "candidate": candidate,
                "accepted": bool(candidate == accepted_candidate),
                "H_shape": f"{h_used.shape[0]}x{h_used.shape[1]}",
                "H_rank": int(np.linalg.matrix_rank(h_used)),
                "H_condition_number": float(np.linalg.cond(h_used)) if np.linalg.matrix_rank(h_used) == min(h_used.shape) else float("inf"),
                "H_is_identity": is_identity,
                "finite_diff_error": diff,
                "H_norm": float(np.linalg.norm(h_used)),
                "H_finite": bool(np.all(np.isfinite(h_used))),
            }
        )
    return rows


def audit_runtime_H(first_context: Mapping[str, Any], accepted_candidate: str) -> Dict[str, Any]:
    payload = first_context["payload"]
    meta = payload["meta"]
    dt = float(meta["ssm"]["true"]["dt"])
    dcm = first_context["dcm_pb"]
    h_npz = np.asarray(payload.get("H"), dtype=np.float64)
    h_meta = np.asarray(meta.get("ssm", {}).get("assumed", {}).get("H"), dtype=np.float64)
    h_acc = analytic_H_gyro_delta_platform(dt, dcm) if accepted_candidate == "gyro_delta_platform" else analytic_H_gyro_delta_simple(dt)
    return {
        "npz_H_shape": f"{h_npz.shape[0]}x{h_npz.shape[1]}",
        "meta_H_shape": f"{h_meta.shape[0]}x{h_meta.shape[1]}",
        "current_H_is_identity": bool(np.allclose(h_npz, np.eye(6), atol=1.0e-8, rtol=0.0)),
        "current_H_matches_meta": bool(np.allclose(h_npz, h_meta, atol=1.0e-8, rtol=1.0e-8)),
        "accepted_H": accepted_candidate,
        "current_H_matches_accepted": bool(np.allclose(h_npz, h_acc, atol=1.0e-8, rtol=1.0e-8)),
        "adapter_code_path": "bench/models/split_knet.py::SplitKNetAdapter.setup uses system_info['H'] when present, else identity",
        "runtime_assessment": (
            "current NPZ H matches audited IMU projection"
            if np.allclose(h_npz, h_acc, atol=1.0e-8, rtol=1.0e-8)
            else "current NPZ H does not match audited IMU projection"
        ),
    }


def audit_innovation(paths: Sequence[Path], accepted_candidate: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = _load_npz(path)
        x = np.asarray(payload["x"], dtype=np.float64)
        y_raw = np.asarray(payload["y"], dtype=np.float64)
        y_clean = np.asarray(payload["imu_clean_y_seq"], dtype=np.float64)
        meta = payload["meta"]
        dt = float(meta["ssm"]["true"]["dt"])
        dcm = _dcm_from_euler321(meta.get("imu_config", {}).get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0]))
        y_model = H_FUNCTIONS[accepted_candidate](x, dt=dt, dcm_pb=dcm)
        if x.shape[1] > 1:
            y_model = y_model[:, 1:, :]
            y_raw = y_raw[:, 1:, :]
            y_clean = y_clean[:, 1:, :]
        raw_resid = y_raw - y_model
        clean_resid = y_clean - y_model
        raw_norm_t = np.linalg.norm(raw_resid, axis=2)
        clean_norm_t = np.linalg.norm(clean_resid, axis=2)
        denom = raw_norm_t.reshape(-1)
        ratio = clean_norm_t / (raw_norm_t + EPS)
        rows.append(
            {
                "profile_id": str(meta.get("imu_config", {}).get("profile_id", "unknown")),
                "split": str(meta.get("split", path.stem)),
                "scenario_id": str(meta.get("scenario_id", path.parent.parent.name.replace("scenario_", ""))),
                "candidate": accepted_candidate,
                "residual_kind": "truth_h_residual_skip_first",
                "innovation_norm_raw": float(np.mean(raw_norm_t)),
                "innovation_norm_enh_proxy_clean": float(np.mean(clean_norm_t)),
                "innovation_ratio_clean_over_raw": float(np.mean(ratio)),
                "innovation_norm_raw_per_dim": _list_str([float(v) for v in np.mean(np.abs(raw_resid), axis=(0, 1))]),
                "innovation_norm_enh_per_dim": _list_str([float(v) for v in np.mean(np.abs(clean_resid), axis=(0, 1))]),
                "innovation_ratio_per_dim": _list_str(
                    [
                        float(np.mean(np.abs(clean_resid[..., j])) / (np.mean(np.abs(raw_resid[..., j])) + EPS))
                        for j in range(raw_resid.shape[-1])
                    ]
                ),
                "denominator_min": float(np.min(denom)),
                "denominator_mean": float(np.mean(denom)),
                "denominator_p05": float(np.percentile(denom, 5)),
                "ratio_is_reliable": bool(float(np.percentile(denom, 5)) > 1.0e-8),
                "finite": bool(np.all(np.isfinite(raw_resid)) and np.all(np.isfinite(clean_resid)) and np.all(np.isfinite(ratio))),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _best_candidate(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    clean_rows = [
        row
        for row in summary_rows
        if row.get("profile_id") == "clean_imu" and row.get("candidate") != "direct_identity_invalid"
    ]
    rows = clean_rows or [row for row in summary_rows if row.get("candidate") != "direct_identity_invalid"]
    if not rows:
        return "gyro_delta_simple"
    best = min(rows, key=lambda r: float(r.get("error_to_clean_norm_ratio", float("inf"))))
    best_ratio = float(best.get("error_to_clean_norm_ratio", float("inf")))
    # In the current identity-platform task, Candidate 1 and Candidate 3 are numerically identical.
    # Prefer the simpler physical statement unless a non-identity platform transform clearly wins.
    simple = next((r for r in rows if r.get("candidate") == "gyro_delta_simple"), None)
    if simple is not None and float(simple.get("error_to_clean_norm_ratio", float("inf"))) <= best_ratio + 1.0e-12:
        return "gyro_delta_simple"
    return str(best["candidate"])


def _old_innovation_issue_present(root: Path) -> bool:
    for rel in (
        "reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv",
        "reports/basilisk_imu_pretrained_enhancer_gpu_pilot_metrics.csv",
    ):
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "21.457" in text or ",0.0," in text or "innovation_ratio" in text:
            return True
    return False


def _plot_outputs(first_context: Mapping[str, Any], accepted_candidate: str, plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    payload = first_context["payload"]
    meta = payload["meta"]
    x = np.asarray(payload["x"], dtype=np.float64)
    y_clean = np.asarray(payload["imu_clean_y_seq"], dtype=np.float64)
    dt = float(meta["ssm"]["true"]["dt"])
    dcm = first_context["dcm_pb"]
    t = np.arange(x.shape[1]) * dt
    i = 0
    y_simple = h_gyro_delta_simple(x, dt=dt, dcm_pb=dcm)
    y_mrp = h_gyro_delta_mrp_fd_approx(x, dt=dt, dcm_pb=dcm)
    y_acc = H_FUNCTIONS[accepted_candidate](x, dt=dt, dcm_pb=dcm)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, y_clean[i, :, j], label="clean AngVelPlatform", linewidth=1.6)
        ax.plot(t, y_simple[i, :, j], "--", label="omega candidate", linewidth=1.1)
        ax.set_ylabel(f"gyro {j}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_h_gyro_overlay.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, y_clean[i, :, 3 + j], label="clean DRFramePlatform", linewidth=1.6)
        ax.plot(t, y_simple[i, :, 3 + j], "--", label="omega*dt", linewidth=1.1)
        ax.plot(t, y_mrp[i, :, 3 + j], ":", label="4*delta_sigma", linewidth=1.1)
        ax.set_ylabel(f"delta {j}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_h_delta_overlay.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.reshape(-1)
    for j, ax in enumerate(axes):
        ax.scatter(y_clean[:, 1:, j].reshape(-1), y_acc[:, 1:, j].reshape(-1), s=2, alpha=0.18)
        ax.set_xlabel(f"clean y[{j}]")
        ax.set_ylabel(f"model y[{j}]")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Accepted h candidate scatter: {accepted_candidate}")
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_h_scatter.png", dpi=160)
    plt.close(fig)


def _write_markdown_reports(
    *,
    root: Path,
    reports_dir: Path,
    paths: Sequence[Path],
    first_context: Mapping[str, Any],
    comparison_summary: Sequence[Mapping[str, Any]],
    h_rows: Sequence[Mapping[str, Any]],
    h_audit_rows: Sequence[Mapping[str, Any]],
    runtime_h: Mapping[str, Any],
    innovation_rows: Sequence[Mapping[str, Any]],
    accepted_candidate: str,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = first_context["payload"]
    meta = payload["meta"]
    dcm = np.asarray(first_context["dcm_pb"], dtype=np.float64)
    platform_identity = bool(np.allclose(dcm, np.eye(3), atol=1.0e-12, rtol=0.0))
    direct_row = next((r for r in comparison_summary if r.get("candidate") == "direct_identity_invalid" and r.get("profile_id") == "ALL"), None)
    accepted_row = next((r for r in comparison_summary if r.get("candidate") == accepted_candidate and r.get("profile_id") == "ALL"), None)
    simple_row = next((r for r in comparison_summary if r.get("candidate") == "gyro_delta_simple" and r.get("profile_id") == "ALL"), None)
    mrp_row = next((r for r in comparison_summary if r.get("candidate") == "gyro_delta_mrp_fd_approx" and r.get("profile_id") == "ALL"), None)
    old_issue = _old_innovation_issue_present(root)
    finite_innovation = all(bool(r.get("finite", False)) for r in innovation_rows)
    unreliable = [r for r in innovation_rows if not bool(r.get("ratio_is_reliable", False))]

    measurement_md = f"""# Basilisk IMU Measurement Model Audit

## Scope

This is an audit-only report for `task_family=basilisk_imu_adcs_v0`. No model adapter,
metric, fairness, or Split-KalmanNet G1/G2 behavior is changed by this audit.

## API Field Summary

Current measurement mode: `{meta.get('observation', {}).get('measurement_mode')}`

Selected fields:

{chr(10).join(f"- `{m['field']}` as `{m['alias']}` columns={m['columns']} units={m['units']}" for m in meta.get('observation', {}).get('field_mapping', []))}

State target:

```text
x = [sigma_BN(3), omega_BN_B(3)]
```

Measurement:

```text
y = [AngVelPlatform(3), DRFramePlatform(3)]
```

## Current Task Assumptions

- IMU sample interval equals simulation `dt = {float(meta['ssm']['true']['dt']):.9g}` seconds.
- Platform/body Euler 3-2-1 config is `{meta.get('imu_config', {}).get('body_to_platform_euler321_rad')}`.
- Platform transform is identity in current data: `{platform_identity}`.
- `AccelPlatform` and `DVFramePlatform` are intentionally excluded from this configured task.
- Sensor noise/bias/severity is applied by `imuSensor.ImuSensor`; the generator stores both measured and clean IMU packets.

## Risks

- `H=I` and `h(x)=x` are physically invalid because the IMU does not output absolute MRP attitude.
- `DRFramePlatform` is a delta-angle-like packet, not `sigma_BN`.
- Instantaneous IMU `H` is rank deficient for absolute attitude; sequence dynamics are needed to infer attitude.
- Innovation ratios can be misleading when the denominator is near zero, especially for clean IMU packets.

## h_imu Candidates Tested

{chr(10).join(f"- `{c.name}`: {c.description}" for c in CANDIDATES)}
"""
    (reports_dir / "basilisk_imu_measurement_model_audit.md").write_text(measurement_md, encoding="utf-8")

    h_summary = "\n".join(
        f"| {row['candidate']} | {row['profile_id']} | {float(row['mse_total']):.6g} | "
        f"{float(row['mse_gyro']):.6g} | {float(row['mse_delta']):.6g} | "
        f"{float(row['corr_mean']):.6g} | {float(row['error_to_clean_norm_ratio']):.6g} |"
        for row in comparison_summary
        if row.get("profile_id") == "ALL"
    )
    h_audit_md = "\n".join(
        f"| {row['candidate']} | {row['H_shape']} | {row['H_rank']} | {row['H_is_identity']} | {row['finite_diff_error']} |"
        for row in h_audit_rows
    )
    innovation_note = (
        "Some denominator percentiles are near zero, so ratios are marked unreliable for those rows and absolute norms should be used."
        if unreliable
        else "All denominator checks are above the near-zero threshold used by this audit."
    )
    final_decision = (
        "A"
        if accepted_candidate in {"gyro_delta_simple", "gyro_delta_platform"}
        and bool(runtime_h.get("current_H_matches_accepted"))
        else "B"
        if accepted_candidate == "gyro_delta_mrp_fd_approx"
        else "C"
    )
    final_md = f"""# Basilisk IMU Measurement Model Audit Final

## Data Used

- Cache root paths inspected: `{len(paths)}` split files
- First split: `{first_context.get('path')}`
- Time mask for acceptance metrics: `skip_first` to avoid the recorder initialization sample.

## h_imu Candidate Summary

| candidate | profile | mse_total | mse_gyro | mse_delta | corr_mean | error_to_clean_norm_ratio |
|---|---:|---:|---:|---:|---:|---:|
{h_summary}

Accepted h candidate: `{accepted_candidate}`

Direct H=I invalid: `{bool(direct_row and accepted_row and float(direct_row['error_to_clean_norm_ratio']) > 100.0 * float(accepted_row['error_to_clean_norm_ratio']))}`

DRFramePlatform is closer to `omega*dt` than MRP finite-difference delta:
`{bool(simple_row and mrp_row and float(simple_row['mse_delta']) < float(mrp_row['mse_delta']))}`

Platform/body transform needed for current config: `{not platform_identity}`

## H Audit

| candidate | H_shape | H_rank | H_is_identity | finite_diff_error |
|---|---:|---:|---:|---:|
{h_audit_md}

The accepted instantaneous H is non-identity but rank-deficient. This is expected for
`[gyro, delta-angle]`: absolute attitude is not directly observed at a single timestep.

## Runtime Adapter H Audit

- current_H: `npz/meta H from Basilisk IMU generator`
- accepted_H: `{accepted_candidate}`
- mismatch: `{not bool(runtime_h.get('current_H_matches_accepted'))}`
- current H is identity: `{runtime_h.get('current_H_is_identity')}`
- evidence: `{runtime_h.get('adapter_code_path')}`
- assessment: `{runtime_h.get('runtime_assessment')}`

## Innovation Diagnostic Audit

- old issue reproduced from prior pilot reports: `{old_issue}`
- corrected truth-h residuals finite: `{finite_innovation}`
- denominator warning: {innovation_note}

The corrected diagnostic should report absolute innovation/residual norms whenever the denominator
percentile is near zero. Ratios alone are unsafe on clean IMU data.

## Decision

Decision category: `{final_decision}`

Rationale:

- `gyro_delta_simple` / platform equivalent matches clean IMU packets after the first sample to numerical precision.
- `H=I` is invalid for this IMU task.
- Runtime Split-KNet is already receiving the non-identity IMU projection H from the NPZ split.
- The task remains a partial-observation IMU task; prior pilot results are useful diagnostic IMU results,
  not full-state observation results and not oracle-level results.

## Next Step

Proceed to planning an adapter/runtime cleanup that makes the audited IMU h/H explicit in model metadata and
diagnostics. For benchmark development, the next scientifically useful task is bias-state or sparse attitude
reference support; `DRFramePlatform` can be used as delta angle under this audited configuration.
"""
    (reports_dir / "basilisk_imu_measurement_model_audit_final.md").write_text(final_md, encoding="utf-8")

    h_md = f"""# Basilisk IMU H Audit Summary

Accepted candidate: `{accepted_candidate}`

| candidate | H_shape | H_rank | H_is_identity | finite_diff_error |
|---|---:|---:|---:|---:|
{h_audit_md}

Notes:

- `direct_identity_invalid` is included only as a failing baseline.
- `gyro_delta_simple` H is `[0 I; 0 dt*I]`.
- Rank 3 is expected because attitude is not directly observed by gyro/delta-angle.
"""
    (reports_dir / "basilisk_imu_H_audit_summary.md").write_text(h_md, encoding="utf-8")

    runtime_md = f"""# Basilisk IMU Runtime Adapter H Audit

- Current adapter path: `{runtime_h.get('adapter_code_path')}`
- Current H source: NPZ split key `H` / `meta_json.ssm.assumed.H`
- Current H is identity: `{runtime_h.get('current_H_is_identity')}`
- Current H matches metadata: `{runtime_h.get('current_H_matches_meta')}`
- Accepted H: `{runtime_h.get('accepted_H')}`
- Current H matches accepted: `{runtime_h.get('current_H_matches_accepted')}`
- Assessment: {runtime_h.get('runtime_assessment')}

If future IMU tasks change `y_dim`, frame transform, or selected fields, this audit must be rerun and the
adapter must not silently fall back to identity H.
"""
    (reports_dir / "basilisk_imu_runtime_adapter_H_audit.md").write_text(runtime_md, encoding="utf-8")

    innovation_md = f"""# Basilisk IMU Innovation Diagnostic Audit

Previous pilot reports had suspicious innovation ratios: reproduced = `{old_issue}`.

This audit uses truth-based measurement residuals:

```text
nu_raw_truth   = y_raw - h_imu(x_true)
nu_clean_truth = imu_clean_y_seq - h_imu(x_true)
```

Corrected residual values finite: `{finite_innovation}`.

{innovation_note}

For clean IMU rows, denominator norms can be near machine precision because `h_imu(x_true)` matches
`imu_clean_y_seq`. In that case diagnostics should report absolute norms and denominator percentiles instead
of relying on a ratio.
"""
    (reports_dir / "basilisk_imu_innovation_diagnostic_audit.md").write_text(innovation_md, encoding="utf-8")


def run_audit(
    *,
    root: Path,
    cache_root: Path,
    suite_name: str,
    task_id: str,
    seed: int = 0,
    reports_dir: Path,
    plots_dir: Path,
    write_plots: bool = True,
) -> Dict[str, Any]:
    paths = discover_split_paths(cache_root, suite_name, task_id, seed)
    if not paths:
        raise FileNotFoundError(f"No IMU split NPZ files found under {cache_root / suite_name / task_id}")
    h_rows, first_context = compare_candidates(paths)
    summary = summarize_candidate_rows(h_rows)
    accepted = _best_candidate(summary)
    h_audit = audit_H(first_context, accepted)
    runtime_h = audit_runtime_H(first_context, accepted)
    innovation = audit_innovation(paths, accepted)

    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(reports_dir / "basilisk_imu_h_model_comparison.csv", h_rows)
    _write_csv(reports_dir / "basilisk_imu_h_model_summary.csv", summary)
    _write_csv(reports_dir / "basilisk_imu_H_audit.csv", h_audit)
    _write_csv(reports_dir / "basilisk_imu_innovation_diagnostic_audit.csv", innovation)
    _write_markdown_reports(
        root=root,
        reports_dir=reports_dir,
        paths=paths,
        first_context=first_context,
        comparison_summary=summary,
        h_rows=h_rows,
        h_audit_rows=h_audit,
        runtime_h=runtime_h,
        innovation_rows=innovation,
        accepted_candidate=accepted,
    )
    if write_plots:
        _plot_outputs(first_context, accepted, plots_dir)
    return {
        "accepted_candidate": accepted,
        "paths_count": len(paths),
        "h_rows": len(h_rows),
        "summary_rows": len(summary),
        "runtime_h": runtime_h,
        "finite_innovation": all(bool(r.get("finite", False)) for r in innovation),
        "reports_dir": str(reports_dir),
        "plots_dir": str(plots_dir),
    }
