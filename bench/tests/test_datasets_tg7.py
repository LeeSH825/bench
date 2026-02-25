from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import yaml

from bench.tasks.generator.datasets.common import DatasetMissingError
from bench.tasks.generator.datasets.nclt import (
    NCLT_ENV_VAR,
    NCLT_DATASET_NAME,
    generate_nclt_v0,
    has_nclt_dataset_root,
    resolve_nclt_root,
)
from bench.tasks.generator.datasets.uzh_fpv import (
    UZH_FPV_DATASET_NAME,
    UZH_FPV_ENV_VAR,
    generate_uzh_fpv_v0,
    has_uzh_fpv_dataset_root,
    resolve_uzh_fpv_root,
)


@dataclass
class DatasetLoadersTG7Result:
    ok: bool
    skipped: bool
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg7_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_tg7_datasets_smoke.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_cfg(suite: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for t in suite.get("tasks", []) or []:
        if str(t.get("task_id")) == str(task_id):
            return dict(t)
    raise KeyError(f"task_id not found in suite: {task_id}")


@contextmanager
def _temp_env(name: str, value: Optional[str]) -> Generator[None, None, None]:
    sentinel = object()
    prev = os.environ.get(name, sentinel)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
        yield
    finally:
        if prev is sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(prev)


def _assert_missing_error(
    exc: DatasetMissingError,
    *,
    dataset: str,
    env_var: str,
    must_contain: Optional[str] = None,
) -> Optional[str]:
    if getattr(exc, "error_code", "") != "io_error":
        return f"error_code mismatch for {dataset}: expected io_error, got {getattr(exc, 'error_code', None)}"
    msg = str(exc)
    low = msg.lower()
    if str(dataset).lower() not in low:
        return f"missing dataset token '{dataset}' in error message: {msg}"
    if str(env_var).lower() not in low:
        return f"missing env var token '{env_var}' in error message: {msg}"
    if must_contain is not None and str(must_contain).lower() not in low:
        return f"missing expected token '{must_contain}' in error message: {msg}"
    return None


def _expect_missing(callable_obj, *, dataset: str, env_var: str, must_contain: Optional[str] = None) -> Optional[str]:
    try:
        callable_obj()
    except DatasetMissingError as exc:
        return _assert_missing_error(exc, dataset=dataset, env_var=env_var, must_contain=must_contain)
    except Exception as exc:  # pragma: no cover - defensive
        return f"expected DatasetMissingError for {dataset}, got {type(exc).__name__}: {exc}"
    return f"expected DatasetMissingError for {dataset}, but call succeeded"


def run_dataset_loaders_tg7(suite_yaml: Optional[Path] = None) -> DatasetLoadersTG7Result:
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg7_smoke()
    suite = _load_yaml(suite_path)

    nclt_task = _find_task_cfg(suite, task_id="TG7_nclt_loader_smoke_v0")
    uzh_task = _find_task_cfg(suite, task_id="TG7_uzh_fpv_loader_smoke_v0")
    if str(nclt_task.get("task_family", "")).lower() not in {"nclt", "nclt_v0", "nclt_segway", "nclt_segway_v0"}:
        return DatasetLoadersTG7Result(ok=False, skipped=False, note="TG7 NCLT smoke task has unexpected task_family")
    if str(uzh_task.get("task_family", "")).lower() not in {"uzh_fpv", "uzh_fpv_v0", "uzh", "uzh_v0"}:
        return DatasetLoadersTG7Result(ok=False, skipped=False, note="TG7 UZH-FPV smoke task has unexpected task_family")

    # 1) missing env vars
    with _temp_env(NCLT_ENV_VAR, None):
        err = _expect_missing(resolve_nclt_root, dataset=NCLT_DATASET_NAME, env_var=NCLT_ENV_VAR, must_contain="not set")
        if err is not None:
            return DatasetLoadersTG7Result(ok=False, skipped=False, note=err)
    with _temp_env(UZH_FPV_ENV_VAR, None):
        err = _expect_missing(resolve_uzh_fpv_root, dataset=UZH_FPV_DATASET_NAME, env_var=UZH_FPV_ENV_VAR, must_contain="not set")
        if err is not None:
            return DatasetLoadersTG7Result(ok=False, skipped=False, note=err)

    # 2) invalid path from env var
    with tempfile.TemporaryDirectory(prefix="tg7_invalid_path_") as tmp:
        invalid = Path(tmp).resolve() / "does_not_exist"
        with _temp_env(NCLT_ENV_VAR, str(invalid)):
            err = _expect_missing(resolve_nclt_root, dataset=NCLT_DATASET_NAME, env_var=NCLT_ENV_VAR, must_contain="does not exist")
            if err is not None:
                return DatasetLoadersTG7Result(ok=False, skipped=False, note=err)
        with _temp_env(UZH_FPV_ENV_VAR, str(invalid)):
            err = _expect_missing(resolve_uzh_fpv_root, dataset=UZH_FPV_DATASET_NAME, env_var=UZH_FPV_ENV_VAR, must_contain="does not exist")
            if err is not None:
                return DatasetLoadersTG7Result(ok=False, skipped=False, note=err)

    # 3) incomplete dataset structure should still produce clear io_error
    with tempfile.TemporaryDirectory(prefix="tg7_incomplete_nclt_") as tmp:
        root = Path(tmp).resolve()
        with _temp_env(NCLT_ENV_VAR, str(root)):
            err = _expect_missing(
                lambda: generate_nclt_v0(
                    suite_name="tg7_missing",
                    task_cfg_dict=nclt_task,
                    scenario_cfg={},
                    seed=0,
                    scenario_id="scenario_missing",
                    task_family="nclt_v0",
                ),
                dataset=NCLT_DATASET_NAME,
                env_var=NCLT_ENV_VAR,
                must_contain="session",
            )
            if err is not None:
                return DatasetLoadersTG7Result(ok=False, skipped=False, note=err)

    with tempfile.TemporaryDirectory(prefix="tg7_incomplete_uzh_") as tmp:
        root = Path(tmp).resolve()
        with _temp_env(UZH_FPV_ENV_VAR, str(root)):
            err = _expect_missing(
                lambda: generate_uzh_fpv_v0(
                    suite_name="tg7_missing",
                    task_cfg_dict=uzh_task,
                    scenario_cfg={},
                    seed=0,
                    scenario_id="scenario_missing",
                    task_family="uzh_fpv_v0",
                ),
                dataset=UZH_FPV_DATASET_NAME,
                env_var=UZH_FPV_ENV_VAR,
                must_contain="session",
            )
            if err is not None:
                return DatasetLoadersTG7Result(ok=False, skipped=False, note=err)

    nclt_present = has_nclt_dataset_root()
    uzh_present = has_uzh_fpv_dataset_root()
    if not nclt_present and not uzh_present:
        return DatasetLoadersTG7Result(
            ok=True,
            skipped=True,
            note=(
                "TG7 loader diagnostics passed; datasets are absent in this environment "
                "(expected CI-safe skip)."
            ),
        )

    return DatasetLoadersTG7Result(
        ok=True,
        skipped=False,
        note=(
            "TG7 loader diagnostics passed; at least one dataset root is configured. "
            "Use smoke_data commands for local cache conversion."
        ),
    )
