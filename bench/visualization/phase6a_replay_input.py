from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml

from .replay_suite_scenario import (
    REPLAY_SCENARIO_FILENAME,
    REPLAY_SCENARIO_META_FILENAME,
    load_suite_yaml,
    materialize_adcs_replay_task,
    save_replay_input_npz,
    select_task_from_suite,
)


SUITE_SNAPSHOT_FILENAME = "suite_snapshot.yaml"
TASK_SNAPSHOT_FILENAME = "task_snapshot.yaml"


def build_phase6a_replay_input(
    suite_yaml: str | Path,
    *,
    task_id: str,
    seed: int,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    suite_path = Path(suite_yaml).expanduser().resolve()
    suite_cfg = load_suite_yaml(suite_path)
    task_cfg = select_task_from_suite(suite_cfg, task_id)
    scenario = materialize_adcs_replay_task(
        suite_cfg,
        task_cfg,
        seed=int(seed),
    )
    output_dir = Path(out_dir).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"failed to create output directory: {output_dir}") from exc

    try:
        with tempfile.TemporaryDirectory(
            prefix=".phase6a_build_",
            dir=output_dir,
        ) as tmp:
            staging_dir = Path(tmp)
            staged_npz, staged_meta = save_replay_input_npz(
                scenario,
                staging_dir,
            )
            staged_suite = staging_dir / SUITE_SNAPSHOT_FILENAME
            staged_task = staging_dir / TASK_SNAPSHOT_FILENAME
            shutil.copy2(suite_path, staged_suite)
            staged_task.write_text(
                yaml.safe_dump(
                    task_cfg,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            final_npz = output_dir / REPLAY_SCENARIO_FILENAME
            final_meta = output_dir / REPLAY_SCENARIO_META_FILENAME
            staged_npz.replace(final_npz)
            staged_meta.replace(final_meta)
            staged_suite.replace(output_dir / SUITE_SNAPSHOT_FILENAME)
            staged_task.replace(output_dir / TASK_SNAPSHOT_FILENAME)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"failed to write Phase 6A replay input under {output_dir}: {exc}"
        ) from exc
    return final_npz, final_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate replay-ready ADCS arrays from an existing-style suite YAML task."
        )
    )
    parser.add_argument("--suite-yaml", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    npz_path, meta_path = build_phase6a_replay_input(
        args.suite_yaml,
        task_id=args.task_id,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(f"wrote {npz_path}")
    print(f"wrote {meta_path}")
    print(f"wrote {npz_path.parent / SUITE_SNAPSHOT_FILENAME}")
    print(f"wrote {npz_path.parent / TASK_SNAPSHOT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
