from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.runners.run_suite import load_suite_yaml, run_one, _expand_sweep


def _try_read_text(p: Path, max_bytes: int = 20000) -> str:
    try:
        if not p.exists():
            return ""
        data = p.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _print_failure_details(run_dir: Path) -> None:
    failure_json = run_dir / "failure.json"
    stderr_log = run_dir / "stderr.log"
    stdout_log = run_dir / "stdout.log"

    if failure_json.exists():
        try:
            obj = json.loads(failure_json.read_text(encoding="utf-8"))
            print("\n[smoke_run] failure.json:")
            print(json.dumps(obj, indent=2, ensure_ascii=False))
        except Exception:
            print("\n[smoke_run] failure.json exists but failed to parse.")
            print(_try_read_text(failure_json))

    tail = _try_read_text(stderr_log)
    if tail.strip():
        print("\n[smoke_run] stderr.log (tail):")
        print(tail)

    tail2 = _try_read_text(stdout_log)
    if tail2.strip():
        print("\n[smoke_run] stdout.log (tail):")
        print(tail2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-yaml", type=str, required=True)
    ap.add_argument("--task-id", type=str, required=True)
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", type=str, default="frozen")
    ap.add_argument("--init-id", type=str, default="untrained")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--precision", type=str, default="fp32")
    args = ap.parse_args()

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = load_suite_yaml(suite_path)

    task = None
    for t in suite.get("tasks", []) or []:
        if t.get("task_id") == args.task_id:
            task = t
            break
    if task is None:
        raise KeyError(f"task_id={args.task_id} not found in suite.tasks")

    model = None
    for m in suite.get("models", []) or []:
        if m.get("model_id") == args.model_id:
            model = m
            break
    if model is None:
        raise KeyError(f"model_id={args.model_id} not found in suite.models")

    scenario_list = _expand_sweep(task.get("sweep"))
    scenario_settings = scenario_list[0] if scenario_list else {}

    res = run_one(
        suite=suite,
        task=task,
        model=model,
        scenario_settings=scenario_settings,
        seed=int(args.seed),
        track_id=str(args.track),
        device_str=str(args.device),
        precision=str(args.precision),
        init_id=str(args.init_id),
    )

    print("[smoke_run] result:")
    for k in ["status", "run_dir", "suite", "task_id", "scenario_id", "seed", "model_id", "track_id"]:
        if k in res:
            print(f"  {k}: {res[k]}")

    if "scenario_settings" in res:
        print(f"  scenario_settings: {res['scenario_settings']}")

    # 항상 찍어서 “failed 원인”이 콘솔에 바로 보이게
    if res.get("status") != "ok":
        if "error" in res and res["error"]:
            print("\n[smoke_run] error:", res["error"])
        run_dir = Path(str(res.get("run_dir", ""))).expanduser()
        if run_dir.exists():
            _print_failure_details(run_dir)
        else:
            print("[smoke_run] run_dir not found on disk:", run_dir)


if __name__ == "__main__":
    main()
