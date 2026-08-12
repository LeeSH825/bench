#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_head(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def copy_template_if_missing(template: Path, output: Path) -> None:
    if not output.exists():
        output.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--study-id", default="side-gyro-mag-comp-v1")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    template = repo / "agent_system/side_gyro_mag_comp/state/STAGE_STATE.template.json"
    output = repo / "agent_system/side_gyro_mag_comp/state/STAGE_STATE.json"
    if output.exists() and not args.reset:
        print(output)
        return 0

    data = json.loads(template.read_text(encoding="utf-8"))
    data["study_id"] = args.study_id
    data["source_commit"] = git_head(repo)
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    copy_template_if_missing(repo / "agent_system/side_gyro_mag_comp/state/DECISION_LEDGER.template.md", repo / "agent_system/side_gyro_mag_comp/state/DECISION_LEDGER.md")
    copy_template_if_missing(repo / "agent_system/side_gyro_mag_comp/state/DEFERRED_REGISTER.template.md", repo / "agent_system/side_gyro_mag_comp/state/DEFERRED_REGISTER.md")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
