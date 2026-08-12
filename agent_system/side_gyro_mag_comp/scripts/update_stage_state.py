#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("state")
    p.add_argument("--stage")
    p.add_argument("--status")
    p.add_argument("--next-actor")
    p.add_argument("--next-stage")
    p.add_argument("--checkpoint")
    p.add_argument("--codex-handoff")
    p.add_argument("--claude-audit")
    p.add_argument("--method-decision")
    p.add_argument("--terminal-reason")
    p.add_argument("--repair-stage")
    args = p.parse_args()

    path = Path(args.state)
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        "current_stage": args.stage,
        "stage_status": args.status,
        "next_actor": args.next_actor,
        "next_allowed_stage": args.next_stage,
        "sealed_checkpoint": args.checkpoint,
        "codex_handoff": args.codex_handoff,
        "claude_audit": args.claude_audit,
        "method_decision": args.method_decision,
        "terminal_reason": args.terminal_reason,
    }
    for key, value in mapping.items():
        if value is not None:
            data[key] = value
    if args.repair_stage:
        current = data.setdefault("repair_round_by_stage", {}).get(args.repair_stage, 0)
        if current >= 1:
            raise SystemExit(f"REPAIR_LIMIT_REACHED: {args.repair_stage}")
        data["repair_round_by_stage"][args.repair_stage] = current + 1
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
