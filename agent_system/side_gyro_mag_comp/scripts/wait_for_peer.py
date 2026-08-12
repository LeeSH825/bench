#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--actor", required=True, choices=["CODEX", "CLAUDE"])
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--timeout-seconds", type=float, default=86400.0)
    args = p.parse_args()
    path = Path(args.state)
    start = time.monotonic()
    while True:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("next_actor") == args.actor:
                print(json.dumps(data, ensure_ascii=False))
                return 0
            if str(data.get("current_stage", "")).startswith("COMPLETE"):
                print(json.dumps(data, ensure_ascii=False))
                return 10
        if time.monotonic() - start >= args.timeout_seconds:
            raise SystemExit("WAIT_TIMEOUT")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
