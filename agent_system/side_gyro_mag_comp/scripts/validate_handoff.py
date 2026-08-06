#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    raise SystemExit(f"INVALID_HANDOFF: {msg}")


def main() -> int:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "SEALED_FOR_AUDIT":
        fail("status must be SEALED_FOR_AUDIT")
    targets = data.get("audit_target_paths")
    if not isinstance(targets, list) or data.get("target_count") != len(targets):
        fail("target_count does not match audit_target_paths")
    if len(targets) == 0:
        fail("TARGET_NOT_FOUND")
    if len(targets) != 1:
        fail("AMBIGUOUS: seal one canonical target bundle manifest")
    if not data.get("checkpoint_digest"):
        fail("missing checkpoint_digest")
    claims = data.get("claims", [])
    if not claims:
        fail("no claims")
    seen = set()
    for claim in claims:
        cid = claim.get("claim_id")
        if not cid or cid in seen:
            fail(f"missing or duplicate claim_id {cid!r}")
        seen.add(cid)
        if not claim.get("predicate"):
            fail(f"{cid}: missing predicate")
        if not claim.get("red_path_test"):
            fail(f"{cid}: missing red_path_test")
        if claim.get("target_population_count", 0) <= 0:
            fail(f"{cid}: empty target population")
        if not claim.get("machine_evidence"):
            fail(f"{cid}: missing machine evidence")
    print("PASS_HANDOFF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
