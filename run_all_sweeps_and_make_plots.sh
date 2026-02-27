#!/usr/bin/env bash
set -euo pipefail

# Run all suite tasks/seeds/scenarios and generate reproducible overlay plots.
#
# Usage example:
# DEVICE=cuda SEEDS="0 1 2" SUITES="bench/configs/suite_all_simple.yaml bench/configs/suite_shift.yaml" ./run_all_sweeps_and_make_plots.sh
#
# Environment configuration:
DEVICE=${DEVICE:-auto}
SEEDS=${SEEDS:-"0"}
RUNS_ROOT=${RUNS_ROOT:-"runs"}
OUT_DIR=${OUT_DIR:-"reports"}
CACHE_ROOT=${CACHE_ROOT:-"/tmp/bench_data_cache"}
SUITES=${SUITES:-"bench/configs/suite_all_simple_tiny.yaml"}
PY=${PY:-".venv/bin/python"}
NCLT_ROOT=${NCLT_ROOT:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -x "$PY" ]]; then
  echo "[FATAL] Python executable not found or not executable: $PY" >&2
  exit 1
fi

if [[ "$DEVICE" == "auto" ]]; then
  if "$PY" - <<'PY'
import sys
import torch

sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

RUNS_ROOT_ABS="$("$PY" - "$RUNS_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
OUT_DIR_ABS="$("$PY" - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
CACHE_ROOT_ABS="$("$PY" - "$CACHE_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
NCLT_ROOT_ABS=""
if [[ -n "${NCLT_ROOT}" ]]; then
  NCLT_ROOT_ABS="$("$PY" - "$NCLT_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
elif [[ -d "$SCRIPT_DIR/external_data/nclt" ]]; then
  NCLT_ROOT_ABS="$("$PY" - "$SCRIPT_DIR/external_data/nclt" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
fi
DEFAULT_RUNS_ABS="$("$PY" - "$SCRIPT_DIR" <<'PY'
from pathlib import Path
import sys
print((Path(sys.argv[1]).resolve() / "runs").resolve())
PY
)"

mkdir -p "$OUT_DIR_ABS" "$CACHE_ROOT_ABS" "$RUNS_ROOT_ABS"

# run_suite currently writes under repo-root "runs". To support configurable RUNS_ROOT,
# we temporarily map ./runs -> $RUNS_ROOT when needed.
RUNS_SYMLINK_MODE=0
RUNS_BACKUP_PATH=""
RUNS_PREV_LINK_TARGET=""
cleanup_runs_mapping() {
  if [[ "$RUNS_SYMLINK_MODE" -eq 1 ]]; then
    rm -f "runs"
    if [[ -n "$RUNS_BACKUP_PATH" && -e "$RUNS_BACKUP_PATH" ]]; then
      mv "$RUNS_BACKUP_PATH" "runs"
    elif [[ -n "$RUNS_PREV_LINK_TARGET" ]]; then
      ln -s "$RUNS_PREV_LINK_TARGET" "runs"
    fi
  fi
}
trap cleanup_runs_mapping EXIT

if [[ "$RUNS_ROOT_ABS" != "$DEFAULT_RUNS_ABS" ]]; then
  RUNS_SYMLINK_MODE=1
  if [[ -L "runs" ]]; then
    RUNS_PREV_LINK_TARGET="$(readlink "runs" || true)"
    rm -f "runs"
  elif [[ -e "runs" ]]; then
    RUNS_BACKUP_PATH="runs.__backup__.$(date +%Y%m%d_%H%M%S)"
    mv "runs" "$RUNS_BACKUP_PATH"
  fi
  ln -s "$RUNS_ROOT_ABS" "runs"
fi

SMOKE_HAS_SWEEP_ALL=0
if "$PY" -m bench.tasks.smoke_data --help 2>&1 | grep -q -- "--sweep-all"; then
  SMOKE_HAS_SWEEP_ALL=1
fi

RUN_SUITE_HAS_RUNS_ROOT=0
if "$PY" -m bench.runners.run_suite --help 2>&1 | grep -q -- "--runs-root"; then
  RUN_SUITE_HAS_RUNS_ROOT=1
fi

read -r -a SUITE_ARR <<< "$SUITES"
read -r -a SEED_ARR <<< "$SEEDS"

if [[ "${#SUITE_ARR[@]}" -eq 0 ]]; then
  echo "[FATAL] SUITES is empty." >&2
  exit 1
fi
if [[ "${#SEED_ARR[@]}" -eq 0 ]]; then
  echo "[FATAL] SEEDS is empty." >&2
  exit 1
fi

declare -a DATA_GEN_FAILURES=()
declare -a FAILED_MODEL_RUNS=()
declare -a SKIPPED_MODELS=()
declare -A CUSTOM_PLOT_COUNT=()
NCLT_DATA_READY=0

if [[ -n "$NCLT_ROOT_ABS" ]]; then
  if "$PY" - "$NCLT_ROOT_ABS" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
if not root.exists():
    print(f"[WARN] NCLT_ROOT does not exist: {root}")
    sys.exit(1)

def session_ok(session: str) -> bool:
    candidates = [
        root / "prepared" / f"{session}.npz",
        root / "sessions" / f"{session}.npz",
        root / "sessions" / session / "xy.npz",
        root / f"{session}.npz",
    ]
    return any(p.exists() for p in candidates)

required = ["2012-01-22", "2012-04-29"]
missing = [s for s in required if not session_ok(s)]
if missing:
    print(f"[WARN] NCLT_ROOT missing prepared session files for: {missing}")
    print(f"[WARN] checked root: {root}")
    sys.exit(1)

print(f"[INFO] NCLT_ROOT ready: {root}")
PY
  then
    NCLT_DATA_READY=1
    export NCLT_ROOT="$NCLT_ROOT_ABS"
  fi
fi

echo "[INFO] Starting sweep run"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] SEEDS=${SEED_ARR[*]}"
echo "[INFO] RUNS_ROOT=$RUNS_ROOT_ABS"
echo "[INFO] OUT_DIR=$OUT_DIR_ABS"
echo "[INFO] CACHE_ROOT=$CACHE_ROOT_ABS"
echo "[INFO] SUITES=${SUITE_ARR[*]}"
if [[ "$NCLT_DATA_READY" -eq 1 ]]; then
  echo "[INFO] NCLT_ROOT=$NCLT_ROOT_ABS"
else
  echo "[INFO] NCLT_ROOT not ready (NCLT tasks will be skipped unless prepared)."
fi

for SUITE in "${SUITE_ARR[@]}"; do
  if [[ ! -f "$SUITE" ]]; then
    echo "[WARN] Suite file not found, skipping: $SUITE"
    continue
  fi

  echo
  echo "============================================================"
  echo "[SUITE] $SUITE"

  declare -a TASKS=()
  declare -a MODELS=()
  declare -A TASK_IS_SHIFT=()
  declare -A TASK_NEEDS_NCLT=()
  declare -A MODEL_REGISTERED=()
  declare -A MODEL_IS_BASELINE=()
  declare -A MODEL_SUPPORTS_ADAPT=()
  SUITE_NAME=""
  SUITE_BUDGETED_ENABLED=0

  while IFS=$'\t' read -r kind c1 c2 c3 c4; do
    case "$kind" in
      SUITE)
        SUITE_NAME="$c1"
        ;;
      BUDGETED)
        SUITE_BUDGETED_ENABLED="$c1"
        ;;
      TASK)
        TASKS+=("$c1")
        TASK_IS_SHIFT["$c1"]="$c2"
        TASK_NEEDS_NCLT["$c1"]="$c3"
        ;;
      MODEL)
        MODELS+=("$c1")
        MODEL_REGISTERED["$c1"]="$c2"
        MODEL_IS_BASELINE["$c1"]="$c3"
        MODEL_SUPPORTS_ADAPT["$c1"]="$c4"
        ;;
    esac
  done < <("$PY" - "$SUITE" <<'PY'
import sys
from pathlib import Path
import yaml

suite_path = Path(sys.argv[1]).expanduser().resolve()
obj = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
suite_name = str(((obj.get("suite") or {}).get("name")) or suite_path.stem)

def as_bool(x, default=True):
    if x is None:
        return bool(default)
    return bool(x)

def get_task_id(item):
    if isinstance(item, str):
        return item, {}
    if isinstance(item, dict):
        return str(item.get("task_id", "")).strip(), item
    return "", {}

def get_model_id(item):
    if isinstance(item, str):
        return item, {}
    if isinstance(item, dict):
        return str(item.get("model_id", "")).strip(), item
    return "", {}

def task_is_shift(task_cfg):
    if not isinstance(task_cfg, dict):
        return False
    n = task_cfg.get("noise") or {}
    t0 = (((n.get("shift") or {}).get("t0")))
    if t0 is not None:
        return True
    # fallback for potential alternate shapes
    s = task_cfg.get("shift") or {}
    return s.get("t0") is not None

def task_needs_nclt(task_id, task_cfg):
    fam = str((task_cfg or {}).get("task_family", "")).strip().lower()
    if fam in {"nclt", "nclt_v0", "nclt_segway", "nclt_segway_v0"}:
        return True
    return "nclt" in str(task_id).lower()

registry_ok = False
def registered_model(mid: str) -> bool:
    global registry_ok
    try:
        from bench.models.registry import get_model_adapter_class
        registry_ok = True
        get_model_adapter_class(mid)
        return True
    except Exception:
        return False

tracks = ((obj.get("runner") or {}).get("tracks") or [])
budgeted_enabled = False
for tr in tracks:
    if not isinstance(tr, dict):
        continue
    if str(tr.get("track_id", "")).strip() == "budgeted" and bool(tr.get("adaptation_enabled", False)):
        budgeted_enabled = True
        break

print(f"SUITE\t{suite_name}")
print(f"BUDGETED\t{1 if budgeted_enabled else 0}")

for t in (obj.get("tasks") or []):
    task_id, tcfg = get_task_id(t)
    if not task_id:
        continue
    enabled = as_bool((tcfg.get("enabled") if isinstance(tcfg, dict) else None), True)
    if not enabled:
        continue
    is_shift = task_is_shift(tcfg)
    needs_nclt = task_needs_nclt(task_id, tcfg)
    print(f"TASK\t{task_id}\t{1 if is_shift else 0}\t{1 if needs_nclt else 0}")

for m in (obj.get("models") or []):
    model_id, mcfg = get_model_id(m)
    if not model_id:
        continue
    enabled = as_bool((mcfg.get("enabled") if isinstance(mcfg, dict) else None), True)
    if not enabled:
        continue
    reg = registered_model(model_id)
    caps = (mcfg.get("capabilities") or {}) if isinstance(mcfg, dict) else {}
    raw_supports_adapt = caps.get("supports_adaptation", None)
    if raw_supports_adapt is None:
        supports_adapt = False
        lo = model_id.lower()
        if any(k in lo for k in ("adaptive", "maml", "split")):
            supports_adapt = True
    else:
        supports_adapt = bool(raw_supports_adapt)
    is_baseline = model_id.startswith("mb_kf") or model_id in {"oracle_kf", "nominal_kf", "oracle_shift_kf"}
    print(f"MODEL\t{model_id}\t{1 if reg else 0}\t{1 if is_baseline else 0}\t{1 if supports_adapt else 0}")
PY
)

  if [[ -z "$SUITE_NAME" ]]; then
    echo "[WARN] Could not resolve suite name from $SUITE; skipping"
    continue
  fi

  if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "[WARN] No enabled tasks found in suite=$SUITE_NAME; skipping"
    continue
  fi
  if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "[WARN] No enabled models found in suite=$SUITE_NAME; skipping"
    continue
  fi

  echo "[INFO] suite_name=$SUITE_NAME"
  echo "[INFO] budgeted_track_enabled=$SUITE_BUDGETED_ENABLED"
  echo "[INFO] tasks=${TASKS[*]}"
  echo "[INFO] models=${MODELS[*]}"

  for TASK in "${TASKS[@]}"; do
    CUSTOM_PLOT_COUNT["$SUITE_NAME::$TASK"]=0

    for SEED in "${SEED_ARR[@]}"; do
      echo
      echo "[TASK] suite=$SUITE_NAME task=$TASK seed=$SEED"

      if [[ "${TASK_NEEDS_NCLT[$TASK]:-0}" == "1" ]] && [[ "$NCLT_DATA_READY" -ne 1 ]]; then
        echo "[WARN] NCLT task requires prepared dataset but NCLT_ROOT is not ready. Skipping."
        echo "[WARN] Fix with:"
        echo "       scripts/datasets/download_nclt_groundtruth.sh --out-root external_data/nclt"
        echo "       python3 scripts/datasets/prepare_nclt_groundtruth_npz.py --raw-root external_data/nclt/raw/ground_truth --out-root external_data/nclt"
        echo "       export NCLT_ROOT=\$PWD/external_data/nclt"
        DATA_GEN_FAILURES+=("$SUITE_NAME/$TASK/seed_$SEED(nclt_root_not_ready)")
        continue
      fi

      echo "[STEP] Data generation (smoke_data)"
      DATA_CMD=(
        "$PY" -m bench.tasks.smoke_data
        --suite-yaml "$SUITE"
        --task "$TASK"
        --seed "$SEED"
      )
      if [[ "$SMOKE_HAS_SWEEP_ALL" -eq 1 ]]; then
        DATA_CMD+=(--sweep-all)
      else
        echo "[WARN] smoke_data --sweep-all not available; generating default scenario only."
      fi
      set +e
      BENCH_DATA_CACHE="$CACHE_ROOT_ABS" "${DATA_CMD[@]}"
      DATA_RC=$?
      set -e
      if [[ "$DATA_RC" -ne 0 ]]; then
        echo "[WARN] smoke_data failed for suite=$SUITE_NAME task=$TASK seed=$SEED (rc=$DATA_RC); skipping this task/seed."
        DATA_GEN_FAILURES+=("$SUITE_NAME/$TASK/seed_$SEED(rc=$DATA_RC)")
        continue
      fi

      # Collect run_dir paths from each successful run_suite invocation,
      # then write one merged manifest so make_report(latest_manifest) sees
      # exactly this task+seed invocation and avoids stale historical mixing.
      RUN_DIR_ACCUM_FILE="$(mktemp)"

      for MODEL_ID in "${MODELS[@]}"; do
        if [[ "${MODEL_REGISTERED[$MODEL_ID]:-0}" != "1" ]]; then
          echo "[WARN] model_id=$MODEL_ID not present in registry; skipping."
          SKIPPED_MODELS+=("$SUITE_NAME/$TASK/seed_$SEED:$MODEL_ID(not_registered)")
          continue
        fi

        declare -a PLANS=()
        if [[ "${MODEL_IS_BASELINE[$MODEL_ID]:-0}" == "1" ]]; then
          # MB-KF baseline policy: inference only
          PLANS=("untrained:frozen")
        else
          # Trainable model policy: untrained + trained frozen
          PLANS=("untrained:frozen" "trained:frozen")
          # For shift tasks with meaningful budgeted track, try trained:budgeted
          if [[ "${TASK_IS_SHIFT[$TASK]:-0}" == "1" ]] \
            && [[ "$SUITE_BUDGETED_ENABLED" == "1" ]] \
            && [[ "${MODEL_SUPPORTS_ADAPT[$MODEL_ID]:-0}" == "1" ]]; then
            PLANS+=("trained:budgeted")
          fi
        fi

        PLANS_CSV="$(IFS=,; echo "${PLANS[*]}")"
        echo "[STEP] run_suite model=$MODEL_ID plans=$PLANS_CSV"

        RUN_CMD=(
          "$PY" -m bench.runners.run_suite
          --suite-yaml "$SUITE"
          --tasks "$TASK"
          --models "$MODEL_ID"
          --seeds "$SEED"
          --plans "$PLANS_CSV"
          --device "$DEVICE"
        )
        if [[ "$RUN_SUITE_HAS_RUNS_ROOT" -eq 1 ]]; then
          RUN_CMD+=(--runs-root "$RUNS_ROOT_ABS")
        fi

        # Continue on model-level failure, summarize later.
        set +e
        BENCH_DATA_CACHE="$CACHE_ROOT_ABS" "${RUN_CMD[@]}"
        RC=$?
        set -e
        if [[ "$RC" -ne 0 ]]; then
          echo "[WARN] run_suite failed for model=$MODEL_ID (rc=$RC); continuing."
          FAILED_MODEL_RUNS+=("$SUITE_NAME/$TASK/seed_$SEED:$MODEL_ID(rc=$RC)")
          continue
        fi

        # Append latest manifest's run_dirs into this task+seed accumulator.
        "$PY" - "$RUNS_ROOT_ABS" "$SUITE_NAME" "$RUN_DIR_ACCUM_FILE" <<'PY'
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1]).expanduser().resolve()
suite_name = sys.argv[2]
accum_file = Path(sys.argv[3]).expanduser().resolve()

mdir = runs_root / suite_name / "_manifests"
if not mdir.exists():
    sys.exit(0)
manifests = sorted([p for p in mdir.glob("*.json") if p.is_file()], key=lambda p: p.stat().st_mtime)
if not manifests:
    sys.exit(0)
latest = manifests[-1]
try:
    obj = json.loads(latest.read_text(encoding="utf-8"))
except Exception:
    sys.exit(0)
raw = obj.get("run_dirs", [])
if not isinstance(raw, list):
    sys.exit(0)

existing = set()
if accum_file.exists():
    existing = {ln.strip() for ln in accum_file.read_text(encoding="utf-8").splitlines() if ln.strip()}

to_add = []
for rd in raw:
    if rd is None:
        continue
    p = str(Path(str(rd)).expanduser().resolve())
    if p in existing:
        continue
    existing.add(p)
    to_add.append(p)

if to_add:
    accum_file.parent.mkdir(parents=True, exist_ok=True)
    with accum_file.open("a", encoding="utf-8") as f:
        for p in to_add:
            f.write(p + "\n")
PY
      done

      MERGED_MANIFEST_PATH="$("$PY" - "$RUNS_ROOT_ABS" "$SUITE_NAME" "$SUITE" "$TASK" "$SEED" "$RUN_DIR_ACCUM_FILE" <<'PY'
import json
import re
import sys
import time
import uuid
from pathlib import Path

runs_root = Path(sys.argv[1]).expanduser().resolve()
suite_name = sys.argv[2]
suite_yaml = str(Path(sys.argv[3]).expanduser().resolve())
task_id = sys.argv[4]
seed = str(sys.argv[5])
accum_file = Path(sys.argv[6]).expanduser().resolve()

run_dirs = []
if accum_file.exists():
    for ln in accum_file.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            run_dirs.append(str(Path(s).expanduser().resolve()))
run_dirs = sorted(set(run_dirs))

mdir = runs_root / suite_name / "_manifests"
mdir.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
name = f"{ts}_merged_{safe_task}_seed{seed}_{uuid.uuid4().hex[:8]}.json"
out_path = mdir / name
payload = {
    "suite_name": suite_name,
    "suite_yaml": suite_yaml,
    "created_at_unix": float(time.time()),
    "source": "run_all_sweeps_and_make_plots.sh",
    "task_id": task_id,
    "seed": seed,
    "run_count": int(len(run_dirs)),
    "run_dirs": run_dirs,
}
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(str(out_path))
PY
)"
      echo "[INFO] merged_manifest=$MERGED_MANIFEST_PATH"
      rm -f "$RUN_DIR_ACCUM_FILE"

      echo "[STEP] make_report (latest_manifest scope)"
      REPORT_CMD=(
        "$PY" -m bench.reports.make_report
        --suite-yaml "$SUITE"
        --runs-root "$RUNS_ROOT_ABS"
        --out-dir "$OUT_DIR_ABS"
        --input-scope latest_manifest
        --fig5a-plot
      )
      "${REPORT_CMD[@]}"

      echo "[STEP] custom plot x=inv_r2_db y=mse_db_mean"
      PLOT_RESULT="$("$PY" - "$SUITE" "$SUITE_NAME" "$OUT_DIR_ABS" "$TASK" <<'PY'
import sys
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

suite_yaml = Path(sys.argv[1]).expanduser().resolve()
suite_name = sys.argv[2]
out_dir = Path(sys.argv[3]).expanduser().resolve()
task_id = sys.argv[4]

task_name = task_id
try:
    suite_obj = yaml.safe_load(suite_yaml.read_text(encoding="utf-8")) or {}
    for t in (suite_obj.get("tasks") or []):
        if not isinstance(t, dict):
            continue
        if str(t.get("task_id", "")).strip() != str(task_id):
            continue
        candidate = str(t.get("task_name", "")).strip()
        if candidate:
            task_name = candidate
        break
except Exception:
    pass

def slugify(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    out = out.strip("._")
    return out or "task"

task_slug = slugify(task_name)

suite_dir = out_dir / suite_name
if not suite_dir.exists():
    print(f"skipped\tno_suite_report_dir:{suite_dir}")
    sys.exit(0)

latest_alias = suite_dir / "latest"
if latest_alias.exists() and latest_alias.is_dir():
    latest_report_dir = latest_alias.resolve()
else:
    time_dirs = []
    for day in suite_dir.iterdir():
        if not day.is_dir() or day.name == "latest":
            continue
        for tdir in day.iterdir():
            if tdir.is_dir():
                time_dirs.append(tdir)
    if not time_dirs:
        print(f"skipped\tno_timestamped_report_dir:{suite_dir}")
        sys.exit(0)
    latest_report_dir = max(time_dirs, key=lambda p: p.stat().st_mtime)
agg = latest_report_dir / "tables" / f"aggregate_{suite_name}.csv"
if not agg.exists():
    found = sorted(latest_report_dir.rglob(f"aggregate_{suite_name}.csv"))
    if found:
        agg = found[0]
    else:
        print(f"skipped\taggregate_csv_not_found_under:{latest_report_dir}")
        sys.exit(0)

df = pd.read_csv(agg)
if "task_id" not in df.columns:
    print(f"skipped\tmissing_task_id_column:{agg}")
    sys.exit(0)

tdf = df[df["task_id"].astype(str) == str(task_id)].copy()
if tdf.empty:
    print(f"skipped\tno_rows_for_task:{task_id}")
    sys.exit(0)

x_col = "inv_r2_db"
y_col = "mse_db_mean"
for c in (x_col, y_col, "model_id", "init_id"):
    if c not in tdf.columns:
        print(f"skipped\tmissing_column:{c}")
        sys.exit(0)

tdf[x_col] = pd.to_numeric(tdf[x_col], errors="coerce")
tdf[y_col] = pd.to_numeric(tdf[y_col], errors="coerce")
tdf["model_id"] = tdf["model_id"].astype(str)
tdf["init_id"] = tdf["init_id"].astype(str)
tdf = tdf.dropna(subset=[x_col, y_col])
if tdf.empty:
    print(f"skipped\tno_numeric_rows_for_task:{task_id}")
    sys.exit(0)

if tdf[x_col].nunique(dropna=True) <= 1:
    print(f"skipped\tinv_r2_db_missing_or_constant_for_task:{task_id}")
    sys.exit(0)

models = sorted(tdf["model_id"].unique())
cmap = plt.get_cmap("tab20")
colors = {m: cmap(i % cmap.N) for i, m in enumerate(models)}
markers = {"untrained": "o", "trained": "s"}
fallback_markers = ["^", "D", "v", "P", "X", "*", "h"]

plt.figure(figsize=(10, 6))

for m in models:
    mdf = tdf[tdf["model_id"] == m].copy()
    init_values = list(sorted(mdf["init_id"].unique()))
    ordered_inits = []
    for key in ("untrained", "trained"):
        if key in init_values:
            ordered_inits.append(key)
    for iv in init_values:
        if iv not in ordered_inits:
            ordered_inits.append(iv)

    fb_idx = 0
    for init_id in ordered_inits:
        sdf = mdf[mdf["init_id"] == init_id].sort_values(by=x_col)
        if sdf.empty:
            continue
        marker = markers.get(init_id)
        if marker is None:
            marker = fallback_markers[fb_idx % len(fallback_markers)]
            fb_idx += 1
        plt.plot(
            sdf[x_col].to_numpy(),
            sdf[y_col].to_numpy(),
            color=colors[m],
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            label=f"{m} | {init_id}",
        )

plt.xlabel("inv_r2_db")
plt.ylabel("mse_db_mean")
plt.title(f"Custom Fig5a-style overlay: {task_name}")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, ncol=1)
plt.tight_layout()

plots_dir = latest_report_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
out_png = plots_dir / f"{suite_name}_{task_slug}.png"
plt.savefig(out_png)
plt.close()

print(f"generated\t{out_png}")
PY
)"
      IFS=$'\t' read -r PLOT_STATUS PLOT_MSG <<< "$PLOT_RESULT"
      if [[ "$PLOT_STATUS" == "generated" ]]; then
        echo "[OK] custom plot: $PLOT_MSG"
        current_count="${CUSTOM_PLOT_COUNT["$SUITE_NAME::$TASK"]:-0}"
        CUSTOM_PLOT_COUNT["$SUITE_NAME::$TASK"]=$((current_count + 1))
      else
        echo "[WARN] custom plot skipped for $SUITE_NAME/$TASK (seed=$SEED): $PLOT_MSG"
      fi
    done
  done
done

echo
echo "============================================================"
echo "[SUMMARY]"
if [[ "${#DATA_GEN_FAILURES[@]}" -gt 0 ]]; then
  echo "[SUMMARY] data generation failures (skipped task/seed):"
  for d in "${DATA_GEN_FAILURES[@]}"; do
    echo "  - $d"
  done
fi
if [[ "${#SKIPPED_MODELS[@]}" -gt 0 ]]; then
  echo "[SUMMARY] skipped models:"
  for s in "${SKIPPED_MODELS[@]}"; do
    echo "  - $s"
  done
fi
if [[ "${#FAILED_MODEL_RUNS[@]}" -gt 0 ]]; then
  echo "[SUMMARY] model run failures (continued):"
  for f in "${FAILED_MODEL_RUNS[@]}"; do
    echo "  - $f"
  done
fi

if [[ "${#CUSTOM_PLOT_COUNT[@]}" -gt 0 ]]; then
  echo "[SUMMARY] custom plot coverage:"
  for k in "${!CUSTOM_PLOT_COUNT[@]}"; do
    count="${CUSTOM_PLOT_COUNT[$k]}"
    if [[ "$count" -lt 1 ]]; then
      echo "  - $k: 0 generated (warn: inv_r2_db may be missing/constant)"
    else
      echo "  - $k: $count generated"
    fi
  done
fi

echo "[DONE] run_all_sweeps_and_make_plots.sh completed."
