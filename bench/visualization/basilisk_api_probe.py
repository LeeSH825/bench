from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any


BASILISK_API_PROBE_FILENAME = "basilisk_api_probe.json"

_CANDIDATE_MODULES = (
    "Basilisk",
    "Basilisk.simulation.vizard.dataFileToViz",
    "Basilisk.simulation.vizard.vizInterface",
    "Basilisk.simulation.dataFileToViz",
    "Basilisk.simulation.vizInterface",
    "Basilisk.utilities.vizSupport",
    "Basilisk.utilities.SimulationBaseClass",
    "Basilisk.utilities.macros",
)
_NAME_TERMS = (
    "datafiletoviz",
    "datafile",
    "viz",
    "vizinterface",
    "save",
    "recorder",
)


def _short_doc(value: Any, limit: int = 500) -> str | None:
    doc = inspect.getdoc(value)
    if not doc:
        return None
    return doc[:limit]


def _signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _candidate_entry(module_name: str, name: str, value: Any) -> dict[str, Any]:
    return {
        "module": module_name,
        "name": name,
        "qualified_name": f"{module_name}.{name}",
        "signature": _signature(value),
        "docstring": _short_doc(value),
    }


def _find_scenario_data_to_viz(
    basilisk_module: Any,
) -> list[str]:
    roots: list[Path] = []
    module_file = getattr(basilisk_module, "__file__", None)
    if module_file:
        package_path = Path(module_file).resolve()
        roots.extend(
            [
                package_path.parent,
                package_path.parent.parent,
                package_path.parent.parent.parent,
            ]
        )
    roots.extend([Path.cwd(), Path.home() / "basilisk"])

    found: list[str] = []
    visited: set[Path] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            continue
        if resolved in visited or not resolved.is_dir():
            continue
        visited.add(resolved)
        direct_candidates = (
            resolved / "examples" / "scenarioDataToViz.py",
            resolved / "bsk_example" / "examples" / "scenarioDataToViz.py",
            resolved
            / "basilisk"
            / "bsk_example"
            / "examples"
            / "scenarioDataToViz.py",
        )
        for candidate in direct_candidates:
            if candidate.is_file():
                path_str = str(candidate.resolve())
                if path_str not in found:
                    found.append(path_str)
    return found


def _probe_payload(require_basilisk: bool) -> dict[str, Any]:
    import_results: dict[str, dict[str, Any]] = {}
    module_files: dict[str, str | None] = {}
    candidate_callables: list[dict[str, Any]] = []
    candidate_classes: list[dict[str, Any]] = []
    errors: list[str] = []
    imported: dict[str, Any] = {}

    for module_name in _CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_name)
            imported[module_name] = module
            public_attributes = sorted(
                name for name in dir(module) if not name.startswith("_")
            )
            import_results[module_name] = {
                "success": True,
                "error": None,
                "public_attributes": public_attributes,
                "docstring": _short_doc(module),
            }
            module_files[module_name] = getattr(module, "__file__", None)

            for name in public_attributes:
                if not any(term in name.lower() for term in _NAME_TERMS):
                    continue
                try:
                    value = getattr(module, name)
                except Exception as exc:
                    errors.append(
                        f"{module_name}.{name}: {type(exc).__name__}: {exc}"
                    )
                    continue
                if inspect.isclass(value):
                    candidate_classes.append(
                        _candidate_entry(module_name, name, value)
                    )
                elif callable(value):
                    candidate_callables.append(
                        _candidate_entry(module_name, name, value)
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            import_results[module_name] = {
                "success": False,
                "error": error,
                "public_attributes": [],
                "docstring": None,
            }
            module_files[module_name] = None
            errors.append(f"{module_name}: {error}")

    basilisk = imported.get("Basilisk")
    basilisk_available = basilisk is not None
    basilisk_version = (
        None
        if basilisk is None
        else str(getattr(basilisk, "__version__", "unknown"))
    )
    examples = (
        [] if basilisk is None else _find_scenario_data_to_viz(basilisk)
    )

    data_module = imported.get("Basilisk.simulation.dataFileToViz")
    viz_module = imported.get("Basilisk.simulation.vizInterface")
    data_attrs: list[str] = []
    viz_attrs: list[str] = []
    if data_module is not None and hasattr(data_module, "DataFileToViz"):
        try:
            data_attrs = sorted(
                name
                for name in dir(data_module.DataFileToViz())
                if not name.startswith("_")
            )
        except Exception as exc:
            errors.append(f"DataFileToViz instance: {type(exc).__name__}: {exc}")
    if viz_module is not None and hasattr(viz_module, "VizInterface"):
        try:
            viz_attrs = sorted(
                name
                for name in dir(viz_module.VizInterface())
                if not name.startswith("_")
            )
        except Exception as exc:
            errors.append(f"VizInterface instance: {type(exc).__name__}: {exc}")

    required_data_attrs = {
        "attitudeType",
        "convertPosToMeters",
        "dataFileName",
        "delimiter",
        "headerLine",
        "scStateOutMsgs",
        "setNumOfSatellites",
    }
    required_viz_attrs = {
        "broadcastStream",
        "liveStream",
        "protoFilename",
        "saveFile",
        "scData",
        "settings",
    }
    native_contract_discoverable = bool(
        examples
        and required_data_attrs.issubset(data_attrs)
        and required_viz_attrs.issubset(viz_attrs)
    )

    return {
        "schema_version": "basilisk_api_probe_v1",
        "basilisk_required": bool(require_basilisk),
        "basilisk_available": bool(basilisk_available),
        "basilisk_version": basilisk_version,
        "import_results": import_results,
        "candidate_modules": list(_CANDIDATE_MODULES),
        "candidate_callables": candidate_callables,
        "candidate_classes": candidate_classes,
        "module_files": module_files,
        "instance_attributes": {
            "DataFileToViz": data_attrs,
            "VizInterface": viz_attrs,
        },
        "discovered_examples": examples,
        "native_contract_discoverable": native_contract_discoverable,
        "native_contract": (
            {
                "attitude_type": 0,
                "attitude_representation": "MRP",
                "delimiter": ",",
                "header_line": True,
                "position_scale_to_meters": 1.0,
                "row_layout": (
                    "time_s followed by, for each spacecraft, "
                    "r_BN_N[3], v_BN_N[3], sigma_BN[3], omega_BN_B[3]"
                ),
                "recording_api": (
                    "VizInterface.saveFile=True and "
                    "VizInterface.protoFilename=<output.bin>"
                ),
            }
            if native_contract_discoverable
            else None
        ),
        "errors": errors,
        "environment": {
            "python_executable": os.sys.executable,
        },
        "notes": (
            "Local, non-executing probe of Basilisk dataFileToViz and "
            "vizInterface APIs. No Vizard process or socket is opened."
        ),
    }


def probe_basilisk_vizard_api(
    out_dir: str | Path,
    *,
    require_basilisk: bool = False,
) -> Path:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / BASILISK_API_PROBE_FILENAME
    payload = _probe_payload(require_basilisk)

    with tempfile.TemporaryDirectory(
        prefix=".basilisk_probe_",
        dir=output_dir,
    ) as tmp:
        staged = Path(tmp) / BASILISK_API_PROBE_FILENAME
        staged.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staged.replace(output_path)

    if require_basilisk and not payload["basilisk_available"]:
        raise RuntimeError(
            "Basilisk is unavailable; see probe artifact for import errors: "
            f"{output_path}"
        )
    return output_path


def load_probe(path: str | Path) -> dict[str, Any]:
    probe_path = Path(path).expanduser().resolve()
    if not probe_path.exists():
        raise FileNotFoundError(f"Basilisk API probe not found: {probe_path}")
    value = json.loads(probe_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{probe_path} must contain a JSON object")
    if value.get("schema_version") != "basilisk_api_probe_v1":
        raise ValueError(
            f"{probe_path} has invalid schema_version="
            f"{value.get('schema_version')!r}"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe locally installed Basilisk/Vizard APIs without launching Vizard."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--require-basilisk", action="store_true")
    args = parser.parse_args(argv)

    path = probe_basilisk_vizard_api(
        args.out_dir,
        require_basilisk=args.require_basilisk,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
