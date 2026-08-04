"""Preset catalog — a tracked allowlist, never a filesystem browser.

The GUI must not become a way to read arbitrary repository files or to feed
arbitrary paths into the runner. So a preset is identified by an opaque
``preset_id`` derived from its tracked relative path, and the catalog only
ever contains files that Git tracks under the approved config root. A path
that is untracked, outside the root, absolute, traversing, or a symlink
escaping the root is not a preset and cannot be addressed at all.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from ..paths import repo_root

#: The only directory presets may come from.
PRESET_ROOT = "bench/configs"

#: Parser limits. A preset is a small declarative document; anything wildly
#: outside these bounds is either a mistake or an attempt to exhaust the
#: parser, and neither should reach the resolver.
MAX_PRESET_BYTES = 512 * 1024
MAX_PRESET_DEPTH = 30
MAX_PRESET_NODES = 20_000

CATALOG_SCHEMA_VERSION = 1


class PresetError(RuntimeError):
    """A preset could not be resolved, parsed, or trusted."""


class PresetNotFound(PresetError):
    pass


class PresetUnsafe(PresetError):
    """The requested preset is outside the tracked allowlist."""


class LaunchSupport:
    """Why a preset can or cannot be launched from the GUI."""

    SUPPORTED = "SUPPORTED"
    MODEL_NOT_LAUNCHABLE = "MODEL_NOT_LAUNCHABLE"
    NO_MODELS = "NO_MODELS"
    NO_TASKS = "NO_TASKS"
    UNPARSEABLE = "UNPARSEABLE"
    SCHEMA_INCOMPATIBLE = "SCHEMA_INCOMPATIBLE"


_SUPPORT_MESSAGES = {
    LaunchSupport.SUPPORTED: "Launchable from the GUI.",
    LaunchSupport.MODEL_NOT_LAUNCHABLE: (
        "No model in this preset has a GUI-launch-certified adapter. Certified "
        "models are kalmannet_tsp and split_knet, plus the model-based KF "
        "baselines; Adaptive/MAML/ME-Split are deferred."
    ),
    LaunchSupport.NO_MODELS: "This preset declares no models.",
    LaunchSupport.NO_TASKS: "This preset declares no tasks.",
    LaunchSupport.UNPARSEABLE: "This preset could not be safely parsed.",
    LaunchSupport.SCHEMA_INCOMPATIBLE: "This preset is not compatible with the config schema.",
}


def preset_id_for(relative_path: str) -> str:
    """Stable opaque id for a tracked path.

    Deliberately *not* the path itself: a path in a request body invites
    traversal attempts and makes the id churn when files move. The mapping
    back to a path is resolved only against the tracked allowlist.
    """
    digest = hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()
    stem = Path(relative_path).stem.replace(".", "_")
    return f"{stem}.{digest[:12]}"


@dataclass(frozen=True)
class PresetEntry:
    preset_id: str
    display_name: str
    relative_path: str
    content_digest: str
    size_bytes: int
    suite_name: Optional[str] = None
    suite_version: Optional[str] = None
    task_ids: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    launchable_model_ids: tuple[str, ...] = ()
    launch_support: str = LaunchSupport.SUPPORTED
    unsupported_reason: Optional[str] = None
    schema_compatible: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "display_name": self.display_name,
            "relative_path": self.relative_path,
            "content_digest": self.content_digest,
            "size_bytes": self.size_bytes,
            "suite_name": self.suite_name,
            "suite_version": self.suite_version,
            "task_ids": list(self.task_ids),
            "model_ids": list(self.model_ids),
            "launchable_model_ids": list(self.launchable_model_ids),
            "launch_support": self.launch_support,
            "launch_supported": self.launch_support == LaunchSupport.SUPPORTED,
            "unsupported_reason": self.unsupported_reason,
            "schema_compatible": self.schema_compatible,
        }


def _tracked_preset_paths(root: Optional[Path] = None) -> list[str]:
    """Paths Git tracks under the preset root. The allowlist, verbatim."""
    base = Path(root) if root is not None else repo_root()
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", PRESET_ROOT],
            cwd=str(base), capture_output=True, text=True, timeout=30, check=False,
        )
    except Exception:  # pragma: no cover - git unavailable
        return []
    if completed.returncode != 0:
        return []
    return sorted(
        line.strip() for line in completed.stdout.splitlines()
        if line.strip().endswith((".yaml", ".yml"))
    )


def safe_load_preset_text(text: str) -> Mapping[str, Any]:
    """Parse a preset document defensively.

    ``yaml.safe_load`` refuses custom tags and arbitrary object construction.
    Size, depth and node budgets bound the rest: a YAML bomb should fail as a
    validation error, not as an OOM inside the API process.
    """
    if len(text.encode("utf-8")) > MAX_PRESET_BYTES:
        raise PresetUnsafe(
            f"preset exceeds {MAX_PRESET_BYTES} bytes; refusing to parse")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PresetError(f"invalid YAML: {_yaml_message(exc)}") from exc
    if document is None:
        raise PresetError("preset is empty")
    if not isinstance(document, Mapping):
        raise PresetError(f"preset must be a mapping, got {type(document).__name__}")
    _assert_bounded(document)
    return document


def _assert_bounded(node: Any, depth: int = 0, counter: Optional[list[int]] = None) -> None:
    counter = counter if counter is not None else [0]
    if depth > MAX_PRESET_DEPTH:
        raise PresetUnsafe(f"preset nesting exceeds depth {MAX_PRESET_DEPTH}")
    counter[0] += 1
    if counter[0] > MAX_PRESET_NODES:
        # Catches alias expansion bombs, which are small on disk but explode
        # into millions of nodes once resolved.
        raise PresetUnsafe(f"preset expands to more than {MAX_PRESET_NODES} nodes")
    if isinstance(node, Mapping):
        for value in node.values():
            _assert_bounded(value, depth + 1, counter)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _assert_bounded(value, depth + 1, counter)


def _yaml_message(exc: yaml.YAMLError) -> str:
    """YAML error with line/column, and without leaking absolute paths."""
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or str(exc)
    if mark is not None:
        return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"
    return str(problem)


def _launchable_models(model_ids: Iterable[str]) -> list[str]:
    """Models this build will launch from the GUI.

    Derived from the capability registry and the training-path contract, never
    from a hard-coded name list.
    """
    from ..capabilities import capabilities_for
    from ..training_path import RESUMABLE_MODEL_IDS

    launchable: list[str] = []
    for model_id in model_ids:
        if model_id in RESUMABLE_MODEL_IDS:
            launchable.append(model_id)
            continue
        try:
            capability = capabilities_for(model_id)
        except Exception:
            continue
        # Model-based baselines have no learning lifecycle; they launch and
        # complete, with no Stop/Resume offered.
        if not getattr(capability, "trainable", True):
            launchable.append(model_id)
    return launchable


def _summarize(relative_path: str, text: str) -> PresetEntry:
    display = Path(relative_path).stem.replace("_", " ")
    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    base = dict(
        preset_id=preset_id_for(relative_path), display_name=display,
        relative_path=relative_path, content_digest=digest,
        size_bytes=len(text.encode("utf-8")),
    )
    try:
        document = safe_load_preset_text(text)
    except PresetError as exc:
        return PresetEntry(**base, launch_support=LaunchSupport.UNPARSEABLE,
                           unsupported_reason=str(exc), schema_compatible=False)

    suite = document.get("suite") or {}
    tasks = [str(t.get("task_id")) for t in (document.get("tasks") or [])
             if isinstance(t, Mapping) and t.get("task_id")]
    models = [str(m.get("model_id")) for m in (document.get("models") or [])
              if isinstance(m, Mapping) and m.get("model_id")]
    launchable = _launchable_models(models)

    if not tasks:
        support, reason = LaunchSupport.NO_TASKS, _SUPPORT_MESSAGES[LaunchSupport.NO_TASKS]
    elif not models:
        support, reason = LaunchSupport.NO_MODELS, _SUPPORT_MESSAGES[LaunchSupport.NO_MODELS]
    elif not launchable:
        support = LaunchSupport.MODEL_NOT_LAUNCHABLE
        reason = _SUPPORT_MESSAGES[support]
    else:
        support, reason = LaunchSupport.SUPPORTED, None

    return PresetEntry(
        **base,
        suite_name=str(suite.get("name")) if isinstance(suite, Mapping) else None,
        suite_version=str(suite.get("version")) if isinstance(suite, Mapping) else None,
        task_ids=tuple(tasks), model_ids=tuple(models),
        launchable_model_ids=tuple(launchable),
        launch_support=support, unsupported_reason=reason,
    )


class PresetCatalog:
    """Read-only view over the tracked preset allowlist."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else repo_root()

    def _read(self, relative_path: str) -> str:
        path = (self.root / relative_path).resolve()
        allowed_root = (self.root / PRESET_ROOT).resolve()
        # Belt and braces: the path came from `git ls-files`, but resolve()
        # also collapses any symlink, so an escaping link is caught here.
        try:
            path.relative_to(allowed_root)
        except ValueError as exc:
            raise PresetUnsafe(
                f"preset path escapes {PRESET_ROOT}; refusing to read") from exc
        if not path.is_file():
            raise PresetNotFound(f"preset file is missing: {relative_path}")
        return path.read_text(encoding="utf-8")

    def list(self) -> list[PresetEntry]:
        entries: list[PresetEntry] = []
        for relative_path in _tracked_preset_paths(self.root):
            try:
                text = self._read(relative_path)
            except PresetError:
                continue
            entries.append(_summarize(relative_path, text))
        return entries

    def resolve_id(self, preset_id: str) -> str:
        """Map an opaque id back to a tracked path, or refuse.

        An id that does not correspond to a tracked file is simply unknown —
        there is no path to construct from user input.
        """
        for relative_path in _tracked_preset_paths(self.root):
            if preset_id_for(relative_path) == preset_id:
                return relative_path
        raise PresetNotFound(f"unknown preset_id {preset_id!r}")

    def get(self, preset_id: str) -> tuple[PresetEntry, str]:
        relative_path = self.resolve_id(preset_id)
        text = self._read(relative_path)
        return _summarize(relative_path, text), text

    def digest_of(self, preset_id: str) -> str:
        return self.get(preset_id)[0].content_digest
