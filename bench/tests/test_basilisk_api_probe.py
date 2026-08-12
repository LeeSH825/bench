from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from bench.visualization.basilisk_api_probe import (
    load_probe,
    probe_basilisk_vizard_api,
)
from bench.visualization.vizard_basilisk_wrapper import (
    detect_basilisk_available,
)


class BasiliskAPIProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_probe_without_requirement_writes_schema(self) -> None:
        path = probe_basilisk_vizard_api(self.root)
        self.assertTrue(path.exists())
        probe = load_probe(path)
        self.assertEqual(probe["schema_version"], "basilisk_api_probe_v1")
        self.assertIn("basilisk_available", probe)
        self.assertIn("import_results", probe)

    def test_probe_strict_matches_environment(self) -> None:
        if detect_basilisk_available():
            path = probe_basilisk_vizard_api(
                self.root,
                require_basilisk=True,
            )
            self.assertTrue(path.exists())
        else:
            with self.assertRaises(RuntimeError):
                probe_basilisk_vizard_api(
                    self.root,
                    require_basilisk=True,
                )

    def test_load_probe_returns_dict(self) -> None:
        path = probe_basilisk_vizard_api(self.root)
        self.assertIsInstance(load_probe(path), dict)


@dataclass
class BasiliskAPIProbeResult:
    ok: bool
    note: str


def run_basilisk_api_probe_tests() -> BasiliskAPIProbeResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        BasiliskAPIProbeTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return BasiliskAPIProbeResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Basilisk API probe tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
