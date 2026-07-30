"""Thin HTTP client for the control-plane API.

Every request has a timeout and every failure returns a structured error dict
instead of raising. A Dash callback that raises renders a red error box and
loses the rest of the page; a callback that receives ``{"error": ...}`` can show
the problem in place and keep the rest of the dashboard usable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Optional

#: Default location of the read-only API.
DEFAULT_BASE_URL = "http://127.0.0.1:8765"

#: Request timeout. Deliberately short: the dashboard polls, so a slow response
#: is better abandoned than queued behind the next poll.
DEFAULT_TIMEOUT_SECONDS = 10.0


class ApiClient:
    """Read-only client. Exposes only GETs, because the API only has GETs."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        query = ""
        if params:
            cleaned = {k: v for k, v in params.items() if v is not None}
            if cleaned:
                query = "?" + urllib.parse.urlencode(cleaned)
        url = f"{self.base_url}{path}{query}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except Exception:
                pass
            return {"error": f"HTTP {exc.code}", "detail": detail, "url": url}
        except urllib.error.URLError as exc:
            return {
                "error": "unreachable",
                "detail": (
                    f"Could not reach the control-plane API at {self.base_url} ({exc.reason}). "
                    "Start it with: python -m bench.control.api.app"
                ),
                "url": url,
            }
        except Exception as exc:
            return {"error": type(exc).__name__, "detail": str(exc), "url": url}

    # -- typed helpers -------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self.get("/api/v1/system/health")

    def gpus(self) -> dict[str, Any]:
        return self.get("/api/v1/system/gpus")

    def workers(self) -> dict[str, Any]:
        return self.get("/api/v1/system/workers")

    def state_machine(self) -> dict[str, Any]:
        return self.get("/api/v1/system/state-machine")

    def capabilities(self) -> dict[str, Any]:
        return self.get("/api/v1/capabilities")

    def runs(self, **params: Any) -> dict[str, Any]:
        return self.get("/api/v1/runs", params)

    def run(self, run_id: str) -> dict[str, Any]:
        return self.get(f"/api/v1/runs/{urllib.parse.quote(run_id)}")

    def events(self, run_id: str, **params: Any) -> dict[str, Any]:
        return self.get(f"/api/v1/runs/{urllib.parse.quote(run_id)}/events", params)

    def metrics(self, run_id: str, **params: Any) -> dict[str, Any]:
        return self.get(f"/api/v1/runs/{urllib.parse.quote(run_id)}/metrics", params)

    def resources(self, run_id: str, **params: Any) -> dict[str, Any]:
        return self.get(f"/api/v1/runs/{urllib.parse.quote(run_id)}/resources", params)

    def artifacts(self, run_id: str) -> dict[str, Any]:
        return self.get(f"/api/v1/runs/{urllib.parse.quote(run_id)}/artifacts")

    def logs(self, run_id: str, **params: Any) -> dict[str, Any]:
        return self.get(f"/api/v1/runs/{urllib.parse.quote(run_id)}/logs", params)

    def orphan_candidates(self) -> dict[str, Any]:
        return self.get("/api/v1/orphan-candidates")


def is_error(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload, Mapping) and "error" in payload
