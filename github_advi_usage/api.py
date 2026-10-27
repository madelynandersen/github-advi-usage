"""Minimal GitHub API client: authenticated GET requests against the GitHub
REST/Code-Search API, with retry/backoff on rate limits and transient errors.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import Settings


class ApiClient:
    def __init__(self, settings: Settings, timeout: int = 60) -> None:
        self.settings = settings
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})

    def github_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        return self._get_json(f"https://api.github.com/{path.lstrip('/')}", params=params, headers=headers)

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(6):
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(60, 2**attempt))
                continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()
        raise RuntimeError("unreachable")
