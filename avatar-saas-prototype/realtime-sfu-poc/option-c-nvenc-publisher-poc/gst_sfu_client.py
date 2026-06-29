"""Minimal Cloudflare Realtime SFU HTTPS API client for Option C POC.

Mirrors the pattern used in phase-b/publisher.py and production
sfu-client.ts, but in pure Python so the GStreamer publisher can call it
directly without a Worker proxy.
"""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"


class SfuClientError(Exception):
    def __init__(self, message: str, status: int, body: str):
        super().__init__(message)
        self.status = status
        self.body = body


class SfuClient:
    def __init__(self, session: aiohttp.ClientSession, app_id: str, app_secret: str):
        self.session = session
        self.app_id = app_id
        self.base = f"{SFU_API_BASE}/{app_id}"
        self.headers = {
            "Authorization": f"Bearer {app_secret}",
            "Content-Type": "application/json",
        }

    async def new_session(self) -> str:
        async with self.session.post(
            f"{self.base}/sessions/new", headers=self.headers
        ) as resp:
            text = await resp.text()
            if resp.status not in (200, 201):
                raise SfuClientError(f"sessions/new failed", resp.status, text)
            data = json.loads(text)
            return data["sessionId"]

    async def new_tracks(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        async with self.session.post(
            f"{self.base}/sessions/{session_id}/tracks/new",
            headers=self.headers,
            json=body,
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise SfuClientError(f"tracks/new failed", resp.status, text)
            return json.loads(text)

    async def close_tracks(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        async with self.session.put(
            f"{self.base}/sessions/{session_id}/tracks/close",
            headers=self.headers,
            json=body,
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise SfuClientError(f"tracks/close failed", resp.status, text)
            return json.loads(text)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        async with self.session.get(
            f"{self.base}/sessions/{session_id}", headers=self.headers
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise SfuClientError(f"get session failed", resp.status, text)
            return json.loads(text)


def require_sfu_creds() -> tuple[str, str]:
    app_id = os.environ.get("CALLS_APP_ID", "")
    app_secret = os.environ.get("CALLS_APP_SECRET", "")
    if not app_id or not app_secret:
        raise RuntimeError("CALLS_APP_ID and CALLS_APP_SECRET must be set")
    return app_id, app_secret
