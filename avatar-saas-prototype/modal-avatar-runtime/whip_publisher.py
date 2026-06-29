"""WHIP publisher for Cloudflare Stream WebRTC proof.

Subclasses GstWebRtcPublisher to replace the Worker-brokered SDP exchange with
standard WHIP (WebRTC-HTTP Ingestion Protocol):

  1. POST offer SDP (application/sdp) to CF Stream WHIP URL
  2. Parse 201 response: body = answer SDP, Location header = resource URL
  3. Set remote description from answer
  4. On stop(): DELETE the WHIP resource URL for cleanup

The encode pipeline, frame source, pre-roll, fail-fast, ICE handling, and push
thread are all inherited unchanged from GstWebRtcPublisher.

Security:
  - CF_STREAM_WHIP_URL is read from env (Modal secret "cf-stream-whip").
  - The WHIP URL and resource URL are NEVER logged or sent to PostHog/browser.
  - Error messages reference status codes only, never the URL.

TURN relay:
  - WHIP/WHIP MUST use TURN on first attempt (same policy as SFU/direct WebRTC).
  - ice_servers from the Worker's TURN credential generation are passed through
    to GstWebRtcPublisher, which configures webrtcbin add-turn-server.

H264 profile: constrained-baseline, level 3.1, profile-level-id=42e01f
  (inherited from GstWebRtcPublisher pipeline — matches CF Stream requirements).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import urljoin

import httpx

from gst_webrtc_publisher import (
    GstWebRtcPublisher,
    BITRATE_DEFAULT,
    _ENCODER_INFO,
)

logger = logging.getLogger("whip-pub")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[whip-pub] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def _sanitize_answer_sdp(sdp: str) -> dict[str, Any]:
    """Extract safe diagnostic counts from the WHIP answer SDP.

    Never logs raw SDP, IPs, or candidate details. Only returns:
      - remote_candidate_count: number of a=candidate lines
      - remote_candidate_types: set of types (host, srflx, relay, etc.)
      - remote_candidate_protocols: set of protocols (udp, tcp)
      - has_candidates: bool
    """
    candidates = re.findall(r"^a=candidate:(.+)$", sdp, re.MULTILINE)
    types: set[str] = set()
    protocols: set[str] = set()
    for c in candidates:
        parts = c.split()
        if len(parts) >= 4:
            protocols.add(parts[2].lower())
        for i, token in enumerate(parts):
            if token == "typ" and i + 1 < len(parts):
                types.add(parts[i + 1])
                break
    return {
        "remote_candidate_count": len(candidates),
        "remote_candidate_types": sorted(types),
        "remote_candidate_protocols": sorted(protocols),
        "has_candidates": len(candidates) > 0,
    }


class WhipPublisher(GstWebRtcPublisher):
    """WHIP publisher — GStreamer webrtcbin + CF Stream WHIP signaling.

    Inherits the full GStreamer/NVENC pipeline, frame push thread, pre-roll,
    fail-fast, ICE handling, and metrics from GstWebRtcPublisher.

    Overrides only the SDP exchange:
      - _post_offer: POST application/sdp to WHIP URL, parse 201 + Location
      - _call_close_callback: DELETE the WHIP resource URL
    """

    def __init__(
        self,
        frame_source: Any,
        track_name: str,
        *,
        bitrate: int = BITRATE_DEFAULT,
        ice_servers: list[dict[str, Any]] | None = None,
        encoder: str = "nvh264enc",
    ) -> None:
        whip_url = os.environ.get("CF_STREAM_WHIP_URL", "").strip()
        if not whip_url:
            raise RuntimeError(
                "CF_STREAM_WHIP_URL not set — create Modal secret 'cf-stream-whip'"
            )

        # Verify env var presence without printing the value.
        # Only log length and first-protocol for debugging.
        _proto = whip_url.split("://")[0] if "://" in whip_url else "?"
        logger.info(
            "CF_STREAM_WHIP_URL present (len=%d, proto=%s)",
            len(whip_url),
            _proto,
        )

        # TURN relay is mandatory — same policy as SFU/direct WebRTC.
        # The Worker generates TURN credentials and passes ice_servers in the
        # Modal start body. If empty/missing, ICE will likely fail.
        effective_ice_servers = list(ice_servers) if ice_servers else []
        if not effective_ice_servers:
            logger.warning(
                "WhipPublisher created with empty ice_servers — "
                "ICE will likely fail without TURN relay"
            )
        else:
            logger.info(
                "WhipPublisher created with %d ice_servers (TURN relay configured)",
                len(effective_ice_servers),
            )

        # Initialize parent with placeholder callback URLs.
        # These are never used because we override _post_offer and
        # _call_close_callback.
        super().__init__(
            frame_source=frame_source,
            track_name=track_name,
            publisher_callback_url="",  # overridden
            close_callback_url="",     # overridden
            bitrate=bitrate,
            ice_servers=effective_ice_servers,
            encoder=encoder,
        )

        self._whip_url = whip_url
        self._whip_resource_url: str | None = None
        self._whip_answer_diagnostics: dict[str, Any] | None = None

    def _verify_env_visible(self) -> bool:
        """Prove CF_STREAM_WHIP_URL is visible without printing the value."""
        val = os.environ.get("CF_STREAM_WHIP_URL", "")
        return bool(val.strip())

    async def _post_offer(self, sdp: str, mid: str) -> dict[str, Any]:
        """Override: POST offer SDP to CF Stream WHIP endpoint.

        Standard WHIP protocol:
          - Content-Type: application/sdp
          - Body: raw SDP text
          - Expects: 201 Created with Location header + answer SDP body
        """
        logger.info(
            "WHIP POST: offer len=%d mid=%s (content-type=application/sdp)",
            len(sdp),
            mid,
        )

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._whip_url,
                headers={"Content-Type": "application/sdp"},
                content=sdp.encode("utf-8"),
            )

        if resp.status_code != 201:
            # Never log the URL. Only the status code and truncated body.
            body_preview = resp.text[:300] if resp.text else "(empty)"
            logger.error("WHIP POST rejected: status=%d body_preview=%s", resp.status_code, body_preview)
            raise RuntimeError(
                f"WHIP POST failed: status={resp.status_code}"
            )

        # Extract resource URL from Location header (for DELETE cleanup).
        # CF Stream may return a RELATIVE URL (e.g., "/resource/abc").
        # Resolve it against the WHIP URL base.
        location = resp.headers.get("location") or resp.headers.get("Location")
        if location:
            if not location.startswith(("http://", "https://")):
                location = urljoin(self._whip_url, location)
                logger.info("WHIP resource Location was relative — resolved to absolute")
            self._whip_resource_url = location
            logger.info("WHIP resource created (Location header present, len=%d)", len(location))
        else:
            logger.warning("WHIP 201 but no Location header — DELETE cleanup will be skipped")

        answer_sdp = resp.text
        if not answer_sdp or "v=0" not in answer_sdp:
            logger.error("WHIP response body does not look like SDP (len=%d)", len(answer_sdp))
            raise RuntimeError("WHIP response is not valid SDP")

        # Extract safe diagnostics from the answer SDP (no raw SDP, no IPs)
        self._whip_answer_diagnostics = _sanitize_answer_sdp(answer_sdp)
        logger.info(
            "WHIP answer received (len=%d) candidates=%d types=%s protocols=%s",
            len(answer_sdp),
            self._whip_answer_diagnostics["remote_candidate_count"],
            self._whip_answer_diagnostics["remote_candidate_types"],
            self._whip_answer_diagnostics["remote_candidate_protocols"],
        )
        return {"answer_sdp": answer_sdp, "answer_type": "answer"}

    async def _call_close_callback(self) -> None:
        """Override: DELETE the WHIP resource URL for cleanup.

        Sends a bounded HTTP DELETE to the resource URL from the Location
        header. Best-effort — errors are logged but do not block teardown.
        """
        if not self._whip_resource_url:
            logger.info("WHIP cleanup: no resource URL stored — skipping DELETE")
            return

        logger.info("WHIP cleanup: sending DELETE (resource len=%d)", len(self._whip_resource_url))
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.delete(self._whip_resource_url)
                if resp.status_code in (200, 204):
                    logger.info("WHIP resource deleted successfully (status=%d)", resp.status_code)
                else:
                    logger.warning(
                        "WHIP DELETE returned status=%d (not 200/204)",
                        resp.status_code,
                    )
        except Exception as exc:
            logger.warning("WHIP DELETE failed: %s", exc)

    def get_metrics(self) -> dict[str, Any]:
        """Return publisher metrics including WHIP-specific fields."""
        base = {
            "frames_generated": self._metrics.frames_generated,
            "frames_emitted": self._metrics.frames_emitted,
            "underflow_count": self._metrics.underflow_count,
            "fail_fast_reason": self._metrics.fail_fast_reason,
            "pipeline_error": self._metrics.pipeline_error,
            "whip_resource_created": self._whip_resource_url is not None,
            "whip_answer_diagnostics": self._whip_answer_diagnostics,
            "encoder": dict(_ENCODER_INFO, bitrate=self._bitrate, whip=True),
        }
        return base
