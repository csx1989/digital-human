"""Standalone render health check — calls deployed 5Q HTTP endpoints with signing.

Usage:
    modal run modal_health_check_standalone.py

This file does NOT import from the production 5Q module. It defines its own
minimal app with just httpx for making signed HTTP calls.
"""

from __future__ import annotations

import json
import time
import uuid
import hashlib
import hmac
import base64

import modal

app = modal.App("avtr1-health-check")

WORKSPACE = "kommathotichinnasubbayamma"
START_SESSION_URL = f"https://{WORKSPACE}--start-session-5q.modal.run"
PUSH_AUDIO_BATCH_URL = f"https://{WORKSPACE}--push-audio-batch-5q.modal.run"
STREAM_FMP4_AV_URL = f"https://{WORKSPACE}--stream-fmp4-av-adapter-5q.modal.run"
END_SESSION_URL = f"https://{WORKSPACE}--end-session-adapter-5q.modal.run"
SESSION_STATUS_URL = f"https://{WORKSPACE}--session-status-5q.modal.run"


def _canonicalize_url(url: str) -> str:
    """Match JS new URL(url).href — ensures trailing slash on root path."""
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    path = parts.path if parts.path else "/"
    return f"{parts.scheme}://{parts.netloc}{path}"


def _sign_request(secret: str, method: str, url: str, body: bytes) -> dict:
    req_id = str(uuid.uuid4())
    ts = int(time.time())
    body_hash = hashlib.sha256(body).hexdigest()
    canon_url = _canonicalize_url(url)
    base = f"{method.upper()}\n{canon_url}\n{ts}\n{req_id}\n{body_hash}"
    sig = hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return {
        "X-AVTR-Timestamp": str(ts),
        "X-AVTR-Request-Id": req_id,
        "X-AVTR-Body-SHA256": body_hash,
        "X-AVTR-Signature": sig,
    }


@app.function(
    image=modal.Image.debian_slim().pip_install("httpx", "numpy"),
    secrets=[modal.Secret.from_name("avtr-modal-request-signing-secret")],
    timeout=120,
)
def render_health_check_http(
    avatar_id: str = "arnold",
    num_silence_chunks: int = 3,
) -> dict:
    import os
    import httpx
    import numpy as np

    secret = os.environ["AVTR_MODAL_REQUEST_SIGNING_SECRET"]
    session_id = f"health-http-{uuid.uuid4().hex[:12]}"
    t_start = time.monotonic()

    def _signed_post(url, body, client):
        body_bytes = json.dumps(body).encode()
        headers = _sign_request(secret, "POST", url, body_bytes)
        return client.post(url, content=body_bytes, headers=headers, timeout=30.0)

    verdict = {
        "ok": False, "session_id": session_id,
        "chunks": 0, "videoBytes": 0, "audioBytes": 0,
        "warmup_ms": -1, "error": None, "engine_backends": {},
    }

    try:
        with httpx.Client() as client:
            # 1. Start session
            print(f"[health] POST start-session session={session_id}", flush=True)
            resp = _signed_post(START_SESSION_URL, {
                "session_id": session_id, "avatar_id": avatar_id,
            }, client)
            if resp.status_code != 200:
                verdict["error"] = f"start HTTP {resp.status_code}: {resp.text[:300]}"
                return verdict
            result = resp.json()
            if "error" in result:
                verdict["error"] = f"start error: {result['error']}"
                return verdict
            verdict["warmup_ms"] = result.get("warmup_ms", -1)
            print(f"[health] started warmup_ms={verdict['warmup_ms']}", flush=True)

            # 2. Push silence audio
            sr = 16000
            for i in range(num_silence_chunks):
                silence = np.zeros(int(sr * 0.2), dtype=np.float32)
                audio_b64 = base64.b64encode(silence.tobytes()).decode()
                _signed_post(PUSH_AUDIO_BATCH_URL, {
                    "session_id": session_id,
                    "chunks": [{"audio_b64": audio_b64, "sample_rate": sr}],
                }, client)
                print(f"[health] pushed silence {i+1}/{num_silence_chunks}", flush=True)

            # 3. Stream fMP4
            stream_body = json.dumps({"session_id": session_id}).encode()
            stream_headers = _sign_request(secret, "POST", STREAM_FMP4_AV_URL, stream_body)
            chunks_seen = 0
            video_bytes = 0
            found_end = False

            with client.stream("POST", STREAM_FMP4_AV_URL,
                               content=stream_body, headers=stream_headers,
                               timeout=90.0) as stream:
                if stream.status_code != 200:
                    verdict["error"] = f"stream HTTP {stream.status_code}"
                    return verdict
                buffer = ""
                for line in stream.iter_lines():
                    buffer += line + "\n"
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        if "event: fmp4_media" in event_str:
                            chunks_seen += 1
                            for l in event_str.split("\n"):
                                if l.startswith("data:"):
                                    try:
                                        d = json.loads(l.split("data:", 1)[1].strip())
                                        if d.get("type") == "fmp4_media":
                                            video_bytes += d.get("byteLength", 0)
                                    except Exception:
                                        pass
                        elif "event: end" in event_str:
                            found_end = True
                            for l in event_str.split("\n"):
                                if l.startswith("data:"):
                                    try:
                                        d = json.loads(l.split("data:", 1)[1].strip())
                                        chunks_seen = d.get("chunksRendered", chunks_seen)
                                        video_bytes = d.get("videoBytes", video_bytes)
                                    except Exception:
                                        pass
                            break
                    if chunks_seen >= 30 or found_end:
                        break

            verdict["chunks"] = chunks_seen
            verdict["videoBytes"] = video_bytes
            print(f"[health] stream done chunks={chunks_seen} videoBytes={video_bytes} end={found_end}", flush=True)

            # 4. End session
            resp = _signed_post(END_SESSION_URL, {"session_id": session_id}, client)
            if resp.status_code == 200:
                end_result = resp.json()
                verdict["chunks"] = end_result.get("chunks_rendered", chunks_seen)
                verdict["videoBytes"] = end_result.get("fmp4_media_total_bytes", video_bytes)
                print(f"[health] ended chunks={verdict['chunks']} videoBytes={verdict['videoBytes']}", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        verdict["error"] = str(e)
        return verdict

    verdict["elapsed_s"] = round(time.monotonic() - t_start, 1)
    verdict["ok"] = (verdict["chunks"] > 0 and verdict["videoBytes"] > 0
                     and verdict["warmup_ms"] > 0 and verdict["error"] is None)
    print(f"[health] VERDICT ok={verdict['ok']} chunks={verdict['chunks']} "
          f"videoBytes={verdict['videoBytes']} warmup_ms={verdict['warmup_ms']}", flush=True)
    return verdict


@app.local_entrypoint()
def main(avatar_id: str = "arnold", num_silence_chunks: int = 3):
    import json as _json
    result = render_health_check_http.remote(avatar_id=avatar_id, num_silence_chunks=num_silence_chunks)
    print(_json.dumps(result, indent=2, default=str))
