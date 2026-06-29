"""HMAC request signing verification for Worker -> Modal internal endpoints.

Signature base string:
  METHOD + "\\n" + URL + "\\n" + TIMESTAMP + "\\n" + REQUEST_ID + "\\n" + BODY_SHA256

Default mode is observe-only: signatures are verified and logged, but requests
are not rejected. Set AVTR_ENFORCE_MODAL_REQUEST_SIGNING=true to reject
invalid/missing signatures.

OBSERVE-MODE VALIDATION REQUIREMENT (full-URL signing):
  Before enabling enforce mode, deployed Modal logs MUST show
  "[modal-auth] valid" for every active endpoint. If URL canonicalization
  causes a mismatch (Worker signs env-var URL without trailing slash; Modal
  sees str(request.url) with trailing slash), all requests will log as
  invalid with reason=signature_mismatch. In that case, fall back to signing
  method + path + bodyHash, or pass a canonical URL via a header.

Never log the secret, signature, raw request body, or audio payloads.
"""

import hashlib
import hmac
import os
import re
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request

SIGNING_SECRET_ENV = "AVTR_MODAL_REQUEST_SIGNING_SECRET"
ENFORCE_ENV = "AVTR_ENFORCE_MODAL_REQUEST_SIGNING"
ALLOWED_SKEW_SECONDS = 300

# Safe characters for request-id logging (UUID v4 is hex + dashes).
_SAFE_REQUEST_ID_RE = re.compile(r"^[0-9a-fA-F-]{1,64}$")


class SignatureError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _sanitize_request_id(raw: str) -> str:
    """Return the request ID if it contains only safe characters, else 'invalid'.

    Prevents log injection from user-controlled headers.
    """
    if raw and _SAFE_REQUEST_ID_RE.match(raw):
        return raw[:64]
    return "invalid"


def _sanitize_endpoint(hostname: str) -> str:
    """Extract endpoint name from Modal function hostname for logging.

    Host looks like 'vallangiprasad--start-session-5q.modal.run'.
    """
    if not hostname:
        return "unknown"
    hostname = hostname.strip().lower()
    if "--" not in hostname:
        return "unknown"
    # vallangiprasad--start-session-5q.modal.run -> start-session-5q
    parts = hostname.split("--", 1)
    if len(parts) < 2:
        return "unknown"
    endpoint_part = parts[1].split(".")[0]
    if re.match(r"^[a-z0-9_-]{1,64}$", endpoint_part):
        return endpoint_part
    return "unknown"


def sign_request(
    secret: str,
    method: str,
    url: str,
    body: bytes,
    request_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> dict:
    """Generate signing headers. Exposed for tests and Worker-side symmetry."""
    req_id = request_id or str(uuid.uuid4())
    ts = timestamp if timestamp is not None else int(time.time())
    body_hash = hashlib.sha256(body).hexdigest()
    base = f"{method.upper()}\n{url}\n{ts}\n{req_id}\n{body_hash}"
    signature = hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return {
        "X-AVTR-Timestamp": str(ts),
        "X-AVTR-Request-Id": req_id,
        "X-AVTR-Body-SHA256": body_hash,
        "X-AVTR-Signature": signature,
    }


async def verify_modal_request(request: Request) -> None:
    """Verify request signature. In observe mode, logs but never rejects.

    Raises HTTPException(401) only when AVTR_ENFORCE_MODAL_REQUEST_SIGNING=true.
    """
    secret = os.environ.get(SIGNING_SECRET_ENV)
    enforce = os.environ.get(ENFORCE_ENV) == "true"

    if not secret:
        if enforce:
            raise HTTPException(
                status_code=401,
                detail="modal_request_signing_secret_not_configured",
            )
        return

    request_id = _sanitize_request_id(request.headers.get("X-AVTR-Request-Id", ""))
    endpoint = _sanitize_endpoint(request.url.hostname or "")

    try:
        await _verify(secret, request)
        print(f"[modal-auth] valid request_id={request_id} endpoint={endpoint}", flush=True)
    except SignatureError as exc:
        reason = exc.reason
        if enforce:
            raise HTTPException(
                status_code=401,
                detail=f"invalid_signature: {reason}",
            )
        print(f"[modal-auth] invalid observe request_id={request_id} endpoint={endpoint} reason={reason}", flush=True)


async def _verify(secret: str, request: Request) -> None:
    timestamp_hdr = request.headers.get("X-AVTR-Timestamp")
    request_id_hdr = request.headers.get("X-AVTR-Request-Id")
    body_hash_hdr = request.headers.get("X-AVTR-Body-SHA256")
    signature_hdr = request.headers.get("X-AVTR-Signature")

    if not timestamp_hdr or not request_id_hdr or not body_hash_hdr or not signature_hdr:
        raise SignatureError("missing_headers")

    try:
        ts = int(timestamp_hdr)
    except ValueError:
        raise SignatureError("invalid_timestamp")

    now = int(time.time())
    if abs(now - ts) > ALLOWED_SKEW_SECONDS:
        raise SignatureError("stale_timestamp")

    try:
        uuid.UUID(request_id_hdr, version=4)
    except ValueError:
        raise SignatureError("invalid_request_id")

    body = await request.body()
    expected_body_hash = hashlib.sha256(body).hexdigest()
    if not hmac.compare_digest(expected_body_hash.encode(), body_hash_hdr.encode()):
        raise SignatureError("body_hash_mismatch")

    base = f"{request.method.upper()}\n{str(request.url)}\n{timestamp_hdr}\n{request_id_hdr}\n{body_hash_hdr}"
    expected_signature = hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_signature.encode(), signature_hdr.encode()):
        raise SignatureError("signature_mismatch")
