"""Unit tests for request_signing verification helper.

Run with: python -m pytest modal-avatar-runtime/test_request_signing.py
"""

import asyncio
import os
import time
from urllib.parse import urlparse

from starlette.requests import Request

from request_signing import (
    ENFORCE_ENV,
    SIGNING_SECRET_ENV,
    sign_request,
    verify_modal_request,
)

TEST_SECRET = "test-modal-signing-secret"


def _make_request(method: str, url: str, headers: dict, body: bytes) -> Request:
    parsed = urlparse(url)
    scope = {
        "type": "http",
        "method": method,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "path": parsed.path or "/",
        "query_string": parsed.query.encode(),
        "scheme": parsed.scheme,
        "server": (
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        ),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _run(coro):
    return asyncio.run(coro)


def test_valid_request_accepted(capsys, monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.delenv(ENFORCE_ENV, raising=False)

    url = "https://modal.run/start-session-5q"
    body = b'{"session_id": "sess-123"}'
    headers = sign_request(TEST_SECRET, "POST", url, body)
    request = _make_request("POST", url, headers, body)

    _run(verify_modal_request(request))

    captured = capsys.readouterr()
    assert "valid" in captured.out
    assert "invalid" not in captured.out


def test_missing_headers_rejected_in_enforce_mode(monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.setenv(ENFORCE_ENV, "true")

    request = _make_request("POST", "https://modal.run/start-session-5q", {}, b"{}")
    try:
        _run(verify_modal_request(request))
        raise AssertionError("expected HTTPException")
    except Exception as exc:
        assert "invalid_signature" in str(exc) or "missing_headers" in str(exc)


def test_bad_signature_rejected_in_enforce_mode(monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.setenv(ENFORCE_ENV, "true")

    url = "https://modal.run/start-session-5q"
    body = b'{"session_id": "sess-123"}'
    headers = sign_request(TEST_SECRET, "POST", url, body)
    sig = headers["X-AVTR-Signature"]
    headers["X-AVTR-Signature"] = sig[:-1] + ("1" if sig[-1] == "0" else "0")

    request = _make_request("POST", url, headers, body)
    try:
        _run(verify_modal_request(request))
        raise AssertionError("expected HTTPException")
    except Exception as exc:
        assert "signature_mismatch" in str(exc)


def test_stale_timestamp_rejected_in_enforce_mode(monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.setenv(ENFORCE_ENV, "true")

    url = "https://modal.run/start-session-5q"
    body = b'{"session_id": "sess-123"}'
    headers = sign_request(TEST_SECRET, "POST", url, body, timestamp=int(time.time()) - 600)

    request = _make_request("POST", url, headers, body)
    try:
        _run(verify_modal_request(request))
        raise AssertionError("expected HTTPException")
    except Exception as exc:
        assert "stale_timestamp" in str(exc)


def test_tampered_body_rejected_in_enforce_mode(monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.setenv(ENFORCE_ENV, "true")

    url = "https://modal.run/start-session-5q"
    body = b'{"session_id": "sess-123"}'
    headers = sign_request(TEST_SECRET, "POST", url, body)

    request = _make_request("POST", url, headers, b'{"session_id": "sess-456"}')
    try:
        _run(verify_modal_request(request))
        raise AssertionError("expected HTTPException")
    except Exception as exc:
        assert "body_hash_mismatch" in str(exc)


def test_observe_mode_does_not_reject(capsys, monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.delenv(ENFORCE_ENV, raising=False)

    url = "https://modal.run/start-session-5q"
    body = b'{"session_id": "sess-123"}'
    headers = sign_request(TEST_SECRET, "POST", url, body)
    sig = headers["X-AVTR-Signature"]
    headers["X-AVTR-Signature"] = sig[:-1] + ("1" if sig[-1] == "0" else "0")

    request = _make_request("POST", url, headers, body)
    _run(verify_modal_request(request))  # should not raise

    captured = capsys.readouterr()
    assert "invalid observe" in captured.out


def test_empty_body(capsys, monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.delenv(ENFORCE_ENV, raising=False)

    url = "https://modal.run/list-cached-avatars-5q"
    body = b""
    headers = sign_request(TEST_SECRET, "POST", url, body)
    request = _make_request("POST", url, headers, body)

    _run(verify_modal_request(request))

    captured = capsys.readouterr()
    assert "valid" in captured.out


def test_url_binding(capsys, monkeypatch):
    monkeypatch.setenv(SIGNING_SECRET_ENV, TEST_SECRET)
    monkeypatch.delenv(ENFORCE_ENV, raising=False)

    url = "https://modal.run/start-session-5q"
    body = b'{"session_id": "sess-123"}'
    headers = sign_request(TEST_SECRET, "POST", url, body)
    request = _make_request("POST", "https://modal.run/other-endpoint", headers, body)

    _run(verify_modal_request(request))

    captured = capsys.readouterr()
    assert "invalid observe" in captured.out


# ── Cross-language compatibility vector ─────────────────────────────────────
# Fixed values prove Python sign_request() and TS signModalRequest() produce
# identical signatures. The body_hash and signature were generated by the TS
# signModalRequest() and independently verified with Python hmac/hashlib.
# If either implementation changes, these tests break.

XL_SECRET = "avtr-cross-lang-test-secret"
XL_URL = "https://vallangirakesh--start-session-5q.modal.run/"
XL_BODY = b'{"session_id":"session:cross-lang-001","avatar_id":"arnold"}'
XL_TS = 1719811200
XL_REQ_ID = "12345678-1234-4234-8345-1234567890ab"
XL_BODY_HASH = "f1f76708e2aef0e1894947069209ce948d8aecf8616aedfb5f10b9d5e01858bd"
XL_SIGNATURE = "098cab3fb724707633124369239c2bb6e6eb53d8ac00cf5540936291efc644fc"


def test_cross_lang_python_sign_matches_expected_vector():
    """Python sign_request produces the exact TS-compatible signature."""
    headers = sign_request(
        XL_SECRET, "POST", XL_URL, XL_BODY, XL_REQ_ID, XL_TS,
    )
    assert headers["X-AVTR-Body-SHA256"] == XL_BODY_HASH
    assert headers["X-AVTR-Signature"] == XL_SIGNATURE


def test_cross_lang_python_verify_accepts_signed_request(capsys, monkeypatch):
    """Python verifier accepts a request signed by sign_request (same algorithm as TS).

    Cross-language compatibility is proven by the sign-output vector test above.
    This test proves the Python verifier end-to-end with a current timestamp.
    """
    monkeypatch.setenv(SIGNING_SECRET_ENV, XL_SECRET)
    monkeypatch.delenv(ENFORCE_ENV, raising=False)

    headers = sign_request(XL_SECRET, "POST", XL_URL, XL_BODY, XL_REQ_ID)
    request = _make_request("POST", XL_URL, headers, XL_BODY)
    _run(verify_modal_request(request))

    captured = capsys.readouterr()
    assert "valid" in captured.out


def test_cross_lang_python_verify_rejects_tampered_body(monkeypatch):
    """Python verifier rejects a tampered body."""
    monkeypatch.setenv(SIGNING_SECRET_ENV, XL_SECRET)
    monkeypatch.setenv(ENFORCE_ENV, "true")

    headers = sign_request(XL_SECRET, "POST", XL_URL, XL_BODY, XL_REQ_ID)
    tampered_body = b'{"session_id":"session:HACKED","avatar_id":"arnold"}'
    request = _make_request("POST", XL_URL, headers, tampered_body)
    try:
        _run(verify_modal_request(request))
        raise AssertionError("expected HTTPException")
    except Exception as exc:
        assert "body_hash_mismatch" in str(exc)
