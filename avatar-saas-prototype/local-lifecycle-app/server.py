"""
Local lifecycle server for AVTR-1 Phase 4A.

Zero-dependency stdlib HTTP server that proxies browser requests to the
deployed Modal AvatarLifecycleL4 runtime.

Usage:
    python3 avatar-saas-prototype/local-lifecycle-app/server.py

Then open http://localhost:8765 in a browser.

Prerequisites:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_app.py
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

MODAL_APP_NAME = "avtr1-lifecycle"
MODAL_CLASS_NAME = "AvatarLifecycleL4"

STATIC_DIR = Path(__file__).resolve().parent / "static"
PORT = int(os.environ.get("AVTR1_LIFECYCLE_PORT", "8765"))

_modal_cls = None
_modal_instance = None
_instance_lock = threading.Lock()
current_session_id: str | None = None


def _get_modal_instance():
    global _modal_cls, _modal_instance
    with _instance_lock:
        if _modal_cls is None:
            import modal
            _modal_cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        if _modal_instance is None:
            _modal_instance = _modal_cls()
        return _modal_instance


def _reset_modal_instance():
    global _modal_instance
    with _instance_lock:
        _modal_instance = None


class LifecycleHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[server] {args[0]}")

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _send_static_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            self.send_error(404, "Not found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._send_static_file(STATIC_DIR / "index.html")
        elif self.path.startswith("/static/"):
            rel = self.path[len("/static/"):]
            self._send_static_file(STATIC_DIR / rel)
        elif self.path == "/api/health":
            self._handle_health()
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/connect":
            self._handle_connect()
        elif self.path == "/api/debug-frame":
            self._handle_debug_frame()
        elif self.path == "/api/stop":
            self._handle_stop()
        else:
            self.send_error(404, "Not found")

    def _handle_health(self) -> None:
        try:
            inst = _get_modal_instance()
            result = inst.health.remote()
            self._send_json({"ok": True, "modal": result})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, 502)

    def _handle_connect(self) -> None:
        global current_session_id
        try:
            inst = _get_modal_instance()
            result = inst.start_session.remote()
            if result.get("status") == "avatar_ready":
                current_session_id = result["sessionId"]
                print(f"[server] Session started: {current_session_id}")
            else:
                current_session_id = None
                print(f"[server] Start session failed: {result}")
            self._send_json(result)
        except Exception as exc:
            current_session_id = None
            _reset_modal_instance()
            self._send_json({"status": "error", "error": str(exc)}, 502)

    def _handle_debug_frame(self) -> None:
        try:
            inst = _get_modal_instance()
            result = inst.render_debug_frame.remote(current_session_id)
            self._send_json(result)
        except Exception as exc:
            _reset_modal_instance()
            self._send_json({"available": False, "note": str(exc)}, 502)

    def _handle_stop(self) -> None:
        global current_session_id
        try:
            inst = _get_modal_instance()
            result = inst.end_session.remote(current_session_id)
            current_session_id = None
            print("[server] Session ended")
            self._send_json(result)
        except Exception as exc:
            current_session_id = None
            _reset_modal_instance()
            self._send_json({"ok": False, "error": str(exc)}, 502)


def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), LifecycleHandler)
    print(f"[server] AVTR-1 Lifecycle server on http://localhost:{PORT}")
    print(f"[server] Static dir: {STATIC_DIR}")
    print(f"[server] Modal app: {MODAL_APP_NAME}, class: {MODAL_CLASS_NAME}")
    print("[server] Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
