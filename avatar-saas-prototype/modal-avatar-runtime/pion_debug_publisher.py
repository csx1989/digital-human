"""Pion WebRTC debug publisher — B2 control test.

Uses GStreamer for H264 encoding (same NVENC, same settings), then sends RTP
via UDP to a Pion Go binary that handles the WebRTC transport. This bypasses
GStreamer's webrtcbin/libnice entirely, using Pion's independent Go WebRTC stack.

The signaling (SDP offer/answer, ICE candidate exchange) flows through the same
Worker direct-webrtc-debug path — only the Modal sender changes.

Pipeline: appsrc → queue → nvh264enc → h264parse → rtph264pay → udpsink → Pion → browser
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
from typing import Any

import httpx

from gst_webrtc_publisher import (
    FrameSource,
    WIDTH,
    HEIGHT,
    FPS,
    GOP_SIZE,
    BITRATE_DEFAULT,
    RTP_AGGREGATE_MODE,
    _black_i420_bytes,
    _GstPublisherMetrics,
    _FailFastPolicy,
    _ENCODER_INFO,
)

logger = logging.getLogger("pion-debug-pub")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[pion-pub] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

_PIPELINE_TEMPLATE = (
    "appsrc name=src is-live=true format=time do-timestamp=false ! "
    "video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 ! "
    "queue max-size-buffers=3 leaky=downstream ! "
    "nvh264enc name=nvenc gop-size={gop} preset=low-latency-hq bitrate={bitrate} "
    "rc-mode=cbr-ld-hq strict-gop=true zerolatency=true ! "
    "video/x-h264,profile=constrained-baseline,level=(string)3.1 ! "
    "h264parse name=h264p config-interval=-1 ! "
    "rtph264pay name=rtppay config-interval=-1 aggregate-mode={agg_mode} pt=96 ! "
    "udpsink name=udpsink host=127.0.0.1 port={rtp_port} sync=false async=false"
)

_PION_BINARY = "/usr/local/bin/pion-forwarder"
_PION_HTTP_PORT = "8090"
_PION_RTP_PORT = "5004"
_PION_BASE = f"http://127.0.0.1:{_PION_HTTP_PORT}"

_NAL_TYPE_NAMES = {
    1: "non-IDR", 2: "DPA", 3: "DPB", 4: "DPC", 5: "IDR",
    6: "SEI", 7: "SPS", 8: "PPS", 9: "AUD", 24: "STAP-A", 28: "FU-A",
}


class _RtpAccounting:
    """Full RTP packet accounting: seq gaps, timestamps, markers, NAL types."""

    def __init__(self):
        self.total: int = 0
        self.has_first: bool = False
        self.first_seq: int = 0
        self.last_seq: int = 0
        self.gaps: int = 0
        self.dups: int = 0
        self.late: int = 0
        self.first_ts: int = 0
        self.last_ts: int = 0
        self.markers: int = 0
        self.nal_types: dict[str, int] = {}
        self.parse_errors: int = 0

    def observe(self, data: bytes) -> None:
        if len(data) < 12:
            self.parse_errors += 1
            return
        b0 = data[0]
        b1 = data[1]
        marker = bool(b1 & 0x80)
        seq = (data[2] << 8) | data[3]
        ts = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        cc = b0 & 0x0F
        hdr_len = 12 + cc * 4
        if (b0 & 0x10) and len(data) >= hdr_len + 4:
            ext_len = (data[hdr_len + 2] << 8) | data[hdr_len + 3]
            hdr_len += 4 + 4 * ext_len
        payload = data[hdr_len:]
        nal_type = (payload[0] & 0x1F) if len(payload) > 0 else 0
        nal_name = _NAL_TYPE_NAMES.get(nal_type, f"type_{nal_type}")

        self.total += 1
        if not self.has_first:
            self.first_seq = seq
            self.last_seq = seq
            self.first_ts = ts
            self.has_first = True
        else:
            fwd = (seq - self.last_seq) & 0xFFFF
            if fwd == 0:
                self.dups += 1
            elif fwd < 32768:
                self.gaps += fwd - 1
                self.last_seq = seq
            else:
                self.late += 1
        self.last_ts = ts
        if marker:
            self.markers += 1
        self.nal_types[nal_name] = self.nal_types.get(nal_name, 0) + 1

    def snapshot(self) -> dict:
        expected = self.total + self.gaps
        return {
            "total": self.total,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "gaps": self.gaps,
            "dups": self.dups,
            "late": self.late,
            "expected": expected,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "markers": self.markers,
            "nal_types": dict(self.nal_types),
            "parse_errors": self.parse_errors,
        }


class PionDebugPublisher:
    """WebRTC publisher using Pion (Go) for transport, GStreamer for encoding."""

    def __init__(
        self,
        frame_source: FrameSource,
        track_name: str,
        publisher_callback_url: str,
        close_callback_url: str,
        *,
        bitrate: int = BITRATE_DEFAULT,
    ) -> None:
        self._source = frame_source
        self._track_name = track_name
        self._callback_url = publisher_callback_url
        self._close_url = close_callback_url
        self._bitrate = int(os.environ.get("PION_BITRATE", "2000"))
        self._base_bitrate = self._bitrate
        self._nvenc: Any = None
        self._congestion_task: asyncio.Task | None = None

        self._Gst: Any = None
        self._GLib: Any = None
        self._pipeline: Any = None
        self._appsrc: Any = None

        self._glib_loop: Any = None
        self._glib_thread: threading.Thread | None = None
        self._push_thread: threading.Thread | None = None
        self._stop_push = threading.Event()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ice_connected = asyncio.Event()
        self._black_frame: bytes = _black_i420_bytes()
        self._push_counter: int = 0

        self._metrics = _GstPublisherMetrics()
        self._fail_fast = _FailFastPolicy()
        self._rtp_accounting: _RtpAccounting | None = None

        self._pion_proc: subprocess.Popen[bytes] | None = None
        self._pion_offer: str | None = None
        self._answer_data: dict[str, Any] | None = None
        self._stopped = False
        self._last_metrics: dict[str, Any] | None = None

    def _import_gstreamer(self) -> None:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GLib", "2.0")
        from gi.repository import Gst, GLib
        Gst.init(None)
        self._Gst = Gst
        self._GLib = GLib

    async def start(self) -> str:
        """Start encoding pipeline + Pion subprocess. Returns video mid."""
        self._loop = asyncio.get_event_loop()
        self._import_gstreamer()

        # Start GLib main loop
        self._glib_loop = self._GLib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run,
            name="pion-glib-loop",
            daemon=True,
        )
        self._glib_thread.start()

        # Build pipeline
        pipe_desc = _PIPELINE_TEMPLATE.format(
            width=WIDTH, height=HEIGHT, fps=FPS, gop=GOP_SIZE,
            bitrate=self._bitrate, agg_mode=RTP_AGGREGATE_MODE,
            rtp_port=_PION_RTP_PORT,
        )
        logger.info("Pipeline:\n%s", pipe_desc)
        self._pipeline = self._Gst.parse_launch(pipe_desc)
        self._appsrc = self._pipeline.get_by_name("src")
        self._nvenc = self._pipeline.get_by_name("nvenc")
        logger.info("Pion path bitrate: %d kbps (PION_BITRATE env)", self._bitrate)

        # ── Luma probe on appsrc:src — verify raw frames contain avatar ──
        _appsrc_pad = self._appsrc.get_static_pad("src")
        if _appsrc_pad is not None:
            _appsrc_pad.add_probe(
                self._Gst.PadProbeType.BUFFER,
                self._make_luma_probe(),
            )
            logger.info("Luma probe installed on appsrc:src")

        # ── H264 capture after h264parse — OFF by default, gated by env ──
        # The probe registers BUFFER | BUFFER_LIST; calling get_buffer() on a
        # BUFFER_LIST event triggers GStreamer-CRITICAL assertion spam that
        # can SIGABRT the container.
        if os.environ.get("AVTR_PION_H264_CAPTURE", "0") == "1":
            h264parse = self._pipeline.get_by_name("h264p")
            if h264parse is not None:
                _cap_dir = os.environ.get("AVTR1_LOCAL_STORAGE", "/tmp")
                _cap_path = os.path.join(_cap_dir, "pion_h264_capture.h264")
                self._h264_cap_file = open(_cap_path, "wb")
                self._h264_cap_count = 0
                self._H264_CAP_MAX = 750
                _h264_pad = h264parse.get_static_pad("src")
                _h264_pad.add_probe(
                    self._Gst.PadProbeType.BUFFER | self._Gst.PadProbeType.BUFFER_LIST,
                    self._h264_capture_probe,
                )
                logger.info("H264 capture probe: %s (max %d bufs) [AVTR_PION_H264_CAPTURE=1]", _cap_path, self._H264_CAP_MAX)
            else:
                self._h264_cap_file = None
                self._h264_cap_count = 0
                self._H264_CAP_MAX = 0
        else:
            self._h264_cap_file = None
            self._h264_cap_count = 0
            self._H264_CAP_MAX = 0
            logger.info("H264 capture probe SKIPPED (AVTR_PION_H264_CAPTURE not set — OFF by default)")

        # ── RTP accounting probe on rtph264pay:src — OFF by default, gated by env ──
        if os.environ.get("AVTR_PION_RTP_ACCOUNTING", "0") == "1":
            rtppay = self._pipeline.get_by_name("rtppay")
            if rtppay is not None:
                _pay_pad = rtppay.get_static_pad("src")
                self._rtp_accounting = _RtpAccounting()
                _pay_pad.add_probe(
                    self._Gst.PadProbeType.BUFFER | self._Gst.PadProbeType.BUFFER_LIST,
                    self._make_rtp_accounting_probe(),
                )
                logger.info("RTP accounting probe installed on rtph264pay:src [AVTR_PION_RTP_ACCOUNTING=1]")
        else:
            self._rtp_accounting = None
            logger.info("RTP accounting probe SKIPPED (AVTR_PION_RTP_ACCOUNTING not set — OFF by default)")

        # Start push thread (pre-roll phase until ICE connects)
        self._push_thread = threading.Thread(
            target=self._push_frames, name="pion-push", daemon=True
        )
        self._push_thread.start()

        ret = self._pipeline.set_state(self._Gst.State.PLAYING)
        logger.info("Pipeline state: %s", ret)

        # Start Pion binary
        env = dict(os.environ)
        env["PION_STUN_SERVER"] = "stun:stun.cloudflare.com:3478"
        self._pion_proc = subprocess.Popen(
            [_PION_BINARY],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        logger.info("Pion process started (pid=%d)", self._pion_proc.pid)

        # Wait for Pion HTTP health
        await self._wait_pion_health()

        # Fetch offer from Pion
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_PION_BASE}/sdp/offer")
            resp.raise_for_status()
            offer_data = resp.json()
        self._pion_offer = offer_data["sdp"]
        logger.info("Pion offer received: %d chars", len(self._pion_offer))

        # Post offer to Worker
        await self._post_offer_to_worker()

        # Start ICE state polling
        asyncio.ensure_future(self._poll_ice_state())

        # Start adaptive bitrate polling
        asyncio.ensure_future(self._poll_congestion())

        return "video0"

    # ── Diagnostic probes ────────────────────────────────────────────────

    def _make_luma_probe(self):
        """Probe on appsrc:src — samples raw I420 luma to detect black frames."""
        state = {"count": 0, "last_log": time.monotonic(), "black_count": 0}

        def _probe(_pad, info):
            buf = info.get_buffer()
            if buf is None:
                return self._Gst.PadProbeReturn.OK
            state["count"] += 1
            if state["count"] % 25 != 0:
                return self._Gst.PadProbeReturn.OK

            success, map_info = buf.map(self._Gst.MapFlags.READ)
            if not success:
                return self._Gst.PadProbeReturn.OK
            try:
                data = bytes(map_info.data)
                y_size = WIDTH * HEIGHT
                if len(data) < y_size:
                    return self._Gst.PadProbeReturn.OK
                y_plane = data[:y_size]
                y_mean = sum(y_plane) / y_size
                non_black = sum(1 for b in y_plane[::100] if b > 16) / max(1, len(y_plane[::100]))
                if y_mean < 8:
                    state["black_count"] += 1
                now = time.monotonic()
                if now - state["last_log"] >= 5.0:
                    logger.info(
                        "[luma_probe] frame#%d y_mean=%.1f non_black=%.2f black_samples=%d/%d",
                        state["count"], y_mean, non_black,
                        state["black_count"], state["count"] // 25,
                    )
                    state["last_log"] = now
            finally:
                buf.unmap(map_info)
            return self._Gst.PadProbeReturn.OK

        return _probe

    def _h264_capture_probe(self, _pad, info):
        """Capture raw H264 NALUs after h264parse for offline analysis."""
        if self._h264_cap_count >= self._H264_CAP_MAX:
            if self._h264_cap_file:
                self._h264_cap_file.close()
                self._h264_cap_file = None
            return self._Gst.PadProbeReturn.OK
        if self._h264_cap_file is None:
            return self._Gst.PadProbeReturn.OK
        # Guard: check info.type before calling get_buffer()/get_buffer_list()
        if info.type & self._Gst.PadProbeType.BUFFER:
            buf = info.get_buffer()
            if buf is not None:
                self._write_h264_buf(buf)
        elif info.type & self._Gst.PadProbeType.BUFFER_LIST:
            buf_list = info.get_buffer_list()
            if buf_list is not None:
                for i in range(buf_list.length()):
                    b = buf_list.get(i)
                    if b is not None:
                        self._write_h264_buf(b)
        return self._Gst.PadProbeReturn.OK

    def _write_h264_buf(self, buf):
        success, map_info = buf.map(self._Gst.MapFlags.READ)
        if success:
            self._h264_cap_file.write(map_info.data)
            self._h264_cap_count += 1
            buf.unmap(map_info)

    def _make_rtp_accounting_probe(self):
        """Probe on rtph264pay:src — full RTP seq/timestamp/marker/NAL accounting."""
        last_log = {"t": time.monotonic()}

        def _process(buf):
            if self._rtp_accounting is None:
                return
            success, map_info = buf.map(self._Gst.MapFlags.READ)
            if not success:
                return
            try:
                data = bytes(map_info.data)
                self._rtp_accounting.observe(data)
            finally:
                buf.unmap(map_info)

        def _probe(_pad, info):
            # Guard: check info.type before calling get_buffer()/get_buffer_list()
            if info.type & self._Gst.PadProbeType.BUFFER:
                buf = info.get_buffer()
                if buf is not None:
                    _process(buf)
            elif info.type & self._Gst.PadProbeType.BUFFER_LIST:
                buf_list = info.get_buffer_list()
                if buf_list is not None:
                    for i in range(buf_list.length()):
                        b = buf_list.get(i)
                        if b is not None:
                            _process(b)

            now = time.monotonic()
            if now - last_log["t"] >= 5.0 and self._rtp_accounting is not None:
                last_log["t"] = now
                snap = self._rtp_accounting.snapshot()
                logger.info(
                    "[rtp_acct] total=%d seq=[%d..%d] gaps=%d dups=%d late=%d "
                    "expected=%d markers=%d parse_err=%d nal=%s",
                    snap["total"], snap["first_seq"], snap["last_seq"],
                    snap["gaps"], snap["dups"], snap["late"],
                    snap["expected"], snap["markers"],
                    snap["parse_errors"], snap["nal_types"],
                )
            return self._Gst.PadProbeReturn.OK

        return _probe

    async def _wait_pion_health(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    resp = await client.get(f"{_PION_BASE}/health")
                    if resp.status_code == 200:
                        logger.info("Pion healthy")
                        return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError("Pion did not become healthy within timeout")

    async def _post_offer_to_worker(self) -> None:
        headers = {"Content-Type": "application/json"}
        secret = os.environ.get("MODAL_WORKER_SECRET")
        if secret:
            headers["x-modal-secret"] = secret
        body = {
            "sdp": self._pion_offer,
            "type": "offer",
            "mid": "video0",
            "track_name": self._track_name,
            "kind": "video",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self._callback_url, headers=headers, json=body)
            logger.info("Offer posted to Worker: status=%d", resp.status_code)

    async def set_remote_answer(self, answer_sdp: str, answer_type: str) -> None:
        """Forward browser's answer to Pion."""
        logger.info("Forwarding answer to Pion: %d chars", len(answer_sdp))
        self._answer_data = {"answer_sdp": answer_sdp, "answer_type": answer_type}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{_PION_BASE}/sdp/answer",
                json={"sdp": answer_sdp, "type": "answer"},
            )
            logger.info("Pion answer set: status=%d", resp.status_code)

    async def _poll_ice_state(self) -> None:
        """Poll Pion's ICE state; set _ice_connected when connected."""
        while not self._stopped:
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    resp = await client.get(f"{_PION_BASE}/state")
                    if resp.status_code == 200:
                        data = resp.json()
                        ice_state = data.get("ice", "new")
                        if ice_state in ("connected", "completed"):
                            if not self._ice_connected.is_set():
                                self._source.start()
                                logger.info("Pion ICE connected — activating frame source (GPU rendering starts)")
                                self._ts_set(self._ice_connected)
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)

    async def _poll_congestion(self) -> None:
        """Poll Pion /congestion endpoint and dynamically adjust nvh264enc bitrate."""
        await asyncio.sleep(8)  # wait for ICE to connect and traffic to flow
        last_bitrate = self._bitrate
        while not self._stopped:
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(f"{_PION_BASE}/congestion")
                    if resp.status_code == 200:
                        data = resp.json()
                        suggested = data.get("suggested_bitrate", 2000)
                        congested = data.get("congested", False)
                        loss_rate = data.get("loss_rate", 0)

                        if congested and suggested != last_bitrate:
                            new_br = max(suggested, 500)
                            if new_br != last_bitrate:
                                self._set_encoder_bitrate(new_br)
                                last_bitrate = new_br
                                logger.warning(
                                    "[adaptive-bitrate] loss=%.1f%% → %d kbps",
                                    loss_rate * 100, new_br,
                                )
                        elif not congested and last_bitrate < self._base_bitrate:
                            ramp = min(last_bitrate + 250, self._base_bitrate)
                            if ramp != last_bitrate:
                                self._set_encoder_bitrate(ramp)
                                last_bitrate = ramp
                                logger.info(
                                    "[adaptive-bitrate] recovering → %d kbps", ramp,
                                )
            except Exception:
                pass
            await asyncio.sleep(5)

    def _set_encoder_bitrate(self, bitrate: int) -> None:
        """Dynamically adjust nvh264enc bitrate without pipeline restart."""
        if self._nvenc is None:
            return
        try:
            self._nvenc.set_property("bitrate", bitrate)
            self._bitrate = bitrate
        except Exception as exc:
            logger.warning("[adaptive-bitrate] set_property failed: %s", exc)

    def _ts_set(self, event: asyncio.Event) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(event.set)
        else:
            event.set()

    def _push_frames(self) -> None:
        """Push frames into appsrc. Pre-roll black frames, then real frames."""
        frame_duration_ns = int(1_000_000_000 / FPS)
        next_deadline = time.monotonic()

        while not self._stop_push.is_set():
            if self._ice_connected.is_set():
                err = self._source.get_error()
                if err is not None:
                    self._fail_fast.record_producer_error(err)

                prev_underflow = getattr(self._source, "underflow_count", 0)
                frame = self._source.next_frame()
                curr_underflow = getattr(self._source, "underflow_count", 0)

                is_underflow = curr_underflow > prev_underflow
                if is_underflow:
                    self._metrics.underflow_count += 1

                trip = self._fail_fast.record_frame(underflow=is_underflow)
                if trip or self._fail_fast.is_tripped():
                    break

                self._metrics.frames_generated += 1
            else:
                frame = self._black_frame
                self._metrics.frames_repeated += 1

            raw = frame.tobytes() if hasattr(frame, "tobytes") else frame

            # Verify frame content (every 25 frames in live phase)
            if self._ice_connected.is_set() and self._metrics.frames_generated % 25 == 0:
                try:
                    _y_slice = raw[:WIDTH * HEIGHT]
                    _y_mean = sum(_y_slice) / max(1, len(_y_slice))
                    logger.info("[frame_luma] gen#%d y_mean=%.1f raw_len=%d", self._metrics.frames_generated, _y_mean, len(raw))
                except Exception as exc:
                    logger.warning("[frame_luma] check failed: %s", exc)
            buf = self._Gst.Buffer.new_wrapped(raw)
            buf.pts = self._push_counter * frame_duration_ns
            buf.dts = buf.pts
            buf.duration = frame_duration_ns
            buf.offset = self._push_counter
            buf.offset_end = self._push_counter + 1
            self._push_counter += 1

            ret = self._appsrc.emit("push-buffer", buf)
            if ret == self._Gst.FlowReturn.OK:
                self._metrics.frames_emitted += 1

            next_deadline += frame_duration_ns / 1_000_000_000.0
            sleep_ns = next_deadline - time.monotonic()
            if sleep_ns > 0:
                time.sleep(sleep_ns)
            else:
                next_deadline = time.monotonic()

    async def stop(self) -> dict[str, Any]:
        if self._stopped:
            return self._last_metrics or {}
        self._stopped = True

        self._stop_push.set()
        if self._push_thread and self._push_thread.is_alive():
            self._push_thread.join(timeout=5)

        try:
            self._source.stop()
        except Exception:
            pass

        # Fetch Pion stats BEFORE killing it
        pion_stats = None
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{_PION_BASE}/stats")
                if resp.status_code == 200:
                    pion_stats = resp.json()
                    logger.info("[pion-stats] udp=%s write=%s rtcp=%s ice=%s",
                                pion_stats.get("udp"),
                                {k: v for k, v in (pion_stats.get("write") or {}).items()
                                 if k in ("attempts", "successes", "errors")},
                                {k: v for k, v in (pion_stats.get("rtcp") or {}).items()
                                 if k in ("nack", "pli", "fir", "rr")},
                                pion_stats.get("ice"))
        except Exception:
            pass

        # Dump RTP accounting from GStreamer side
        if self._rtp_accounting is not None:
            snap = self._rtp_accounting.snapshot()
            logger.info(
                "[rtp_acct FINAL] total=%d seq=[%d..%d] gaps=%d dups=%d late=%d "
                "expected=%d markers=%d parse_err=%d nal=%s",
                snap["total"], snap["first_seq"], snap["last_seq"],
                snap["gaps"], snap["dups"], snap["late"],
                snap["expected"], snap["markers"],
                snap["parse_errors"], snap["nal_types"],
            )

        # Kill Pion — capture stdout for logs
        if self._pion_proc:
            try:
                self._pion_proc.terminate()
                try:
                    stdout, _ = self._pion_proc.communicate(timeout=3)
                    if stdout:
                        for line in stdout.decode("utf-8", errors="replace").splitlines()[-50:]:
                            logger.info("[pion-stdout] %s", line)
                except Exception:
                    self._pion_proc.kill()
            except Exception:
                pass
            self._pion_proc = None

        # Close H264 capture
        if self._h264_cap_file:
            try:
                self._h264_cap_file.close()
                logger.info("H264 capture closed: %d buffers", self._h264_cap_count)
            except Exception:
                pass
            self._h264_cap_file = None

        if self._pipeline is not None:
            try:
                # Send EOS first to flush pipeline cleanly (prevents terminate exception)
                self._pipeline.send_event(self._Gst.Event.new_eos())
                # Brief wait for EOS to propagate
                time.sleep(0.5)
                self._pipeline.set_state(self._Gst.State.NULL)
            except Exception as exc:
                logger.warning("[cleanup] pipeline teardown: %s", exc)

        if self._glib_loop:
            try:
                self._glib_loop.quit()
            except Exception:
                pass

        # Best-effort close callback
        if self._close_url:
            try:
                headers = {"Content-Type": "application/json"}
                secret = os.environ.get("MODAL_WORKER_SECRET")
                if secret:
                    headers["x-modal-secret"] = secret
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.put(self._close_url, headers=headers, json={
                        "track_name": self._track_name,
                        "video_mid": "video0",
                    })
            except Exception:
                pass

        # Metrics
        err = self._source.get_error()
        if err is not None:
            self._metrics.producer_error = f"{type(err).__name__}: {str(err)[:300]}"
        if self._fail_fast.fail_fast_reason:
            self._metrics.fail_fast_reason = self._fail_fast.fail_fast_reason

        source_metrics = getattr(self._source, "metrics", None)
        if source_metrics and hasattr(source_metrics, "snapshot"):
            snap = source_metrics.snapshot()
            if snap.get("queue_depth_min") is not None:
                self._metrics.queue_depth_min = snap["queue_depth_min"]
            if snap.get("queue_depth_max") is not None:
                self._metrics.queue_depth_max = snap["queue_depth_max"]

        self._last_metrics = self._metrics.snapshot(
            encoder_info=dict(_ENCODER_INFO, bitrate=self._bitrate, pion_debug=True)
        )
        if pion_stats:
            self._last_metrics["pion_stats"] = pion_stats
        if self._rtp_accounting:
            self._last_metrics["rtp_accounting"] = self._rtp_accounting.snapshot()
        return self._last_metrics
