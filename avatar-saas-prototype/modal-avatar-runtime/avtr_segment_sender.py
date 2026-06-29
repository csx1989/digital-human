"""
Avtr-1 shared segment sender + media event stamping.

Contains the EXACT _SegmentSender class and stamp_media_event function
used by both the production Modal pipeline and the isolated test suite.

When this module is updated, both production and tests automatically
use the same code — no copy drift possible.
"""

import base64
import collections
import threading
import time


class _SegmentSender:
    """Server-side source playout buffer.

    Buffers encoded fMP4 segments and yields them at real-time cadence.
    Init segment yields immediately.  Media segments:
      - Burst-fill until target_lead_s is reached.
      - Then yield at ~chunk_media_s intervals (pacing sleep).
      - Back-pressure break when source lead exceeds target + half-chunk.
    """

    def __init__(self, chunk_media_s: float, target_lead_s: float,
                 stream_start_mono: float):
        self.chunk_media_s = chunk_media_s
        self.target_lead_s = target_lead_s
        self.stream_start_mono = stream_start_mono
        self.pending: collections.deque[tuple[str, bytes]] = collections.deque()
        self.pending_meta: collections.deque[dict] = collections.deque()
        self.produced_media_s: float = 0.0
        self.target_lead_reached: bool = False
        self.last_yield_mono: float = 0.0
        self.media_sequence: int = 0
        self._lock = threading.Lock()
        self._drain_fast: bool = False
        self.collected_init: dict | None = None
        self.collected_media: list[dict] = []
        self._inter_yield_gaps: list[float] = []
        self._total_pacing_sleep_ms: float = 0.0
        self._min_queue_depth: int = 999999
        self._max_queue_depth: int = 0
        self._dry_spell_count: int = 0
        self._meta_added_count: int = 0
        self._meta_popped_count: int = 0
        self._missing_meta_count: int = 0
        self._speech_media_count: int = 0
        self._idle_media_count: int = 0
        self._media_count: int = 0

    def add(self, seg_type: str, seg_data: bytes) -> None:
        with self._lock:
            self.pending.append((seg_type, seg_data))

    def add_meta(self, meta: dict) -> None:
        with self._lock:
            self.pending_meta.append(meta)
            self._meta_added_count += 1

    def set_drain_fast(self, fast: bool) -> None:
        self._drain_fast = fast

    def _next_seq(self) -> int:
        self.media_sequence += 1
        self._media_count += 1
        return self.media_sequence

    def _pop_meta(self) -> dict:
        """Pop one metadata item.  Must be called under self._lock."""
        if self.pending_meta:
            self._meta_popped_count += 1
            meta = self.pending_meta.popleft()
            if meta.get("mode") == "speech":
                self._speech_media_count += 1
            else:
                self._idle_media_count += 1
            return meta
        self._missing_meta_count += 1
        self._idle_media_count += 1
        return {"mode": "idle"}

    def yield_events(self):
        results = []
        now = time.monotonic()

        with self._lock:
            qd = len(self.pending)
            if qd < self._min_queue_depth:
                self._min_queue_depth = qd
            if qd > self._max_queue_depth:
                self._max_queue_depth = qd

            if not self.pending:
                self._dry_spell_count += 1
                return results

            while self.pending:
                seg_type, seg_data = self.pending[0]

                if seg_type == "init":
                    self.pending.popleft()
                    b64 = base64.b64encode(seg_data).decode("ascii")
                    evt = {
                        "type": "fmp4_init",
                        "dataBase64": b64,
                        "byteLength": len(seg_data),
                        "timing": {
                            "timeToInitMs": round(
                                (now - self.stream_start_mono) * 1000, 1
                            ),
                        },
                    }
                    self.collected_init = {"byteLength": len(seg_data), "dataBase64": b64}
                    results.append(evt)
                    self.last_yield_mono = time.monotonic()
                    continue

                # --- Media segment ---
                wall_elapsed = time.monotonic() - self.stream_start_mono
                media_lead = self.produced_media_s - wall_elapsed

                if not self.target_lead_reached:
                    self.pending.popleft()
                    self.produced_media_s += self.chunk_media_s
                    meta = self._pop_meta()
                    b64 = base64.b64encode(seg_data).decode("ascii")
                    evt = {
                        "type": "fmp4_media",
                        "dataBase64": b64,
                        "byteLength": len(seg_data),
                        "sequence": self._next_seq(),
                    }
                    results.append(("media", seg_data, evt, meta))
                    self.last_yield_mono = time.monotonic()
                    wall_after = time.monotonic() - self.stream_start_mono
                    if self.produced_media_s - wall_after >= self.target_lead_s:
                        self.target_lead_reached = True
                    continue

                if not self._drain_fast and media_lead > self.target_lead_s + self.chunk_media_s * 0.5:
                    break

                if not self._drain_fast:
                    desired_interval = self.chunk_media_s
                    if self.last_yield_mono > 0:
                        elapsed_since = time.monotonic() - self.last_yield_mono
                        if elapsed_since < desired_interval:
                            break

                self.pending.popleft()
                self.produced_media_s += self.chunk_media_s
                meta = self._pop_meta()

                if self.last_yield_mono > 0:
                    gap_ms = (time.monotonic() - self.last_yield_mono) * 1000
                    self._inter_yield_gaps.append(gap_ms)

                b64 = base64.b64encode(seg_data).decode("ascii")
                evt = {
                    "type": "fmp4_media",
                    "dataBase64": b64,
                    "byteLength": len(seg_data),
                    "sequence": self._next_seq(),
                }
                results.append(("media", seg_data, evt, meta))
                self.last_yield_mono = time.monotonic()

                if not self._drain_fast:
                    break

            return results

    def drain_all(self):
        results = []
        with self._lock:
            qd = len(self.pending)
            if qd > self._max_queue_depth:
                self._max_queue_depth = qd
            if qd < self._min_queue_depth:
                self._min_queue_depth = qd
            while self.pending:
                seg_type, seg_data = self.pending.popleft()
                if seg_type == "init":
                    t_now = time.monotonic()
                    b64 = base64.b64encode(seg_data).decode("ascii")
                    evt = {
                        "type": "fmp4_init",
                        "dataBase64": b64,
                        "byteLength": len(seg_data),
                        "timing": {
                            "timeToInitMs": round(
                                (t_now - self.stream_start_mono) * 1000, 1
                            ),
                        },
                    }
                    self.collected_init = {"byteLength": len(seg_data), "dataBase64": b64}
                    results.append(evt)
                else:
                    self.produced_media_s += self.chunk_media_s
                    meta = self._pop_meta()
                    b64 = base64.b64encode(seg_data).decode("ascii")
                    evt = {
                        "type": "fmp4_media",
                        "dataBase64": b64,
                        "byteLength": len(seg_data),
                        "sequence": self._next_seq(),
                    }
                    results.append(("media", seg_data, evt, meta))
                    self.collected_media.append({"byteLength": len(seg_data)})
        return results

    @property
    def max_inter_yield_gap_ms(self):
        return round(max(self._inter_yield_gaps), 1) if self._inter_yield_gaps else 0

    @property
    def avg_inter_yield_gap_ms(self):
        n = len(self._inter_yield_gaps)
        return round(sum(self._inter_yield_gaps) / n, 1) if n else 0

    @property
    def total_pacing_sleep_ms(self):
        return round(self._total_pacing_sleep_ms, 1)

    @property
    def queue_depth(self):
        with self._lock:
            return len(self.pending)

    @property
    def min_queue_depth(self):
        return self._min_queue_depth if self._min_queue_depth != 999999 else 0

    @property
    def max_queue_depth(self):
        return self._max_queue_depth

    @property
    def dry_spell_count(self):
        return self._dry_spell_count

    @property
    def source_lead_ms(self):
        wall = time.monotonic() - self.stream_start_mono
        return max(0, round((self.produced_media_s - wall) * 1000))

    @property
    def meta_added_count(self):
        with self._lock:
            return self._meta_added_count

    @property
    def meta_popped_count(self):
        with self._lock:
            return self._meta_popped_count

    @property
    def missing_meta_count(self):
        with self._lock:
            return self._missing_meta_count

    @property
    def speech_media_count(self):
        with self._lock:
            return self._speech_media_count

    @property
    def idle_media_count(self):
        with self._lock:
            return self._idle_media_count


def stamp_media_event(evt: dict, meta: dict, sess: dict) -> None:
    """Stamp a media event dict with metadata and one-shot speech flags."""
    evt["mode"] = meta.get("mode", "idle")
    evt["turnId"] = meta.get("turnId")
    evt["chunkIndex"] = meta.get("chunkIndex")

    timing = {
        "chunkRenderMs": round(meta.get("chunkRenderMs", 0), 1),
        "feedMs": round(meta.get("feedMs", 0), 1),
        "framesFed": meta.get("framesFed", 0),
        "audioSamplesPerChunk": meta.get("audioSamplesPerChunk", 0),
    }
    evt["timing"] = timing

    is_speech_start = bool(meta.get("isFirst", False))
    is_speech_end = bool(meta.get("isLast", False))
    turn_id = meta.get("turnId")

    current_turn_id = sess.get("_current_turn_id")
    if is_speech_start and turn_id is not None and turn_id != current_turn_id:
        sess["_speech_first_segment_emitted"] = False
        sess["_speech_end_segment_emitted"] = False
        current_turn_id = turn_id

    if (meta.get("mode") == "speech"
            and is_speech_start
            and not sess.get("_speech_first_segment_emitted", False)):
        evt["isSpeechStart"] = True
        sess["_speech_first_segment_emitted"] = True
    else:
        evt["isSpeechStart"] = False

    if (meta.get("mode") == "speech"
            and is_speech_end
            and not sess.get("_speech_end_segment_emitted", False)):
        evt["isSpeechEnd"] = True
        sess["_speech_end_segment_emitted"] = True
    else:
        evt["isSpeechEnd"] = False

    sess["_current_turn_id"] = current_turn_id
