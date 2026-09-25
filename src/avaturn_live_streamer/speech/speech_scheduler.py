# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import asyncio
from collections import deque
from fractions import Fraction
from typing import Literal

from attr import dataclass

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.core.logs import get_logger

from .duration import Duration
from .speech_buffer import SpeechBuffer

_LOGGER = get_logger()

LatencyKind = Literal["turn_start", "interruption"]


@dataclass(slots=True, frozen=True)
class SegmentStarted:
    id: str


@dataclass(slots=True, frozen=True)
class SegmentEnded:
    id: str


@dataclass(slots=True, frozen=True)
class SegmentInterrupted:
    id: str
    played_duration: Fraction


@dataclass(slots=True, frozen=True)
class SegmentCancelled:
    id: str


SegmentEvent = SegmentStarted | SegmentEnded | SegmentInterrupted | SegmentCancelled


@dataclass(slots=True, frozen=True)
class _Arrival:
    """When a marker or the first fragment of a segment entered the input queue."""

    media: Fraction
    wall: float
    clock_started: bool


@dataclass(slots=True, frozen=True)
class SegmentLatencies:
    kind: LatencyKind
    t_wait: Fraction
    t_pad: Fraction
    t_lead: Fraction
    t_present: Fraction
    t_total: Fraction
    arrival_monotonic: float
    no_fragment_at_admit: bool
    clock_started: bool
    consistent: bool


@dataclass(slots=True, frozen=True)
class TimestampedEvent:
    event: SegmentEvent
    timestamp: Fraction
    latencies: SegmentLatencies | None = None


class SpeechScheduler:
    @dataclass(slots=True, frozen=True)
    class _StepResult:
        present: SpeechBuffer
        future: SpeechBuffer
        events: list[TimestampedEvent]

    def __init__(
        self,
        sample_rate: int,
        present_duration: Fraction,
        future_duration: Fraction,
        clocks: StreamClocks,
    ):
        self.sample_rate = sample_rate
        self.present_duration = present_duration
        self.future_duration = future_duration
        self._clocks = clocks

        self._write_active_segment: str | None = None
        self._queue = deque[SegmentEvent | SpeechBuffer]()

        self._max_padding_buffer = SpeechBuffer.silence(
            future_duration + present_duration, sample_rate
        )
        self._buffer = SpeechBuffer.empty()

        self._event_queue = deque[TimestampedEvent]()
        self._next_present_timestamp = Fraction(denominator=sample_rate)
        self._active_segment: str | None = None
        self._active_segment_start: Fraction = Fraction()

        self._new_arrival = asyncio.Condition()

        self._autoclosed_segments: set[str] = set[str]()
        self._stop: bool = False

        self._marker_arrivals: dict[str, _Arrival] = {}
        self._fragment_arrivals: dict[str, _Arrival] = {}
        self._interrupt_arrival: _Arrival | None = None
        self._interrupt_ended_segment: str | None = None
        self._pending_interrupt_latencies: SegmentLatencies | None = None

    def _arrival_now(self) -> _Arrival:
        return _Arrival(
            media=Fraction(self._clocks.now),
            wall=self._clocks.now_wall,
            clock_started=self._clocks.started,
        )

    @property
    def _next_event_timestamp(self) -> Fraction:
        return self._next_present_timestamp + self._buffer.duration

    def _build_latencies(
        self,
        kind: LatencyKind,
        arrival: _Arrival,
        step_media: Fraction,
        present_start: Fraction,
        lookahead: Fraction,
        t_pad: Fraction,
        playback_timestamp: Fraction,
        *,
        no_fragment_at_admit: bool,
    ) -> SegmentLatencies:
        t_wait = step_media - arrival.media
        t_lead = present_start - step_media
        t_total = playback_timestamp - arrival.media

        # The report names the last term T_present: the constant present stride the turn commits to
        # move forward. The raw backlog (`lookahead`) is that stride plus a constant tail (the suffix
        # of the previous future window) and any oversized-fragment overage; fold that tail into
        # T_pad so the two variable terms carry it and T_present stays constant.
        t_present = self.present_duration
        t_pad = t_pad + (lookahead - t_present)

        consistent = t_wait + t_pad + t_lead + t_present == t_total
        if not consistent:
            _LOGGER.warning(
                "Latency terms do not sum to the total, instrumentation is out of sync with do_step",
                kind=kind,
                t_wait=float(t_wait),
                t_pad=float(t_pad),
                t_lead=float(t_lead),
                t_present=float(t_present),
                t_total=float(t_total),
            )

        return SegmentLatencies(
            kind=kind,
            t_wait=t_wait,
            t_pad=t_pad,
            t_lead=t_lead,
            t_present=t_present,
            t_total=t_total,
            arrival_monotonic=arrival.wall,
            no_fragment_at_admit=no_fragment_at_admit,
            clock_started=arrival.clock_started,
            consistent=consistent,
        )

    def _attach_turn_start_latencies(
        self,
        events: list[TimestampedEvent],
        segment_id: str,
        step_media: Fraction,
        present_start: Fraction,
        lookahead: Fraction,
        t_pad: Fraction,
    ) -> list[TimestampedEvent]:
        marker = self._marker_arrivals.get(segment_id)
        if marker is None:
            return events

        # The report defines the turn as starting once both the start marker and the first
        # fragment are in hand. A fragment that only arrives after admittance cannot have
        # delayed us, so fall back to the marker and flag the sample.
        fragment = self._fragment_arrivals.get(segment_id)
        if fragment is None or fragment.media > step_media:
            no_fragment_at_admit = True
            arrival = marker
        else:
            no_fragment_at_admit = False
            arrival = max(marker, fragment, key=lambda a: a.media)

        result = list[TimestampedEvent]()
        for e in events:
            if isinstance(e.event, SegmentStarted) and e.event.id == segment_id:
                e = TimestampedEvent(
                    e.event,
                    e.timestamp,
                    self._build_latencies(
                        "turn_start",
                        arrival,
                        step_media,
                        present_start,
                        lookahead,
                        t_pad,
                        e.timestamp,
                        no_fragment_at_admit=no_fragment_at_admit,
                    ),
                )
            result.append(e)
        return result

    def _forget_segment(self, segment_id: str) -> None:
        self._marker_arrivals.pop(segment_id, None)
        self._fragment_arrivals.pop(segment_id, None)

    async def interrupt(self):
        self._buffer = self._buffer.slice(
            0, self.future_duration
        )  # Drop overage as it will not be played back

        new_queue = deque[SpeechBuffer | SegmentEvent]()

        if self._active_segment is not None:
            self._interrupt_arrival = self._arrival_now()
            self._interrupt_ended_segment = self._active_segment
            new_queue.append(SegmentEnded(self._active_segment))
            new_queue.append(
                SegmentInterrupted(
                    self._active_segment,
                    played_duration=self._next_event_timestamp - self._active_segment_start,
                )
            )
        if self._write_active_segment is not None:
            if self._write_active_segment != self._active_segment:
                self._autoclosed_segments.add(self._write_active_segment)
            self._write_active_segment = None

        for c in self._queue:
            if isinstance(c, SegmentStarted):
                new_queue.append(SegmentCancelled(c.id))

        self._queue = new_queue

        async with self._new_arrival:
            self._new_arrival.notify()

    async def stop(self) -> None:
        if self._write_active_segment is not None:
            await self.end_segment(self._write_active_segment)
        self._stop = True

    def is_stopped(self) -> bool:
        return self._stop and len(self._queue) == 0 and len(self._event_queue) == 0

    @property
    def unconsumed_duration(self) -> Duration:
        return self._buffer.duration + sum(
            [item.duration for item in self._queue if isinstance(item, SpeechBuffer)]
        )

    async def _append_queue(self, item: SegmentEvent | SpeechBuffer):
        self._queue.append(item)
        async with self._new_arrival:
            self._new_arrival.notify()

    async def append(self, chunk: SpeechBuffer, segment_id: str):
        if self._stop:
            return
        if self._write_active_segment != segment_id:
            _LOGGER.warning(
                "Current write-active segment id mismatch, write_active_segment_id=%s, segment_id=%s, chunk ignored",
                self._write_active_segment,
                segment_id,
            )
            return
        assert chunk.sample_rate == self.sample_rate, "Sample rates must match"
        if segment_id not in self._fragment_arrivals:
            self._fragment_arrivals[segment_id] = self._arrival_now()
        await self._append_queue(chunk)

    async def start_segment(self, segment_id: str):
        if self._stop:
            return
        if self._write_active_segment is not None:
            _LOGGER.warning(
                "Double segment start, write_active_segment_id=%s, start_segment_id=%s, start ignored",
                self._write_active_segment,
                segment_id,
            )
            return
        self._write_active_segment = segment_id
        self._marker_arrivals[segment_id] = self._arrival_now()
        await self._append_queue(SegmentStarted(segment_id))

    async def end_segment(self, segment_id: str):
        if self._write_active_segment is None or self._write_active_segment != segment_id:
            _LOGGER.warning(
                "Unexpected segment end, write_active_segment_id=%s, end_segment_id=%s, end ignored",
                self._write_active_segment,
                segment_id,
                stack_info=True,
            )
            return
        self._write_active_segment = None
        await self._append_queue(SegmentEnded(segment_id))

    async def do_step(self) -> _StepResult:
        bufs = list[SpeechBuffer]()
        new_events = list[TimestampedEvent]()
        is_segment_start = False

        step_media = Fraction(self._clocks.now)
        present_start = self._next_present_timestamp
        lookahead = self._buffer.duration
        started_segment: str | None = None

        # Buffer can already contain enough because of last chunk size
        duration_to_consume = max(
            0, (self.present_duration + self.future_duration) - self._buffer.duration
        )
        consumed_duration = 0

        # Consume queue until we have enough duration or until we hit
        while duration_to_consume > 0:
            if len(self._queue) == 0:
                if self._active_segment is None or is_segment_start:
                    # Stop as we can pad it to the left or to the right
                    break

                # Block and wait for new arrivals to the queue if we are midsegment
                async with self._new_arrival:
                    await self._new_arrival.wait()

            item = self._queue.popleft()
            if isinstance(item, SegmentStarted):
                timestamp = self._next_event_timestamp + consumed_duration
                new_events.append(TimestampedEvent(item, timestamp))
                self._active_segment = item.id
                is_segment_start = True
                self._active_segment_start = timestamp
                started_segment = item.id
            elif isinstance(item, SegmentEnded):
                timestamp = self._next_event_timestamp + consumed_duration
                new_events.append(TimestampedEvent(item, timestamp))
                # An interrupt queues SegmentEnded ahead of SegmentInterrupted, and playback
                # actually stops at the former. Measure here, attach when SegmentInterrupted is
                # admitted on a later step.
                if self._interrupt_arrival is not None and item.id == self._interrupt_ended_segment:
                    self._pending_interrupt_latencies = self._build_latencies(
                        "interruption",
                        self._interrupt_arrival,
                        step_media,
                        present_start,
                        lookahead,
                        Fraction(0),
                        timestamp,
                        no_fragment_at_admit=False,
                    )
                    self._interrupt_arrival = None
                    self._interrupt_ended_segment = None
                self._forget_segment(item.id)
                self._active_segment = None
                # We don't want to end one segment and start another in the same step because we
                # will not be able to pad left or right if there is not enough data
                break
            elif isinstance(item, SpeechBuffer):
                bufs.append(item)
                duration_to_consume -= item.duration
                consumed_duration += item.duration
            else:
                latencies = None
                if isinstance(item, SegmentInterrupted):
                    latencies = self._pending_interrupt_latencies
                    self._pending_interrupt_latencies = None
                else:
                    self._forget_segment(item.id)
                self._event_queue.append(
                    TimestampedEvent(
                        item, self._next_event_timestamp + consumed_duration, latencies
                    )
                )

        # Now we are in one of three situations:
        # 1. We have too little duration in started segment , we need to add silence either on the left if it's a segment start or on the right if it's a segment end
        # 2. We have too much duration because the last chunk was too long (it's ok, we keep it in present+future buffer)
        # 3. We have the exact duration we need
        # PS If duration is too little but midsegment -- we already blocked and waited above

        t_pad = Fraction(0)
        if duration_to_consume > 0:
            pad = self._max_padding_buffer.slice(0, duration_to_consume)
            if self._active_segment is None:
                # We either don't have current active segment and pad it right
                pad_left, pad_right = [], [pad]
            elif is_segment_start:
                # Or we just started segment and didn't have enough of it so we pad it left
                pad_left, pad_right = [pad], []
                t_pad = pad.duration

                # We need to shift event timestamps right
                new_events = [
                    TimestampedEvent(i.event, i.timestamp + pad.duration) for i in new_events
                ]
            else:
                # In other cases we
                raise Exception("Should never get here")
        else:  # duration_to_consume == 0 or duration_to_consume < 0
            # If it's negative -- we keep overage in self._buffer, but we don't send it as future
            pad_left, pad_right = [], []

        if started_segment is not None:
            new_events = self._attach_turn_start_latencies(
                new_events,
                started_segment,
                step_media,
                present_start,
                lookahead,
                t_pad,
            )

        self._event_queue.extend(new_events)
        self._buffer = SpeechBuffer.concat([self._buffer, *pad_left, *bufs, *pad_right])

        self._next_present_timestamp += self.present_duration
        present, self._buffer = self._buffer.split_in_two_parts_with_duration(self.present_duration)
        future, _ = self._buffer.split_in_two_parts_with_duration(self.future_duration)

        actual_events: list[TimestampedEvent] = []
        while len(self._event_queue) > 0:
            if self._event_queue[0].timestamp >= self._next_present_timestamp:
                break
            actual_events.append(self._event_queue.popleft())
        return SpeechScheduler._StepResult(present, future, actual_events)
