"""Worklet that records streamer-own turn latency metrics from playback events."""

from attrs import define

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    SegmentPlaybackInterrupted,
    SegmentPlaybackStarted,
    Shutdown,
)
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()

@define
class LatencyWorklet:
    """
    Listener-only worklet: turns the latency breakdown carried by playback events into OTel
    histograms and one structured log line per turn. Publishes nothing.
    """

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        async with bus.subscribe(
            SegmentPlaybackStarted, SegmentPlaybackInterrupted, Shutdown
        ) as sub:
            bus.ready()

            async for event in sub:
                match event:
                    case SegmentPlaybackStarted() | SegmentPlaybackInterrupted():
                        self._record(event, clocks)

                    case Shutdown():
                        return

    def _record(
        self, event: SegmentPlaybackStarted | SegmentPlaybackInterrupted, clocks: StreamClocks
    ) -> None:
        latencies = event.latencies
        if latencies is None:
            return

        kind = latencies.kind

        t_total_wall = clocks.now_wall - latencies.arrival_monotonic

        log_extra = {"t_total_wall": round(t_total_wall, ndigits=4)} if kind == "turn_start" else {}
        _LOGGER.info(
            "Latency contribution measured",
            kind=kind,
            segment_id=event.segment_id,
            t_wait=round(latencies.t_wait, ndigits=4),
            t_pad=round(latencies.t_pad, ndigits=4),
            t_lead=round(latencies.t_lead, ndigits=4),
            t_present=round(latencies.t_present, ndigits=4),
            t_total=round(latencies.t_total, ndigits=4),
            **log_extra,
            no_fragment_at_admit=latencies.no_fragment_at_admit,
            clock_started=latencies.clock_started,
            consistent=latencies.consistent,
        )
