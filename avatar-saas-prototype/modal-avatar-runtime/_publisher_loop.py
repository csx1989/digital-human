"""Persistent event loop for aiortc WebRTC publisher (Phase D1).

Provides _PublisherLoop — a dedicated background thread with its own asyncio
event loop that stays alive for the entire session lifetime.  This is critical
because aiortc's RTCPeerConnection needs a running loop to maintain RTP
sending, ICE keepalives, and track recv() calls.

The old approach used asyncio.run() which creates a fresh loop, runs the
coroutine, then CLOSES the loop.  After asyncio.run() returned, the aiortc
PC had no running loop — killing all RTP sending.

With _PublisherLoop:
  1. pub_loop = _PublisherLoop()       — starts thread + loop
  2. pub_loop.run(publisher.start())   — runs startup on the persistent loop
  3. ... loop stays alive for session lifetime ...
  4. pub_loop.run(publisher.stop())    — runs cleanup on the persistent loop
  5. pub_loop.shutdown()               — stops loop + joins thread

AvtrFrameSource captures the persistent loop via asyncio.get_running_loop()
during step 2.  Its producer thread uses run_coroutine_threadsafe() to bridge
frames to that loop.  As long as the loop is running, frames flow.
"""

from __future__ import annotations

import asyncio
import threading


class _PublisherLoop:
    """Dedicated thread + event loop for aiortc RTCPeerConnection lifetime.

    The loop persists for the entire session, keeping the PC alive for
    RTP sending, ICE keepalives, and track recv() calls.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="webrtc-publisher-loop",
        )
        self._thread.start()
        # Wait until the loop is actually running before returning
        if not self._ready.wait(timeout=5):
            raise RuntimeError("publisher loop thread failed to start")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def run(self, coro, timeout: float = 60.0):
        """Run a coroutine on this loop, blocking until done.

        Returns the coroutine's result or re-raises its exception.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def schedule(self, coro) -> None:
        """Schedule a coroutine on this loop without blocking (fire-and-forget)."""
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def is_alive(self) -> bool:
        """Returns True if the loop is running and the thread is alive."""
        return self._loop.is_running() and self._thread.is_alive()

    def shutdown(self) -> None:
        """Stop the loop and join the thread (safe to call multiple times)."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
