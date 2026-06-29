# Future AV Polish Cleanup

## Current Status

AV fMP4 muxed pipeline is working.

- `ENABLE_MODAL_FMP4_AV_STREAM` is live and functional.
- Browser no longer plays separate TTS audio in AV mode.
- Modal/Bridge/Browser path now supports idle + speech through one muxed A/V stream.
- CPU-only Modal control endpoint (`timeline_control`) handles enqueue/stop independently of the GPU stream container, resolving the GPU-blocking deadlock.

## Known Minor Polish Issues

- Hair-split static pause can appear at speech -> idle boundary.
- Duplicate `avatar.speech.start` / `avatar_fmp4_av_speech_started` events were observed for one long turn.
- Playback ack may arrive slightly after speech video end, making idle resume feel marginally late.
- Browser can emit MSE/empty-src cleanup noise after `session.end`.
- Segment cadence/buffering may still produce tiny `waiting`/`canplay` transitions.

## Evidence Session

Session: `session:7ba63b1c-371a-40f2-855f-8451c1cfaaf1`

Important observed evidence:

- `renderMode=fmp4_av`
- `assistant.audio` was metadata-only / no browser AudioContext playback
- `fmp4_speech_enqueue_requested` happened
- `fmp4_enqueue_result` happened
- `avatar_fmp4_av_speech_started` happened
- `avatar_fmp4_av_speech_ended` happened
- `avatar_fmp4_av_speech_video_started_browser` happened
- `avatar_fmp4_av_speech_video_ended_browser` happened

## Future Cleanup Checklist

- De-dupe speech start events per turn in browser and/or bridge.
- Measure speech video end -> first idle segment displayed.
- Suppress MSE init/play attempts after `session.end` or `fmp4Cleanup()`.
- Avoid empty-src `MEDIA_ELEMENT_ERROR` during shutdown.
- Add explicit transition-gap telemetry.
- Optionally pre-buffer or prioritize one idle segment immediately after speech end.
- Keep all fixes bounded and do not re-architect the AV path unless hard evidence demands it.

## Next Active Roadmap Phase

These polish fixes are intentionally deferred. The next active phase is:

**Performance Baseline + OmniVoice Same-GPU Feasibility**
