#!/usr/bin/env python3
"""AV Baseline Report — summarizes PostHog events for a session.

Usage:
    python3 avatar-saas-prototype/scripts/av-baseline-report.py <session-uuid>
    python3 avatar-saas-prototype/scripts/av-baseline-report.py 7ba63b1c-371a-40f2-855f-8451c1cfaaf1

Reads .env.posthog from project root for credentials.
"""

import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def load_env():
    env_file = Path(__file__).resolve().parents[2] / ".env.posthog"
    if env_file.is_file():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("POSTHOG_PERSONAL_API_KEY", "")
    project_id = os.environ.get("POSTHOG_PROJECT_ID", "")
    api_host = os.environ.get("POSTHOG_API_HOST", "https://eu.posthog.com")
    if not api_key or not project_id:
        print("ERROR: Set POSTHOG_PERSONAL_API_KEY, POSTHOG_PROJECT_ID in .env.posthog")
        sys.exit(1)
    return api_key, project_id, api_host


def hogql_query(api_host: str, api_key: str, project_id: str, query: str) -> list:
    url = f"{api_host}/api/projects/{project_id}/query/"
    payload = json.dumps({"query": {"kind": "HogQLQuery", "query": query}}).encode()
    req = Request(url, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("results", [])
    except HTTPError as e:
        print(f"ERROR: PostHog API returned {e.code}")
        print(e.read().decode()[:500])
        return []


def fetch_events(api_host: str, api_key: str, project_id: str, distinct_id: str, limit: int = 500) -> list:
    query = (
        f"SELECT timestamp, event, "
        f"JSONExtractString(properties, 'turnId') as turnId, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'latencyMs')) as latencyMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'audioDurationMs')) as audioDurationMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'enqueueCallDurationMs')) as enqueueCallDurationMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'enqueueToSpeechStartMs')) as enqueueToSpeechStartMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'timeToInitMs')) as timeToInitMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'timeSinceSessionReadyMs')) as timeSinceSessionReadyMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'chunkRenderMs')) as chunkRenderMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'speechEndToIdleResumeMs')) as speechEndToIdleResumeMs, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'gpuSmiUsedMaxMb')) as gpuSmiUsedMaxMb, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'gpuSmiTotalMb')) as gpuSmiTotalMb, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'gpuTorchPeakMb')) as gpuTorchPeakMb, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'gpuTorchAllocMaxMb')) as gpuTorchAllocMaxMb, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'ffmpegReturnCode')) as ffmpegReturnCode, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'ffmpegStderrErrorCount')) as ffmpegStderrErrorCount, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'waitingCount')) as waitingCount, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'stalledCount')) as stalledCount, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'segmentsForwarded')) as segmentsForwarded, "
        f"toFloatOrDefault(JSONExtractRaw(properties, 'encodedBytes')) as encodedBytes, "
        f"toString(JSONExtractRaw(properties, 'renderMode')) as renderMode, "
        f"toString(JSONExtractRaw(properties, 'mode')) as mode, "
        f"toString(JSONExtractRaw(properties, 'provider')) as provider, "
        f"toString(JSONExtractRaw(properties, 'error')) as error "
        f"FROM events WHERE distinct_id = '{distinct_id}' "
        f"ORDER BY timestamp ASC LIMIT {limit}"
    )
    return hogql_query(api_host, api_key, project_id, query)


def fetch_event_counts(api_host: str, api_key: str, project_id: str, distinct_id: str) -> list:
    query = (
        f"SELECT event, count() as cnt FROM events "
        f"WHERE distinct_id = '{distinct_id}' "
        f"GROUP BY event ORDER BY cnt DESC"
    )
    return hogql_query(api_host, api_key, project_id, query)


def fmt_ms(val) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
        if v == 0:
            return "0ms"
        return f"{v:.0f}ms"
    except (ValueError, TypeError):
        return "—"


def fmt_mb(val) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.0f}MB"
    except (ValueError, TypeError):
        return "—"


def generate_report(session_id: str, api_key: str, project_id: str, api_host: str):
    distinct_id = f"session:{session_id}"
    print(f"{'=' * 70}")
    print(f"AV BASELINE REPORT")
    print(f"{'=' * 70}")
    print(f"Session ID : {session_id}")
    print(f"Distinct ID: {distinct_id}")
    print()

    events = fetch_events(api_host, api_key, project_id, distinct_id)
    if not events:
        print("No events found for this session.")
        return

    print(f"Total events fetched: {len(events)}")
    print()

    render_mode = None
    session_ready_at = None
    session_ended_at = None
    session_ready_latency = None

    stt_latencies = {}
    llm_latencies = {}
    tts_latencies = {}
    tts_provider = {}
    enqueue_durations = {}
    enqueue_to_speech_start = {}
    speech_video_start_at = {}
    speech_video_end_at = {}
    speech_end_to_idle = {}
    mse_ready_time_since_session = None
    first_frame_time = None
    waiting_count = 0
    stalled_count = 0
    gpu_smi_max = None
    gpu_smi_total = None
    gpu_torch_peak = None
    gpu_torch_alloc_max = None
    ffmpeg_rc = None
    ffmpeg_errors = None
    total_segments = None
    encoded_bytes = None
    errors = []

    for row in events:
        ts, event = row[0], row[1]
        turn_id = row[2] if len(row) > 2 else None
        latency_ms = row[3] if len(row) > 3 else None
        audio_dur = row[4] if len(row) > 4 else None
        enqueue_dur = row[5] if len(row) > 5 else None
        enqueue_to_speech = row[6] if len(row) > 6 else None
        time_to_init = row[7] if len(row) > 7 else None
        time_since_ready = row[8] if len(row) > 8 else None
        chunk_render = row[9] if len(row) > 9 else None
        speech_end_idle = row[10] if len(row) > 10 else None
        smi_max = row[11] if len(row) > 11 else None
        smi_total = row[12] if len(row) > 12 else None
        torch_peak = row[13] if len(row) > 13 else None
        torch_alloc_max = row[14] if len(row) > 14 else None
        ff_rc = row[15] if len(row) > 15 else None
        ff_err = row[16] if len(row) > 16 else None
        wait_cnt = row[17] if len(row) > 17 else None
        stall_cnt = row[18] if len(row) > 18 else None
        segs = row[19] if len(row) > 19 else None
        enc_bytes = row[20] if len(row) > 20 else None
        r_mode = row[21] if len(row) > 21 else None
        evt_mode = row[22] if len(row) > 22 else None
        prov = row[23] if len(row) > 23 else None
        evt_error = row[24] if len(row) > 24 else None

        if event == "session_ready" or event == "session_ready_received":
            session_ready_at = ts
            session_ready_latency = latency_ms
            if r_mode and r_mode not in ("", "null"):
                render_mode = r_mode.strip('"')

        if event == "session_ended" or event == "session_ended_received":
            session_ended_at = ts

        if event == "stt_completed" and turn_id:
            stt_latencies[turn_id] = latency_ms

        if event == "llm_completed" and turn_id:
            llm_latencies[turn_id] = latency_ms

        if event == "tts_completed" and turn_id:
            tts_latencies[turn_id] = latency_ms
            if prov and prov not in ("", "null"):
                tts_provider[turn_id] = prov.strip('"')

        if event == "fmp4_enqueue_result" and turn_id:
            enqueue_durations[turn_id] = enqueue_dur

        if event == "avatar_fmp4_av_speech_first_segment_latency" and turn_id:
            enqueue_to_speech_start[turn_id] = enqueue_to_speech

        if event == "avatar_fmp4_av_speech_started" and turn_id:
            speech_video_start_at[turn_id] = ts

        if event == "avatar_fmp4_av_speech_ended" and turn_id:
            speech_video_end_at[turn_id] = ts

        if event == "avatar_fmp4_av_segment_forwarded" and speech_end_idle is not None:
            speech_end_to_idle.setdefault("value", speech_end_idle)

        if event == "avatar_fmp4_mse_ready" and time_since_ready is not None:
            mse_ready_time_since_session = time_since_ready

        if event == "avatar_fmp4_first_frame" and time_since_ready is not None:
            first_frame_time = time_since_ready

        if wait_cnt is not None:
            try:
                waiting_count = max(waiting_count, int(float(wait_cnt)))
            except (ValueError, TypeError):
                pass
        if stall_cnt is not None:
            try:
                stalled_count = max(stalled_count, int(float(stall_cnt)))
            except (ValueError, TypeError):
                pass

        if event == "avatar_fmp4_av_stream_summary":
            if smi_max is not None:
                gpu_smi_max = float(smi_max)
            if smi_total is not None:
                gpu_smi_total = float(smi_total)
            if torch_peak is not None:
                gpu_torch_peak = float(torch_peak)
            if torch_alloc_max is not None:
                gpu_torch_alloc_max = float(torch_alloc_max)
            if ff_rc is not None:
                ffmpeg_rc = int(float(ff_rc))
            if ff_err is not None:
                ffmpeg_errors = int(float(ff_err))
            if segs is not None:
                total_segments = int(float(segs))
            if enc_bytes is not None:
                encoded_bytes = int(float(enc_bytes))

        if event in ("pipeline_error", "avatar_fmp4_av_stream_failed",
                      "avatar_fmp4_mse_error", "stt_failed", "llm_failed",
                      "tts_failed", "fmp4_enqueue_failed"):
            errors.append(f"{ts} {event}: {evt_error}")

    session_duration = "—"
    if session_ready_at and session_ended_at:
        try:
            from datetime import datetime
            t1 = datetime.fromisoformat(session_ready_at.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(session_ended_at.replace("Z", "+00:00"))
            session_duration = f"{(t2 - t1).total_seconds():.1f}s"
        except Exception:
            pass

    print(f"{'─' * 70}")
    print(f"SESSION OVERVIEW")
    print(f"{'─' * 70}")
    print(f"  Render Mode       : {render_mode or 'unknown'}")
    print(f"  Session Duration  : {session_duration}")
    print(f"  Session Ready Lat : {fmt_ms(session_ready_latency)}")
    print()

    turns = sorted(set(
        list(stt_latencies.keys()) + list(llm_latencies.keys()) +
        list(tts_latencies.keys()) + list(enqueue_durations.keys()) +
        list(speech_video_start_at.keys())
    ))

    if turns:
        print(f"{'─' * 70}")
        print(f"PER-TURN LATENCIES")
        print(f"{'─' * 70}")
        print(f"  {'Turn':<8} {'STT':>8} {'LLM':>8} {'TTS':>8} {'Enqueue':>10} {'Enq→Start':>12} {'Provider':>10}")
        print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*12} {'─'*10}")
        for t in turns:
            stt = fmt_ms(stt_latencies.get(t))
            llm = fmt_ms(llm_latencies.get(t))
            tts = fmt_ms(tts_latencies.get(t))
            enq = fmt_ms(enqueue_durations.get(t))
            e2s = fmt_ms(enqueue_to_speech_start.get(t))
            prov = tts_provider.get(t, "—")
            print(f"  {t:<8} {stt:>8} {llm:>8} {tts:>8} {enq:>10} {e2s:>12} {prov:>10}")

        for t in turns:
            start = speech_video_start_at.get(t)
            end = speech_video_end_at.get(t)
            if start and end:
                try:
                    from datetime import datetime
                    t1 = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    dur_ms = (t2 - t1).total_seconds() * 1000
                    print(f"\n  Turn {t} speech video duration: {dur_ms:.0f}ms")
                except Exception:
                    pass
        print()

    print(f"{'─' * 70}")
    print(f"MSE / BROWSER TIMING")
    print(f"{'─' * 70}")
    print(f"  Session Ready → MSE Ready : {fmt_ms(mse_ready_time_since_session)}")
    print(f"  Session Ready → First Frame: {fmt_ms(first_frame_time)}")
    print(f"  Waiting events (cumulative): {waiting_count}")
    print(f"  Stalled events (cumulative): {stalled_count}")
    print()

    print(f"{'─' * 70}")
    print(f"SPEECH → IDLE TRANSITION")
    print(f"{'─' * 70}")
    if "value" in speech_end_to_idle:
        print(f"  Speech end → idle resume : {fmt_ms(speech_end_to_idle['value'])}")
    else:
        print(f"  Speech end → idle resume : — (not captured in this session)")
    print()

    print(f"{'─' * 70}")
    print(f"GPU MEMORY")
    print(f"{'─' * 70}")
    print(f"  nvidia-smi peak used   : {fmt_mb(gpu_smi_max)}")
    print(f"  nvidia-smi total       : {fmt_mb(gpu_smi_total)}")
    print(f"  torch peak allocated   : {fmt_mb(gpu_torch_peak)}")
    print(f"  torch max allocated    : {fmt_mb(gpu_torch_alloc_max)}")
    if gpu_smi_max and gpu_smi_total:
        pct = (float(gpu_smi_max) / float(gpu_smi_total)) * 100
        print(f"  Peak utilization       : {pct:.1f}%")
    print()

    print(f"{'─' * 70}")
    print(f"FFMPEG / STREAM HEALTH")
    print(f"{'─' * 70}")
    print(f"  FFmpeg return code     : {ffmpeg_rc if ffmpeg_rc is not None else '—'}")
    print(f"  FFmpeg stderr errors   : {ffmpeg_errors if ffmpeg_errors is not None else '—'}")
    print(f"  Total segments         : {total_segments if total_segments is not None else '—'}")
    if encoded_bytes is not None:
        print(f"  Total encoded bytes    : {encoded_bytes:,}")
    print()

    if errors:
        print(f"{'─' * 70}")
        print(f"ERRORS ({len(errors)})")
        print(f"{'─' * 70}")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        print()

    counts = fetch_event_counts(api_host, api_key, project_id, distinct_id)
    if counts:
        print(f"{'─' * 70}")
        print(f"EVENT VOLUME")
        print(f"{'─' * 70}")
        for row in counts:
            print(f"  {row[1]:>5}  {row[0]}")
        print()

    print(f"{'=' * 70}")
    print(f"END OF REPORT")
    print(f"{'=' * 70}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_id = sys.argv[1]
    api_key, project_id, api_host = load_env()
    generate_report(session_id, api_key, project_id, api_host)


if __name__ == "__main__":
    main()
