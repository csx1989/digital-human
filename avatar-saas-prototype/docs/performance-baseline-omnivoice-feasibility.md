# Performance Baseline + OmniVoice Feasibility

## Measurement Goals

We are measuring the current fMP4 AV pipeline to establish a performance baseline before evaluating whether OmniVoice TTS can run on the same Modal L4 GPU as the AVTR avatar renderer.

Key questions:
1. What is the current GPU, CPU, and memory usage during idle and speech rendering?
2. What are the end-to-end turn latencies at each pipeline stage?
3. How much VRAM headroom exists on the L4 GPU?
4. Can OmniVoice (2.3 GB peak VRAM on L4) coexist with AVTR (3.5 GB) without degrading the real-time avatar experience?

## Current Architecture Summary

```
Browser → WebSocket → Cloudflare DO → SSE → Modal L4 GPU
                                    ↕
                              VAD / Smart Turn / STT / LLM / TTS
```

- **Modal GPU**: L4 (24 GB VRAM), CUDA 12.8.1, PyTorch 2.5-2.8, FFmpeg 6.1.1 with NVENC
- **AVTR renderer**: p95=188ms per 5-frame chunk (0.2s), ~3.5 GB VRAM
- **FFmpeg**: h264_nvenc + AAC-LC, fMP4 muxed A/V stream, 1280x720 @ 25fps
- **Transport**: SSE from Modal → DO → WebSocket → Browser MSE SourceBuffer

## Instrumentation Added (Phase A)

### Modal GPU (`modal_lifecycle_app.py`)
- `_gpu_memory_snapshot()`: Dual measurement via `torch.cuda.memory_allocated()` + `nvidia-smi`
- GPU snapshots at: stream_start, fmp4_init, speech_dequeue, every 25 chunks (~5s), speech_end, first_idle_after_speech, stream_end
- `speechEndToIdleResumeMs` timing: last speech chunk → first idle chunk
- FFmpeg health in SSE end event: return code, stderr error count, bytes, GPU summary

### Bridge DO (`avatar-session-do.ts`)
- `enqueueCallDurationMs`: time the enqueue HTTP call to control endpoint
- `avatar_fmp4_av_speech_first_segment_latency`: bridge-level enqueue→speech start
- GPU stats forwarding from SSE end event via `avatar_fmp4_av_stream_summary`
- `speechEndToIdleResumeMs` forwarded in sampled segment events
- Cleanup tracking for enqueue correlation maps

### Browser (`main.ts`)
- `timeSinceSessionReadyMs` in MSE ready event
- `avatar_fmp4_first_frame`: first `playing` event after MSE ready
- `waitingCount` / `stalledCount` in segment summaries

### New PostHog Events
| Event | Source | Frequency |
|-------|--------|-----------|
| `avatar_fmp4_av_stream_summary` | Bridge | Per stream (session end) |
| `avatar_fmp4_av_speech_first_segment_latency` | Bridge | Per speech turn |
| `avatar_fmp4_first_frame` | Browser | Once per session |

## Preliminary Baseline (Pre-Instrumentation)

Session: `session:7ba63b1c-371a-40f2-855f-8451c1cfaaf1`

| Metric | Value |
|--------|-------|
| Render mode | fmp4_av |
| Session duration | 100.6s |
| STT latency (turn 1 / turn 2) | 539ms / 380ms |
| LLM latency (turn 1 / turn 2) | 1056ms / 894ms |
| TTS latency (turn 1 / turn 2) | 765ms / 894ms |
| TTS provider | sarvam |
| Speech video duration (turn 1) | 2925ms |
| Total events | 218 |
| Video events | 38 |
| Segment summaries | 20 |
| Segment forwards (sampled) | 48 |
| MSE ready count | 2 (init + retry) |
| Speech turns with video | 4 started, 1 ended event |
| GPU memory | Not captured (pre-instrumentation) |
| FFmpeg health | Not captured (pre-instrumentation) |
| Enqueue call duration | Not captured (pre-instrumentation) |
| Speech→idle transition | Not captured (pre-instrumentation) |

**Note**: This session was captured before Phase A instrumentation. GPU memory, FFmpeg health, enqueue timing, and speech→idle transition data will be available from the next instrumented session.

## Official Baseline (Post-Instrumentation)

_Session pending — to be filled after deployment and fresh session._

Session ID: _pending_
Date: _pending_

| Metric | Value |
|--------|-------|
| Render mode | — |
| Session duration | — |
| Session ready latency | — |
| STT latency per turn | — |
| LLM latency per turn | — |
| TTS latency per turn | — |
| TTS completed → enqueue result | — |
| Enqueue call duration | — |
| Enqueue result → speech video start | — |
| Speech video duration | — |
| Speech end → idle resume | — |
| MSE ready time (since session ready) | — |
| First frame time (since session ready) | — |
| Waiting events | — |
| Stalled events | — |
| GPU nvidia-smi peak | — |
| GPU nvidia-smi total | — |
| GPU torch peak | — |
| FFmpeg return code | — |
| FFmpeg stderr errors | — |
| Total segments | — |
| Total encoded bytes | — |
| Event volume | — |

## OmniVoice Research Findings (Phase C)

### Location
- **Full implementation**: `/Users/ayya/developer/my-toy/omnivoice-modal/`
- **Bridge integration**: `/Users/ayya/developer/my-toy/cloudflare_toy_agent/any-toy-agent/src/pcm-tts.ts` → `#synthesizeOmniVoice()`
- **Deployed endpoint**: `https://vallangirakesh--omnivoice-tts-tts-api.modal.run`

### Model Details
| Property | Value |
|----------|-------|
| Model ID | `k2-fsa/OmniVoice` (HuggingFace) |
| Git commit | `f7f5040646f4fb8eef84e011adebec0c375ea6ae` |
| Install | `pip install git+https://github.com/k2-fsa/OmniVoice.git@{commit}` |
| Precision | BF16 (TF32 matmul/cudnn enabled) |
| torch.compile | Skipped for GPU snapshot stability |

### Performance on L4 (from benchmarks)
| Metric | Value |
|--------|-------|
| Model load time | 18.125s (one-time, mitigated by snapshots) |
| Peak VRAM during generation | 2.254 GB |
| Inference time (32 steps, ~5.35s audio) | 1.086s |
| Real-time factor | ~4.9x |
| Output format | WAV at 24kHz |
| Streaming support | No — full-utterance only |
| Sample rate | 24,000 Hz (needs resampling to 16kHz) |

### VRAM Feasibility Estimate
| Component | VRAM |
|-----------|------|
| AVTR renderer (idle) | ~3.5 GB |
| OmniVoice (peak during generation) | ~2.3 GB |
| **Combined estimate** | **~5.8 GB** |
| L4 total | 24 GB |
| **Headroom** | **~18.2 GB (76%)** |

### Compatibility
- AVTR image: `nvidia/cuda:12.8.1-devel-ubuntu24.04` + PyTorch 2.5-2.8
- OmniVoice image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- CUDA versions are nearly identical — good sign for same-GPU coexistence
- OmniVoice requires additional pip packages: `soundfile`, `huggingface-hub`
- OmniVoice needs `attn_implementation="eager"` — avoids Flash Attention

### Dependencies
```
git+https://github.com/k2-fsa/OmniVoice.git@f7f5040646f4fb8eef84e011adebec0c375ea6ae
huggingface-hub>=0.30
soundfile
```

### Input/Output Format
- **Input**: JSON `{ text, language, num_step: 32, guidance_scale: 2.0, speed: 1.0 }`
- **Output**: WAV binary at 24kHz (response body)
- **Language mapping**: ISO 639-1 → human-readable name (English, Hindi, Bengali, Tamil, Telugu, etc.)
- **Resampling**: 24kHz → 16kHz linear interpolation in bridge pcm-tts.ts

## Probe Results (Phase D)

_Pending — to be filled after probe deployment._

## Decision Criteria

Same-GPU OmniVoice is viable only if ALL of the following are met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| No OOM | Zero OOM events | Hard blocker |
| Combined VRAM | < 80% of L4's 24GB (~19.2GB) | Safety headroom for CUDA, FFmpeg, OS |
| OmniVoice load time | < 30s | Acceptable one-time cost with snapshots |
| TTS real-time factor | ≥ 3x (generate 3s audio in ≤1s) | Conversational use; L4 benchmark shows 4.9x |
| AVTR render latency regression | < 15% p95 increase | Must not degrade user experience |
| End-to-end turn latency | < 5s total (STT+LLM+TTS+enqueue+first frame) | Conversational acceptability |
| Scale-to-zero | Still works | No zombie containers |
| No thread/process leaks | Confirmed | Hard blocker |
| No production AV path regression | Confirmed | Hard blocker |

### If Same-GPU Fails
- Report blocker: VRAM / inference speed / dependency conflict / startup time / integration complexity
- Recommend one of:
  1. Same GPU integration (if viable)
  2. Separate Modal GPU/container for OmniVoice
  3. Keep current external TTS provider (Sarvam → MeloTTS)

## Next Steps After This Phase

1. Deploy Phase A instrumentation (Modal + Bridge + Browser)
2. Run fresh instrumented baseline session
3. Populate official baseline results table
4. Deploy OmniVoice probe (`modal_omnivoice_probe_app.py`)
5. Run probe, verify VRAM + inference timing
6. If probe passes, run AVTR+OmniVoice stress probe
7. Report recommendation
