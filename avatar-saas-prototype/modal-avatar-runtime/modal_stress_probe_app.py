"""
AVTR + OmniVoice Same-GPU Stress Probe.

Isolated probe to measure whether OmniVoice TTS can coexist with the AVTR
avatar renderer on the same L4 GPU without degrading real-time performance.

NOT wired into production. Measurement only.

Phases:
  1. Load AVTR pipeline under torch 2.8.0+cu128 (bail if fails).
  2. AVTR-alone baseline: 10 idle render chunks, measure avg/min/max render_ms.
  3. VRAM snapshot after AVTR loaded.
  4. Load OmniVoice model.
  5. VRAM snapshot with both loaded.
  6. OmniVoice-alone inference (2 texts), measure time/RTF/peak VRAM.
  7. Concurrent: AVTR render loop while OmniVoice inference runs (threading).
  8. Measure AVTR latency regression and OmniVoice slowdown.
  9. Cleanup OmniVoice, verify VRAM returns to AVTR-only levels.
  10. Return JSON results.

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_stress_probe_app.py

Run:
    modal run avatar-saas-prototype/modal-avatar-runtime/modal_stress_probe_app.py
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

_SCRIPT_DIR = Path(__file__).resolve().parent

try:
    _REPO_ROOT = _SCRIPT_DIR.parents[1]
    _SRC_DIR = _REPO_ROOT / "src"
    _SCRIPTS_DIR = _REPO_ROOT / "scripts"
    assert (_SRC_DIR / "avtr1_renderer").is_dir(), (
        f"Not in repo root: expected {_SRC_DIR / 'avtr1_renderer'}"
    )
except (IndexError, AssertionError):
    _SRC_DIR = Path("/nonexistent/src")
    _SCRIPTS_DIR = Path("/nonexistent/scripts")

L4_VOLUME_NAME = "avtr1-artifacts-l4"
L4_MOUNT_PATH = "/data/avtr1-l4"
HF_CACHE_PATH = "/root/.cache/huggingface"

OMNIVOICE_COMMIT = "f7f5040646f4fb8eef84e011adebec0c375ea6ae"
MODEL_ID = "k2-fsa/OmniVoice"

app = modal.App("avtr1-stress-probe")

l4_volume = modal.Volume.from_name(L4_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

stress_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "libglib2.0-0",
        "libgl1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ffmpeg",
        "git",
        "gcc",
        "g++",
    )
    .pip_install(
        "torch>=2.8,<2.9",
        "torchaudio>=2.8,<2.9",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "tensorrt<=10.12",
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install(
        "onnxruntime-gpu>=1.20",
        "numpy>=2.0,<3",
        "safetensors>=0.5,<1",
        "huggingface_hub>=0.30,<1",
        "kornia>=0.8.1,<0.9",
        "roma>=1.5.4,<2",
        "opencv-python-headless>=4.11,<5",
        "einops>=0.8.1,<0.9",
        "scikit-image>=0.25,<0.26",
        "soxr>=0.4,<0.5",
        "imageio>=2.37,<3",
        "attrs>=25.3,<26",
        "tenacity>=9.1,<10",
        "av>=17.0,<18",
        "soundfile>=0.13,<0.14",
        "httpx>=0.28,<0.29",
        "fastapi>=0.116,<0.117",
        "uvicorn>=0.35,<0.36",
        "python-multipart>=0.0.20,<0.0.21",
        "anyio>=4.11,<5",
        "tqdm>=4.67,<5",
        "onnx>=1.17,<2",
        "onnx-graphsurgeon>=0.5",
    )
    .pip_install(
        f"git+https://github.com/k2-fsa/OmniVoice.git@{OMNIVOICE_COMMIT}",
    )
    .add_local_dir(
        str(_SRC_DIR / "avtr1_renderer"),
        remote_path="/opt/avtr1/avtr1_renderer",
        copy=True,
    )
    .add_local_dir(
        str(_SCRIPTS_DIR),
        remote_path="/opt/avtr1/scripts",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/opt/avtr1:/opt/avtr1/scripts",
            "AVTR1_LOCAL_STORAGE": L4_MOUNT_PATH,
            "PYTHONUNBUFFERED": "1",
        }
    )
)


def _nvidia_smi_dict(label: str) -> dict:
    snap: dict = {"label": label}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                snap["smiMemoryUsedMb"] = int(parts[0])
                snap["smiMemoryTotalMb"] = int(parts[1])
                snap["smiGpuUtilPct"] = int(parts[2])
                snap["smiMemoryUtilPct"] = int(parts[3])
    except Exception:
        pass
    print(f"[smi] {label}: used={snap.get('smiMemoryUsedMb')}MB "
          f"total={snap.get('smiMemoryTotalMb')}MB util={snap.get('smiGpuUtilPct')}%")
    return snap


def _torch_mem_dict(label: str) -> dict:
    import torch
    return {
        "label": label,
        "torchAllocatedMb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "torchPeakMb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }


def _pick_first_bg(backgrounds: dict) -> str:
    for bg_id in sorted(backgrounds):
        if bg_id != "transparent":
            return bg_id
    raise RuntimeError("No non-transparent backgrounds registered")


def _pick_first_avatar(storage_root: Path) -> str:
    portraits_dir = storage_root / "avatars_artifacts" / "reference_frames"
    if not portraits_dir.is_dir():
        raise FileNotFoundError(f"No reference_frames at {portraits_dir}")
    all_pngs = sorted(portraits_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No portrait PNGs in {portraits_dir}")
    stems = [p.stem for p in all_pngs]
    if "arnold" in stems:
        return "arnold"
    return stems[0]


RENDERER_ENGINE_REL = "renderer_runtime_artifacts_cc"
RENDERER_ENGINES = [
    f"{RENDERER_ENGINE_REL}/decoder_b5_fp16.engine",
    f"{RENDERER_ENGINE_REL}/warp_network_b5_fp16.engine",
    f"{RENDERER_ENGINE_REL}/stitch_network_b5_fp16.engine",
    f"{RENDERER_ENGINE_REL}/modnet_b5_fp16.engine",
]
AVTR1_ENGINE_REL = "speech2motion_runtime_artifacts_cc"
AVTR1_ENGINES = [
    f"{AVTR1_ENGINE_REL}/hubert_lbs_fp16.engine",
    f"{AVTR1_ENGINE_REL}/avtr1_encode_fp16.engine",
    f"{AVTR1_ENGINE_REL}/avtr1_decode_fp16.engine",
]
NORMALIZER_FILE = "avtr1_normalizer.safetensors"
WARP_PLUGIN_REL = "renderer_runtime_artifacts/libgrid_sample_3d_plugin.so"
REQUIRED_ARTIFACT_RELS = (
    RENDERER_ENGINES + AVTR1_ENGINES + [NORMALIZER_FILE, WARP_PLUGIN_REL]
)


@app.function(
    image=stress_image,
    gpu="L4",
    timeout=900,
    scaledown_window=60,
    min_containers=0,
    max_containers=1,
    volumes={
        L4_MOUNT_PATH: l4_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
)
def run_stress_probe() -> str:
    import torch
    import numpy as np
    from avtr1_renderer.types import Chunk, RenderOptions
    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.avtr1_artifact_manager import (
        get_artifact_manager,
        get_storage_root,
    )

    results: dict = {
        "phase": "init",
        "torchVersion": torch.__version__,
        "torchCudaVersion": torch.version.cuda,
    }
    if torch.cuda.is_available():
        results["gpuName"] = torch.cuda.get_device_name(0)
        results["gpuCap"] = list(torch.cuda.get_device_capability(0))

    # ── Phase 1: Load AVTR pipeline ──────────────────────────────────────
    print("=" * 60)
    print("Phase 1: Load AVTR pipeline under torch 2.8.0+cu128")
    print("=" * 60)
    results["phase"] = "avtr_load"

    try:
        os.environ["AVTR1_LOCAL_STORAGE"] = str(L4_MOUNT_PATH)
        storage_root = get_storage_root()
        print(f"[stress] Storage root: {storage_root}")

        missing = [rel for rel in REQUIRED_ARTIFACT_RELS
                    if not (storage_root / rel).is_file()]
        if missing:
            print(f"[stress] Missing artifacts, running ensure_all_artifacts...")
            t0 = time.monotonic()
            mgr = get_artifact_manager()
            mgr.ensure_all_artifacts(workers=4)
            print(f"[stress] Artifacts ready in {time.monotonic() - t0:.0f}s")

        missing = [rel for rel in REQUIRED_ARTIFACT_RELS
                    if not (storage_root / rel).is_file()]
        if missing:
            results["avtrError"] = f"Missing artifacts after ensure: {missing[:3]}"
            results["phase"] = "avtr_load_failed"
            return json.dumps(results, default=str)

        avatar_id = _pick_first_avatar(storage_root)
        print(f"[stress] avatar_id={avatar_id}")

        t_load = time.monotonic()
        pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
        load_s = time.monotonic() - t_load
        results["avtrLoadTimeS"] = round(load_s, 3)
        print(f"[stress] AVTR pipeline loaded in {load_s:.1f}s")

        avatar = registry[avatar_id]
        bg_id = _pick_first_bg(pipeline._backgrounds)
        options = RenderOptions(pixel_format="yuv_i420", bg_id=bg_id)

        mg = pipeline._motion_generator
        expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        silence = np.zeros(expected_len, dtype=np.float32)
        chunk = Chunk(audio_speech=silence, audio_listen=silence)

        # Warmup
        print("[stress] Warmup render...")
        t_warm = time.monotonic()
        state, frames_iter = pipeline.process_chunk(
            avatar, chunk, state=None, options=options,
        )
        warmup_frames = list(frames_iter)
        warmup_ms = (time.monotonic() - t_warm) * 1000
        results["avtrWarmupMs"] = round(warmup_ms, 1)
        results["avtrWarmupFrames"] = len(warmup_frames)
        print(f"[stress] Warmup: {warmup_ms:.0f}ms, {len(warmup_frames)} frames")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        results["avtrError"] = str(exc)
        results["phase"] = "avtr_load_failed"
        print(f"[stress] AVTR FAILED under torch {torch.__version__}: {exc}")
        return json.dumps(results, default=str)

    print("[stress] AVTR loaded and rendering successfully under torch "
          f"{torch.__version__}+cu{torch.version.cuda}")

    # ── Phase 2: AVTR-alone baseline ─────────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 2: AVTR-alone render baseline (10 chunks)")
    print("=" * 60)
    results["phase"] = "avtr_baseline"
    torch.cuda.reset_peak_memory_stats()

    N_BASELINE = 10
    baseline_render_ms = []
    for i in range(N_BASELINE):
        t0 = time.monotonic()
        state, frames_iter = pipeline.process_chunk(
            avatar, chunk, state=state, options=options,
        )
        frames = list(frames_iter)
        ms = (time.monotonic() - t0) * 1000
        baseline_render_ms.append(ms)
        print(f"  chunk {i}: {ms:.1f}ms, {len(frames)} frames")

    results["avtrBaseline"] = {
        "chunks": N_BASELINE,
        "avgMs": round(sum(baseline_render_ms) / len(baseline_render_ms), 1),
        "minMs": round(min(baseline_render_ms), 1),
        "maxMs": round(max(baseline_render_ms), 1),
        "allMs": [round(m, 1) for m in baseline_render_ms],
    }
    print(f"  avg={results['avtrBaseline']['avgMs']}ms "
          f"min={results['avtrBaseline']['minMs']}ms "
          f"max={results['avtrBaseline']['maxMs']}ms")

    smi_avtr = _nvidia_smi_dict("avtr_baseline")
    torch_avtr = _torch_mem_dict("avtr_baseline")
    results["vramAfterAvtr"] = {
        "smi": smi_avtr,
        "torch": torch_avtr,
    }

    # ── Phase 3: Load OmniVoice ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 3: Load OmniVoice on same GPU")
    print("=" * 60)
    results["phase"] = "omnivoice_load"

    from omnivoice import OmniVoice

    torch.cuda.reset_peak_memory_stats()
    t_omni_load = time.monotonic()
    device = "cuda:0"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[stress] Loading OmniVoice device={device} dtype={dtype}")
    omni_model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
    )
    omni_load_s = time.monotonic() - t_omni_load
    results["omniLoadTimeS"] = round(omni_load_s, 3)
    omni_sr = int(getattr(omni_model, "sampling_rate", 24000))
    results["omniSamplingRate"] = omni_sr
    print(f"[stress] OmniVoice loaded in {omni_load_s:.1f}s, sr={omni_sr}")

    smi_both = _nvidia_smi_dict("both_loaded")
    torch_both = _torch_mem_dict("both_loaded")
    results["vramBothLoaded"] = {
        "smi": smi_both,
        "torch": torch_both,
    }
    total_mb = smi_both.get("smiMemoryTotalMb", 23034)
    used_mb = smi_both.get("smiMemoryUsedMb", 0)
    pct = round(used_mb / total_mb * 100, 1) if total_mb else 0
    results["vramBothPct"] = pct
    print(f"[stress] Combined VRAM: {used_mb}MB / {total_mb}MB = {pct}%")

    # ── Phase 4: OmniVoice-alone inference ───────────────────────────────
    print()
    print("=" * 60)
    print("Phase 4: OmniVoice-alone inference")
    print("=" * 60)
    results["phase"] = "omni_alone"

    omni_texts = [
        "Hello, how are you doing today?",
        "This is a longer test sentence to measure inference scalability with more tokens and see how the model handles extended utterances on this GPU.",
    ]

    omni_alone_results = []
    for i, text in enumerate(omni_texts):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            audio = omni_model.generate(
                text,
                preprocess_prompt=True,
                postprocess_output=True,
                denoise=True,
                t_shift=0.1,
                layer_penalty_factor=5.0,
                position_temperature=5.0,
                class_temperature=0.0,
            )
        elapsed = time.time() - t0
        audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
        audio_dur = audio_data.shape[-1] / omni_sr if hasattr(audio_data, 'shape') else 0
        rtf = audio_dur / elapsed if elapsed > 0 else 0
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        item = {
            "textLength": len(text),
            "inferenceS": round(elapsed, 3),
            "audioDurS": round(audio_dur, 3),
            "rtf": round(rtf, 2),
            "peakVramGb": round(peak_gb, 3),
        }
        omni_alone_results.append(item)
        print(f"  text {i} ({len(text)} chars): {elapsed:.3f}s, "
              f"{audio_dur:.2f}s audio, RTF={rtf:.2f}x, peak={peak_gb:.3f}GB")

    results["omniAlone"] = omni_alone_results

    # ── Phase 5: Concurrent stress test ──────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 5: Concurrent AVTR render + OmniVoice inference")
    print("=" * 60)
    results["phase"] = "concurrent"

    N_CONCURRENT_AVTR = 15
    concurrent_avtr_ms = []
    concurrent_errors: list[str] = []
    omni_concurrent_result: dict | None = None

    def avtr_render_thread():
        nonlocal state
        try:
            for i in range(N_CONCURRENT_AVTR):
                t0 = time.monotonic()
                state, frames_iter = pipeline.process_chunk(
                    avatar, chunk, state=state, options=options,
                )
                frames = list(frames_iter)
                ms = (time.monotonic() - t0) * 1000
                concurrent_avtr_ms.append(ms)
                print(f"  [avtr] chunk {i}: {ms:.1f}ms, {len(frames)} frames")
        except Exception as exc:
            concurrent_errors.append(f"avtr: {exc}")
            import traceback
            traceback.print_exc()

    def omnivoice_thread():
        nonlocal omni_concurrent_result
        try:
            text = omni_texts[1]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.inference_mode():
                audio = omni_model.generate(
                    text,
                    preprocess_prompt=True,
                    postprocess_output=True,
                    denoise=True,
                    t_shift=0.1,
                    layer_penalty_factor=5.0,
                    position_temperature=5.0,
                    class_temperature=0.0,
                )
            elapsed = time.time() - t0
            audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data)
            audio_dur = audio_data.shape[-1] / omni_sr if hasattr(audio_data, 'shape') else 0
            rtf = audio_dur / elapsed if elapsed > 0 else 0
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            omni_concurrent_result = {
                "textLength": len(text),
                "inferenceS": round(elapsed, 3),
                "audioDurS": round(audio_dur, 3),
                "rtf": round(rtf, 2),
                "peakVramGb": round(peak_gb, 3),
            }
            print(f"  [omni] concurrent: {elapsed:.3f}s, {audio_dur:.2f}s audio, "
                  f"RTF={rtf:.2f}x, peak={peak_gb:.3f}GB")
        except Exception as exc:
            concurrent_errors.append(f"omni: {exc}")
            import traceback
            traceback.print_exc()

    smi_before_concurrent = _nvidia_smi_dict("before_concurrent")

    t_avtr = threading.Thread(target=avtr_render_thread)
    t_omni = threading.Thread(target=omnivoice_thread)

    t_avtr.start()
    t_omni.start()
    t_avtr.join(timeout=300)
    t_omni.join(timeout=300)

    smi_after_concurrent = _nvidia_smi_dict("after_concurrent")

    # Compute AVTR regression
    avtr_baseline_avg = results["avtrBaseline"]["avgMs"]
    avtr_concurrent_avg = round(
        sum(concurrent_avtr_ms) / len(concurrent_avtr_ms), 1
    ) if concurrent_avtr_ms else 0
    avtr_regression_pct = round(
        (avtr_concurrent_avg - avtr_baseline_avg) / avtr_baseline_avg * 100, 1
    ) if avtr_baseline_avg > 0 else 0

    results["concurrentAvtr"] = {
        "chunks": N_CONCURRENT_AVTR,
        "avgMs": avtr_concurrent_avg,
        "minMs": round(min(concurrent_avtr_ms), 1) if concurrent_avtr_ms else 0,
        "maxMs": round(max(concurrent_avtr_ms), 1) if concurrent_avtr_ms else 0,
        "allMs": [round(m, 1) for m in concurrent_avtr_ms],
        "regressionPct": avtr_regression_pct,
    }

    # Compute OmniVoice slowdown
    omni_alone_ms = round(omni_alone_results[1]["inferenceS"] * 1000, 1)
    omni_concurrent_ms = round(
        omni_concurrent_result["inferenceS"] * 1000, 1
    ) if omni_concurrent_result else 0
    omni_slowdown_pct = round(
        (omni_concurrent_ms - omni_alone_ms) / omni_alone_ms * 100, 1
    ) if omni_alone_ms > 0 else 0

    results["concurrentOmni"] = omni_concurrent_result
    if omni_concurrent_result:
        results["concurrentOmni"]["slowdownPct"] = omni_slowdown_pct
        results["concurrentOmni"]["aloneMs"] = omni_alone_ms

    results["concurrentVram"] = {
        "before": smi_before_concurrent,
        "after": smi_after_concurrent,
    }
    results["concurrentErrors"] = concurrent_errors if concurrent_errors else None

    print(f"  AVTR regression: {avtr_regression_pct}% "
          f"({avtr_baseline_avg}ms -> {avtr_concurrent_avg}ms)")
    print(f"  OmniVoice slowdown: {omni_slowdown_pct}% "
          f"({omni_alone_ms}ms -> {omni_concurrent_ms}ms)")

    # ── Phase 6: Cleanup OmniVoice ───────────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 6: Cleanup OmniVoice, verify VRAM")
    print("=" * 60)
    results["phase"] = "cleanup"

    del omni_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    smi_cleanup = _nvidia_smi_dict("after_omni_cleanup")
    torch_cleanup = _torch_mem_dict("after_omni_cleanup")
    results["vramAfterCleanup"] = {
        "smi": smi_cleanup,
        "torch": torch_cleanup,
    }
    print(f"[stress] After OmniVoice cleanup: smi={smi_cleanup.get('smiMemoryUsedMb')}MB, "
          f"torch_alloc={torch_cleanup['torchAllocatedMb']}MB")

    # Verify AVTR still works after cleanup
    print("[stress] Verifying AVTR still renders after OmniVoice cleanup...")
    t_verify = time.monotonic()
    state, frames_iter = pipeline.process_chunk(
        avatar, chunk, state=state, options=options,
    )
    verify_frames = list(frames_iter)
    verify_ms = (time.monotonic() - t_verify) * 1000
    results["avtrPostCleanupRenderMs"] = round(verify_ms, 1)
    results["avtrPostCleanupFrames"] = len(verify_frames)
    print(f"[stress] Post-cleanup AVTR render: {verify_ms:.1f}ms, {len(verify_frames)} frames")

    # ── Final verdict ────────────────────────────────────────────────────
    results["phase"] = "complete"

    decision = {
        "avtrLoadedUnderTorch28": True,
        "oom": False,
        "cudaErrors": len(concurrent_errors) == 0,
        "peakVramPct": pct,
        "peakVramUnder80pct": pct < 80,
        "avtrRegressionPct": avtr_regression_pct,
        "avtrRegressionUnder15pct": abs(avtr_regression_pct) < 15,
        "omniRtfAlone": omni_alone_results[1]["rtf"] if len(omni_alone_results) > 1 else 0,
        "omniRtfConcurrent": omni_concurrent_result["rtf"] if omni_concurrent_result else 0,
    }
    results["decision"] = decision

    print()
    print("=" * 60)
    print("DECISION GATES")
    print("=" * 60)
    for k, v in decision.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {v} [{status}]")

    results["success"] = True
    return json.dumps(results, default=str)


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("AVTR + OmniVoice Same-GPU Stress Probe")
    print("=" * 60)
    raw = run_stress_probe.remote()
    result = json.loads(raw)

    print()
    print("=" * 60)
    print("STRESS PROBE RESULTS")
    print("=" * 60)

    print(f"  Phase:     {result.get('phase')}")
    print(f"  Success:   {result.get('success')}")
    print(f"  Torch:     {result.get('torchVersion')}+cu{result.get('torchCudaVersion')}")
    print(f"  GPU:       {result.get('gpuName')} cap={result.get('gpuCap')}")
    print()

    if result.get("avtrError"):
        print(f"  *** AVTR FAILED: {result['avtrError']}")
        print(f"  *** Cannot proceed with stress test.")
        return

    b = result.get("avtrBaseline", {})
    print(f"  AVTR baseline ({b.get('chunks')} chunks):")
    print(f"    avg={b.get('avgMs')}ms  min={b.get('minMs')}ms  max={b.get('maxMs')}ms")
    print()

    va = result.get("vramAfterAvtr", {}).get("smi", {})
    vb = result.get("vramBothLoaded", {}).get("smi", {})
    print(f"  VRAM after AVTR:  {va.get('smiMemoryUsedMb')}MB / {va.get('smiMemoryTotalMb')}MB")
    print(f"  VRAM both loaded: {vb.get('smiMemoryUsedMb')}MB / {vb.get('smiMemoryTotalMb')}MB "
          f"({result.get('vramBothPct')}%)")
    print()

    for i, o in enumerate(result.get("omniAlone", [])):
        print(f"  OmniVoice alone text {i}: {o.get('inferenceS')}s, "
              f"{o.get('audioDurS')}s audio, RTF={o.get('rtf')}x, peak={o.get('peakVramGb')}GB")
    print()

    ca = result.get("concurrentAvtr", {})
    co = result.get("concurrentOmni") or {}
    print(f"  Concurrent AVTR ({ca.get('chunks')} chunks):")
    print(f"    avg={ca.get('avgMs')}ms  regression={ca.get('regressionPct')}%")
    print(f"  Concurrent OmniVoice:")
    print(f"    {co.get('inferenceS')}s  RTF={co.get('rtf')}x  "
          f"slowdown={co.get('slowdownPct')}%  alone={co.get('aloneMs')}ms")
    print()

    vc = result.get("vramAfterCleanup", {}).get("smi", {})
    print(f"  VRAM after OmniVoice cleanup: {vc.get('smiMemoryUsedMb')}MB")
    print(f"  AVTR post-cleanup render: {result.get('avtrPostCleanupRenderMs')}ms")
    print()

    d = result.get("decision", {})
    print("  DECISION GATES:")
    for k, v in d.items():
        tag = "PASS" if v else "*** FAIL ***"
        print(f"    {k}: {v}  [{tag}]")

    if result.get("concurrentErrors"):
        print(f"  ERRORS: {result['concurrentErrors']}")
