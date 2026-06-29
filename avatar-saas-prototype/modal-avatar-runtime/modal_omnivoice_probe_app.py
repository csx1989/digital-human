"""
OmniVoice GPU Probe for AVTR-1.

Standalone probe app to test OmniVoice TTS on the same Modal L4 GPU type
used by the AVTR avatar renderer. Completely isolated from the production
avtr1-lifecycle app.

Reports:
- Cold import time
- Model load time
- VRAM before/after load
- Peak VRAM during inference
- Inference time and real-time factor
- Output audio format/sample rate/duration
- Cleanup behavior
- Scale-to-zero behavior

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_omnivoice_probe_app.py

Run:
    modal run avatar-saas-prototype/modal-avatar-runtime/modal_omnivoice_probe_app.py
"""

from __future__ import annotations

import gc
import io
import json
import subprocess
import time

import modal

OMNIVOICE_COMMIT = "f7f5040646f4fb8eef84e011adebec0c375ea6ae"
MODEL_ID = "k2-fsa/OmniVoice"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App("avtr1-omnivoice-probe")

probe_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ffmpeg", "git", "gcc", "g++")
    .pip_install(
        "torch>=2.8,<2.9",
        "torchaudio>=2.8,<2.9",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy>=2.0,<3",
        "soundfile",
        "huggingface-hub>=0.30",
        f"git+https://github.com/k2-fsa/OmniVoice.git@{OMNIVOICE_COMMIT}",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)


def _nvidia_smi(label: str) -> dict:
    snap: dict = {"label": label}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                snap["smiMemoryUsedMb"] = int(parts[0])
                snap["smiMemoryTotalMb"] = int(parts[1])
                snap["smiGpuUtilPct"] = int(parts[2])
    except Exception:
        pass
    return snap


def _torch_mem(label: str) -> dict:
    import torch
    return {
        "label": label,
        "torchAllocatedMb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "torchPeakMb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }


@app.function(
    image=probe_image,
    gpu="L4",
    timeout=600,
    scaledown_window=60,
    min_containers=0,
    max_containers=1,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
)
def run_omnivoice_probe() -> dict:
    import torch
    import torchaudio
    import numpy as np

    results: dict = {
        "gpu": "L4",
        "image": "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        "torchVersion": torch.__version__,
        "torchCudaVersion": torch.version.cuda,
        "torchaudioVersion": torchaudio.__version__,
    }
    if torch.cuda.is_available():
        results["gpuNameTorch"] = torch.cuda.get_device_name(0)
        results["gpuCapTorch"] = torch.cuda.get_device_capability(0)

    smi_before = _nvidia_smi("before_import")
    results["smiBefore"] = smi_before

    import_start = time.time()
    from omnivoice import OmniVoice
    import_elapsed = time.time() - import_start
    results["coldImportTimeS"] = round(import_elapsed, 3)
    print(f"[probe] OmniVoice import: {import_elapsed:.3f}s")

    torch.cuda.reset_peak_memory_stats()
    mem_before_load = _torch_mem("before_load")
    smi_before_load = _nvidia_smi("before_load")
    results["memBeforeLoad"] = mem_before_load
    results["smiBeforeLoad"] = smi_before_load

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    results["device"] = device
    results["gpuName"] = gpu_name

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    results["dtype"] = str(dtype)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    load_start = time.time()
    print(f"[probe] Loading model device={device} dtype={dtype}")
    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=dtype,
        attn_implementation="eager" if torch.cuda.is_available() else None,
    )
    load_elapsed = time.time() - load_start
    results["modelLoadTimeS"] = round(load_elapsed, 3)
    print(f"[probe] Model loaded: {load_elapsed:.3f}s")

    sampling_rate = int(getattr(model, "sampling_rate", 24000))
    results["samplingRate"] = sampling_rate

    mem_after_load = _torch_mem("after_load")
    smi_after_load = _nvidia_smi("after_load")
    results["memAfterLoad"] = mem_after_load
    results["smiAfterLoad"] = smi_after_load

    test_texts = [
        "Hello, how are you doing today?",
        "This is a longer test sentence to measure inference scalability with more tokens.",
    ]

    inference_results = []
    for i, text in enumerate(test_texts):
        torch.cuda.reset_peak_memory_stats()
        gen_start = time.time()
        with torch.inference_mode():
            audio = model.generate(
                text,
                preprocess_prompt=True,
                postprocess_output=True,
                denoise=True,
                t_shift=0.1,
                layer_penalty_factor=5.0,
                position_temperature=5.0,
                class_temperature=0.0,
            )
        gen_elapsed = time.time() - gen_start

        audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
        audio_duration_s = audio_data.shape[-1] / sampling_rate if hasattr(audio_data, 'shape') else 0
        rtf = audio_duration_s / gen_elapsed if gen_elapsed > 0 else 0
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

        output_buf = io.BytesIO()
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        torchaudio.save(output_buf, audio_data.cpu(), sampling_rate, format="wav")
        wav_bytes = output_buf.getvalue()

        result_item = {
            "textLength": len(text),
            "inferenceTimeS": round(gen_elapsed, 3),
            "audioDurationS": round(audio_duration_s, 3),
            "realTimeFactor": round(rtf, 2),
            "peakVramGb": round(peak_mem_gb, 3),
            "outputFormat": "wav",
            "outputSampleRate": sampling_rate,
            "outputBytes": len(wav_bytes),
        }
        inference_results.append(result_item)
        print(f"[probe] Text {i}: {gen_elapsed:.3f}s, {audio_duration_s:.2f}s audio, "
              f"RTF={rtf:.2f}x, peak={peak_mem_gb:.3f}GB")

    results["inference"] = inference_results

    smi_after_gen = _nvidia_smi("after_gen")
    results["smiAfterGen"] = smi_after_gen

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    mem_after_cleanup = _torch_mem("after_cleanup")
    smi_after_cleanup = _nvidia_smi("after_cleanup")
    results["memAfterCleanup"] = mem_after_cleanup
    results["smiAfterCleanup"] = smi_after_cleanup
    print(f"[probe] Cleanup done, torch alloc={mem_after_cleanup['torchAllocatedMb']}MB")

    results["success"] = True
    return json.dumps(results, default=str)


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("OmniVoice Probe — L4 GPU")
    print("=" * 60)
    result = json.loads(run_omnivoice_probe.remote())
    print()
    print("=" * 60)
    print("PROBE RESULTS")
    print("=" * 60)

    for key in ["gpu", "gpuName", "gpuNameTorch", "device", "dtype", "success"]:
        print(f"  {key}: {result.get(key)}")
    print()
    print(f"  torch version:    {result.get('torchVersion')}")
    print(f"  torch CUDA:       {result.get('torchCudaVersion')}")
    print(f"  torchaudio:       {result.get('torchaudioVersion')}")
    if result.get("gpuCapTorch"):
        print(f"  GPU capability:   {result.get('gpuCapTorch')}")
    print()
    print(f"  Cold import time: {result.get('coldImportTimeS')}s")
    print(f"  Model load time:  {result.get('modelLoadTimeS')}s")
    print(f"  Sampling rate:    {result.get('samplingRate')}Hz")
    print()

    if "smiBeforeLoad" in result:
        b = result["smiBeforeLoad"]
        a = result["smiAfterLoad"]
        print(f"  VRAM before load: {b.get('smiMemoryUsedMb')}MB / {b.get('smiMemoryTotalMb')}MB")
        print(f"  VRAM after load:  {a.get('smiMemoryUsedMb')}MB / {a.get('smiMemoryTotalMb')}MB")
        delta = (a.get('smiMemoryUsedMb', 0) or 0) - (b.get('smiMemoryUsedMb', 0) or 0)
        print(f"  VRAM delta:       {delta}MB")
    print()

    if "memBeforeLoad" in result:
        b = result["memBeforeLoad"]
        a = result["memAfterLoad"]
        print(f"  Torch alloc before: {b.get('torchAllocatedMb')}MB")
        print(f"  Torch alloc after:  {a.get('torchAllocatedMb')}MB")
        print(f"  Torch peak after:   {a.get('torchPeakMb')}MB")
    print()

    for i, inf in enumerate(result.get("inference", [])):
        print(f"  Inference {i}:")
        print(f"    Text length:     {inf.get('textLength')} chars")
        print(f"    Inference time:  {inf.get('inferenceTimeS')}s")
        print(f"    Audio duration:  {inf.get('audioDurationS')}s")
        print(f"    Real-time factor: {inf.get('realTimeFactor')}x")
        print(f"    Peak VRAM:       {inf.get('peakVramGb')}GB")
        print(f"    Output:          {inf.get('outputFormat')} @ {inf.get('outputSampleRate')}Hz, {inf.get('outputBytes')}B")
        print()

    if "smiAfterCleanup" in result:
        c = result["smiAfterCleanup"]
        print(f"  VRAM after cleanup: {c.get('smiMemoryUsedMb')}MB / {c.get('smiMemoryTotalMb')}MB")
    if "memAfterCleanup" in result:
        c = result["memAfterCleanup"]
        print(f"  Torch alloc after cleanup: {c.get('torchAllocatedMb')}MB")
    print()
    print("=" * 60)
