"""
OmniVoice TTS endpoint for AVTR-1 browser-test integration.

Separate Modal app on its own L4 GPU container. The Bridge Worker calls this
endpoint when ENABLE_OMNIVOICE_TTS_ON_L4="true". On failure or timeout, the
Bridge falls back to Sarvam automatically.

Architecture: separate-GPU
  - OmniVoice runs in avtr1-omnivoice-tts on its own L4.
  - AVTR runs in avtr1-lifecycle on its own L4.
  - This test validates OmniVoice UX/latency/voice quality, NOT same-GPU contention.

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_omnivoice_tts_app.py

Smoke test:
    curl -X POST <web_endpoint_url> \\
      -H "Content-Type: application/json" \\
      -d '{"text": "Hello, how are you?"}'
"""

from __future__ import annotations

import base64
import io
import json
import time

import modal

from fastapi import Request, HTTPException

from request_signing import verify_modal_request

OMNIVOICE_COMMIT = "f7f5040646f4fb8eef84e011adebec0c375ea6ae"
MODEL_ID = "k2-fsa/OmniVoice"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App("avtr1-omnivoice-tts")

tts_image = (
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


@app.cls(
    image=tts_image,
    gpu="L4",
    timeout=120,
    scaledown_window=120,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("avtr-modal-request-signing-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class OmniVoiceTTS:
    """Separate-GPU OmniVoice TTS endpoint for browser testing.

    Uses GPU memory snapshot for fast warm starts. Model loads once
    via @modal.enter(snap=True) and is captured in the snapshot.
    """

    @modal.enter(snap=True)
    def _load_model(self) -> None:
        import torch

        print("[omnivoice-tts] === @modal.enter(snap=True) loading model ===")
        print(f"[omnivoice-tts] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[omnivoice-tts] GPU: {torch.cuda.get_device_name(0)}")

        self._model_load_error: str | None = None

        try:
            from omnivoice import OmniVoice

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            load_start = time.time()
            self._model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
                attn_implementation="eager" if torch.cuda.is_available() else None,
            )
            load_elapsed = time.time() - load_start

            self._sampling_rate = int(getattr(self._model, "sampling_rate", 24000))
            self._device = device
            self._dtype = dtype

            print(f"[omnivoice-tts] Model loaded in {load_elapsed:.1f}s, sr={self._sampling_rate}Hz")
        except Exception as exc:
            self._model_load_error = str(exc)
            print(f"[omnivoice-tts] ERROR loading model: {exc}")

    def _do_synthesize(self, text: str, target_sample_rate: int = 16000) -> dict:
        """Shared inference helper used by both @modal.method and @modal.fastapi_endpoint."""
        if not text or not text.strip():
            return {"ok": False, "error": "empty_text"}

        if self._model_load_error:
            return {"ok": False, "error": f"model_not_loaded: {self._model_load_error}"}

        import torch
        import torchaudio
        import numpy as np

        t_start = time.time()

        try:
            with torch.inference_mode():
                audio = self._model.generate(
                    text,
                    preprocess_prompt=True,
                    postprocess_output=True,
                    denoise=True,
                    t_shift=0.1,
                    layer_penalty_factor=5.0,
                    position_temperature=5.0,
                    class_temperature=0.0,
                )

            audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data)
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0)

            native_sr = self._sampling_rate
            audio_duration_s = audio_data.shape[-1] / native_sr

            if target_sample_rate != native_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=native_sr,
                    new_freq=target_sample_rate,
                )
                audio_data = resampler(audio_data)
                resample_note = f"{native_sr}Hz -> {target_sample_rate}Hz"
            else:
                resample_note = f"{native_sr}Hz (no resample)"

            output_buf = io.BytesIO()
            torchaudio.save(output_buf, audio_data.cpu(), target_sample_rate, format="wav")
            wav_bytes = output_buf.getvalue()

            inference_elapsed = time.time() - t_start
            rtf = audio_duration_s / inference_elapsed if inference_elapsed > 0 else 0

            wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

            print(
                f"[omnivoice-tts] synthesized: {len(text)} chars, "
                f"{audio_duration_s:.2f}s audio, {inference_elapsed:.3f}s inference, "
                f"RTF={rtf:.2f}x, {resample_note}, {len(wav_bytes)} bytes"
            )

            return {
                "ok": True,
                "wav_base64": wav_b64,
                "sample_rate": target_sample_rate,
                "duration_ms": round(audio_duration_s * 1000),
                "latency_ms": round(inference_elapsed * 1000),
                "rtf": round(rtf, 2),
                "text_length": len(text),
                "output_bytes": len(wav_bytes),
                "provider": "omnivoice",
            }
        except Exception as exc:
            inference_elapsed = time.time() - t_start
            print(f"[omnivoice-tts] inference error: {exc}")
            return {
                "ok": False,
                "error": str(exc),
                "latency_ms": round(inference_elapsed * 1000),
                "text_length": len(text),
            }

    @modal.method()
    def synthesize(self, text: str, target_sample_rate: int = 16000) -> dict:
        return self._do_synthesize(text, target_sample_rate)

    @modal.fastapi_endpoint(method="POST")
    async def web(self, request: Request) -> dict:
        await verify_modal_request(request)
        body = await request.json()
        text = body.get("text", "")
        target_sr = int(body.get("target_sample_rate", 16000))
        return self._do_synthesize(text, target_sr)


@app.local_entrypoint()
def main() -> None:
    print("=" * 60)
    print("OmniVoice TTS — smoke test")
    print("=" * 60)
    inst = OmniVoiceTTS()
    for text in [
        "Hello, how are you?",
        "This is a longer sentence to measure how OmniVoice handles more tokens during browser testing.",
    ]:
        result = inst.synthesize.remote(text)
        if result.get("ok"):
            print(
                f"  ok: {result['ok']}  provider: {result['provider']}  "
                f"latency: {result['latency_ms']}ms  duration: {result['duration_ms']}ms  "
                f"RTF: {result['rtf']}x  sr: {result['sample_rate']}Hz  "
                f"bytes: {result['output_bytes']}"
            )
        else:
            print(f"  FAIL: {result.get('error')}")
    print("=" * 60)
