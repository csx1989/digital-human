"""
Modal avatar runtime skeleton.

This is a placeholder. It does not load or run the AVTR-1 model.
All functions are stubs with TODO comments indicating where real
implementation will go.
"""

import modal

app = modal.App("avatar-runtime")

AVATAR_IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04",
    add_python="3.11",
).pip_install(
    "torch",
    "tensorrt",
    # TODO: Add avtr1_renderer and its dependencies once we define
    # the Modal image build strategy (pixi vs pip vs conda).
)


@app.function(image=AVATAR_IMAGE, gpu="T4", timeout=600)
@modal.web_server(port=8000)
def serve():
    """Start the avatar runtime HTTP server.

    TODO: Implement using FastAPI or equivalent.
    The server should expose endpoints for:
      - POST /session/start  - initialize a rendering session
      - POST /session/chunk  - send audio, receive frames + state
      - POST /session/end    - tear down session, release GPU
      - GET  /health         - GPU health check
    """
    pass


class AvatarSession:
    """Manages a single avatar rendering session.

    TODO: This class will hold:
      - The loaded AVTR-1 pipeline (Pipeline.from_artifacts)
      - Per-session autoregressive state (AVTR1State)
      - Avatar registration data (KPInfo, appearance features, etc.)
      - Session timing and latency metrics
    """

    def __init__(self, avatar_id: str, background_id: str) -> None:
        self.avatar_id = avatar_id
        self.background_id = background_id
        self.state: bytes | None = None

    def start(self) -> dict:
        """Initialize the rendering session.

        TODO:
          1. Load the avatar's registered assets from artifact storage
          2. Initialize cold-start autoregressive state
          3. Warm up the pipeline with a dummy chunk if needed
          4. Return session metadata (frame dimensions, format, etc.)
        """
        return {"status": "placeholder", "avatarId": self.avatar_id}

    def process_chunk(self, audio_bytes: bytes) -> dict:
        """Process a single audio chunk and return rendered frames.

        TODO:
          1. Accept TTS audio (int16 PCM at 16 kHz, likely candidate)
          2. Prepare dual-stream audio (speech + silence for listen)
          3. Build the multipart request matching the renderer's
             /process-audio-v3 contract:
               - current_chunk (present speech audio)
               - future_chunk (future speech audio)
               - current_chunk_listen (present listen audio)
               - future_chunk_listen (future listen audio)
               - state (safetensors blob from previous call)
          4. Call pipeline.process_chunk()
          5. Serialize new state to safetensors
          6. Return frames + state + timing metadata

        The exact chunk duration and present/future split will be defined
        after analyzing the SpeechScheduler's timing constants:
          - present_duration: 5 frames = 200ms
          - future_duration: 5 frames + 80 samples

        These are observations from the existing renderer, not final protocol.
        """
        return {
            "status": "placeholder",
            "frames": 0,
            "stateSize": 0,
        }

    def end(self) -> dict:
        """Tear down the session and release GPU resources.

        TODO:
          1. Free CUDA tensors (state, pipeline caches)
          2. Log session metrics (total chunks, avg latency, etc.)
          3. Return final session summary
        """
        return {"status": "placeholder", "avatarId": self.avatar_id}
