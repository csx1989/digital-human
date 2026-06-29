"""Prep/maintenance helpers for the 5Q production app.

NOT deployed as part of production. Invoke via:

    modal run modal_prep_helpers_5q.py::build_trt_engines_standalone
    modal run modal_prep_helpers_5q.py::build_warp_trt_engine_standalone
    modal run modal_prep_helpers_5q.py::run_render_health_check

The render health check calls the DEPLOYED 5Q HTTP endpoints with proper
Modal request signing — it does NOT use internal RPC. This tests the real
auth + transport path end-to-end.

Production app surface is not affected by this file.
"""

from __future__ import annotations

import modal

# Reuse the production image + volume definitions (module-level objects,
# no deployment side effects from importing).
from modal_lifecycle_pytorch_cpu_snapshot_session_l40s_5q import (
    image,
    l40s_volume,
    L40S_MOUNT_PATH,
)

app = modal.App("avtr1-prep-helpers-5q")


# ============================================================================
# Build helper: first-time TRT engine build (avtr1_encode + avtr1_decode)
# ============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={L40S_MOUNT_PATH: l40s_volume},
)
def build_trt_engines_standalone() -> dict:
    import sys
    import os
    sys.path.insert(0, "/opt/avtr1")

    print(f"[trt-build] AVTR1_LOCAL_STORAGE={os.environ.get('AVTR1_LOCAL_STORAGE')}", flush=True)
    print("[trt-build] starting engine build", flush=True)

    from build_avtr1_engines import main as build_main
    build_main([])

    from avtr1_renderer.avtr1_artifact_manager import get_trt_engine_path
    enc = get_trt_engine_path("avtr1_encode")
    dec = get_trt_engine_path("avtr1_decode")
    print(f"[trt-build] encode exists={enc.exists()}", flush=True)
    print(f"[trt-build] decode exists={dec.exists()}", flush=True)

    if not (enc.exists() and dec.exists()):
        return {"ok": False, "error": "engines not found after build"}

    print(f"[trt-build] encode={enc.stat().st_size / 1e6:.1f}MB decode={dec.stat().st_size / 1e6:.1f}MB", flush=True)
    print("[trt-build] committing volume", flush=True)
    l40s_volume.commit()
    print("[trt-build] DONE \u2014 volume committed", flush=True)

    return {
        "ok": True,
        "encode_size_mb": enc.stat().st_size / 1e6,
        "decode_size_mb": dec.stat().st_size / 1e6,
    }


# ============================================================================
# Build helper: patch ONNX GridSample3D -> GridSample
# ============================================================================

@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=120,
    volumes={L40S_MOUNT_PATH: l40s_volume},
)
def patch_warp_gridsample_standalone() -> dict:
    import onnx
    model_path = "/data/avtr1-l40s/main/build_artifacts/warp_network.onnx"
    print(f"[patch] Loading {model_path}", flush=True)
    m = onnx.load(model_path)

    valid_attrs = {"mode", "padding_mode", "align_corners"}

    renamed = 0
    attrs_stripped = 0
    for node in m.graph.node:
        if node.op_type in ("GridSample3D", "GridSample"):
            if node.op_type == "GridSample3D":
                node.op_type = "GridSample"
                renamed += 1
            orig_count = len(node.attribute)
            kept = [a for a in node.attribute if a.name in valid_attrs]
            if len(kept) != orig_count:
                attrs_stripped += orig_count - len(kept)
                del node.attribute[:]
                node.attribute.extend(kept)
            print(f"  patched node: {node.name} attrs={[a.name for a in node.attribute]}", flush=True)

    print(f"[patch] Renamed {renamed} GridSample3D -> GridSample", flush=True)
    print(f"[patch] Stripped {attrs_stripped} non-standard attributes", flush=True)

    if renamed > 0 or attrs_stripped > 0:
        onnx.save(m, model_path)

    print("[patch] Committing volume", flush=True)
    l40s_volume.commit()
    print("[patch] DONE", flush=True)
    return {"ok": True, "renamed": renamed, "attrs_stripped": attrs_stripped}


# ============================================================================
# Build helper: warp_network TRT engine with GridSample3D plugin
# ============================================================================

@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=600,
    volumes={L40S_MOUNT_PATH: l40s_volume},
)
def build_warp_trt_engine_standalone() -> dict:
    from pathlib import Path
    import sys
    import os

    sys.path.insert(0, "/opt/avtr1")
    os.environ["AVTR1_LOCAL_STORAGE"] = L40S_MOUNT_PATH

    root = Path(L40S_MOUNT_PATH) / "main"
    onnx_path = root / "build_artifacts" / "warp_network_ori.onnx"
    plugin_path = root / "renderer_runtime_artifacts" / "libgrid_sample_3d_plugin.so"
    engine_dir = root / "renderer_runtime_artifacts_cc"
    engine_path = engine_dir / "warp_network_b5_fp16.engine"

    if not onnx_path.is_file():
        return {"ok": False, "error": f"ONNX not found: {onnx_path}"}
    if not plugin_path.is_file():
        return {"ok": False, "error": f"Plugin not found: {plugin_path}"}

    engine_dir.mkdir(parents=True, exist_ok=True)

    import ctypes
    import glob as _glob

    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs
    import tensorrt as trt

    from build_renderer_engines import _surge_warp, _symbolize_batch

    _trt_dir = str(Path(trt.__file__).parent)
    _nvinfer_paths = (
        _glob.glob(f"{_trt_dir}/libnvinfer.so*")
        + _glob.glob("/usr/lib/x86_64-linux-gnu/libnvinfer.so*")
        + _glob.glob("/opt/tensorrt/lib/libnvinfer.so*")
        + _glob.glob("/usr/local/lib/**/libnvinfer.so*", recursive=True)
    )
    if _nvinfer_paths:
        _best = sorted(_nvinfer_paths, key=lambda p: p, reverse=True)[0]
        print(f"[warp-build] Preloading {_best}", flush=True)
        ctypes.cdll.LoadLibrary(_best)
    else:
        print("[warp-build] libnvinfer.so not found by glob; tensorrt import may suffice", flush=True)

    print(f"[warp-build] Loading plugin: {plugin_path}", flush=True)
    try:
        ctypes.cdll.LoadLibrary(str(plugin_path))
    except OSError as e:
        return {"ok": False, "error": f"Failed to load plugin: {e}"}

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"[warp-build] Loading ONNX: {onnx_path}", flush=True)
    model = onnx.load(str(onnx_path))

    # Rename GridSample -> GridSample3D so the custom plugin registers.
    renamed_gs = 0
    for node in model.graph.node:
        if node.op_type == "GridSample":
            node.op_type = "GridSample3D"
            renamed_gs += 1
    if renamed_gs > 0:
        print(f"[warp-build] Renamed {renamed_gs} GridSample \u2192 GridSample3D", flush=True)

    print("[warp-build] Applying warp graph surgery for dynamic batch", flush=True)
    graph = gs.import_onnx(model)
    _surge_warp(graph)
    for tensor in list(graph.inputs) + list(graph.outputs):
        new_shape = list(tensor.shape)
        if len(new_shape) > 0:
            new_shape[0] = "batch"
        tensor.shape = new_shape
    graph.cleanup().toposort()
    new_model = gs.export_onnx(graph)
    _symbolize_batch(new_model)

    print("[warp-build] Parsing ONNX into TensorRT", flush=True)
    if not parser.parse(new_model.SerializeToString()):
        for i in range(parser.num_errors):
            print(f"  [trt] {parser.get_error(i)}", flush=True)
        return {"ok": False, "error": "ONNX parse failed"}

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))

    profile = builder.create_optimization_profile()
    profile.set_shape("feature_3d", (1,32,16,64,64), (5,32,16,64,64), (5,32,16,64,64))
    profile.set_shape("kp_source", (1,21,3), (5,21,3), (5,21,3))
    profile.set_shape("kp_driving", (1,21,3), (5,21,3), (5,21,3))
    config.add_optimization_profile(profile)

    print("[warp-build] Building engine (this may take a few minutes)...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return {"ok": False, "error": "build_serialized_network returned None"}

    with open(engine_path, "wb") as f:
        f.write(serialized)

    if not engine_path.is_file():
        return {"ok": False, "error": "engine file not created"}

    size_mb = engine_path.stat().st_size / 1e6
    print(f"[warp-build] Engine built: {size_mb:.1f}MB", flush=True)

    print("[warp-build] Committing volume", flush=True)
    l40s_volume.commit()
    print("[warp-build] DONE \u2014 volume committed", flush=True)

    return {"ok": True, "engine_size_mb": size_mb}


@app.function(image=image, timeout=600)
def _run_warp_build_entrypoint():
    import json
    result = build_warp_trt_engine_standalone.remote()
    print(json.dumps(result, indent=2, default=str))
    return result


# ============================================================================
# Render health check via signed HTTP endpoints (NOT internal RPC)
# ============================================================================

WORKSPACE = "kommathotichinnasubbayamma"
BASE_URL = f"https://{WORKSPACE}"

START_SESSION_URL = f"{BASE_URL}--start-session-5q.modal.run"
PUSH_AUDIO_BATCH_URL = f"{BASE_URL}--push-audio-batch-5q.modal.run"
STREAM_FMP4_AV_URL = f"{BASE_URL}--stream-fmp4-av-adapter-5q.modal.run"
END_SESSION_URL = f"{BASE_URL}--end-session-adapter-5q.modal.run"


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("avtr-modal-request-signing-secret"),
    ],
    timeout=120,
)
def render_health_check_http(
    avatar_id: str = "arnold",
    num_silence_chunks: int = 3,
) -> dict:
    """Call deployed 5Q HTTP endpoints with Modal request signing.

    Proves:
      - auth signing is accepted ([modal-auth] valid in logs)
      - session starts + warmup succeeds
      - fMP4 stream produces chunks > 0, videoBytes > 0
      - end-session cleans up
    """
    import json
    import uuid
    import numpy as np
    import httpx
    from request_signing import sign_request

    secret = __import__("os").environ["AVTR_MODAL_REQUEST_SIGNING_SECRET"]
    session_id = f"health-http-{uuid.uuid4().hex[:12]}"
    t_start = __import__("time").monotonic()

    def _canonicalize_url(url: str) -> str:
        """Match JS new URL(url).href — ensures trailing slash on root path."""
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        path = parts.path if parts.path else "/"
        return f"{parts.scheme}://{parts.netloc}{path}"

    def _signed_post(url: str, body: dict, client: httpx.Client) -> httpx.Response:
        body_bytes = json.dumps(body).encode()
        headers = sign_request(secret, "POST", _canonicalize_url(url), body_bytes)
        return client.post(url, content=body_bytes, headers=headers, timeout=30.0)

    verdict = {
        "ok": False,
        "session_id": session_id,
        "chunks": 0,
        "videoBytes": 0,
        "audioBytes": 0,
        "warmup_ms": -1,
        "error": None,
        "engine_backends": {},
    }

    try:
        with httpx.Client() as client:
            # 1. Start session
            print(f"[health-http] POST {START_SESSION_URL} session={session_id}", flush=True)
            resp = _signed_post(START_SESSION_URL, {
                "session_id": session_id,
                "avatar_id": avatar_id,
            }, client)

            if resp.status_code != 200:
                verdict["error"] = f"start_session HTTP {resp.status_code}: {resp.text[:500]}"
                return verdict

            result = resp.json()
            if "error" in result:
                verdict["error"] = f"start_session error: {result['error']}"
                return verdict

            verdict["warmup_ms"] = result.get("warmup_ms", -1)
            print(f"[health-http] session started warmup_ms={verdict['warmup_ms']}", flush=True)

            # 2. Push silence audio chunks
            sample_rate = 16000
            chunk_ms = 200
            samples = int(sample_rate * chunk_ms / 1000)
            for i in range(num_silence_chunks):
                silence = np.zeros(samples, dtype=np.float32)
                audio_b64 = __import__("base64").b64encode(silence.tobytes()).decode()
                resp = _signed_post(PUSH_AUDIO_BATCH_URL, {
                    "session_id": session_id,
                    "chunks": [{"audio_b64": audio_b64, "sample_rate": sample_rate}],
                }, client)
                if resp.status_code == 200:
                    print(f"[health-http] pushed silence chunk {i+1}/{num_silence_chunks}", flush=True)
                else:
                    print(f"[health-http] push chunk {i+1} HTTP {resp.status_code}", flush=True)

            # 3. Stream fMP4 and collect events
            stream_body = json.dumps({"session_id": session_id}).encode()
            stream_headers = sign_request(secret, "POST", _canonicalize_url(STREAM_FMP4_AV_URL), stream_body)
            print(f"[health-http] POST stream {STREAM_FMP4_AV_URL}", flush=True)

            chunks_seen = 0
            video_bytes = 0
            audio_bytes = 0
            found_end = False
            max_iterations = 100

            with client.stream(
                "POST", STREAM_FMP4_AV_URL,
                content=stream_body,
                headers=stream_headers,
                timeout=90.0,
            ) as stream:
                if stream.status_code != 200:
                    body = stream.read()
                    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
                    verdict["error"] = f"stream HTTP {stream.status_code}: {text[:500]}"
                    return verdict

                buffer = ""
                for line in stream.iter_lines():
                    buffer += line + "\n"
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        if "event: fmp4_media" in event_str:
                            chunks_seen += 1
                            data_line = [l for l in event_str.split("\n") if l.startswith("data:")]
                            if data_line:
                                try:
                                    data = json.loads(data_line[0].removeprefix("data: ").strip())
                                    if data.get("type") == "fmp4_media":
                                        video_bytes += data.get("byteLength", 0)
                                except Exception:
                                    pass
                        elif "event: end" in event_str:
                            found_end = True
                            data_line = [l for l in event_str.split("\n") if l.startswith("data:")]
                            if data_line:
                                try:
                                    end_data = json.loads(data_line[0].removeprefix("data: ").strip())
                                    chunks_seen = end_data.get("chunksRendered", chunks_seen)
                                    video_bytes = end_data.get("videoBytes", video_bytes)
                                    audio_bytes = end_data.get("audioBytes", audio_bytes)
                                except Exception:
                                    pass
                            break

                    if chunks_seen >= max_iterations or found_end:
                        break

            verdict["chunks"] = chunks_seen
            verdict["videoBytes"] = video_bytes
            verdict["audioBytes"] = audio_bytes
            verdict["found_end"] = found_end
            print(f"[health-http] stream done chunks={chunks_seen} videoBytes={video_bytes} found_end={found_end}", flush=True)

            # 4. End session
            resp = _signed_post(END_SESSION_URL, {"session_id": session_id}, client)
            if resp.status_code == 200:
                end_result = resp.json()
                verdict["chunks"] = end_result.get("chunks_rendered", chunks_seen)
                verdict["videoBytes"] = end_result.get("fmp4_media_total_bytes", video_bytes)
                verdict["engine_backends"] = "see_logs"
                print(f"[health-http] session ended chunks={verdict['chunks']} videoBytes={verdict['videoBytes']}", flush=True)
            else:
                print(f"[health-http] end HTTP {resp.status_code}", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        verdict["error"] = str(e)
        return verdict

    elapsed_s = __import__("time").monotonic() - t_start
    verdict["elapsed_s"] = round(elapsed_s, 1)
    verdict["ok"] = (
        verdict["chunks"] > 0
        and verdict["videoBytes"] > 0
        and verdict["warmup_ms"] > 0
        and verdict["error"] is None
    )

    print(f"[health-http] VERDICT ok={verdict['ok']} chunks={verdict['chunks']} "
          f"videoBytes={verdict['videoBytes']} warmup_ms={verdict['warmup_ms']}", flush=True)

    return verdict


@app.local_entrypoint()
def run_render_health_check(
    avatar_id: str = "arnold",
    num_silence_chunks: int = 3,
):
    """Local entrypoint: modal run modal_prep_helpers_5q.py::run_render_health_check"""
    import json
    result = render_health_check_http.remote(avatar_id=avatar_id, num_silence_chunks=num_silence_chunks)
    print(json.dumps(result, indent=2, default=str))
    return result


@app.local_entrypoint()
def run_warp_build():
    """Local entrypoint: modal run modal_prep_helpers_5q.py::run_warp_build"""
    import json
    result = build_warp_trt_engine_standalone.remote()
    print(json.dumps(result, indent=2, default=str))
    return result


# ============================================================================
# Build helper: pre-cache avatar tensors on the L40S volume
# (Reuses the production image; avoids building a separate heavy prep image.)
# ============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=600,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={L40S_MOUNT_PATH: l40s_volume},
)
def build_avatar_cache_standalone(avatar_id: str | None = None) -> dict:
    from pathlib import Path
    import json
    import time as _time
    import os

    os.environ["AVTR1_LOCAL_STORAGE"] = L40S_MOUNT_PATH

    import torch
    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.backgrounds import load_background
    from avtr1_renderer.avtr1_artifact_manager import (
        get_artifact_manager,
        get_storage_root,
    )
    from safetensors.torch import load_file

    print("[cache] === Avatar Cache Prep (5Q image) ===")
    storage_root = get_storage_root()
    print(f"[cache] Storage root: {storage_root}")

    mgr = get_artifact_manager()
    mgr.ensure_all_artifacts(workers=4)
    print("[cache] Artifacts ensured")

    portraits_dir = storage_root / "avatars_artifacts" / "reference_frames"
    if not portraits_dir.is_dir():
        return {"ok": False, "error": f"No reference_frames at {portraits_dir}"}

    all_pngs = sorted(portraits_dir.glob("*.png"))
    if not all_pngs:
        return {"ok": False, "error": f"No portrait PNGs in {portraits_dir}"}

    if avatar_id is None:
        avatar_id = all_pngs[0].stem
    print(f"[cache] Using avatar_id={avatar_id}")

    print("[cache] Loading pipeline...")
    t0 = _time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    print(f"[cache] Pipeline loaded in {_time.monotonic() - t0:.1f}s")

    avatar = registry[avatar_id]
    print(f"[cache] Avatar loaded: {avatar.id}")
    print(f"[cache]   f_s shape: {avatar.f_s.shape}")
    print(f"[cache]   M_grid shape: {avatar.M_grid.shape}")
    print(f"[cache]   mask shape: {avatar.mask.shape}")
    print(f"[cache]   source shape: {avatar.source.shape}")
    print(f"[cache]   kp_info.kp shape: {avatar.kp_info.kp.shape}")
    print(f"[cache]   no_matting: {avatar.no_matting}")

    cache_dir = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache" / avatar_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.save(avatar.kp_info.kp.cpu(), cache_dir / "kp_info_kp.pt")
    torch.save(avatar.kp_info.exp.cpu(), cache_dir / "kp_info_exp.pt")
    torch.save(avatar.kp_info.scale.cpu(), cache_dir / "kp_info_scale.pt")
    torch.save(avatar.kp_info.t.cpu(), cache_dir / "kp_info_t.pt")
    torch.save(avatar.kp_info.pitch.cpu(), cache_dir / "kp_info_pitch.pt")
    torch.save(avatar.kp_info.yaw.cpu(), cache_dir / "kp_info_yaw.pt")
    torch.save(avatar.kp_info.roll.cpu(), cache_dir / "kp_info_roll.pt")
    torch.save(avatar.kp_info.R.cpu(), cache_dir / "kp_info_R.pt")
    torch.save(avatar.f_s.cpu(), cache_dir / "f_s.pt")
    torch.save(avatar.M_grid.cpu(), cache_dir / "M_grid.pt")
    torch.save(avatar.mask.cpu(), cache_dir / "mask.pt")
    torch.save(avatar.source.cpu(), cache_dir / "source.pt")

    H, W = avatar.source.shape[-2:]
    bg = load_background(None, H, W)
    torch.save(bg.cpu(), cache_dir / "bg.pt")

    normalizer_path = storage_root / "avtr1_normalizer.safetensors"
    normalizer_tensors = load_file(str(normalizer_path), device="cpu")
    torch.save(normalizer_tensors, cache_dir / "normalizer.pt")

    with open(cache_dir / "no_matting.json", "w") as f:
        json.dump({"no_matting": bool(avatar.no_matting)}, f)

    l40s_volume.commit()

    cached_files = sorted(p.name for p in cache_dir.iterdir())
    total_mb = sum(p.stat().st_size for p in cache_dir.iterdir()) / 1e6

    print(f"[cache] Cached {len(cached_files)} files ({total_mb:.1f} MB)")
    print(f"[cache] Files: {cached_files}")
    print("[cache] Volume committed.")

    return {
        "ok": True,
        "avatar_id": avatar_id,
        "cache_dir": str(cache_dir),
        "files": cached_files,
        "total_mb": round(total_mb, 1),
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=120,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={L40S_MOUNT_PATH: l40s_volume},
)
def probe_nvh264enc_properties() -> dict:
    """Probe actual nvh264enc element properties inside the deployed Modal image.

    Uses GStreamer GObject introspection to enumerate the exact properties,
    their types, default values, and (for enums) the valid nicknames.
    """
    import importlib
    import json

    gi = importlib.import_module("gi")
    gi.require_version("Gst", "1.0")
    ns = {}
    exec("from gi.repository import Gst, GLib", ns)
    Gst = ns["Gst"]
    Gst.init(None)

    factory = Gst.ElementFactory.find("nvh264enc")
    if factory is None:
        return {"ok": False, "error": "nvh264enc factory not found"}

    elem = factory.create()
    if elem is None:
        return {"ok": False, "error": "nvh264enc.create() returned None"}

    TARGET_PROPS = [
        "rc-mode", "strict-gop", "zerolatency", "gop-size",
        "bitrate", "preset", "max-bitrate", "vbv-buffer-size",
        "bframes", "repeat-sequence-header", "multi-pass",
        "rc-lookahead", "tune", "aud",
    ]

    properties: dict = {}
    param_specs = elem.list_properties()
    all_prop_names = {ps.name for ps in param_specs}

    for name in TARGET_PROPS:
        if name not in all_prop_names:
            properties[name] = {"supported": False}
            continue
        ps = [p for p in param_specs if p.name == name][0]
        try:
            current = elem.get_property(name)
        except Exception:
            current = None

        entry: dict = {
            "supported": True,
            "type": ps.value_type.name,
            "default": repr(current),
        }
        # For GEnums, list all valid nicknames
        type_name = ps.value_type.name
        if "Enum" in type_name or type_name.startswith("G"):
            try:
                from gi.repository import GObject
                gtype = ps.value_type
                enum_class = GObject.GEnum
                # Try to get enum values via the param spec's GType
                enum_values = []
                # Access via __enum_values__ if available
                gv = getattr(gtype, "pytype", None)
                if gv is not None:
                    for nick, val in sorted(
                        [(v.value_nick, int(v)) for v in gv.__enum_values__.values()],
                        key=lambda x: x[1],
                    ):
                        enum_values.append(f"{int(val)}={nick}")
                if enum_values:
                    entry["enum_values"] = enum_values
            except Exception:
                pass

        properties[name] = entry

    # Also get Gst version + nvcodec plugin version
    try:
        gst_version = f"{Gst.version().major}.{Gst.version().minor}.{Gst.version().micro}"
    except Exception:
        gst_version = "unknown"

    result = {
        "ok": True,
        "gst_version": gst_version,
        "element_name": "nvh264enc",
        "total_properties_on_element": len(all_prop_names),
        "target_properties": properties,
    }
    print(json.dumps(result, indent=2, default=str))
    return result


@app.local_entrypoint()
def run_avatar_cache_build(avatar_id: str = "arnold"):
    """Local entrypoint: modal run modal_prep_helpers_5q.py::run_avatar_cache_build"""
    import json
    result = build_avatar_cache_standalone.remote(avatar_id=avatar_id)
    print(json.dumps(result, indent=2, default=str))
    return result


@app.local_entrypoint()
def run_nvh264enc_probe():
    """Local entrypoint: modal run modal_prep_helpers_5q.py::run_nvh264enc_probe"""
    result = probe_nvh264enc_properties.remote()
    return result
