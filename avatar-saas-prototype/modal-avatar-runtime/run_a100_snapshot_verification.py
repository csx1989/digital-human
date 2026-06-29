#!/usr/bin/env python3
"""A100-40GB Snapshot Verification Runner.

Isolated test that proves whether Modal GPU snapshot restore works on A100-40GB
for our exact AVTR + OmniVoice stack.

Steps:
  1. First cold call to health (triggers snap=True init on fresh container)
  2. Wait 120s for scaledown_window=90 to force scale-to-zero
  3. Second call (triggers snap=False restore from GPU snapshot)
  4. Wait 120s, third call (second restore)
  5. Wait 120s, fourth call (third restore)
  6. After snapshot proof: run functional smokes
  7. Print structured verdict

Usage:
    /Users/ayya/.local/share/uv/tools/modal/bin/python run_a100_snapshot_verification.py

DO NOT run with system python3.14 — use the Modal SDK python at the path above.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

APP_NAME = "avtr1-lifecycle-parallel-a100"
CLASS_NAME = "AvatarLifecycleA100Parallel"
SCALE_DOWN_WAIT_S = 120
RESTORE_ATTEMPTS = 3

ERROR_PATTERNS = [
    "SIGSEGV",
    "exit code 139",
    "Transient snapshot error",
    "failed to restore container from snapshot",
    "snapshot restore failed",
    "cuda",
    "checkpoint",
    "restore",
]

SMOKE_TESTS = [
    "fmp4_av_live_smoke",
    "tts_same_gpu_parallel_standalone",
    "tts_handoff_smoke",
]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _banner(text: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def _get_cls():
    import modal

    print(f"[{_ts()}] Resolving modal.Cls.from_name(\"{APP_NAME}\", \"{CLASS_NAME}\")...")
    cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    return cls


def _call_health(inst, label: str) -> dict:
    t0 = time.monotonic()
    print(f"[{_ts()}] Calling health.remote() [{label}]...")
    result = inst.health.remote()
    wall_ms = (time.monotonic() - t0) * 1000
    result["_wall_ms"] = round(wall_ms, 1)
    result["_label"] = label
    result["_timestamp"] = datetime.now(timezone.utc).isoformat()

    gpu_name = result.get("gpu_name") or result.get("nvidia_smi", {}).get("gpu_name", "?")
    smi = result.get("nvidia_smi", {})
    vram_total = smi.get("memory_total_mb", "?")
    vram_used = smi.get("memory_used_mb", "?")
    snap_ok = result.get("snap_ok")
    pipeline = result.get("pipeline_loaded")
    omnivoice = result.get("omnivoice_loaded")
    cid = result.get("container_instance_id")
    snap_error = result.get("snap_error")

    print(f"[{_ts()}] [{label}] wall={wall_ms:.0f}ms gpu={gpu_name} vram={vram_total}MB "
          f"snap_ok={snap_ok} pipeline={pipeline} omnivoice={omnivoice} "
          f"cid={cid} error={snap_error}")

    return result


def _snapshot_phase(inst) -> list[dict]:
    results = []

    _banner("PHASE 1: COLD INIT (first call, triggers snap=True)")
    cold = _call_health(inst, "cold_init_1")
    results.append(cold)

    if not cold.get("snap_ok"):
        print(f"\n[FAIL] Cold init failed: snap_ok=False, error={cold.get('snap_error')}")
        print("[ABORT] Cannot proceed to restore tests without successful init.")
        return results

    print(f"\n[{_ts()}] Cold init successful. Waiting {SCALE_DOWN_WAIT_S}s for scale-to-zero...")
    for i in range(SCALE_DOWN_WAIT_S, 0, -10):
        print(f"  ... {i}s remaining", flush=True)
        time.sleep(10)

    for attempt in range(1, RESTORE_ATTEMPTS + 1):
        _banner(f"PHASE {attempt + 1}: RESTORE ATTEMPT {attempt}/{RESTORE_ATTEMPTS}")
        restore = _call_health(inst, f"restore_{attempt}")
        results.append(restore)

        if not restore.get("snap_ok"):
            print(f"\n[WARN] Restore {attempt} shows snap_ok=False: {restore.get('snap_error')}")
            print("  This may indicate snapshot restore failed and container re-initialized.")

        if attempt < RESTORE_ATTEMPTS:
            print(f"\n[{_ts()}] Waiting {SCALE_DOWN_WAIT_S}s for scale-to-zero...")
            for i in range(SCALE_DOWN_WAIT_S, 0, -10):
                print(f"  ... {i}s remaining", flush=True)
                time.sleep(10)

    return results


def _smoke_phase(inst) -> dict:
    _banner("PHASE 5: FUNCTIONAL SMOKE TESTS")
    smoke_results = {}

    print(f"\n[{_ts()}] Running fmp4_av_live_smoke...")
    t0 = time.monotonic()
    try:
        smoke = inst.fmp4_av_live_smoke.remote()
        smoke["_wall_ms"] = round((time.monotonic() - t0) * 1000, 1)
        smoke_results["fmp4_av_live_smoke"] = smoke
        print(f"[{_ts()}] fmp4_av_live_smoke: success={smoke.get('success')} "
              f"elapsed={smoke.get('smokeElapsedMs')}ms wall={smoke['_wall_ms']}ms")
    except Exception as exc:
        smoke_results["fmp4_av_live_smoke"] = {"error": str(exc)}
        print(f"[{_ts()}] fmp4_av_live_smoke FAILED: {exc}")

    print(f"\n[{_ts()}] Running TTS standalone (no timeline)...")
    t0 = time.monotonic()
    try:
        tts_result = inst.tts_same_gpu_parallel.remote(
            body={
                "action": "tts_same_gpu_parallel",
                "text": "Hello, this is an A100 snapshot verification test.",
                "target_sample_rate": 16000,
            }
        )
        tts_result["_wall_ms"] = round((time.monotonic() - t0) * 1000, 1)
        smoke_results["tts_same_gpu_parallel_standalone"] = tts_result
        print(f"[{_ts()}] TTS standalone: ok={tts_result.get('ok')} "
              f"duration={tts_result.get('duration_ms')}ms latency={tts_result.get('latency_ms')}ms")
    except Exception as exc:
        smoke_results["tts_same_gpu_parallel_standalone"] = {"error": str(exc)}
        print(f"[{_ts()}] TTS standalone FAILED: {exc}")

    print(f"\n[{_ts()}] Running tts_handoff_smoke...")
    t0 = time.monotonic()
    try:
        handoff = inst.tts_handoff_smoke.remote(body={"action": "tts_handoff_smoke"})
        handoff["_wall_ms"] = round((time.monotonic() - t0) * 1000, 1)
        smoke_results["tts_handoff_smoke"] = handoff
        print(f"[{_ts()}] tts_handoff_smoke: success={handoff.get('success')} "
              f"elapsed={handoff.get('smokeElapsedMs')}ms")
    except Exception as exc:
        smoke_results["tts_handoff_smoke"] = {"error": str(exc)}
        print(f"[{_ts()}] tts_handoff_smoke FAILED: {exc}")

    return smoke_results


def _analyze_snapshot_results(results: list[dict]) -> dict:
    analysis = {
        "cold_init": None,
        "restores": [],
        "verdict": "INCONCLUSIVE",
        "verdict_reason": "",
        "gpu_evidence": {},
        "snapshot_errors": [],
        "container_ids": [],
    }

    if not results:
        analysis["verdict"] = "FAIL"
        analysis["verdict_reason"] = "No results collected"
        return analysis

    cold = results[0]
    analysis["cold_init"] = {
        "wall_ms": cold.get("_wall_ms"),
        "snap_ok": cold.get("snap_ok"),
        "snap_error": cold.get("snap_error"),
        "pipeline_loaded": cold.get("pipeline_loaded"),
        "omnivoice_loaded": cold.get("omnivoice_loaded"),
        "gpu_name": cold.get("gpu_name"),
        "nvidia_smi": cold.get("nvidia_smi"),
        "container_instance_id": cold.get("container_instance_id"),
    }

    smi = cold.get("nvidia_smi", {})
    gpu_name = smi.get("gpu_name") or cold.get("gpu_name") or "unknown"
    vram_total = smi.get("memory_total_mb", "?")
    gpu_uuid = smi.get("gpu_uuid", "?")

    analysis["gpu_evidence"] = {
        "gpu_name": gpu_name,
        "memory_total_mb": vram_total,
        "gpu_uuid": gpu_uuid,
        "is_a100": "A100" in gpu_name.upper(),
        "is_40gb": "40" in str(vram_total) or (isinstance(vram_total, int) and vram_total <= 45000),
    }

    if not cold.get("snap_ok"):
        analysis["verdict"] = "FAIL"
        analysis["verdict_reason"] = f"Cold init failed: {cold.get('snap_error')}"
        return analysis

    all_container_ids = [cold.get("container_instance_id")]
    restore_successes = 0
    restore_failures = 0

    for i, r in enumerate(results[1:], 1):
        restore_info = {
            "attempt": i,
            "wall_ms": r.get("_wall_ms"),
            "snap_ok": r.get("snap_ok"),
            "snap_error": r.get("snap_error"),
            "pipeline_loaded": r.get("pipeline_loaded"),
            "omnivoice_loaded": r.get("omnivoice_loaded"),
            "container_instance_id": r.get("container_instance_id"),
        }
        analysis["restores"].append(restore_info)
        all_container_ids.append(r.get("container_instance_id"))

        if r.get("snap_ok"):
            restore_successes += 1
        else:
            restore_failures += 1
            analysis["snapshot_errors"].append({
                "attempt": i,
                "error": r.get("snap_error"),
            })

    analysis["container_ids"] = all_container_ids

    unique_cids = set(cid for cid in all_container_ids if cid)
    analysis["unique_container_ids"] = len(unique_cids)
    analysis["container_id_changed_between_restores"] = len(unique_cids) > 1

    cold_wall = cold.get("_wall_ms", 0)
    restore_walls = [r.get("_wall_ms", 0) for r in results[1:]]
    avg_restore_wall = sum(restore_walls) / len(restore_walls) if restore_walls else 0

    analysis["timing"] = {
        "cold_init_wall_ms": cold_wall,
        "restore_walls_ms": restore_walls,
        "avg_restore_wall_ms": round(avg_restore_wall, 1),
        "restore_speedup_ratio": round(cold_wall / avg_restore_wall, 1) if avg_restore_wall > 0 else None,
    }

    if restore_failures == 0 and restore_successes == RESTORE_ATTEMPTS:
        analysis["verdict"] = "A100 GPU SNAPSHOT RESTORE WORKS"
        analysis["verdict_reason"] = (
            f"All {RESTORE_ATTEMPTS}/{RESTORE_ATTEMPTS} restore attempts succeeded. "
            f"Cold init {cold_wall:.0f}ms, avg restore {avg_restore_wall:.0f}ms. "
            f"Container IDs changed: {analysis['container_id_changed_between_restores']}."
        )
    elif restore_failures > 0 and restore_successes > 0:
        analysis["verdict"] = "A100 GPU SNAPSHOT RESTORE UNRELIABLE"
        analysis["verdict_reason"] = (
            f"{restore_successes}/{RESTORE_ATTEMPTS} restores succeeded, "
            f"{restore_failures}/{RESTORE_ATTEMPTS} failed. "
            f"Errors: {analysis['snapshot_errors']}"
        )
    elif restore_failures == RESTORE_ATTEMPTS:
        analysis["verdict"] = "A100 GPU SNAPSHOT RESTORE FAILS"
        analysis["verdict_reason"] = (
            f"All {RESTORE_ATTEMPTS} restore attempts failed. "
            f"Errors: {analysis['snapshot_errors']}"
        )

    return analysis


def _print_final_report(analysis: dict, snapshot_results: list[dict], smoke_results: dict) -> None:
    _banner("FINAL REPORT")

    gpu = analysis["gpu_evidence"]
    print("GPU EVIDENCE:")
    print(f"  GPU Name:          {gpu.get('gpu_name')}")
    print(f"  VRAM Total:        {gpu.get('memory_total_mb')} MB")
    print(f"  GPU UUID:          {gpu.get('gpu_uuid')}")
    print(f"  Is A100:           {gpu.get('is_a100')}")
    print(f"  Is 40GB:           {gpu.get('is_40gb')}")
    print()

    cold = analysis.get("cold_init", {})
    print("COLD INIT (first call):")
    print(f"  Wall Time:         {cold.get('wall_ms', 0):.0f} ms")
    print(f"  snap_ok:           {cold.get('snap_ok')}")
    print(f"  pipeline_loaded:   {cold.get('pipeline_loaded')}")
    print(f"  omnivoice_loaded:  {cold.get('omnivoice_loaded')}")
    print(f"  Container ID:      {cold.get('container_instance_id')}")
    if cold.get("snap_error"):
        print(f"  ERROR:             {cold.get('snap_error')}")
    print()

    restores = analysis.get("restores", [])
    if restores:
        print("RESTORE ATTEMPTS:")
        for r in restores:
            status = "OK" if r.get("snap_ok") else "FAIL"
            print(f"  Restore {r['attempt']}: {status} "
                  f"wall={r.get('wall_ms', 0):.0f}ms "
                  f"cid={r.get('container_instance_id')}")
            if r.get("snap_error"):
                print(f"    ERROR: {r.get('snap_error')}")
        print()

    timing = analysis.get("timing", {})
    if timing:
        print("TIMING:")
        print(f"  Cold Init:         {timing.get('cold_init_wall_ms', 0):.0f} ms")
        print(f"  Restore Walls:     {timing.get('restore_walls_ms')}")
        print(f"  Avg Restore:       {timing.get('avg_restore_wall_ms', 0):.0f} ms")
        if timing.get("restore_speedup_ratio"):
            print(f"  Speedup Ratio:     {timing.get('restore_speedup_ratio')}x")
        print()

    print(f"Container IDs seen: {analysis.get('container_ids')}")
    print(f"Unique Container IDs: {analysis.get('unique_container_ids')}")
    print(f"Container ID changed between restores: {analysis.get('container_id_changed_between_restores')}")
    print()

    print("SMOKE TEST RESULTS:")
    for name, result in smoke_results.items():
        if "error" in result and "success" not in result:
            print(f"  {name}: ERROR - {result['error']}")
        else:
            success = result.get("success", result.get("ok"))
            elapsed = result.get("smokeElapsedMs", result.get("_wall_ms"))
            print(f"  {name}: {'PASS' if success else 'FAIL'} ({elapsed}ms)")
    print()

    _banner(f"VERDICT: {analysis['verdict']}")
    print(f"  {analysis['verdict_reason']}")
    print()

    print("MODAL LOG INSPECTION (manual):")
    print("  Run: npx wrangler tail avatar-session-bridge --format json  # (not needed for Modal)")
    print("  Run: modal app logs avtr1-lifecycle-parallel-a100")
    print("  Search for these patterns:")
    for pat in ERROR_PATTERNS:
        print(f"    - \"{pat}\"")
    print()

    print("FULL JSON DATA:")
    report = {
        "analysis": analysis,
        "snapshot_results": snapshot_results,
        "smoke_results": smoke_results,
    }
    print(json.dumps(report, indent=2, default=str))


def main():
    _banner("A100-40GB SNAPSHOT VERIFICATION")
    print(f"App:    {APP_NAME}")
    print(f"Class:  {CLASS_NAME}")
    print(f"Scale-down wait: {SCALE_DOWN_WAIT_S}s")
    print(f"Restore attempts: {RESTORE_ATTEMPTS}")
    print(f"Python: {sys.executable}")
    print(f"Time:   {datetime.now(timezone.utc).isoformat()}")

    try:
        import modal
        print(f"Modal SDK: {modal.__version__}")
    except ImportError:
        print("[FATAL] modal SDK not found. Use: /Users/ayya/.local/share/uv/tools/modal/bin/python")
        sys.exit(1)

    cls = _get_cls()
    inst = cls()

    snapshot_results = _snapshot_phase(inst)
    analysis = _analyze_snapshot_results(snapshot_results)

    smoke_results = {}
    if analysis["cold_init"] and analysis["cold_init"].get("snap_ok"):
        smoke_results = _smoke_phase(inst)
    else:
        print("\n[SKIP] Smoke tests skipped — cold init failed.")

    _print_final_report(analysis, snapshot_results, smoke_results)


if __name__ == "__main__":
    main()
