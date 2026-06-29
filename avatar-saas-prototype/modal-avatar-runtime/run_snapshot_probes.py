"""
Invoke the deployed snapshot isolation probes.

Usage:
    python run_snapshot_probes.py

Prerequisites:
    modal deploy modal_l40s_snapshot_isolation_probes.py

This script calls the DEPLOYED app (not modal run), so snapshots are created/restored properly.
"""

import json
import time

import modal

APP_NAME = "avtr1-l40s-snapshot-isolation"

PROBE_CLASSES = [
    ("1-minimal-cuda", "MinimalCudaSnapshotL40S"),
    ("2-avtr-load-only", "MinimalAvtrLoadSnapshotL40S"),
    ("3-avtr-warmup", "MinimalAvtrWarmupSnapshotL40S"),
    ("4-omnivoice-load-only", "MinimalOmniLoadSnapshotL40S"),
    ("5-omnivoice-generate", "MinimalOmniGenerateSnapshotL40S"),
    ("6-combined-load-only", "MinimalCombinedLoadSnapshotL40S"),
    ("7-combined-warmup", "MinimalCombinedWarmupSnapshotL40S"),
]


def call_probe(label: str, cls_name: str) -> dict:
    cls = modal.Cls.from_name(APP_NAME, cls_name)
    instance = cls()
    try:
        t0 = time.monotonic()
        result = instance.health.remote()
        wall_s = time.monotonic() - t0
        result["_wall_s"] = round(wall_s, 2)
        return result
    except Exception as exc:
        return {"probe": label, "_error": str(exc)}


def main():
    results = {}

    print("=" * 72)
    print("PHASE 1: Snapshot creation (first call per probe)")
    print("=" * 72)

    for label, cls_name in PROBE_CLASSES:
        print(f"\n--- {label} ({cls_name}) ---")
        result = call_probe(label, cls_name)
        result["_phase"] = "snapshot-creation"
        print(f"  wall={result.get('_wall_s', '?')}s snap_ok={result.get('snap_ok')} restore_ok={result.get('restore_ok')}")
        if result.get("snap_error"):
            print(f"  snap_error: {result['snap_error']}")
        if result.get("restore_error"):
            print(f"  restore_error: {result['restore_error']}")
        if result.get("_error"):
            print(f"  call_error: {result['_error']}")
        results[f"{label}-phase1"] = result

    print("\n" + "=" * 72)
    print("Waiting 100s for containers to scale to zero...")
    print("=" * 72)
    time.sleep(100)

    print("\n" + "=" * 72)
    print("PHASE 2: Restore test 1")
    print("=" * 72)

    for label, cls_name in PROBE_CLASSES:
        print(f"\n--- {label} ({cls_name}) ---")
        result = call_probe(label, cls_name)
        result["_phase"] = "restore-1"
        print(f"  wall={result.get('_wall_s', '?')}s snap_ok={result.get('snap_ok')} restore_ok={result.get('restore_ok')}")
        if result.get("restore_error"):
            print(f"  restore_error: {result['restore_error']}")
        if result.get("_error"):
            print(f"  call_error: {result['_error']}")
        results[f"{label}-phase2"] = result

    print("\n" + "=" * 72)
    print("Waiting 100s for containers to scale to zero...")
    print("=" * 72)
    time.sleep(100)

    print("\n" + "=" * 72)
    print("PHASE 3: Restore test 2")
    print("=" * 72)

    for label, cls_name in PROBE_CLASSES:
        print(f"\n--- {label} ({cls_name}) ---")
        result = call_probe(label, cls_name)
        result["_phase"] = "restore-2"
        print(f"  wall={result.get('_wall_s', '?')}s snap_ok={result.get('snap_ok')} restore_ok={result.get('restore_ok')}")
        if result.get("restore_error"):
            print(f"  restore_error: {result['restore_error']}")
        if result.get("_error"):
            print(f"  call_error: {result['_error']}")
        results[f"{label}-phase3"] = result

    print("\n" + "=" * 72)
    print("RESULTS MATRIX")
    print("=" * 72)

    header = (
        f"{'Probe':<28} {'AVTR':>5} {'AVTR':>7} {'Omni':>5} {'Omni':>6} "
        f"{'TRT/':>6} {'Snap':>5} {'Rstr1':>6} {'Rstr2':>6} {'Wall1':>7} {'Wall2':>7} {'Wall3':>7}"
    )
    subheader = (
        f"{'':>28} {'load':>5} {'warmup':>7} {'load':>5} {'gen':>6} "
        f"{'ONNX':>6} {'ok':>5} {'ok':>6} {'ok':>6} {'s':>7} {'s':>7} {'s':>7}"
    )
    print(header)
    print(subheader)
    print("-" * len(header))

    for label, cls_name in PROBE_CLASSES:
        phase1 = results.get(f"{label}-phase1", {})
        phase2 = results.get(f"{label}-phase2", {})
        phase3 = results.get(f"{label}-phase3", {})

        avtr_load = "Y" if label in ("2-avtr-load-only", "3-avtr-warmup", "6-combined-load-only", "7-combined-warmup") else "-"
        avtr_warm = "Y" if label in ("3-avtr-warmup", "7-combined-warmup") else "-"
        omni_load = "Y" if label in ("4-omnivoice-load-only", "5-omnivoice-generate", "6-combined-load-only", "7-combined-warmup") else "-"
        omni_gen = "Y" if label in ("5-omnivoice-generate",) else "-"
        trt_onnx = "Y" if label not in ("1-minimal-cuda", "4-omnivoice-load-only", "5-omnivoice-generate") else "-"

        snap_ok = str(phase1.get("snap_ok", "?"))
        rstr1_ok = "FAIL" if phase1.get("_error") else str(phase2.get("restore_ok", "?"))
        rstr2_ok = "FAIL" if phase1.get("_error") else str(phase3.get("restore_ok", "?"))

        wall1 = f"{phase1.get('_wall_s', '?')}"
        wall2 = f"{phase2.get('_wall_s', '?')}"
        wall3 = f"{phase3.get('_wall_s', '?')}"

        print(
            f"{label:<28} {avtr_load:>5} {avtr_warm:>7} {omni_load:>5} {omni_gen:>6} "
            f"{trt_onnx:>6} {snap_ok:>5} {rstr1_ok:>6} {rstr2_ok:>6} {wall1:>7} {wall2:>7} {wall3:>7}"
        )

    print()
    print("CONCLUSIONS:")
    for label, cls_name in PROBE_CLASSES:
        phase1 = results.get(f"{label}-phase1", {})
        phase2 = results.get(f"{label}-phase2", {})
        phase3 = results.get(f"{label}-phase3", {})

        if phase1.get("_error"):
            verdict = "CALL FAILED (timeout/crash during invocation)"
        elif not phase1.get("snap_ok", False):
            verdict = f"SNAP INIT FAILED: {phase1.get('snap_error', 'unknown')}"
        elif phase2.get("_error") or phase3.get("_error"):
            verdict = f"RESTORE CRASH: p2={phase2.get('_error','ok')} p3={phase3.get('_error','ok')}"
        elif phase2.get("restore_ok") and phase3.get("restore_ok"):
            verdict = "PASS — snapshot + restore both work"
        elif not phase2.get("restore_ok") or not phase3.get("restore_ok"):
            err = phase2.get("restore_error") or phase3.get("restore_error") or "unknown"
            verdict = f"RESTORE ERROR: {err}"
        else:
            verdict = "UNKNOWN"
        print(f"  {label}: {verdict}")

    with open("/tmp/snapshot_probe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to /tmp/snapshot_probe_results.json")


if __name__ == "__main__":
    main()
