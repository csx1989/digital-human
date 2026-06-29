"""
Invoke a deployed AVTR1Snapshot class (T4 or L4).

GPU memory snapshots are only created for DEPLOYED apps — `modal run`
does NOT create/restore snapshots.

Usage:
    python invoke_snapshot.py --method health
    python invoke_snapshot.py --method startup
    python invoke_snapshot.py --method benchmark --chunks 10
    python invoke_snapshot.py --method health --class AVTR1SnapshotL4
    python invoke_snapshot.py --method startup --class AVTR1SnapshotL4
    python invoke_snapshot.py --method benchmark --chunks 10 --class AVTR1SnapshotL4
    python invoke_snapshot.py --method profile --chunks 10 --class AVTR1SnapshotL4

Prerequisites:
    modal deploy modal_snapshot_smoke.py
"""

from __future__ import annotations

import argparse
import time

import modal

APP_NAME = "avtr1-snapshot-smoke"
DEFAULT_CLASS = "AVTR1Snapshot"
L4_BASELINE_COLD_START_S = 6.5
T4_BASELINE_COLD_START_S = 24.0


def _get_cls(class_name: str) -> type:
    return modal.Cls.from_name(APP_NAME, class_name)


def cmd_health(class_name: str) -> None:
    inst = _get_cls(class_name)()
    result = inst.health.remote()
    print(f"Health ({class_name}): {result}")
    if not result.get("pipeline_loaded"):
        print("WARNING: Pipeline not loaded. Snap init may have failed.")


def cmd_startup(class_name: str) -> None:
    inst = _get_cls(class_name)()
    print(f"Measuring first-render latency ({class_name}, includes snapshot restore)...")
    print("  Timing from caller perspective (includes Modal overhead)...")

    baseline_s = (L4_BASELINE_COLD_START_S
                  if "L4" in class_name
                  else T4_BASELINE_COLD_START_S)

    t0 = time.monotonic()
    result = inst.benchmark_first_render.remote()
    caller_elapsed = time.monotonic() - t0
    print()
    print(f"  Caller-side total:   {caller_elapsed:.1f}s")
    if "error" in result:
        print(f"  Container-side:      ERROR: {result['error']}")
    else:
        print(f"  Container render:    {result['render_ms']:.0f}ms")
        print(f"  Frames:              {result['frame_count']}")
    print()
    print(f"  Baseline cold start (no snapshot): ~{baseline_s:.1f}s")
    if "error" not in result:
        speedup = baseline_s / caller_elapsed if caller_elapsed > 0 else float("inf")
        print(f"  Snapshot speedup:    ~{speedup:.1f}x")
    print()
    print("  Look for 'Snapshot created' / 'Restoring Function from memory snapshot'")
    print("  in Modal logs to confirm snapshot was used.")


def cmd_benchmark(class_name: str, chunks: int) -> None:
    inst = _get_cls(class_name)()
    print(f"Benchmarking snapshot-restored pipeline ({class_name}, {chunks} chunks)...")
    stats = inst.benchmark_render.remote(chunks=chunks)
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return
    print()
    print(f"  Snapshot p95: {stats['p95_ms']:.1f}ms")
    print(f"  Snapshot p50: {stats['p50_ms']:.1f}ms")
    print(f"  Snapshot FPS: {stats['fps']:.1f}")
    baseline_p95 = 188.0 if "L4" in class_name else 367.7
    print(f"  Baseline p95: {baseline_p95:.1f}ms")


def cmd_profile(class_name: str, chunks: int) -> None:
    inst = _get_cls(class_name)()
    print(f"Profiling snapshot-restored pipeline ({class_name}, {chunks} chunks)...")
    results = inst.benchmark_render_profiled.remote(chunks=chunks)
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    if "error" in results.get("profiled", {}):
        print(f"ERROR (profiled): {results['profiled']['error']}")
        return
    prof = results.get("profiled", {})
    non_prof = results.get("non_profiled", {})
    print()
    print(f"  Non-profiled p95: {non_prof.get('p95_ms', 0):.1f}ms")
    print(f"  Profiled p95:     {prof.get('p95_ms', 0):.1f}ms")
    print(f"  Sync overhead:    {results.get('profiling_overhead_ms', 0):+.1f}ms "
          f"({results.get('profiling_overhead_pct', 0):+.1f}%)")
    print()
    print("  NOTE: Profiled timings are DIAGNOSTIC, not realtime verdict numbers.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoke a deployed AVTR1Snapshot class on Modal.",
    )
    parser.add_argument(
        "--method",
        choices=["health", "startup", "benchmark", "profile"],
        required=True,
    )
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument(
        "--class",
        dest="class_name",
        default=DEFAULT_CLASS,
        choices=[DEFAULT_CLASS, "AVTR1SnapshotL4"],
        help=f"Snapshot class to invoke (default: {DEFAULT_CLASS})",
    )
    args = parser.parse_args()

    if args.method == "profile" and args.class_name != "AVTR1SnapshotL4":
        parser.error(
            "--method profile requires --class AVTR1SnapshotL4 "
            "(AVTR1Snapshot has no benchmark_render_profiled method)"
        )

    if args.method == "health":
        cmd_health(args.class_name)
    elif args.method == "startup":
        cmd_startup(args.class_name)
    elif args.method == "benchmark":
        cmd_benchmark(args.class_name, args.chunks)
    elif args.method == "profile":
        cmd_profile(args.class_name, args.chunks)


if __name__ == "__main__":
    main()
