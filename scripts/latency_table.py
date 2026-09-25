#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Report streamer-side turn latency from a session log.

The streamer's LatencyWorklet writes one ``Latency contribution measured`` line
to stdout per avatar reply and per interruption. This script parses those lines
and reports the decomposition in the style of the technical report's measured
streamer-latency table: for the response latency contribution (RLC, ``kind =
turn_start``) and the interruption latency contribution (ILC, ``kind =
interruption``), the mean and ``[min, max]`` of each term in milliseconds.

It pairs each reply with its interruption by segment id and keeps the first N
complete exchanges, so the RLC and ILC samples are paired and equal in size (as
the report specifies). If no exchange carries both, it falls back to reporting
every reply and every interruption independently.

Usage:
    pixi run interactive-demo > run.log        # collect a session
    python scripts/latency_table.py run.log    # one agent
    cat run.log | python scripts/latency_table.py
    python scripts/latency_table.py cartesia.log openai.log
"""

from __future__ import annotations

import argparse
import re
import sys
from statistics import mean

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_KV = re.compile(r"(\w+)=(\S+)")
_MARKER = "Latency contribution measured"

# Row order and labels, matching the report table.
_TERMS = ["t_wait", "t_pad", "t_present", "t_lead"]
_ROW_LABEL = {
    "t_wait": "T_wait",
    "t_pad": "T_pad",
    "t_present": "T_present",
    "t_lead": "T_lead",
    "t_total": "Total",
}
_AGENT_NAMES = {"cartesia": "Cartesia Line", "openai": "OpenAI Realtime API"}


def parse_records(lines):
    """Yield one dict per latency line: kind, segment_id, and the term seconds."""
    for raw in lines:
        line = _ANSI.sub("", raw)
        if _MARKER not in line:
            continue
        kv = dict(_KV.findall(line))
        try:
            yield {
                "kind": kv["kind"],
                "segment_id": kv["segment_id"],
                "t_wait": float(kv["t_wait"]),
                "t_pad": float(kv["t_pad"]),
                "t_present": float(kv["t_present"]),
                "t_lead": float(kv["t_lead"]),
                "t_total": float(kv["t_total"]),
            }
        except (KeyError, ValueError):
            continue  # a line missing a field is skipped, not fatal


def agent_of(segment_id):
    prefix = segment_id.split("-", 1)[0].lower()
    return _AGENT_NAMES.get(prefix, prefix.capitalize() or "unknown")


def select(records, n):
    """Return (rlc, ilc) record lists for up to n exchanges of one agent.

    Keeps only segments seen as both a reply and an interruption, in first-seen
    order, so the two lists have equal length. If none carry both, falls back to
    the first n of each kind independently.
    """
    starts, interrupts, order = {}, {}, []
    for r in records:
        sid = r["segment_id"]
        if sid not in starts and sid not in interrupts:
            order.append(sid)
        if r["kind"] == "turn_start":
            starts.setdefault(sid, r)
        elif r["kind"] == "interruption":
            interrupts.setdefault(sid, r)

    both = [s for s in order if s in starts and s in interrupts][:n]
    if both:
        return [starts[s] for s in both], [interrupts[s] for s in both]
    # No complete exchanges: report each kind independently.
    rlc = [starts[s] for s in order if s in starts][:n]
    ilc = [interrupts[s] for s in order if s in interrupts][:n]
    return rlc, ilc


def cell(records, key):
    """Format one term's mean and [min, max] in ms, or a single constant value."""
    if not records:
        return "--"
    vals = [r[key] * 1000.0 for r in records]
    m, lo, hi = mean(vals), min(vals), max(vals)
    if abs(hi - lo) < 0.5:  # constant within the 1 ms logging resolution
        return f"{m:.1f}"
    return f"{m:.1f} [{round(lo)}, {round(hi)}]"


def report(agent, rlc, ilc):
    print(f"\n=== {agent}   RLC n={len(rlc)}   ILC n={len(ilc)} ===")
    print(f"{'Term':<11}{'RLC':<22}{'ILC':<22}")
    for key in _TERMS + ["t_total"]:
        if key == "t_total":
            print("-" * 55)
        print(f"{_ROW_LABEL[key]:<11}{cell(rlc, key):<22}{cell(ilc, key):<22}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="*", help="log files (default: read stdin)")
    ap.add_argument("-n", type=int, default=10, help="exchanges per agent (default: 10)")
    args = ap.parse_args(argv)

    if args.logs:
        lines = []
        for path in args.logs:
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines.extend(fh)
    else:
        lines = sys.stdin.readlines()

    # Group records by agent (from the segment-id prefix).
    by_agent: dict[str, list] = {}
    for rec in parse_records(lines):
        by_agent.setdefault(agent_of(rec["segment_id"]), []).append(rec)

    if not by_agent:
        print("No 'Latency contribution measured' lines found in the input.", file=sys.stderr)
        return 1

    for agent in sorted(by_agent):
        rlc, ilc = select(by_agent[agent], args.n)
        report(agent, rlc, ilc)
        if not ilc:
            print(f"note: no interruptions for {agent}; showing all replies (RLC) unpaired.",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
