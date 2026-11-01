#!/usr/bin/env python3
"""
run.py — CFR pipeline orchestrator.

Runs the data-preparation stages in order and then the WAP scoring step, all
sharing ONE data root and ONE run tag, so every stage reads from and writes to
the same ``cfr_data/processed/<run_tag>/`` folder.

Each stage is launched as its own process with ``CFR_DATA_ROOT``,
``CFR_RUN_TAG`` and ``CFR_ITERATIONS`` set in its environment — the exact same
contract the stages already use when run by hand. Fixing the run tag once here
means a run started late at night can't drift onto two different default dates
part-way through.

Stage order (stage2a MUST precede stage2, which reads stage2a's output):

    stage0  -> stage1 -> stage2a -> stage2_map_coordinates -> stage2b -> WAP

Examples
--------
    # Full run, GPU scoring, default data root (./cfr_data), today's run tag:
    python3 run.py --backend gpu

    # CPU scoring, custom root + tag, 10^6 permutations:
    python3 run.py --backend cpu --data-root /data/cfr --run-tag run_2026-11-01 --iterations 1000000

    # Raw inputs already downloaded; smoke-test on 20 genes:
    python3 run.py --skip-stage0 --limit 20

    # Only the data stages, no scoring:
    python3 run.py --no-wap
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Which WAP scorer runs for each backend. Rename here if your files differ.
WAP_FILES = {"cpu": "wap_cpu.py", "gpu": "wap_gpu.py"}


def run_step(script: str, env: dict, extra_args: list[str] | None = None) -> None:
    """Launch one stage as a child process; abort the whole run if it fails."""
    extra_args = extra_args or []
    path = HERE / script
    if not path.exists():
        sys.exit(f"[run] missing script: {path}")
    cmd = [sys.executable, str(path), *extra_args]
    print(f"\n{'=' * 70}\n[run] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"[run] step FAILED ({script}): exit code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CFR pipeline orchestrator")
    ap.add_argument(
        "--data-root",
        default=os.environ.get("CFR_DATA_ROOT", "./cfr_data"),
        help="where raw/ and processed/ live (default ./cfr_data)",
    )
    ap.add_argument(
        "--run-tag",
        default=os.environ.get("CFR_RUN_TAG", f"run_{date.today().isoformat()}"),
        help="output folder name under processed/ (default run_<today>)",
    )
    ap.add_argument(
        "--backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="which WAP scorer to run (default cpu)",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=10**6,
        help="permutations for the WAP null model (default 1000000)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap genes/structures for a quick smoke test (0 = all)",
    )
    ap.add_argument(
        "--skip-stage0",
        action="store_true",
        help="skip the bulk download (raw inputs already present)",
    )
    ap.add_argument(
        "--no-wap",
        action="store_true",
        help="stop after the data stages (2b); do not run WAP",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()

    # One environment, inherited by every stage.
    env = dict(os.environ)
    env["CFR_DATA_ROOT"] = str(data_root)
    env["CFR_RUN_TAG"] = args.run_tag
    env["CFR_ITERATIONS"] = str(args.iterations)

    print(f"[run] CFR_DATA_ROOT = {data_root}")
    print(f"[run] CFR_RUN_TAG   = {args.run_tag}")
    print(f"[run] backend       = {args.backend}")
    print(f"[run] iterations    = {args.iterations}")
    if args.limit:
        print(f"[run] limit         = {args.limit} (smoke test)")

    # stage2_map_coordinates takes a positional limit; stage2b takes --limit.
    limit_positional = [str(args.limit)] if args.limit else []
    limit_flag = ["--limit", str(args.limit)] if args.limit else []

    # --- Data-preparation stages, in order ---
    if args.skip_stage0:
        print("[run] skipping stage0 (--skip-stage0)")
    else:
        run_step("stage0.py", env)
    run_step("stage1.py", env)
    run_step("stage2a_structselect.py", env)
    run_step("stage2_map_coordinates.py", env, limit_positional)
    run_step("stage2b_residues.py", env, limit_flag)

    # --- Scoring ---
    if args.no_wap:
        print("[run] done (data stages only, --no-wap)")
        return
    run_step(WAP_FILES[args.backend], env)

    print(f"\n[run] pipeline complete. Outputs in "
          f"{data_root / 'processed' / args.run_tag}")


if __name__ == "__main__":
    main()