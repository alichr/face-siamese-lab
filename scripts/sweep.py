"""Round-robin sweep runner: keep all GPUs busy until the matrix is done (plan §10).

    .venv/bin/python scripts/sweep.py configs/e1_*.yaml configs/e2_*.yaml
    .venv/bin/python scripts/sweep.py --all

Each config is a separate single-GPU process pinned with `CUDA_VISIBLE_DEVICES`.
When one finishes, the next queued config starts on that GPU. A live status table
is rewritten to `results/sweep_status.md` after every state change, so progress
is inspectable without attaching to the process.

Operational rules from the Phase 5 brief:
  * a crashed or NaN run is recorded and retried **once** with the same config;
  * if it fails twice it is flagged and the sweep moves on -- hyperparameters are
    never silently changed to make a run succeed;
  * runs already holding a complete `metrics.json` are skipped, so the sweep is
    resumable after an interruption.

The DDP row (`negatives: global`) needs all GPUs at once and is skipped here;
run it separately with torchrun.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

STATUS_PATH = Path("results/sweep_status.md")
LOG_DIR = Path("results/_logs")


@dataclass
class Job:
    config: Path
    name: str
    attempts: int = 0
    status: str = "queued"  # queued | running | done | skipped | FAILED
    gpu: int | None = None
    started: float | None = None
    seconds: float | None = None
    headline: str = ""
    error: str = field(default="")


def is_complete(config: Path) -> bool:
    """True when the run already has a metrics.json with a real result."""
    cfg = yaml.safe_load(config.read_text())
    metrics = Path(cfg.get("output_dir", f"results/{config.stem}")) / "metrics.json"
    if not metrics.exists():
        return False
    try:
        data = json.loads(metrics.read_text())
    except json.JSONDecodeError:
        return False
    return "internal" in data or "final" in data


def is_ddp(config: Path) -> bool:
    cfg = yaml.safe_load(config.read_text())
    return cfg.get("loss", {}).get("negatives") == "global"


def headline_of(config: Path) -> str:
    """Short result string for the status table."""
    cfg = yaml.safe_load(config.read_text())
    metrics = Path(cfg.get("output_dir", f"results/{config.stem}")) / "metrics.json"
    if not metrics.exists():
        return ""
    try:
        data = json.loads(metrics.read_text())
    except json.JSONDecodeError:
        return ""
    lfw = data.get("lfw", {})
    internal = data.get("internal", {})
    parts = []
    if isinstance(lfw, dict) and "mean" in lfw:
        parts.append(f"LFW {lfw['mean']:.4f}±{lfw['std']:.4f}")
    if "auc" in internal:
        parts.append(f"AUC {internal['auc']:.4f}")
    return " · ".join(parts)


def write_status(jobs: list[Job], started: float) -> None:
    """Rewrite the live status table."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1

    elapsed = time.time() - started
    lines = [
        "# Sweep status",
        "",
        f"Elapsed **{elapsed / 60:.1f} min** · "
        + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| run | status | gpu | min | result |",
        "|---|---|---|---|---|",
    ]
    for job in jobs:
        minutes = f"{job.seconds / 60:.1f}" if job.seconds else ""
        gpu = "" if job.gpu is None else str(job.gpu)
        result = job.headline or (job.error[:60] if job.error else "")
        lines.append(f"| `{job.name}` | {job.status} | {gpu} | {minutes} | {result} |")

    STATUS_PATH.write_text("\n".join(lines) + "\n")


def launch(job: Job, gpu: int) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOG_DIR / f"{job.name}.log").open("w")
    env_prefix = {"CUDA_VISIBLE_DEVICES": str(gpu)}

    import os

    env = {**os.environ, **env_prefix}
    return subprocess.Popen(
        [sys.executable, "-m", "src.engine.train", "--config", str(job.config)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="every configs/e*.yaml")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="rerun even if complete")
    args = ap.parse_args()

    paths = sorted(Path("configs").glob("e[1-8]_*.yaml")) if args.all else sorted(args.configs)
    if not paths:
        ap.error("no configs given (use --all)")

    jobs: list[Job] = []
    for path in paths:
        if is_ddp(path):
            print(f"skipping DDP config {path.name} — run it with torchrun")
            continue
        job = Job(config=path, name=path.stem)
        if not args.force and is_complete(path):
            job.status = "skipped"
            job.headline = headline_of(path)
        jobs.append(job)

    queue = [j for j in jobs if j.status == "queued"]
    running: dict[int, tuple[Job, subprocess.Popen]] = {}
    started = time.time()

    print(f"{len(queue)} run(s) queued, {len(jobs) - len(queue)} already complete")
    print(f"GPUs: {args.gpus} · status table: {STATUS_PATH}\n")
    write_status(jobs, started)

    while queue or running:
        # Fill idle GPUs.
        for gpu in args.gpus:
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            job.attempts += 1
            job.status = "running"
            job.gpu = gpu
            job.started = time.time()
            running[gpu] = (job, launch(job, gpu))
            print(f"[gpu {gpu}] start {job.name} (attempt {job.attempts})", flush=True)
            write_status(jobs, started)

        time.sleep(5)

        for gpu, (job, process) in list(running.items()):
            code = process.poll()
            if code is None:
                continue

            del running[gpu]
            job.seconds = time.time() - (job.started or time.time())

            if code == 0 and is_complete(job.config):
                job.status = "done"
                job.headline = headline_of(job.config)
                print(
                    f"[gpu {gpu}] done  {job.name} in {job.seconds / 60:.1f} min "
                    f"— {job.headline}",
                    flush=True,
                )
            elif job.attempts < args.max_attempts:
                # Retry once with the SAME config -- never adjust hyperparameters
                # to make a run succeed (plan §14.7).
                job.status = "queued"
                job.gpu = None
                queue.append(job)
                print(f"[gpu {gpu}] FAIL  {job.name} (exit {code}) — retrying once", flush=True)
            else:
                job.status = "FAILED"
                job.error = f"exit {code} after {job.attempts} attempts"
                print(f"[gpu {gpu}] FAILED {job.name} — {job.error}", flush=True)

            write_status(jobs, started)

    write_status(jobs, started)
    failed = [j.name for j in jobs if j.status == "FAILED"]
    print(f"\nsweep finished in {(time.time() - started) / 60:.1f} min")
    print(f"  done {sum(j.status == 'done' for j in jobs)} · "
          f"skipped {sum(j.status == 'skipped' for j in jobs)} · failed {len(failed)}")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
