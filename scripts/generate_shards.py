#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_shard_file(path: Path):
    header = []
    shards = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if "|" in stripped:
                shards.append(stripped)
            else:
                header.append(stripped)
    return header, shards


def write_jobs(header, shards, queue_dir: Path):
    pending_dir = queue_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    job_id = 0
    for line in shards:
        job_id += 1
        job_path = pending_dir / f"job_{job_id:06d}.txt"
        with job_path.open("w", encoding="utf-8") as f:
            for h in header:
                f.write(h + "\n")
            f.write(line + "\n")
    return job_id


def run_frontier(frontier_bin: str, out_path: Path, min_on, max_on, steps):
    cmd = [frontier_bin, "--steps", str(steps), "--out", str(out_path)]
    if min_on is not None:
        cmd += ["--min-on", str(min_on)]
    if max_on is not None:
        cmd += ["--max-on", str(max_on)]
    print(f"[frontier] running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"frontier failed (exit {res.returncode})")


def tune_frontier(frontier_bin: str, out_path: Path, min_on, max_on, steps, target, tolerance, max_iters):
    steps = max(1, steps)
    for _ in range(max_iters):
        run_frontier(frontier_bin, out_path, min_on, max_on, steps)
        _, shards = parse_shard_file(out_path)
        count = len(shards)
        print(f"[frontier] steps={steps} shards={count}")
        if target is None or count == 0:
            return steps, count
        ratio = target / count
        if abs(1.0 - ratio) <= tolerance:
            return steps, count
        new_steps = max(1, int(steps * ratio))
        if new_steps == steps:
            new_steps = steps + 1 if ratio > 1 else max(1, steps - 1)
        steps = new_steps
    return steps, count


def main():
    parser = argparse.ArgumentParser(description="Generate frontier shards and populate a queue.")
    parser.add_argument("--frontier-bin", default="./three_frontier",
                        help="Path to the frontier executable")
    parser.add_argument("--queue-dir", default="queue", help="Queue directory root")
    parser.add_argument("--shards-file", default=None, help="Use an existing shard list instead of running frontier")
    parser.add_argument("--out", default="shards.txt", help="Output shard list path (when generating)")
    parser.add_argument("--min-on", type=int, default=None)
    parser.add_argument("--max-on", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--target-shards", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=10)
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "pending").mkdir(exist_ok=True)
    (queue_dir / "running").mkdir(exist_ok=True)
    (queue_dir / "done").mkdir(exist_ok=True)
    (queue_dir / "failed").mkdir(exist_ok=True)

    if args.shards_file:
        shard_path = Path(args.shards_file)
        if not shard_path.exists():
            print(f"[error] shard file not found: {shard_path}", file=sys.stderr)
            return 1
    else:
        shard_path = Path(args.out)
        steps, count = tune_frontier(
            args.frontier_bin,
            shard_path,
            args.min_on,
            args.max_on,
            args.steps,
            args.target_shards,
            args.tolerance,
            args.max_iters,
        )
        print(f"[frontier] final steps={steps} shards={count}")

    header, shards = parse_shard_file(shard_path)
    if not shards:
        print("[error] no shards found to enqueue", file=sys.stderr)
        return 1

    total_jobs = write_jobs(header, shards, queue_dir)
    print(f"[queue] wrote {len(shards)} shards into {total_jobs} jobs at {queue_dir / 'pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
