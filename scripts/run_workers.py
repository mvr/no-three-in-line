#!/usr/bin/env python3
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def detect_gpus():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        # Remapped to 0..N-1 inside the process.
        entries = [v for v in visible.split(",") if v.strip()]
        return list(range(len(entries)))
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        count = len([line for line in output.splitlines() if line.strip().startswith("GPU")])
        if count > 0:
            return list(range(count))
    except (OSError, subprocess.SubprocessError):
        pass
    return [0]


def format_eta(seconds: float) -> str:
    if seconds < 0:
        return "unknown"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days > 0:
        return f"{days}d{hours:02d}h"
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def update_shard_times(stats_path: Path, state: dict):
    if not stats_path.exists():
        return
    with stats_path.open("r", encoding="utf-8") as f:
        f.seek(state["offset"])
        for line in f:
            try:
                duration = float(line.strip())
            except ValueError:
                continue
            state["total_seconds"] += duration
            state["total_shards"] += 1
        state["offset"] = f.tell()


def extract_job_id(name: str):
    parts = name.split("job_")
    if len(parts) < 2:
        return None
    tail = parts[-1]
    if not tail.endswith(".txt"):
        return None
    return "job_" + tail


def claim_job(pending_dir: Path, running_dir: Path):
    candidates = sorted(pending_dir.glob("*.txt"))
    if not candidates:
        return None
    host = socket.gethostname()
    pid = os.getpid()
    for path in candidates:
        dest = running_dir / f"{host}-{pid}-{path.name}"
        try:
            path.rename(dest)
            return dest
        except OSError:
            continue
    return None


def parse_job(path: Path):
    shards = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or "|" not in stripped:
                continue
            shards.append(stripped)
    return shards


def run_shard(three_bin: str, on_rle: str, off_rle: str, gpu: int, out_file):
    cmd = [three_bin, "--seed-on", on_rle, "--seed-off", off_rle]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    cmd_str = " ".join(cmd)
    print(f"[GPU {str(gpu)}] ON {on_rle}, OFF {off_rle}", flush=True)
    out_file.write(f"# cmd: {' '.join(cmd)}\n")
    out_file.flush()
    res = subprocess.run(cmd, stdout=out_file, stderr=subprocess.STDOUT)
    return res.returncode


def process_job(job_path: Path, results_dir: Path, three_bin: str, gpu: int, shard_times_path: Path):
    shards = parse_job(job_path)
    if not shards:
        return True
    results_dir.mkdir(parents=True, exist_ok=True)
    job_id = extract_job_id(job_path.name)
    out_name = (job_id if job_id else job_path.name) + ".out"
    out_path = results_dir / out_name
    ok = True
    with out_path.open("w", encoding="utf-8") as out_file:
        for idx, shard in enumerate(shards):
            on_rle, off_rle = shard.split("|", 1)
            out_file.write(f"# shard {idx}\n")
            shard_start = time.time()
            rc = run_shard(three_bin, on_rle, off_rle, gpu, out_file)
            shard_end = time.time()
            duration = shard_end - shard_start
            out_file.write(f"# shard_stats idx={idx} seconds={duration:.3f} rc={rc}\n")
            if rc != 0:
                ok = False
                out_file.write(f"# shard {idx} failed: rc={rc}\n")
                out_file.flush()
                try:
                    with out_path.open("r", encoding="utf-8") as failed_log:
                        lines = failed_log.readlines()
                    tail = "".join(lines[-20:])
                    print(f"[worker] shard {idx} failed (rc={rc}). Last output:\n{tail}", flush=True)
                except OSError:
                    print(f"[worker] shard {idx} failed (rc={rc}). Unable to read log.", flush=True)
                break
            with shard_times_path.open("a", encoding="utf-8") as times_file:
                times_file.write(f"{duration:.6f}\n")
    return ok


def worker_loop(args):
    queue_dir = Path(args.queue_dir)
    pending_dir = queue_dir / "pending"
    running_dir = queue_dir / "running"
    done_dir = queue_dir / "done"
    failed_dir = queue_dir / "failed"
    results_dir = queue_dir / "results"
    shard_times_path = queue_dir / "shard_times.txt"
    for d in (pending_dir, running_dir, done_dir, failed_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    while True:
        job = claim_job(pending_dir, running_dir)
        if job is None:
            time.sleep(args.poll_interval)
            continue

        ok = process_job(job, results_dir, args.three, args.gpu, shard_times_path)
        if ok:
            job.rename(done_dir / job.name)
        else:
            job.rename(failed_dir / job.name)


def main():
    parser = argparse.ArgumentParser(description="Run multiple worker processes on one machine.")
    parser.add_argument("--three", default="./three", help="Path to solver executable")
    parser.add_argument("--queue-dir", default="queue", help="Queue directory root")
    # One worker per GPU; no --once mode.
    # Workers are always respawned if they exit unexpectedly.
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Worker poll interval seconds")
    parser.add_argument("--requeue-running", action="store_true",
                        help="Move any existing running jobs back to pending before starting")
    parser.add_argument("--stats-interval", type=float, default=60.0,
                        help="Seconds between queue status reports (0 to disable)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=None, help=argparse.SUPPRESS)
    # Defaults: exit when queue drains and concatenate solutions.
    args = parser.parse_args()

    if args.worker:
        if args.gpu is None:
            print("[worker] missing --gpu", file=sys.stderr)
            return 1
        worker_loop(args)
        return 0

    gpus = detect_gpus()
    if not gpus:
        print("[error] no GPUs specified", file=sys.stderr)
        return 1

    queue_dir = Path(args.queue_dir)
    if args.requeue_running:
        running_dir = queue_dir / "running"
        pending_dir = queue_dir / "pending"
        running_dir.mkdir(parents=True, exist_ok=True)
        pending_dir.mkdir(parents=True, exist_ok=True)
        for path in running_dir.glob("*.txt"):
            dest = pending_dir / path.name
            try:
                path.rename(dest)
            except OSError:
                continue

    pending_dir = queue_dir / "pending"
    running_dir = queue_dir / "running"
    done_dir = queue_dir / "done"
    failed_dir = queue_dir / "failed"
    shard_times_path = queue_dir / "shard_times.txt"
    combined_path = queue_dir / "solutions.txt"
    done_dir.mkdir(parents=True, exist_ok=True)
    results_dir = queue_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    last_stats = 0.0
    empty_checks = 0
    stats_state = {"offset": 0, "total_seconds": 0.0, "total_shards": 0}

    def spawn_worker(gpu_id: int):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--three", args.three,
            "--queue-dir", args.queue_dir,
            "--gpu", str(gpu_id),
            "--poll-interval", str(args.poll_interval),
        ]
        return subprocess.Popen(cmd)

    procs = []
    for gpu in gpus:
        procs.append((gpu, spawn_worker(gpu)))

    while True:
        alive = []
        for gpu, proc in procs:
            ret = proc.poll()
            if ret is None:
                alive.append((gpu, proc))
                continue
            alive.append((gpu, spawn_worker(gpu)))
        procs = alive
        if args.stats_interval > 0:
            now = time.time()
            if now - last_stats >= args.stats_interval:
                pending = len(list(pending_dir.glob("*.txt")))
                running = len(list(running_dir.glob("*.txt")))
                done = len(list(done_dir.glob("*.txt")))
                failed = len(list(failed_dir.glob("*.txt")))
                total = pending + running + done + failed
                update_shard_times(shard_times_path, stats_state)
                eta = "unknown"
                avg_shard = "unknown"
                if stats_state["total_shards"] > 0:
                    remaining_shards = pending + running
                    avg = stats_state["total_seconds"] / stats_state["total_shards"]
                    eta_seconds = remaining_shards * avg
                    eta = format_eta(eta_seconds)
                    avg_shard = f"{avg:.1f}s"
                print(f"[queue] pending={pending} running={running} done={done} failed={failed} total={total} "
                      f"avg_shard={avg_shard} eta={eta}", flush=True)
                last_stats = now

        pending = len(list(pending_dir.glob("*.txt")))
        running = len(list(running_dir.glob("*.txt")))
        if pending + running == 0:
            empty_checks += 1
        else:
            empty_checks = 0
        if empty_checks >= 2:
            for _, proc in procs:
                proc.terminate()
            solutions_count = 0
            done_job_ids = set()
            for done_path in done_dir.glob("*.txt"):
                job_id = extract_job_id(done_path.name)
                if job_id:
                    done_job_ids.add(job_id)
            with combined_path.open("w", encoding="utf-8") as out_file:
                for shard_path in sorted(results_dir.glob("*.out")):
                    if done_job_ids:
                        base = shard_path.name
                        if base.endswith(".out"):
                            base = base[:-4]
                        if base not in done_job_ids:
                            continue
                    with shard_path.open("r", encoding="utf-8") as shard_file:
                        for line in shard_file:
                            if line.startswith("#"):
                                continue
                            if line.startswith("[stats]"):
                                continue
                            out_file.write(line)
                            solutions_count += 1
            update_shard_times(shard_times_path, stats_state)
            elapsed = stats_state["total_seconds"]
            done = len(list(done_dir.glob("*.txt")))
            print(f"[queue] done shards={done} solutions={solutions_count} elapsed={format_eta(elapsed)}", flush=True)
            return 0

        if not procs:
            return 0

        time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())
