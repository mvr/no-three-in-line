#!/usr/bin/env python3
import argparse
import os
import socket
import sqlite3
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def detect_gpus():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
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


def claim_shard(conn: sqlite3.Connection, worker_id: str):
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    row = cur.execute(
        "SELECT id,on_rle,off_rle FROM shards WHERE status='pending' ORDER BY id LIMIT 1;"
    ).fetchone()
    if row is None:
        conn.execute("COMMIT;")
        return None
    shard_id = row[0]
    now = time.time()
    cur.execute(
        "UPDATE shards SET status='running', claimed_by=?, claimed_at=?, started_at=? WHERE id=?;",
        (worker_id, now, now, shard_id),
    )
    conn.commit()
    return row


def finish_shard(conn: sqlite3.Connection, shard_id: int, rc: int, duration: float, solutions, log_tail):
    status = "done" if rc == 0 else "failed"
    finished = time.time()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    cur.execute(
        "UPDATE shards SET status=?, finished_at=?, duration=?, rc=?, log_tail=? WHERE id=?;",
        (status, finished, duration, rc, log_tail, shard_id),
    )
    if rc == 0 and solutions:
        cur.executemany(
            "INSERT INTO solutions (shard_id, rle) VALUES (?, ?);",
            [(shard_id, rle) for rle in solutions],
        )
    conn.commit()


def run_shard_stream(three_bin: str, on_rle: str, off_rle: str, gpu: int):
    cmd = [three_bin, "--seed-on", on_rle, "--seed-off", off_rle]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    print(f"[GPU {str(gpu)}] ON {on_rle}, OFF {off_rle}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    solutions = []
    tail = deque(maxlen=50)
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if not line:
                continue
            if line.startswith("#") or line.startswith("[stats]") or line.startswith("["):
                continue
            solutions.append(line)
    rc = proc.wait()
    return rc, solutions, "\n".join(tail)


def worker_loop(args):
    queue_dir = Path(args.queue_dir)
    db_path = queue_dir / "queue.db"
    conn = sqlite3.connect(db_path, timeout=5, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000;")
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{args.gpu}"

    if not args.no_requeue_running:
        conn.execute("UPDATE shards SET status='pending', claimed_by=NULL, claimed_at=NULL, started_at=NULL WHERE status='running';")
        conn.commit()

    while True:
        row = claim_shard(conn, worker_id)
        if row is None:
            time.sleep(args.poll_interval)
            continue
        shard_id, on_rle, off_rle = row
        start = time.time()
        rc, solutions, log_tail = run_shard_stream(args.three, on_rle, off_rle, args.gpu)
        duration = time.time() - start
        finish_shard(conn, shard_id, rc, duration, solutions, log_tail)
        if rc != 0:
            print(f"[worker] shard {shard_id} failed (rc={rc}). Tail:\n{log_tail}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run multiple worker processes on one machine.")
    parser.add_argument("--three", default="./three", help="Path to solver executable")
    parser.add_argument("--queue-dir", default="queue", help="Queue directory root")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Worker poll interval seconds")
    parser.add_argument("--no-requeue-running", action="store_true",
                        help="Do not move running shards back to pending on startup")
    parser.add_argument("--stats-interval", type=float, default=60.0,
                        help="Seconds between queue status reports (0 to disable)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.gpu is None:
            print("[worker] missing --gpu", file=sys.stderr)
            return 1
        worker_loop(args)
        return 0

    gpus = detect_gpus()
    if not gpus:
        print("[error] no GPUs detected", file=sys.stderr)
        return 1

    queue_dir = Path(args.queue_dir)
    db_path = queue_dir / "queue.db"
    if not db_path.exists():
        print(f"[error] queue database not found: {db_path}", file=sys.stderr)
        return 1

    if not args.no_requeue_running:
        conn = sqlite3.connect(db_path, timeout=5, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("UPDATE shards SET status='pending', claimed_by=NULL, claimed_at=NULL, started_at=NULL WHERE status='running';")
        conn.commit()
        conn.close()

    combined_path = queue_dir / "solutions.txt"
    conn = sqlite3.connect(db_path, timeout=5, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000;")

    last_stats = 0.0
    empty_checks = 0

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

        counts = dict(conn.execute("SELECT status, COUNT(*) FROM shards GROUP BY status;").fetchall())
        pending = counts.get("pending", 0)
        running = counts.get("running", 0)
        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        total = pending + running + done + failed

        if args.stats_interval > 0:
            now = time.time()
            if now - last_stats >= args.stats_interval:
                eta = "unknown"
                avg_shard = "unknown"
                avg = conn.execute(
                    "SELECT AVG(duration) FROM shards WHERE status='done' AND duration IS NOT NULL;"
                ).fetchone()[0]
                solutions_total = conn.execute("SELECT COUNT(*) FROM solutions;").fetchone()[0]
                if avg and avg > 0:
                    remaining_shards = pending + running
                    eta_seconds = (remaining_shards * avg) / max(len(gpus), 1)
                    eta = format_eta(eta_seconds)
                    avg_shard = f"{avg:.1f}s"
                print(f"[queue] pending={pending} running={running} done={done} failed={failed} total={total} "
                      f"solutions={solutions_total} avg_shard={avg_shard} eta={eta}", flush=True)
                last_stats = now

        if pending + running == 0:
            empty_checks += 1
        else:
            empty_checks = 0
        if empty_checks >= 2:
            for _, proc in procs:
                proc.terminate()
            solutions_count = 0
            with combined_path.open("w", encoding="utf-8") as out_file:
                for (rle,) in conn.execute("SELECT rle FROM solutions ORDER BY id;"):
                    out_file.write(rle + "\n")
                    solutions_count += 1
            elapsed = conn.execute(
                "SELECT SUM(duration) FROM shards WHERE status='done' AND duration IS NOT NULL;"
            ).fetchone()[0] or 0.0
            done = counts.get("done", 0)
            print(f"[queue] done shards={done} solutions={solutions_count} elapsed={format_eta(elapsed)}", flush=True)
            return 0

        time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())
