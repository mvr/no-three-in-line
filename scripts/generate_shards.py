#!/usr/bin/env python3
import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def parse_shard_file(path: Path):
    shards = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if "|" in stripped:
                shards.append(stripped)
    return shards


def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shards (
          id INTEGER PRIMARY KEY,
          on_rle TEXT NOT NULL,
          off_rle TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending',
          claimed_by TEXT,
          claimed_at REAL,
          started_at REAL,
          finished_at REAL,
          rc INTEGER,
          duration REAL,
          log_tail TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shards_status ON shards(status);")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS solutions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          shard_id INTEGER NOT NULL,
          rle TEXT NOT NULL
        );
        """
    )
    return conn


def run_frontier(frontier_bin: str, out_path: Path, min_on, steps, grid_n):
    cmd = [frontier_bin, "--steps", str(steps)]
    if min_on is not None:
        cmd += ["--min-on", str(min_on)]
    print(f"[frontier] running: {' '.join(cmd)} > {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        if grid_n is not None:
            f.write(f"N={grid_n}\n")
            width = 64 if grid_n > 32 else 32
            f.write(f"W={width}\n")
        if min_on is not None:
            f.write(f"MIN_ON={min_on}\n")
        if steps is not None:
            f.write(f"MAX_STEPS={steps}\n")
        res = subprocess.run(cmd, stdout=f)
    if res.returncode != 0:
        raise RuntimeError(f"frontier failed (exit {res.returncode})")

def detect_grid_size(explicit: int | None, params_path: Path) -> int | None:
    if explicit is not None:
        return explicit
    env_n = _read_env_int("GRID_N")
    if env_n is not None:
        return env_n
    try:
        text = params_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("const unsigned N"):
            parts = line.split("=")
            if len(parts) >= 2:
                value = parts[1].split(";")[0].strip()
                try:
                    return int(value)
                except ValueError:
                    return None
    return None


def _read_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def tune_frontier(frontier_bin: str, out_path: Path, min_on, steps, target, tolerance, max_iters, grid_n):
    steps = max(1, steps)
    for _ in range(max_iters):
        run_frontier(frontier_bin, out_path, min_on, steps, grid_n)
        shards = parse_shard_file(out_path)
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
    parser.add_argument("--out", default="shards.txt",
                        help="Output shard list path (captured from frontier stdout)")
    parser.add_argument("--min-on", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--target-shards", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--grid-n", type=int, default=None, help="Override grid size for header")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing shard DB before writing")
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    db_path = queue_dir / "queue.db"

    if args.shards_file:
        shard_path = Path(args.shards_file)
        if not shard_path.exists():
            print(f"[error] shard file not found: {shard_path}", file=sys.stderr)
            return 1
    else:
        shard_path = Path(args.out)
        grid_n = detect_grid_size(args.grid_n, Path("params.hpp"))
        steps, count = tune_frontier(
            args.frontier_bin,
            shard_path,
            args.min_on,
            args.steps,
            args.target_shards,
            args.tolerance,
            args.max_iters,
            grid_n,
        )
        print(f"[frontier] final steps={steps} shards={count}")

    shards = parse_shard_file(shard_path)
    if not shards:
        print("[error] no shards found to enqueue", file=sys.stderr)
        return 1

    conn = init_db(db_path)
    if args.overwrite:
        conn.execute("DELETE FROM solutions;")
        conn.execute("DELETE FROM shards;")
    else:
        existing = conn.execute("SELECT COUNT(*) FROM shards;").fetchone()[0]
        if existing > 0:
            print(f"[error] queue already has {existing} shards (use --overwrite to replace)", file=sys.stderr)
            conn.close()
            return 1
    rows = []
    for line in shards:
        on_rle, off_rle = line.split("|", 1)
        rows.append((on_rle, off_rle))
    conn.executemany("INSERT INTO shards (on_rle, off_rle) VALUES (?, ?);", rows)
    conn.commit()
    conn.close()
    print(f"[queue] wrote {len(shards)} shards into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
