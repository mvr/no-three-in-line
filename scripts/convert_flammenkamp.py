#!/usr/bin/env python3
import sys


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def encode_idx(i: int) -> str:
    if i < 0 or i >= len(ALPHABET):
        raise ValueError(f"index {i} out of range for alphabet")
    return ALPHABET[i]


def parse_rle(line: str, n: int | None = None) -> set[tuple[int, int]]:
    x = 0
    y = 0
    cells: set[tuple[int, int]] = set()
    num = 0
    for ch in line.strip():
        if ch.isdigit():
            num = num * 10 + int(ch)
            continue
        if ch == "!":
            break
        count = num if num else 1
        num = 0
        if ch == "b":
            x += count
        elif ch == "o":
            for _ in range(count):
                if n is None or (0 <= x < n and 0 <= y < n):
                    cells.add((x, y))
                x += 1
        elif ch == "$":
            y += count
            x = 0
        else:
            # ignore unknown characters
            pass
    return cells


def infer_n_from_cells(cells: set[tuple[int, int]]) -> int | None:
    if not cells:
        return None
    max_x = max(x for x, _ in cells)
    max_y = max(y for _, y in cells)
    return max(max_x, max_y) + 1


def transform_point(n: int, x: int, y: int, t: int) -> tuple[int, int]:
    if t == 0:  # id
        return x, y
    if t == 1:  # rot90
        return n - 1 - y, x
    if t == 2:  # rot180
        return n - 1 - x, n - 1 - y
    if t == 3:  # rot270
        return y, n - 1 - x
    if t == 4:  # reflect vertical
        return n - 1 - x, y
    if t == 5:  # reflect horizontal
        return x, n - 1 - y
    if t == 6:  # reflect diag
        return y, x
    if t == 7:  # reflect anti-diag
        return n - 1 - y, n - 1 - x
    raise ValueError(t)


def is_invariant(n: int, cells: set[tuple[int, int]], t: int) -> bool:
    return all(transform_point(n, x, y, t) in cells for x, y in cells)


def near_rot4(n: int, cells: set[tuple[int, int]]) -> bool:
    filtered = {(x, y) for (x, y) in cells if x != y and x + y != n - 1}
    if not filtered:
        return True
    return all(transform_point(n, x, y, 1) in filtered for x, y in filtered)


def sym_class(n: int, cells: set[tuple[int, int]]) -> str:
    rot90 = is_invariant(n, cells, 1)
    rot180 = is_invariant(n, cells, 2)
    rot270 = is_invariant(n, cells, 3)
    refl_v = is_invariant(n, cells, 4)
    refl_h = is_invariant(n, cells, 5)
    refl_d = is_invariant(n, cells, 6)
    refl_a = is_invariant(n, cells, 7)

    full = rot90 and refl_v and refl_h and refl_d and refl_a
    if full:
        return "*"
    if rot90:
        return "o"
    if near_rot4(n, cells):
        return "c"
    if refl_d and refl_a:
        return "x"
    if refl_v and refl_h:
        return "+"
    if rot180:
        return ":"
    if refl_d or refl_a:
        return "/"
    if refl_v or refl_h:
        return "-"
    return "."


def encode_solution(n: int, cells: set[tuple[int, int]]) -> str:
    rows: list[list[int]] = [[] for _ in range(n)]
    for x, y in cells:
        rows[y].append(x)
    out_chars: list[str] = []
    for y in range(n):
        cols = sorted(rows[y])
        for x in cols:
            out_chars.append(encode_idx(x))
    return "".join(out_chars)


def main() -> int:
    lines: list[str] = []
    header_n: int | None = None
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("N="):
            try:
                header_n = int(line.split("=", 1)[1].strip())
            except ValueError:
                raise ValueError(f"invalid N header: {line!r}")
            continue
        if line.startswith("MIN_ON") or line.startswith("MAX_STEPS") or line.startswith("MAX_ON") or line.startswith("W="):
            continue
        lines.append(line)

    if not lines:
        print("[flammenkamp] no input lines", file=sys.stderr)
        return 1

    inferred = []
    for line in lines:
        n_line = infer_n_from_cells(parse_rle(line))
        if n_line is not None:
            inferred.append(n_line)

    if header_n is not None:
        if inferred and any(n != header_n for n in inferred):
            unique = ", ".join(str(v) for v in sorted(set(inferred)))
            raise ValueError(
                f"inferred grid sizes [{unique}] do not match header N={header_n}"
            )
        n = header_n
    else:
        if not inferred:
            raise ValueError("unable to infer grid size from input")
        unique = sorted(set(inferred))
        if len(unique) != 1:
            joined = ", ".join(str(v) for v in unique)
            raise ValueError(f"inconsistent inferred grid sizes: {joined}")
        n = unique[0]

    counts: dict[str, int] = {}
    total = 0
    for line in lines:
        cells = parse_rle(line, n)
        sym = sym_class(n, cells)
        code = encode_solution(n, cells)
        sys.stdout.write(sym + code + "\n")
        total += 1
        counts[sym] = counts.get(sym, 0) + 1

    print(f"[flammenkamp] N={n} wrote {total} solutions", file=sys.stderr)
    if counts:
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"[flammenkamp] symmetry counts: {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
