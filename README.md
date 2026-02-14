# No-Three-In-Line

This project is a CUDA searcher that [enumerates maximal
solutions](https://oeis.org/A000769) to the no-three-in-line problem
on an $n \times n$ grid.

The [No-three-in-line problem] asks how many points can be placed on a
$n \times n$ grid so that no three points are on the same line, where
the lines considered are of any slope and not just orthogonal and
diagonal. Each row/column can contain at most 2 points, so clearly the
answer is at most $2n$. The real question is, can we actually achieve
$2n$ for every grid size? It's [conjectured] that the answer is "no"
for grids large enough, but we don't know where the crossover point is
and there's [no indication] that the number of $2n$-point solutions
is falling away from exponential growth, at least up to 19 × 19!

[No-three-in-line problem]: https://en.wikipedia.org/wiki/No-three-in-line_problem
[conjectured]: https://doi.org/10.4153%2FCMB-1968-062-3
[no indication]: http://web.archive.org/web/20131027174807/http://wso.williams.edu/~bchaffin/no_three_in_line/index.htm

For more details, see the following blog posts:
* [No-Three-In-Line](https://mvr.github.io/posts/no-three-in-line.html)
* [No-Three-In-Line, Quicker](https://mvr.github.io/posts/no-three-in-line-quicker.html)

## Results

Solutions are available under `results/` in
[RLE](https://conwaylife.com/wiki/Run_Length_Encoded) format, so they
can be pasted into [Golly](https://golly.sourceforge.io/), for
example. So far a few new values have been calculated over what was
previously known:

* Any symmetry:
  * 19 × 19: 32577
* 2-fold orthogonal symmetry:
  * 32 × 32: 0
  * 34 × 34: 0*
* 4-fold rotational symmetry:
  * 44 × 44: 1016
  * 46 × 46: 1366
  * 48 × 48: 2124*
  * 50 × 50: 3381*
  * 52 × 52: 5062*
* "Near" 4-fold rotational symmetry (except the main diagonals):
  * 43 × 43: 63
  * 45 × 45: 106
  * 47 × 47: 105

*Thanks to [Thomas
Prellberg](https://webspace.maths.qmul.ac.uk/t.prellberg/) and [Queen
Mary's Apocrita HPC facility](https://docs.hpc.qmul.ac.uk/)!

## Compiling

The desired grid size $n$ is set in `params.hpp`. Then:

```
cmake .
make
./three
```
