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
  * 49 × 49: 196*
  * 51 × 51: 264*
  * 53 × 53: 377*
* 4-fold diagonal symmetry:
  * 46 × 46: 0
  * 47 × 47: 1
  * 48 × 48: 0

As well as some solutions with record-breaking sizes*:

* 61 × 61:
  ```
  27bo16bo$28bo8bo$25bo10bo$4bo12bo$43bo13bo$20bo4bo$14bobo$42bo5bo$11bo
  35bo$30bo8bo$33bo4bo$15bo36bo$7bo12bo$8bo13bo$29bo24bo$34bo14bo$o53bo$
  4bo52bo$7bo34bo$21bobo$48bo6bo$9bo31bo$10bo36bo$bo39bo$2bo31bo$55bo2bo
  $15bo8bo$10bo49bo$29bo29bo$32bo13bo$9bo41bo$14bo13bo$bo29bo$o49bo$36bo
  8bo$2bo2bo$26bo31bo$19bo39bo$13bo36bo$19bo31bo$5bo6bo$37bobo$18bo34bo$
  3bo52bo$6bo53bo$11bo14bo$6bo24bo$38bo13bo$40bo12bo$8bo36bo$22bo4bo$21b
  o8bo$13bo35bo$12bo5bo$44bobo$35bo4bo$3bo13bo$43bo12bo$24bo10bo$23bo8bo
  $16bo16bo!
  ```
* 63 × 63:
  ```
  30bo7bo$14bo11bo$16bo12bo$25bo11bo$41bo5bo$14bo19bo$50bo4bo$6bo47bo$7b
  o19bo$40bo3bo$15b2o$27bo8bo$6bo13bo$21bo9bo$57bo3bo$4bo47bo$52bo7bo$
  33bo11bo$9bo13bo$22b2o$32bo17bo$4bo44bo$9bo33bo$43b2o$o27bo$3bo55bo$
  11bo49bo$51bo2bo$5bo32bo$17bo42bo$20bo41bo$13bo35bo$o41bo$2bo42bo$24bo
  32bo$8bo2bo$bo49bo$3bo55bo$34bo27bo$18b2o$19bo33bo$13bo44bo$12bo17bo$
  39b2o$39bo13bo$17bo11bo$2bo7bo$10bo47bo$bo3bo$31bo9bo$42bo13bo$26bo8bo
  $46b2o$18bo3bo$35bo19bo$8bo47bo$7bo4bo$28bo19bo$15bo5bo$25bo11bo$33bo
  12bo$36bo11bo$24bo7bo!
  ```
* 64 × 64:
  ```
  32bo16bo$34bo6bo$24bobo$16bo17bo$38bo13bo$7bo33bo$12bo22bo$31bo26bo$
  35bo16bo$20bo26bo$21bobo$4bo3bo$49bo7bo$42bo3bo$o11bo$33bo9bo$9bo50bo$
  13bo9bo$27bo5bo$27bo11bo$15bo38bo$13bo39bo$bo3bo$46bo6bo$19bo41bo$4bo
  32bo$25bo35bo$44b2o$6bobo$bobo$15bo2bo$o55bo$7bo55bo$45bo2bo$60bobo$
  55bobo$18b2o$2bo35bo$26bo32bo$2bo41bo$10bo6bo$58bo3bo$10bo39bo$9bo38bo
  $24bo11bo$30bo5bo$40bo9bo$3bo50bo$20bo9bo$51bo11bo$17bo3bo$6bo7bo$55bo
  3bo$40bobo$16bo26bo$11bo16bo$5bo26bo$28bo22bo$22bo33bo$11bo13bo$29bo
  17bo$37bobo$22bo6bo$14bo16bo!
  ```
* 66 x 66:
  ```
  31bo4bo$36bo14bo$2bo60bo$20bo2bo$15bo24bo$15bo19bo$13bo25bo$8bo28bo$
  45bo12bo$24bo27bo$21bo21bo$12bo25bo$21bo32bo$9bo49bo$bo47bo$60b2o$14bo
  17bo$25bo17bo$24bo17bo$32bo5bo$8bo53bo$53bobo$10bo6bo$18bo43bo$47bo8bo
  $4bo43bo$6bo23bo$11bo7bo$7bo23bo$2o$5bo33bo$37bo27bo$46bo2bo$16bo2bo$o
  27bo$26bo33bo$64b2o$34bo23bo$46bo7bo$35bo23bo$17bo43bo$9bo8bo$3bo43bo$
  48bo6bo$10bobo$3bo53bo$27bo5bo$23bo17bo$22bo17bo$33bo17bo$4b2o$16bo47b
  o$6bo49bo$11bo32bo$27bo25bo$22bo21bo$13bo27bo$7bo12bo$28bo28bo$26bo25b
  o$30bo19bo$25bo24bo$42bo2bo$2bo60bo$14bo14bo$29bo4bo!
  ```
* 68 x 68:
  ```
  12bobo$26bo11bo$35bo10bo$36bobo$25bo7bo$18bo29bo$40bo3bo$44bo9bo$14bo
  17bo$11bo7bo$25bo13bo$45bo12bo$52bo14bo$7bo44bo$59bo7bo$12b2o$24bo6bo$
  28bo4bo$40bo21bo$5bo52bo$43bo2bo$2bo17bo$11bo14bo$6b2o$20bo30bo$57bo5b
  o$45bo20bo$6bo11bo$10bo39bo$bobo$30bo6bo$3bo47bo$2bo56bo$50bo12bo$4bo
  12bo$8bo56bo$16bo47bo$30bo6bo$64bobo$17bo39bo$49bo11bo$bo20bo$4bo5bo$
  16bo30bo$60b2o$41bo14bo$47bo17bo$21bo2bo$9bo52bo$5bo21bo$34bo4bo$36bo
  6bo$54b2o$o7bo$15bo44bo$o14bo$9bo12bo$28bo13bo$48bo7bo$35bo17bo$13bo9b
  o$23bo3bo$19bo29bo$34bo7bo$29bobo$21bo10bo$29bo11bo$53bobo!
  ```

*Thanks to [Thomas Prellberg] and Queen Mary's [Apocrita] HPC
facility!

[Thomas Prellberg]: https://webspace.maths.qmul.ac.uk/t.prellberg/
[Apocrita]: https://docs.hpc.qmul.ac.uk/

## Compiling

The desired grid size $n$ is set in `params.hpp`. Then:

```
cmake .
make
./three
```
