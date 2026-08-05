# HW 2 — Pthreads and AVX2

Two CPU-parallel workloads: Monte Carlo estimation of π and multi-threaded Mandelbrot rendering.

## Highlights

- Thread-local pseudo-random number generation with AVX2
- Cache-line-aware per-thread state to reduce false sharing
- Dynamic row scheduling for uneven Mandelbrot workloads

`src/part1/` contains the π estimator and Makefile; `src/part2/` contains the Mandelbrot implementation. Course headers and image utilities are not redistributed.
