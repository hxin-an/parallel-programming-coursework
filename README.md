# Parallel Programming Coursework

Six implementations that apply parallelism at different levels: vector instructions, CPU threads, shared memory, distributed memory, CUDA, and OpenCL.

## What this shows

- Vectorized irregular loops with masks and handled tails where the input size is not a multiple of the vector width.
- Reduced CPU-side overhead with AVX2 PRNG, cache-line padding, task scheduling, and OpenMP frontier construction.
- Implemented distributed and accelerator workloads with MPI collectives / RMA, CUDA kernel optimization, and OpenCL host-device memory management.

## Work

| Assignment | Model | Implementation |
| --- | --- | --- |
| [`hw01-vectorization/`](hw01-vectorization/) | SIMD | Masked vector intrinsics, clamped exponent, array sum |
| [`hw02-pthreads/`](hw02-pthreads/) | Pthreads + AVX2 | Monte Carlo π and multi-threaded Mandelbrot |
| [`hw03-openmp/`](hw03-openmp/) | OpenMP | BFS, PageRank, conjugate gradient |
| [`hw04-mpi/`](hw04-mpi/) | MPI | Collective / one-sided π and distributed matrix multiplication |
| [`hw05-cuda/`](hw05-cuda/) | CUDA | Four Mandelbrot kernels with ILP, unrolling, early-exit checks, and launch tuning |
| [`hw06-opencl/`](hw06-opencl/) | OpenCL | Image convolution kernel and host-side buffer / launch pipeline |

## Final project

[Parallel Sudoku Solver](https://github.com/hxin-an/pp_final) compares serial, OpenMP, AVX2 SIMD, and hybrid OpenMP + SIMD solvers. The project uses constraint propagation, MRV ordering, and backtracking; the hybrid version reached about **7× speedup on 16×16 hard cases**, with larger search-order gains on expert cases.

## Notes

These files are the submitted implementation snapshots. The original course harness, shared headers, test data, and assignment statements are not redistributed, so some folders are intended for code review rather than standalone compilation.

