# HW 1 — SIMD Vectorization

Vectorized implementations of clamped exponentiation and array reduction using the course's SIMD intrinsic interface.

## Highlights

- Per-lane masks for irregular exponent counts
- Correct tail handling when input length is not a multiple of vector width
- Horizontal reduction for the final array sum

The implementation is in `src/vectorOP.cpp`. It depends on the course-provided intrinsic headers and test harness, which are not redistributed.
