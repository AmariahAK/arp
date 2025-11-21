# Requirements

## Prerequisites
- Zig 0.12+ (master branch for latest SIMD improvements)
- Intel CPU with AVX-512 or ARM CPU with NEON
- Understanding of SIMD programming (vectorization, data alignment)
- Understanding of computer vision algorithms
- Benchmark tools (hyperfine, perf)
- Image processing knowledge (convolutions, filters, transforms)

## Initial Setup
The developer should provide:
1. Zig master or 0.12+ installed
2. Test image datasets (various sizes: 640x480, 1920x1080, 4K, 8K)
3. Reference implementations for correctness validation (OpenCV)
4. CPU with AVX2 minimum, AVX-512 preferred
5. Ability to pin CPU frequency for consistent benchmarking

## Dependencies
- No external image processing libraries allowed (must implement from scratch)
- Allowed: `std.builtin` for CPU feature detection
- For testing only: Compare against OpenCV for correctness
- No BLAS, cuBLAS, or vendor-optimized libraries

## Testing Environment
- x86-64 CPU with AVX2 (AVX-512 for bonus optimizations)
- OR ARM64 CPU with NEON
- Minimum 8GB RAM (16GB for large image processing)
- SSD for fast image I/O
- Ability to disable CPU frequency scaling

## Performance Requirements
- All operations must use SIMD (no scalar fallbacks)
- Zero dynamic allocation in hot paths
- Process 1920x1080 image in <5ms for simple filters
- Process 4K image in <20ms for complex convolutions
- Memory bandwidth utilization >80% of theoretical peak
- Support real-time video processing (60 FPS minimum)
