# Requirements

## Prerequisites
- C++20 or later compiler (GCC 11+, Clang 14+, or MSVC 2022+)
- Valgrind for memory debugging
- Google Benchmark for performance testing
- CMake 3.20+
- ASan/MSan/TSan support for sanitizers
- Linux kernel 5.0+ (for `mmap` and memory management APIs)

## Initial Setup
The developer should provide:
1. A C++20 development environment
2. Valgrind and sanitizer tools installed
3. Access to performance profiling tools (perf, vtune, or similar)
4. Ability to run multi-threaded benchmarks

## Dependencies
- No external memory allocation libraries allowed (must implement from scratch)
- Standard library usage restricted (cannot use `std::allocator` or `std::pmr`)
- Only system calls: `mmap`, `munmap`, `madvise`, `mremap`
- Testing: Google Test, Google Benchmark
- Profiling: Valgrind, perf

## Testing Environment
- Minimum 8 CPU cores for concurrency tests
- At least 16GB RAM available
- Support for huge pages (2MB/1GB)
- Ability to limit memory with cgroups or ulimit

## Performance Requirements
- All allocations must be O(1) amortized time
- Zero fragmentation after defragmentation pass
- Thread-safe operations with minimal lock contention
- Memory overhead <5% of allocated space
- Support sustained 10M+ allocations/sec across 8 threads
