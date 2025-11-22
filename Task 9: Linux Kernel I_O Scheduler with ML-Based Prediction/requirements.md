# Requirements

**Task Created:** November 22, 2025  
**Linux Kernel Version:** 6.11+ (as of creation date)  
**Repository:** https://github.com/torvalds/linux

> **Note:** This task is based on Linux kernel 6.11+ codebase as of November 2025. If the kernel has been significantly updated since then, some APIs and subsystems may have changed. The core concepts and difficulty level remain valid.

## Prerequisites
- Deep understanding of Linux kernel internals (block layer, VFS, scheduler)
- Expert-level C programming (kernel coding standards)
- Understanding of I/O scheduling algorithms (CFQ, Deadline, BFQ, mq-deadline, Kyber)
- Knowledge of NVMe, SSD internals, and storage performance
- Understanding of machine learning basics (regression, neural networks)
- Experience with kernel module development
- Ability to debug kernel code (printk, ftrace, eBPF)
- Understanding of concurrency primitives (spinlocks, RCU, atomics)

## Initial Setup
The developer should provide:
1. Linux kernel 6.11+ source code
2. Kernel development environment (can compile custom kernels)
3. Test machine with NVMe SSD
4. Ability to boot custom kernels
5. Understanding of kernel build system (Kconfig, Makefiles)
6. fio for I/O benchmarking
7. blktrace/blkparse for I/O tracing

## Dependencies
- Linux kernel 6.11+ source tree
- GCC 11+ or Clang 14+
- For ML model: Lightweight inference library (or implement from scratch)
- For testing: fio, blktrace, iotop
- For profiling: perf, ftrace
- NVMe SSD for realistic testing

## Testing Environment
- Minimum 8 CPU cores
- NVMe SSD (for testing)
- At least 16GB RAM
- Ability to boot custom kernels (QEMU/KVM acceptable for initial dev)
- Root access for kernel module loading/unloading

## Performance Requirements
- I/O scheduler must match or exceed existing schedulers (BFQ, mq-deadline, Kyber)
- <5% CPU overhead for scheduling decisions
- <100 microseconds average scheduling latency
- Must scale to 100k+ IOPS
- ML predictions must complete in <10 microseconds
- Improve tail latency (p99) by >20% vs baseline
- Handle mixed workloads (sequential + random) effectively

## Code Quality Requirements
- Follow Linux kernel coding style exactly (checkpatch.pl clean)
- All code must be GPL-2.0
- Commit messages follow kernel conventions
- Full documentation in kernel-doc format
- Integration with existing block layer APIs
- Should be production-ready quality
- Proper handling of all error paths
- No memory leaks or race conditions
