# Requirements

## Prerequisites
- Rust 1.75+ (nightly for inline assembly)
- LLVM 17+ development libraries
- Capstone disassembler library
- Valgrind for JIT memory validation
- perf or Intel VTune for profiling
- Understanding of x86-64 assembly and calling conventions
- Understanding of compiler theory (SSA, register allocation, optimizations)

## Initial Setup
The developer should provide:
1. Rust nightly toolchain installed
2. LLVM development headers and libraries
3. Access to x86-64 CPU with AVX2+ support
4. Ability to mark memory as executable (mprotect)
5. perf or similar profiling tools

## Dependencies
- `cranelift-codegen` (study only - must implement JIT from scratch)
- `inkwell` (LLVM bindings - for comparison only)
- `dynasmrt` (study only - cannot use directly)
- `capstone` for disassembly verification
- `criterion` for benchmarking
- Standard Rust testing framework

## Testing Environment
- Minimum 4 CPU cores
- x86-64 CPU with SSE4.2, AVX2 support
- Linux or macOS (for mprotect/mmap)
- At least 8GB RAM
- Ability to disable ASLR for deterministic testing

## Performance Requirements
- JIT compilation: <1ms for 1000 bytecode instructions
- Generated code: Within 2x of handwritten assembly
- Optimization passes: Complete in <10ms for typical functions
- Memory overhead: <100 bytes per compiled function
- Support sustained 100k+ function invocations/second
