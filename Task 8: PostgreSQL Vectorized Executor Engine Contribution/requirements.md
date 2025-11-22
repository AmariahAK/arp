# Requirements

**Task Created:** November 22, 2025  
**PostgreSQL Version:** 17.0 (as of creation date)  
**Repository:** https://github.com/postgres/postgres

> **Note:** This task is based on PostgreSQL 17.0 codebase as of November 2025. If the repository has been significantly updated since then, some implementation details may have changed. The core concepts and difficulty level remain valid.

## Prerequisites
- Deep understanding of PostgreSQL internals (executor, planner, storage)
- Proficiency in C (PostgreSQL codebase style)
- Understanding of columnar storage and vectorized execution
- Understanding of SIMD programming (SSE/AVX for x86-64)
- Knowledge of database query execution models
- Experience with large codebases (500k+ lines)
- Ability to write PostgreSQL regression tests
- Understanding of PostgreSQL's extension mechanism

## Initial Setup
The developer should provide:
1. PostgreSQL 17.0 source code cloned from official repository
2. PostgreSQL development environment set up (can compile from source)
3. Understanding of PostgreSQL build system (Autoconf/Meson)
4. Access to PostgreSQL regression test suite
5. Performance profiling tools (perf, pg_stat_statements)
6. Ability to run TPC-H or TPC-DS benchmarks

## Dependencies
- PostgreSQL 17.0 source tree
- GCC 11+ or Clang 14+ (for SIMD intrinsics)
- GNU Make or Ninja
- Flex, Bison (for parser)
- For testing: pgbench, pg_prove
- For benchmarking: TPC-H dataset and queries

## Testing Environment
- Minimum 8 CPU cores (for parallel query testing)
- At least 16GB RAM (32GB recommended for benchmarks)
- SSD storage (for I/O-bound query testing)
- Linux (PostgreSQL's primary platform)
- Ability to build PostgreSQL with various compiler flags

## Performance Requirements
- Vectorized executor must show >3x speedup on analytical queries
- Must pass all existing PostgreSQL regression tests (no breakage)
- Must handle all SQL data types correctly
- Memory overhead <10% vs row-based execution
- Must integrate cleanly with existing optimizer
- TPC-H queries 1, 6, 12 should show >5x speedup
- Scale to tables with billions of rows

## Code Quality Requirements
- Follow PostgreSQL coding standards exactly
- All code must pass pgindent, cpluspluscheck
- Commit messages follow PostgreSQL conventions
- Full documentation in SGML (PostgreSQL doc format)
- Comprehensive regression tests
- Should be acceptable for upstream merge
