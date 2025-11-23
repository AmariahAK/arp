# Requirements

**Task Created:** November 23, 2025  
**Language:** C++ (17+)  
**Repository:** Custom implementation from scratch

> **Note:** This task requires implementing a production-grade neural network compression system from scratch. The implementation must achieve state-of-the-art compression ratios while maintaining accuracy within strict tolerances.

## Prerequisites
- Expert-level C++ (17 or higher)
- Deep understanding of neural network architectures (CNNs, RNNs, Transformers)
- Understanding of quantization, pruning, and knowledge distillation
- SIMD programming (AVX2/AVX-512 for x86-64, NEON for ARM)
- CUDA programming for GPU acceleration
- Numerical analysis and floating-point arithmetic
- Understanding of ONNX format and model interchange
- Experience with inference optimization

## Initial Setup
The developer should provide:
1. Modern C++ compiler (GCC 11+, Clang 14+, or MSVC 2022+)
2. CUDA Toolkit 12.0+ (for GPU acceleration)
3. cuDNN 8.9+ (for optimized kernels)
4. Eigen library for linear algebra
5. Google Protocol Buffers for model serialization
6. Ability to profile code (perf, nvprof, Nsight)
7. Reference models for testing (ResNet50, BERT-base)

## Dependencies
- Eigen 3.4+ (linear algebra)
- Protocol Buffers 3.20+ (model serialization)
- CUDA 12.0+ and cuDNN 8.9+ (GPU acceleration)
- GoogleTest for unit testing
- Google Benchmark for performance testing
- ONNX Runtime (for comparison/validation only)
- No ML frameworks allowed (PyTorch, TensorFlow, etc. - must implement from scratch)

## Testing Environment
- Minimum 16GB RAM (32GB recommended)
- NVIDIA GPU with Compute Capability 7.0+ (V100, RTX 3090, A100)
- x86-64 CPU with AVX2 support (AVX-512 preferred)
- SSD storage for model checkpoints
- Linux or Windows with CUDA support

## Performance Requirements
- **Quantization (INT8):** \u003c1% accuracy loss, 4x speedup vs FP32
- **Pruning (90% sparsity):** \u003c2% accuracy loss, 3x speedup with structured pruning
- **Knowledge Distillation:** Student model achieves \u003e95% of teacher accuracy
- **Compression Ratio:** 10-50x model size reduction
- **Inference Latency:** Match or beat ONNX Runtime on same hardware
- **Memory Footprint:** \u003c10MB overhead for compression engine
- **Numerical Stability:** \u003c1e-5 error in gradient computation

## Code Quality Requirements
- Follow C++ Core Guidelines
- Zero memory leaks (validate with Valgrind/AddressSanitizer)
- Thread-safe for concurrent model compression
- Exception-safe (RAII, no raw pointers)
- Comprehensive unit tests (\u003e90% coverage)
- Doxygen documentation for all public APIs
- CMake build system with proper dependency management
