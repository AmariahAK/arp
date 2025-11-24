# Requirements: Real-Time Neural Ray Tracing Denoiser in CUDA/C++

## Task Overview

Build a production-grade neural network-based denoiser for real-time path-traced rendering. The denoiser must run at 60+ FPS at 1080p resolution while preserving fine details and maintaining temporal stability across frames.

## Prerequisites

### Required Knowledge
- **Expert-level CUDA programming** (custom kernels, memory optimization, Tensor Cores)
- **Deep learning fundamentals** (CNNs, U-Nets, attention mechanisms, training loops)
- **Computer graphics** (path tracing, Monte Carlo integration, G-buffers, temporal reprojection)
- **Signal processing** (bilateral filtering, variance estimation, frequency analysis)
- **Linear algebra** (matrix operations, convolutions, transformations)
- **C++17+** (templates, RAII, move semantics, constexpr)

### Hardware Requirements
- NVIDIA GPU with Tensor Cores (RTX 3090 or better recommended)
- CUDA 12.0+
- 32GB+ RAM (for training dataset)
- Multi-core CPU for data preprocessing

### Software Dependencies
- **CUDA Toolkit 12.0+**
- **cuDNN 8.9+** (for training)
- **OptiX 7.7+** (for ray tracing integration)
- **PyTorch 2.0+** (for initial model training)
- **OpenEXR** (for HDR image I/O)
- **CMake 3.20+**
- **GCC 11+ or Clang 14+**
- **Python 3.10+** (for training scripts)

**Note:** No existing denoising frameworks (NVIDIA NRD, Intel OIDN) can be used directly. Must implement from scratch.

## Performance Requirements

### Inference Performance (Critical)
- **Latency:** <16ms per frame at 1920×1080 (60 FPS minimum)
- **Latency:** <8ms per frame at 1920×1080 (120 FPS target)
- **Throughput:** Process 4K frames in <33ms (30 FPS at 4K)
- **Memory:** <500MB GPU memory for model weights and activations
- **Batch processing:** Support batch size 1-4 for multi-viewport rendering

### Quality Requirements
- **PSNR:** >40 dB on test scenes (vs ground truth 10k SPP)
- **SSIM:** >0.95 on test scenes
- **Temporal stability:** <1% flickering artifacts (measured by temporal variance)
- **Detail preservation:** Maintain high-frequency details (textures, edges)
- **No ghosting:** Proper handling of disocclusions and moving objects

### Training Requirements
- **Dataset:** 50k+ path-traced images with 1-16 SPP (samples per pixel)
- **Training time:** <24 hours on 4x RTX 4090 for convergence
- **Validation:** Real-time validation during training
- **Generalization:** Work on unseen scenes without fine-tuning

## Technical Constraints

### Model Architecture
- **Input:** Low SPP image (1-16 SPP) + auxiliary buffers (albedo, normal, depth, motion vectors)
- **Output:** Denoised image matching 1000+ SPP quality
- **Network size:** <50M parameters (for real-time inference)
- **Architecture:** Custom U-Net variant with temporal components
- **Precision:** FP16 inference with Tensor Cores, FP32 training

### CUDA Optimization
- **Kernel fusion:** Fuse preprocessing, inference, and postprocessing
- **Memory bandwidth:** Minimize DRAM accesses, maximize L2 cache usage
- **Tensor Cores:** Utilize for all GEMM operations
- **Asynchronous execution:** Overlap compute and memory transfers
- **Multi-stream:** Support concurrent frame processing

### Temporal Coherence
- **Reprojection:** Implement robust temporal reprojection using motion vectors
- **History accumulation:** Blend current and previous frames intelligently
- **Disocclusion handling:** Detect and handle newly visible regions
- **Motion blur:** Preserve intentional motion blur from path tracer

## Code Quality Standards

### Production Requirements
- **Zero memory leaks:** Verified with CUDA-MEMCHECK and Valgrind
- **Thread safety:** Support concurrent denoising of multiple frames
- **Error handling:** Graceful degradation on CUDA errors
- **Determinism:** Reproducible results with fixed random seeds
- **Portability:** Support NVIDIA GPUs from Turing (SM 7.5) to Ada (SM 8.9)

### Testing
- **Unit tests:** >200 tests covering all components
- **Integration tests:** End-to-end denoising pipeline
- **Performance tests:** Regression detection for latency/quality
- **Visual tests:** Automated artifact detection
- **Coverage:** >90% code coverage

### Documentation
- **API documentation:** Doxygen for all public interfaces
- **Architecture guide:** Network design, training procedure
- **Integration guide:** How to integrate with existing renderers
- **Performance tuning:** Optimization tips and profiling results

## Deliverables

### Core Implementation
1. **Neural network architecture** (custom U-Net with temporal components)
2. **CUDA inference engine** (optimized kernels, Tensor Core utilization)
3. **Training pipeline** (PyTorch, data augmentation, loss functions)
4. **Temporal reprojection** (motion vector-based history accumulation)
5. **G-buffer preprocessing** (feature extraction from auxiliary buffers)
6. **Model export/import** (save trained weights, load in C++)

### Integration
1. **OptiX integration** (denoiser as OptiX denoiser callback)
2. **Standalone library** (C++ API for custom renderers)
3. **Python bindings** (for research and experimentation)

### Validation
1. **Benchmark suite** (standard test scenes, metrics)
2. **Comparison with baselines** (NVIDIA OptiX AI Denoiser, Intel OIDN)
3. **Ablation studies** (impact of each component)

## Success Criteria

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Inference latency (1080p) | <16ms | <8ms |
| PSNR (vs 10k SPP) | >40 dB | >42 dB |
| SSIM | >0.95 | >0.97 |
| Temporal stability | <1% flicker | <0.5% flicker |
| Model size | <50M params | <30M params |
| Training time (4x 4090) | <24h | <12h |
| Memory usage | <500MB | <300MB |
| Performance vs OptiX AI | Within 20% | Match or beat |

## Estimated Difficulty

**Time estimate:** 60-75 hours for expert CUDA/ML engineer

**Difficulty level:** EXTREME

**Why it's hard:**
- Requires expertise in 3 distinct domains (CUDA, deep learning, graphics)
- Real-time constraint is extremely challenging (16ms budget)
- Temporal stability is notoriously difficult to achieve
- Must balance quality, speed, and memory simultaneously
- Training dataset generation is non-trivial
- Debugging CUDA kernels and neural networks together is complex
