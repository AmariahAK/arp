# Expected Results: Neural Network Compression Engine in C++

## Final Deliverables

### 1. Core Implementation Files
```
compress_engine/
├── include/
│   ├── tensor.h                 # Core tensor class with reference counting
│   ├── ops.h                    # Tensor operations (SIMD optimized)
│   ├── quantization.h           # INT8/INT4 quantization algorithms
│   ├── pruning.h                # Structured/unstructured pruning
│   ├── distillation.h           # Knowledge distillation framework
│   ├── autograd.h               # Automatic differentiation
│   ├── onnx_exporter.h          # ONNX model export
│   ├── tensorrt_deployer.h      # TensorRT integration
│   └── cuda/
│       ├── kernels.cuh          # Custom CUDA kernels
│       ├── tensor_ops.cuh       # GPU tensor operations
│       └── tensorcore_gemm.cuh  # Tensor Core INT8 GEMM
├── src/
│   ├── tensor.cpp               # Tensor implementation
│   ├── ops_cpu.cpp              # CPU ops (AVX2/AVX-512 SIMD)
│   ├── ops_cuda.cu              # CUDA ops with Tensor Cores
│   ├── quantization.cpp         # PTQ and QAT
│   ├── quantization_int4.cpp    # INT4 k-means clustering
│   ├── quantization_dynamic.cpp # Dynamic quantization for RNNs
│   ├── pruning.cpp              # Pruning implementations
│   ├── distillation.cpp         # Distillation loss
│   ├── autograd.cpp             # Reverse-mode AD
│   ├── onnx_exporter.cpp        # ONNX export
│   └── tensorrt_deployer.cpp    # TensorRT deployment
├── tests/
│   ├── test_tensor.cpp          # Tensor tests (52 tests)
│   ├── test_quantization.cpp    # Quantization tests (68 tests)
│   ├── test_pruning.cpp         # Pruning tests (41 tests)
│   ├── test_distillation.cpp    # Distillation tests (28 tests)
│   ├── test_autograd.cpp        # Autograd tests (45 tests)
│   ├── test_cuda.cpp            # CUDA tests (39 tests)
│   └── benchmarks/
│       ├── bench_gemm.cpp
│       ├── bench_quantization.cpp
│       └── bench_end_to_end.cpp
├── examples/
│   ├── quantize_resnet50.cpp
│   ├── prune_bert.cpp
│   ├── distill_mobilenet.cpp
│   └── deploy_tensorrt.cpp
└── CMakeLists.txt
```

### 2. Performance Benchmarks

**Hardware: NVIDIA A100 80GB GPU, Intel Xeon Platinum 8380 (AVX-512)**

#### Turn 1: Tensor Library with SIMD GEMM
```
Benchmark: Matrix Multiplication 1024×1024 (FP32)

Naive (no SIMD):          182ms     11.5 GFLOPS
Our AVX2 implementation:  17.8ms    118 GFLOPS
Our AVX-512 + blocking:   14.2ms    148 GFLOPS
Intel MKL (baseline):     12.1ms    173 GFLOPS

Ratio: 148/173 = 85.5% of MKL ✅ (target: >80%)

Memory:
- Zero leaks (Valgrind verified) ✅
- Proper alignment for SIMD (32-byte) ✅
- Cache-blocking: 2.3x speedup vs naive SIMD
```

#### Turn 2: INT8 Post-Training Quantization
```
Model: ResNet50, ImageNet

Accuracy:
- FP32 baseline: 76.13%
- INT8 (PTQ, per-tensor): 74.28% (Δ=1.85%, fails target)
- INT8 (PTQ, per-channel): 75.87% (Δ=0.26%) ✅ (target: <1%)

Inference Latency (batch=1, 224×224, CPU AVX-512):
- FP32: 4.20ms
- INT8 with VNNI: 1.10ms (3.82x speedup) ✅ (target: 3-4x)

Model Size:
- FP32: 98 MB
- INT8: 25 MB (3.92x compression) ✅ (target: 4x)

INT8 GEMM Performance:
- AVX512_VNNI: 387 GOPS (INT8)
- Equivalent FP32: ~97 GFLOPS
- 3.8x faster than our FP32 GEMM ✅
```

#### Turn 3: Force Failure → Quantization Error Accumulation
```
ResNet50 (50 layers), INT8 PTQ without QAT:
- Accuracy: 69.2% ❌ (7.3% drop, exceeds 1% tolerance)
- Failure cause: Error compounds through 50 layers

Math: Each layer ε ≈ scale/2, total ε ≈ √50 * ε_layer
After 50 layers: cumulative error >> single layer

Fix: Quantization-Aware Training (QAT)
- Fake quantization during forward
- Straight-through estimator for backward
- Result: 75.91% accuracy ✅ (0.22% drop)
```

#### Turn 4: Structured Pruning with Channel-wise Sparsity
```
ResNet50, 50% channel pruning

Strategy Comparison:
- L1 Norm:  74.92% accuracy, 6.4M params
- L2 Norm:  74.76% accuracy, 6.4M params
- Taylor:   75.18% accuracy, 6.5M params ✅ Best
- FPGM:     74.89% accuracy, 6.3M params

Selected: Taylor pruning
- Accuracy: 75.18% (Δ=0.95% from FP32) ✅ (target: <2%)
- Inference: 2.78ms (1.51x speedup) ✅ (target: 1.5-2x)
- Parameters: 25.5M → 6.5M (74.5% reduction)
- Speedup: Real measured speedup, not theoretical
```

#### Turn 5: Knowledge Distillation
```
Teacher: ResNet50 (76.13%, 25M params)
Student: ResNet18 (11M params)

Training Configurations:
- Baseline (no distillation): 69.76%
- Distillation (T=3, α=0.7): 73.12% ✅
- Student/Teacher ratio: 96.0% ✅ (target: >95%)

Model Compression:
- Size: 98MB → 44MB → 11MB (8.9x total compression)
- Inference: 4.2ms → 1.8ms (2.33x speedup)

Temperature Sensitivity:
- T=1 (hard labels): 70.2%
- T=3: 73.12% ✅ Best
- T=5: 72.8%
- T=10: 71.9%
```

#### Turn 6: Automatic Differentiation Engine
```
Autograd Validation (gradient correctness):
- Test: MLP (3 layers, 256 hidden)
- JAX autograd vs numerical: max error < 1e-5 ✅
- Backward pass memory: O(depth), not O(depth²)

Performance:
- 50-layer network gradient computation: 285ms
- Memory efficient (activation checkpointing integrated)
- Supports all common ops: MatMul, Conv2D, ReLU, BatchNorm
```

#### Turn 7: CUDA GPU Acceleration
```
INT8 GEMM with Tensor Cores (NVIDIA A100)

Benchmark: 4096×4096×4096 INT8 GEMM
- Our Tensor Core kernel: 0.85ms (25.6 TOPS)
- cuBLAS INT8: 0.78ms (27.8 TOPS)
- Ratio: 92.1% of cuBLAS ✅ (target: >80%)

Kernel Fusion (ReLU + Quantize):
- Separate kernels: 0.42ms
- Fused kernel: 0.18ms (2.33x speedup)
- Memory bandwidth: 1.2 TB/s (near peak)
```

#### Turn 8: Force Failure → GPU Memory Overflow
```
BERT-Large (340M params), batch=32, seq=512, training:

Without gradient checkpointing:
- Forward pass memory: 18.3 GB
- Activations stored: 24 layers × 760MB = 18.2 GB
- Total: OOM on 80GB GPU ❌

With gradient checkpointing (every 4 layers):
- Forward pass memory: 7.9 GB ✅
- Reduction: 2.32x ✅ (target: 2x)
- Slowdown: 28% (due to recomputation) ✅ (<30%)
- Can train batch=96 with same memory
```

#### Turn 9: INT4 Quantization with Weight Clustering
```
Model: ResNet50

K-means clustering (16 clusters per channel):
- Cluster convergence: <100 iterations
- Compression: 8x vs FP32

Accuracy:
- FP32: 76.13%
- INT8: 75.87% (Δ=0.26%)
- INT4 (per-tensor): 71.42% (Δ=4.71%, too high ❌)
- INT4 (per-channel k-means): 74.15% (Δ=1.98%) ✅ (target: <3%)

Model Size:
- FP32: 98 MB
- INT4: 12.3 MB (7.97x compression) ✅

Inference (with LUT decompression):
- Decompression overhead: 15%
- Still faster than INT8 in memory-bound scenarios
```

#### Turn 10: Dynamic Quantization for RNNs/LSTMs
```
Model: 2-layer LSTM (256 hidden), PTB dataset

Static INT8 quantization (fails):
- Activations vary wildly across timesteps
- Accuracy: 65.2% ❌ (vs 78.4% FP32)

Dynamic INT8 quantization:
- Compute scale/zero-point per timestep
- Accuracy: 77.8% ✅ (0.6% drop, target: <1%)
- Inference: 3.1ms vs 9.7ms FP32 (3.13x speedup) ✅

Memory:
- Weights: INT8 (static)
- Activations: INT8 (dynamic scale)
- Hidden states: FP32 (for stability)
```

#### Turn 11: Quantization Error Analysis Tools
```
Layer-wise Error Analysis (ResNet50 INT8):

Worst 5 layers by activation error:
1. conv1: MSE=0.0142 (high input variance)
2. layer4.2.conv3: MSE=0.0098
3. fc (final): MSE=0.0087
4. layer3.5.conv2: MSE=0.0063
5. layer2.3.conv1: MSE=0.0051

Outlier Detection:
- Layer conv1: 234 outliers (>3σ)
- Recommendation: Use per-channel quantization ✅

Calibration Data Coverage:
- 100 batches covers 95.2% of weight range
- 1000 batches covers 99.1% ✅
- Recommendation: Use 100-500 batches
```

#### Turn 12: Mixed-Precision Inference (INT8 + FP16)
```
ResNet50 with automatic precision selection:

All INT8 (aggressive):
- Accuracy: 74.82% (Δ=1.31%)
- Latency: 1.08ms

Sensitive layers in FP16 (auto-selected):
- conv1, layer1.0: FP16
- Rest: INT8
- Accuracy: 75.96% (Δ=0.17%) ✅ (target: <0.5%)
- Latency: 1.34ms
- Best accuracy/performance trade-off

Manual tuning can achieve 76.02% at 1.28ms
```

#### Turn 13: ONNX Model Export
```
Export Capabilities:
- INT8 quantized models with QDQ nodes ✅
- Preserves per-layer quantization params ✅
- Supports DynamicQuantizeLinear for RNNs ✅

Validation:
- Export ResNet50 INT8 to ONNX
- Import in ONNX Runtime
- Accuracy: exact match (75.87%) ✅
- Latency: within 3% of our engine ✅

Compatibility:
- ONNX opset 13+
- ONNX Runtime 1.12+
- Works with Netron for visualization
```

#### Turn 14: TensorRT Production Deployment
```
ResNet50 INT8, NVIDIA A100

Build TensorRT Engine:
- Parse ONNX: 2.3s
- Calibration (INT8): 12.7s
- Engine build: 8.4s
- Total: 23.4s

Inference Performance:
- Our engine (CUDA): 1.10ms
- TensorRT FP32: 1.82ms
- TensorRT INT8: 0.87ms

Comparison: Our engine 79% speed of TensorRT INT8
- TensorRT applies many optimizations (fusion, etc.)
- Our engine competitive for custom workflows ✅
```

#### Turn 15: End-to-End Benchmark (ResNet50 + BERT)
```
=================================
End-to-End Benchmark Results
=================================

Model                Precision            Accuracy  Latency(ms)  Throughput  Size(MB)  Mem(MB)
--------------------------------------------------------------------------------------------------
ResNet50             FP32                    76.13%        4.20        238.1        98       412
ResNet50             INT8 (PTQ)              75.87%        1.10        909.1        25       184
ResNet50             INT8 + 50% Pruned       74.92%        0.85       1176.5        13       142
ResNet50             INT4                    74.15%        0.78       1282.1         7        98
ResNet50             Mixed (INT8+FP16)       75.98%        1.25        800.0        31       198
BERT-base            FP32                    84.50%       12.30         81.3       438      3200
BERT-base            Dynamic INT8            84.21%        4.10        243.9       110      1100
BERT-base            INT8 Distilled          83.85%        2.95        339.0        85       890

Speedup Analysis:
INT8 (PTQ): 3.82x speedup, 3.92x compression
INT8 + 50% Pruned: 4.94x speedup, 7.54x compression ✅
INT4: 5.38x speedup, 14.00x compression ✅
Mixed (INT8+FP16): 3.36x speedup, 3.16x compression

Overall: 10-50x compression achieved ✅
```

### 3. Correctness Validation

#### Memory Safety (AddressSanitizer + Valgrind)
```bash
# Test with AddressSanitizer
cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZE=address ..
make -j$(nproc)
ctest --output-on-failure

# Expected: 0 memory leaks ✅
# Expected: 0 buffer overflows ✅
# Expected: 0 use-after-free ✅

# Valgrind full check
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./bin/test_all

# Result: 0 bytes leaked ✅
```

#### Thread Safety
```cpp
TEST(ThreadSafetyTest, ConcurrentQuantization) {
    // Quantize 20 models concurrently
    std::vector<std::thread> threads;
    std::atomic<int> successes{0};
    
    for (int i = 0; i < 20; ++i) {
        threads.emplace_back([&]() {
            auto model = load_resnet50();
            ModelQuantizer quantizer;
            quantizer.calibrate(model, calib_data, 50);
            auto model_q = quantizer.quantize(model);
            
            float acc = evaluate(model_q, test_data);
            if (acc > 75.5) successes++;
        });
    }
    
    for (auto& t : threads) t.join();
    
    EXPECT_EQ(successes, 20);  // ✅ All succeed, no races
}
```

#### Numerical Correctness
```cpp
TEST(NumericalTest, AutogradVsFiniteDifference) {
    auto model = create_mlp({784, 256, 128, 10});
    auto input = Tensor::randn({32, 784});
    auto labels = Tensor::randint({32}, 0, 10);
    
    // Compute gradients with autograd
    auto loss = cross_entropy(model.forward(input), labels);
    loss.backward();
    auto grad_auto = model.param("fc1.weight").grad();
    
    // Compute numerical gradients
    auto grad_numerical = compute_finite_difference(
        [&](Tensor w) {
            model.set_param("fc1.weight", w);
            return cross_entropy(model.forward(input), labels);
        },
        model.param("fc1.weight"),
        1e-4  // eps
    );
    
    float max_rel_error = compute_max_relative_error(grad_auto, grad_numerical);
    EXPECT_LT(max_rel_error, 1e-4);  // ✅
}
```

### 4. Comparison with Baselines

#### Quantization: Ours vs ONNX Runtime vs TensorRT
```
Model: ResNet50, ImageNet validation (50k images)
Hardware: NVIDIA A100 GPU

Framework              | Accuracy | Latency (ms) | Throughput (img/s)
-----------------------|----------|--------------|-------------------
PyTorch FP32 (eager)   | 76.13%   | 4.82         | 207
PyTorch FP32 (compile) | 76.13%   | 4.20         | 238
Ours FP32 (CPU)        | 76.13%   | 4.20         | 238
Ours INT8 (CPU)        | 75.87%   | 1.10         | 909   ✅
ONNX Runtime INT8      | 75.91%   | 1.02         | 980
TensorRT FP32 (GPU)    | 76.13%   | 1.82         | 549
TensorRT INT8 (GPU)    | 75.94%   | 0.87         | 1149

→ Our INT8 CPU: 93% of ONNX Runtime speed ✅
→ Accuracy within 0.04% of baselines ✅
→ Goal: Match or beat ONNX Runtime ✅ Nearly achieved
```

#### CUDA Kernels vs cuBLAS/cuDNN
```
Operation             | Ours (ms) | cuBLAS/cuDNN (ms) | Ratio
----------------------|-----------|-------------------|-------
INT8 GEMM 4096³       | 0.85      | 0.78              | 92%  ✅
FP32 Conv2D 256ch     | 2.34      | 2.18              | 93%  ✅
Fused ReLU+Quant      | 0.18      | 0.21 (unfused)    | 117% ✅

→ All kernels >80% of vendor libraries ✅
→ Fusion provides additional wins
```

#### Pruning Methods Comparison
```
ResNet50, 50% channel sparsity, ImageNet

Method               | Accuracy | Inference | FLOPs Reduction
---------------------|----------|-----------|----------------
L1 Norm (baseline)   | 74.92%   | 2.78ms    | 49.2%
L2 Norm              | 74.76%   | 2.78ms    | 49.2%
Taylor Expansion     | 75.18%   | 2.88ms    | 48.7%  ✅ Best
FPGM (geometric)     | 74.89%   | 2.71ms    | 50.1%

Recommendation: Taylor for accuracy, FPGM for speed
```

### 5. Edge Cases Handled

- [x] **Non-contiguous tensors** → Automatic copy to contiguous
- [x] **Misaligned memory** → Aligned allocator (32-byte for AVX)
- [x] **Quantization overflow/underflow** → Clamping to INT8 range
- [x] **Deep models (100+ layers)** → Gradient checkpointing
- [x] **Mixed CPU/GPU tensors** → Automatic device sync
- [x] **Thread-local state corruption** → Proper isolation
- [x] **FP16 underflow in distillation** → Loss scaling
- [x] **Pruning to 0 channels** → Validation prevents empty layers
- [x] **INT4 numerical instability** → Per-channel scales
- [x] **cuDNN version mismatch** → Runtime check with fallback
- [x] **ONNX export edge cases** → Dynamic axes, custom ops
- [x] **TensorRT builder failures** → Graceful degradation
- [x] **RNN variable sequence lengths** → Proper masking
- [x] **Quantization calibration OOM** → Batch-wise calibration

### 6. Known Limitations

1. **INT4 Quantization Accuracy**
   - Typical accuracy loss: 2-3% (vs <1% for INT8)
   - Requires extensive per-channel tuning
   - Best for: edge deployment, models with redundancy
   - Not recommended for: critical applications

2. **Unstructured Pruning Speedup**
   - 90% unstructured sparsity → only 1.2x speedup
   - Sparse matrix operations not well-optimized on CPUs
   - Recommendation: Use structured pruning for inference

3. **CUDA Kernel Coverage**
   - Custom kernels: GEMM, Conv2D, ReLU, Quantize
   - Other ops: cuDNN/cuBLAS fallback
   - Future work: Custom LayerNorm, Softmax kernels

4. **Platform Support**
   - **Full support:** Linux x86-64 with AVX2+
   - **Partial support:** ARM NEON (no INT8 acceleration)
   - **CUDA:** Requires SM 7.0+ for Tensor Cores
   - **No support:** Windows (planned for v2.0)

5. **Dynamic Batching**
   - Fixed batch size at compile time
   - Dynamic batching requires rebuilding
   - TensorRT has better dynamic shape support

### 7. Test Coverage

```
Total test files: 10
Total test count: 312
Code coverage: 94.7% (gcov)

Unit tests by module:
- tensor.cpp: 95% coverage (52 tests)
- quantization.cpp: 94% coverage (68 tests)
- pruning.cpp: 93% coverage (41 tests)
- distillation.cpp: 96% coverage (28 tests)
- autograd.cpp: 95% coverage (45 tests)
- ops_cuda.cu: 91% coverage (39 tests)
- onnx_exporter.cpp: 92% coverage (18 tests)
- tensorrt_deployer.cpp: 89% coverage (21 tests)

Integration tests: 24
End-to-end benchmarks: 12

Performance regression suite:
- Tracked metrics: latency, throughput, memory
- Alert threshold: >5% regression
- Runs nightly on dedicated hardware

All 312 tests pass ✅
Zero flaky tests ✅
```

### 8. Documentation Completeness

#### API Documentation (Doxygen)
- ✅ 100% of public APIs documented
- ✅ Code examples for all major features
- ✅ Architecture diagrams (quantization flow, etc.)
- ✅ Generated docs: `build/docs/html/index.html`

#### User Guides
1. **Quickstart.md** - Quantize your first model in 10 minutes
2. **Quantization_Guide.md** - PTQ, QAT, INT4, dynamic quantization
3. **Pruning_Guide.md** - Structured, unstructured, iterative pruning
4. **Distillation_Guide.md** - Teacher-student setup, hyperparams
5. **CUDA_Optimization.md** - Kernel tuning, profiling
6. **ONNX_Export_Guide.md** - Export workflow, validation
7. **TensorRT_Deployment.md** - Production deployment checklist
8. **Error_Analysis.md** - Debugging quantization failures

### 9. Final Validation Checklist

#### Turns 1-8: Foundation
- [x] Zero memory leaks (Valgrind clean)
- [x] SIMD GEMM within 20% of MKL
- [x] INT8 accuracy <1% loss on ResNet50, MobileNet, BERT
- [x] INT8 3-4x speedup achieved
- [x] Quantization error accumulation identified and fixed
- [x] Structured pruning <2% accuracy loss
- [x] Knowledge distillation >95% student/teacher ratio
- [x] Autograd matches numerical gradients
- [x] CUDA kernels >80% of cuBLAS performance
- [x] GPU OOM fixed with gradient checkpointing

#### Turns 9-15: Advanced Features
- [x] INT4 quantization <3% accuracy loss
- [x] Dynamic quantization for RNNs <1% loss
- [x] Layer-wise error analysis tool implemented
- [x] Mixed precision auto-selection working
- [x] ONNX export with QDQ nodes
- [x] TensorRT integration functional
- [x] End-to-end benchmarks complete
- [x] ResNet50 INT8: 3.82x speedup, 0.26% loss
- [x] BERT-base Dynamic INT8: 3.0x speedup, 0.29% loss

### 10. Production Readiness

#### Build & Test
```bash
# Clone and build
git clone https://github.com/your-org/compress-engine.git
cd compress-engine
mkdir build && cd build

# Release build with CUDA
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_CUDA=ON \
      -DUSE_TENSORRT=ON \
      -DCMAKE_CXX_FLAGS="-march=native" \
      ..

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure -j$(nproc)

# Expected: All 312 tests pass ✅
```

#### Continuous Integration
```yaml
# GitHub Actions workflow
- CPU tests: Ubuntu 20.04, GCC 11, AVX2
- GPU tests: CUDA 12.0, A100 (nightly)
- Benchmarks: vs ONNX Runtime (weekly)
- Performance regression: >5% triggers alert
- Documentation: Auto-generated and deployed
```

#### Code Quality Checks
```bash
# Static analysis
clang-tidy src/*.cpp include/*.h

# Formatting (Google C++ Style)
clang-format -i -style=Google src/*.cpp include/*.h

# Memory leaks
valgrind --leak-check=full --show-leak-kinds=all ./bin/test_all

# Thread safety
valgrind --tool=helgrind ./bin/test_threading

# undefined behavior
clang++ -fsanitize=undefined ...

# All checks pass ✅
```

### 11. Success Criteria Summary

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Turn 1: Tensor Library** |
| GEMM vs MKL | >80% | 85.5% | ✅ |
| Memory leaks | 0 | 0 | ✅ |
| **Turn 2: INT8 Quantization** |
| Accuracy loss | <1% | 0.26% | ✅ |
| Inference speedup | 3-4x | 3.82x | ✅ |
| Compression ratio | 4x | 3.92x | ✅ |
| **Turn 4: Structured Pruning** |
| Accuracy loss | <2% | 0.95% | ✅ |
| Inference speedup | 1.5-2x | 1.51x | ✅ |
| **Turn 5: Knowledge Distillation** |
| Student/teacher ratio | >95% | 96.0% | ✅ |
| **Turn 7: CUDA Acceleration** |
| CUDA vs cuBLAS | >80% | 92.1% | ✅ |
| **Turn 9: INT4 Quantization** |
| Accuracy loss | <3% | 1.98% | ✅ |
| Compression | 8x | 7.97x | ✅ |
| **Turn 10: Dynamic Quantization** |
| RNN accuracy loss | <1% | 0.6% | ✅ |
| RNN speedup | ~3x | 3.13x | ✅ |
| **Turn 14: TensorRT Integration** |
| Deployment works | Yes | Yes | ✅ |
| **Turn 15: End-to-End** |
| Overall compression | 10-50x | 14-15.8x | ✅ |
| Performance vs ONNX | Match | 93-98% | ✅ |
| Test coverage | >90% | 94.7% | ✅ |

**All critical success criteria met ✅**

**Estimated completion time:** 40-50 hours across 15 turns for expert C++ engineer with CUDA experience

**Difficulty:** EXTREME - Requires deep expertise in:
- Modern C++ (17+), templates, RAII
- SIMD programming (AVX2/AVX-512)
- CUDA programming, Tensor Cores
- Linear algebra, numerical methods
- Machine learning theory (quantization, pruning, distillation)
- Production software engineering (testing, CI/CD, documentation)
