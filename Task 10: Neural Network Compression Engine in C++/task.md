# Task: Build a Production-Grade Neural Network Compression Engine from Scratch

**Language:** C++ (17+)  
**Created:** November 23, 2025  
**Difficulty:** EXTREME

## Overview
Implement a production-ready neural network compression system in C++ that supports quantization (INT8/INT4), structured/unstructured pruning, knowledge distillation, and mixed-precision inference. The system must achieve 10-50x model compression with <2% accuracy loss, match or exceed ONNX Runtime performance, handle modern architectures (ResNet, BERT, Vision Transformers), and provide bit-exact reproducibility across runs.

**Key Challenge:** Build everything from scratch - tensor operations, CUDA kernels, quantization algorithms, automatic differentiation for distillation. No ML frameworks allowed. Must be production-quality with proper memory management, thread safety, and numerical stability.

---

## TURN 1 — Core Tensor Library with SIMD Optimization

**Role:** You are a high-performance computing engineer who has built tensor libraries for ML frameworks (like PyTorch's ATen or TensorFlow's Eigen Tensor). You understand cache hierarchies, memory layout optimization, and can write SIMD code that rivals vendor libraries.

**Background:** Before compression, we need a fast tensor library. This forms the foundation for all operations. Must support arbitrary-dimensional tensors, broadcasting, in-place operations, and achieve performance within 20% of MKL/cuBLAS.

**Reference:** Study:
- Eigen's tensor module architecture
- PyTorch's ATen tensor implementation
- Intel MKL and cuBLAS APIs
- SIMD intrinsics (AVX2/AVX-512)

**VERY IMPORTANT:**
- Zero-copy operations whenever possible
- Memory alignment (32-byte for AVX2, 64-byte for AVX-512)
- No memory leaks (validate with AddressSanitizer)
- Cache-friendly data layouts (row-major with stride support)
- SIMD operations must handle edge cases correctly
- Thread-safe (immutable tensors or proper synchronization)

**Goal:** Implement core tensor library with CPU SIMD optimization.

**Instructions:**

1. **Define tensor structure:**
```cpp
#include <vector>
#include <memory>
#include <algorithm>
#include <immintrin.h>  // AVX2/AVX-512

namespace compress {

enum class DType {
    Float32,
    Float16,
    Int32,
    Int8,
    UInt8
};

class Tensor {
public:
    // Constructor
    explicit Tensor(
        const std::vector<int64_t>& shape,
        DType dtype = DType::Float32,
        bool requires_grad = false
    );
    
    // Factory methods
    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f);
    
    // Data access
    template<typename T>
    T* data() { return static_cast<T*>(data_.get()); }
    
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_.get()); }
    
    // Shape information
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t numel() const;
    int64_t ndim() const { return shape_.size(); }
    
    // Operations
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor transpose(int dim1, int dim2) const;
    Tensor contiguous() const;  // Make tensor memory-contiguous
    bool is_contiguous() const;
    
private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::shared_ptr<void> data_;  // Aligned memory
    DType dtype_;
    bool requires_grad_;
    size_t offset_;  // For views
    
    void* allocate_aligned(size_t bytes, size_t alignment = 64);
    static void deallocate_aligned(void* ptr);
};

} // namespace compress
```

2. **Implement SIMD-optimized operations:**
```cpp
namespace compress {
namespace ops {

// Element-wise operations with AVX2
class SIMDOps {
public:
    // Addition: C = A + B (element-wise)
    static void add_float32_avx2(
        const float* a, 
        const float* b, 
        float* c, 
        size_t n
    ) {
        size_t i = 0;
        
        #ifdef __AVX2__
        // Process 8 floats at once with AVX2
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&c[i], vc);
        }
        #endif
        
        // Scalar fallback for remaining elements
        for (; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
    
    // Multiplication: C = A * B
    static void mul_float32_avx2(
        const float* a,
        const float* b,
        float* c,
        size_t n
    ) {
        size_t i = 0;
        
        #ifdef __AVX2__
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&c[i], vc);
        }
        #endif
        
        for (; i < n; ++i) {
            c[i] = a[i] * b[i];
        }
    }
    
    // ReLU: out = max(0, in)
    static void relu_float32_avx2(
        const float* in,
        float* out,
        size_t n
    ) {
        size_t i = 0;
        
        #ifdef __AVX2__
        __m256 zero = _mm256_setzero_ps();
        
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            __m256 result = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(&out[i], result);
        }
        #endif
        
        for (; i < n; ++i) {
            out[i] = std::max(0.0f, in[i]);
        }
    }
    
    // Matrix multiplication (GEMM) using blocking for cache efficiency
    static void gemm_float32(
        const float* A,     // M x K
        const float* B,     // K x N
        float* C,           // M x N
        int M, int K, int N,
        bool trans_a = false,
        bool trans_b = false,
        float alpha = 1.0f,
        float beta = 0.0f
    );
};

// High-level tensor operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& x);
Tensor conv2d(
    const Tensor& input,    // [N, C_in, H, W]
    const Tensor& weight,   // [C_out, C_in, KH, KW]
    const Tensor& bias,     // [C_out]
    int stride = 1,
    int padding = 0
);

} // namespace ops
} // namespace compress
```

3. **Implement cache-optimized GEMM:**
```cpp
void SIMDOps::gemm_float32(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    bool trans_a, bool trans_b,
    float alpha, float beta
) {
    // Blocking parameters (tuned for L1/L2 cache)
    constexpr int MC = 256;  // M block size
    constexpr int KC = 128;  // K block size
    constexpr int NC = 4096; // N block size
    constexpr int MR = 4;    // Micro-kernel M size
    constexpr int NR = 8;    // Micro-kernel N size (AVX2 = 8 floats)
    
    // Temporary buffer for packing
    std::vector<float> packed_a(MC * KC, 0.0f);
    std::vector<float> packed_b(KC * NC, 0.0f);
    
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);
        
        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(KC, K - pc);
            
            // Pack B into packed_b
            pack_b(B, packed_b.data(), pc, jc, kc, nc, K, N, trans_b);
            
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);
                
                // Pack A into packed_a
                pack_a(A, packed_a.data(), ic, pc, mc, kc, K, trans_a);
                
                // Micro-kernel: compute C[ic:ic+mc, jc:jc+nc] += A[ic:ic+mc, pc:pc+kc] * B[pc:pc+kc, jc:jc+nc]
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(NR, nc - jr);
                    
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(MR, mc - ir);
                        
                        // AVX2 micro-kernel (MR x NR)
                        gemm_micro_kernel_avx2(
                            packed_a.data() + ir * kc,
                            packed_b.data() + jr * kc,
                            C + (ic + ir) * N + (jc + jr),
                            kc, N, mr, nr,
                            alpha, beta
                        );
                    }
                }
            }
        }
    }
}

// AVX2 micro-kernel (4x8 = 4 rows x 8 cols)
static void gemm_micro_kernel_avx2(
    const float* A_packed,  // [MR x KC]
    const float* B_packed,  // [NR x KC]
    float* C,               // [MR x N]
    int KC, int ldc,
    int MR, int NR,
    float alpha, float beta
) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    
    for (int k = 0; k < KC; ++k) {
        // Load B[k, 0:8]
        __m256 b0 = _mm256_loadu_ps(&B_packed[k * 8]);
        
        // Broadcast A[0, k]
        __m256 a0 = _mm256_broadcast_ss(&A_packed[0 * KC + k]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        
        if (MR > 1) {
            __m256 a1 = _mm256_broadcast_ss(&A_packed[1 * KC + k]);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
        }
        
        if (MR > 2) {
            __m256 a2 = _mm256_broadcast_ss(&A_packed[2 * KC + k]);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
        }
        
        if (MR > 3) {
            __m256 a3 = _mm256_broadcast_ss(&A_packed[3 * KC + k]);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
        }
    }
    
    // Scale by alpha
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    c00 = _mm256_mul_ps(c00, alpha_vec);
    c10 = _mm256_mul_ps(c10, alpha_vec);
    c20 = _mm256_mul_ps(c20, alpha_vec);
    c30 = _mm256_mul_ps(c30, alpha_vec);
    
    // Add beta * C and store
    if (beta != 0.0f) {
        __m256 beta_vec = _mm256_set1_ps(beta);
        __m256 c_old;
        
        c_old = _mm256_loadu_ps(&C[0 * ldc]);
        c00 = _mm256_fmadd_ps(beta_vec, c_old, c00);
        _mm256_storeu_ps(&C[0 * ldc], c00);
        
        if (MR > 1) {
            c_old = _mm256_loadu_ps(&C[1 * ldc]);
            c10 = _mm256_fmadd_ps(beta_vec, c_old, c10);
            _mm256_storeu_ps(&C[1 * ldc], c10);
        }
        
        if (MR > 2) {
            c_old = _mm256_loadu_ps(&C[2 * ldc]);
            c20 = _mm256_fmadd_ps(beta_vec, c_old, c20);
            _mm256_storeu_ps(&C[2 * ldc], c20);
        }
        
        if (MR > 3) {
            c_old = _mm256_loadu_ps(&C[3 * ldc]);
            c30 = _mm256_fmadd_ps(beta_vec, c_old, c30);
            _mm256_storeu_ps(&C[3 * ldc], c30);
        }
    } else {
        _mm256_storeu_ps(&C[0 * ldc], c00);
        if (MR > 1) _mm256_storeu_ps(&C[1 * ldc], c10);
        if (MR > 2) _mm256_storeu_ps(&C[2 * ldc], c20);
        if (MR > 3) _mm256_storeu_ps(&C[3 * ldc], c30);
    }
}
```

4. **Write comprehensive tests:**
```cpp
#include <gtest/gtest.h>
#include \"tensor.h\"

namespace compress {
namespace test {

TEST(TensorTest, BasicConstruction) {
    Tensor t({2, 3, 4}, DType::Float32);
    
    EXPECT_EQ(t.shape(), std::vector<int64_t>({2, 3, 4}));
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.ndim(), 3);
}

TEST(TensorTest, MemoryAlignment) {
    Tensor t({1024, 1024}, DType::Float32);
    
    // Verify 64-byte alignment for AVX-512
    auto* ptr = t.data<float>();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 64, 0) << \"Tensor memory not 64-byte aligned\";
}

TEST(TensorTest, ZeroMemoryLeaks) {
    // Allocate and deallocate many tensors
    for (int i = 0; i < 1000; ++i) {
        Tensor t = Tensor::randn({100, 100});
        // Automatic deallocation via RAII
    }
    // Run with AddressSanitizer to verify no leaks
}

TEST(SIMDOpsTest, AdditionCorrectness) {
    Tensor a = Tensor::ones({1024});
    Tensor b = Tensor::ones({1024});
    
    Tensor c = ops::add(a, b);
    
    auto* c_data = c.data<float>();
    for (int i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(c_data[i], 2.0f);
    }
}

TEST(SIMDOpsTest, GEMMCorrectness) {
    // Test GEMM against naive implementation
    const int M = 64, K = 128, N = 96;
    
    Tensor A = Tensor::randn({M, K});
    Tensor B = Tensor::randn({K, N});
    
    // SIMD-optimized version
    Tensor C_fast = ops::matmul(A, B);
    
    // Naive implementation for verification
    Tensor C_ref = naive_matmul(A, B);
    
    // Compare results
    auto* fast_data = C_fast.data<float>();
    auto* ref_data = C_ref.data<float>();
    
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(fast_data[i], ref_data[i], 1e-4);
    }
}

BENCHMARK(BM_GEMM_1024x1024) {
    Tensor A = Tensor::randn({1024, 1024});
    Tensor B = Tensor::randn({1024, 1024});
    
    for (auto _ : state) {
        Tensor C = ops::matmul(A, B);
        benchmark::DoNotOptimize(C.data<float>());
    }
    
    // Expected: \u003e100 GFLOPS on modern CPU
    state.SetItemsProcessed(
        state.iterations() * 2LL * 1024 * 1024 * 1024  // 2*N^3 FLOPs
    );
}

} // namespace test
} // namespace compress
```

**Deliverables:**
- Core tensor library with proper memory management
- SIMD-optimized operations (add, mul, ReLU, GEMM)
- Cache-blocked GEMM achieving \u003e80% of MKL performance
- Comprehensive tests (correctness + performance)
- Zero memory leaks validated with sanitizers

---

## TURN 2 — Post-Training Quantization (PTQ) to INT8

**Instructions:**

Implement post-training quantization that converts FP32 models to INT8 with \u003c1% accuracy loss.

**Background:** Quantization reduces model size 4x and speeds up inference 4x on CPUs with VNNI (AVX512_VNNI) and on GPUs with Tensor Cores. Per-tensor and per-channel quantization strategies. Calibration using representative dataset.

**Requirements:**
- Asymmetric quantization: q = round(x/scale + zero_point)
- Per-tensor and per-channel strategies
- Calibration using min-max or percentile methods
- INT8 GEMM using VNNI or DP4A instructions
- Accuracy within 1% of FP32 baseline

**Implement:**

1. **Quantization parameters:**
```cpp
struct QuantParams {
    float scale;
    int32_t zero_point;
    int32_t qmin;  // Typically 0 for uint8 or -128 for int8
    int32_t qmax;  // Typically 255 for uint8 or 127 for int8
};

class Quantizer {
public:
    // Calibration: compute scale and zero_point from FP32 tensor
    static QuantParams calibrate_min_max(const Tensor& x);
    static QuantParams calibrate_percentile(const Tensor& x, float percentile = 99.99f);
    
    // Quantization: FP32 → INT8
    static Tensor quantize(const Tensor& x, const QuantParams& params);
    
    // Dequantization: INT8 → FP32
    static Tensor dequantize(const Tensor& x_q, const QuantParams& params);
    
    // Quantized operations
    static Tensor quantized_matmul_int8(
        const Tensor& a_q,       // INT8 quantized
        const Tensor& b_q,       // INT8 quantized
        const QuantParams& a_params,
        const QuantParams& b_params,
        const QuantParams& out_params
    );
};
```

2. **INT8 GEMM with AVX512_VNNI:**
```cpp
// INT8 matrix multiplication using VNNI (Vector Neural Network Instructions)
void gemm_int8_vnni(
    const int8_t* A,    // [M x K]
    const int8_t* B,    // [K x N]
    int32_t* C,         // [M x N] (accumulate in INT32)
    int M, int K, int N
) {
    #ifdef __AVX512VNNI__
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; j += 16) {
            __m512i acc = _mm512_setzero_si512();
            
            for (int k = 0; k < K; k += 4) {
                // Load 4 elements from A[i, k:k+4]
                __m128i a_vec = _mm_loadu_si32(&A[i * K + k]);
                __m512i a_broadcast = _mm512_broadcast_i32x4(a_vec);
                
                // Load 4x16 elements from B[k:k+4, j:j+16]
                __m512i b_vec = _mm512_loadu_si512(&B[k * N + j]);
                
                // VNNI: acc += a * b (4-way dot product)
                acc = _mm512_dpbusd_epi32(acc, a_broadcast, b_vec);
            }
            
            // Store result
            _mm512_storeu_si512(&C[i * N + j], acc);
        }
    }
    #else
    // Fallback for non-VNNI CPUs
    naive_gemm_int8(A, B, C, M, K, N);
    #endif
}
```

3. **Layer-wise quantization:**
```cpp
class QuantizedLinear {
public:
    QuantizedLinear(
        const Tensor& weight_fp32,
        const Tensor& bias_fp32,
        const QuantParams& weight_qparams,
        const QuantParams& input_qparams,
        const QuantParams& output_qparams
    );
    
    Tensor forward(const Tensor& input_q) {
        // input_q: [N, in_features] INT8
        // weight_q: [out_features, in_features] INT8
        // Compute: output = input @ weight^T + bias
        
        // INT8 GEMM → INT32
        Tensor output_i32 = quantized_matmul_int8(
            input_q, weight_q_,
            input_qparams_, weight_qparams_, output_qparams_
        );
        
        // Rescale and add bias
        Tensor output_fp32 = requantize(output_i32);
        
        // Quantize output to INT8
        return Quantizer::quantize(output_fp32, output_qparams_);
    }
    
private:
    Tensor weight_q_;  // INT8
    Tensor bias_;
    QuantParams weight_qparams_;
    QuantParams input_qparams_;
    QuantParams output_qparams_;
};
```

4. **End-to-end quantization workflow:**
```cpp
class ModelQuantizer {
public:
    // Calibrate quantization parameters using representative dataset
    void calibrate(
        Model& model,
        DataLoader& calibration_data,
        int num_batches = 100
    ) {
        model.eval();
        
        // Insert observers to collect statistics
        for (auto& layer : model.layers()) {
            layer->attach_observer(std::make_unique<MinMaxObserver>());
        }
        
        // Run calibration
        for (int i = 0; i < num_batches; ++i) {
            auto batch = calibration_data.next();
            model.forward(batch);
        }
        
        // Compute quantization parameters
        for (auto& layer : model.layers()) {
            auto stats = layer->get_observer_stats();
            layer_qparams_[layer->name()] = Quantizer::calibrate_min_max(stats);
        }
    }
    
    // Convert FP32 model to INT8
    std::unique_ptr<QuantizedModel> quantize(const Model& model_fp32) {
        auto model_q = std::make_unique<QuantizedModel>();
        
        for (const auto& layer : model_fp32.layers()) {
            if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
                model_q->add_layer(std::make_unique<QuantizedLinear>(
                    linear->weight(),
                    linear->bias(),
                    layer_qparams_[layer->name()].weight,
                    layer_qparams_[layer->name()].input,
                    layer_qparams_[layer->name()].output
                ));
            }
            // Handle other layer types...
        }
        
        return model_q;
    }
    
private:
    std::unordered_map<std::string, LayerQuantParams> layer_qparams_;
};
```

**Tests:**
```cpp
TEST(QuantizationTest, SymmetricQuantization) {
    Tensor x = Tensor::randn({100, 100});
    
    auto qparams = Quantizer::calibrate_min_max(x);
    Tensor x_q = Quantizer::quantize(x, qparams);
    Tensor x_dq = Quantizer::dequantize(x_q, qparams);
    
    // Check quantization error
    float max_error = compute_max_abs_error(x, x_dq);
    EXPECT_LT(max_error, qparams.scale * 0.5f);  // Within half a quantization step
}

TEST(QuantizationTest, INT8GEMMCorrectness) {
    // Compare INT8 GEMM with FP32 GEMM
    Tensor A_fp32 = Tensor::randn({64, 128});
    Tensor B_fp32 = Tensor::randn({128, 96});
    
    // Quantize inputs
    auto a_qparams = Quantizer::calibrate_min_max(A_fp32);
    auto b_qparams = Quantizer::calibrate_min_max(B_fp32);
    
    Tensor A_q = Quantizer::quantize(A_fp32, a_qparams);
    Tensor B_q = Quantizer::quantize(B_fp32, b_qparams);
    
    // INT8 matmul
    // (will need to requantize result)
    Tensor C_q = Quantizer::quantized_matmul_int8(A_q, B_q, a_qparams, b_qparams, ...);
    Tensor C_dq = Quantizer::dequantize(C_q, c_qparams);
    
    // FP32 reference
    Tensor C_fp32 = ops::matmul(A_fp32, B_fp32);
    
    // Compare
    float error = compute_relative_error(C_fp32, C_dq);
    EXPECT_LT(error, 0.01f);  // <1% error
}

BENCHMARK(BM_INT8_GEMM_vs_FP32) {
    Tensor A = Tensor::randn({512, 512});
    Tensor B = Tensor::randn({512, 512});
    
    auto a_qparams = Quantizer::calibrate_min_max(A);
    auto b_qparams = Quantizer::calibrate_min_max(B);
    
    Tensor A_q = Quantizer::quantize(A, a_qparams);
    Tensor B_q = Quantizer::quantize(B, b_qparams);
    
    for (auto _ : state) {
        Tensor C_q = Quantizer::quantized_matmul_int8(A_q, B_q, ...);
        benchmark::DoNotOptimize(C_q.data<int8_t>());
    }
    
    // Expected: 3-4x speedup on AVX512_VNNI CPUs
}
```

---

## TURN 3 — Force Failure: Quantization Error Accumulation

**Ask the AI:**
> \"Your INT8 quantization works well for single layers, but what happens when you stack 50 quantized layers (like ResNet50)? Show a test where quantization errors accumulate and cause accuracy to drop by \u003e5%. Explain the mathematical reason and implement a fix using QAT (Quantization-Aware Training).\"

**Expected failure:**
```cpp
TEST(QuantizationTest, DeepModelQuantizationFails) {
    // Load ResNet50 FP32 (50 layers)
    auto model_fp32 = load_resnet50();
    
    // Post-training quantization
    ModelQuantizer quantizer;
    quantizer.calibrate(model_fp32, calibration_data, 100);
    auto model_q = quantizer.quantize(model_fp32);
    
    // Evaluate on test set
    float acc_fp32 = evaluate(model_fp32, test_data);  // e.g., 76.5%
    float acc_int8 = evaluate(model_q, test_data);     // e.g., 69.2%
    
    // FAILURE: Accuracy drop = 7.3% (exceeds 1% tolerance)
    EXPECT_LT(std::abs(acc_fp32 - acc_int8), 0.01 * acc_fp32);  // FAILS
}
```

**Mathematical explanation:**
- Each layer has quantization error ε_layer ≈ scale/2
- For 50 layers, errors compound: ε_total ≈ sqrt(50) * ε_layer (random walk)
- Final output has much larger error than single layer

**Fix:** Implement fake quantization for QAT:
```cpp
// Simulate quantization during forward pass but keep gradients flowing
Tensor fake_quantize(const Tensor& x, const QuantParams& qparams) {
    // Forward: quantize then dequantize
    Tensor x_q = quantize_implementation(x, qparams);
    Tensor x_dq = dequantize_implementation(x_q, qparams);
    
    // Backward: use straight-through estimator (gradient flows through unchanged)
    // This requires custom autograd (see Turn 6)
    
    return x_dq;
}
```

---

## TURN 4 — Structured Pruning with Channel-wise Sparsity

**Instructions:**

Implement structured pruning that removes entire channels/neurons while maintaining accuracy.

**Background:** Unstructured pruning (random weights) doesn't speedup inference much. Structured pruning (entire channels) can be accelerated with dense operations. Target: 50-70% sparsity with \u003c2% accuracy loss.

**Strategies:**
- L1 norm-based channel selection
- Iterative magnitude pruning
- Gradual pruning schedule
- Fine-tuning after pruning

**Implement:**
```cpp
class StructuredPruner {
public:
    enum class PruningStrategy {
        L1Norm,           // Prune channels with smallest L1 norm
        L2Norm,           // Prune channels with smallest L2 norm
        Taylor,           // Prune channels with smallest Taylor expansion
        FPGM              // Filter Pruning via Geometric Median
    };
    
    StructuredPruner(PruningStrategy strategy, float target_sparsity);
    
    // Compute importance scores for each channel
    std::vector<float> compute_channel_importance(
        const Tensor& weight,  // [out_channels, in_channels, H, W]
        const Tensor& gradients = {}  // Optional: for Taylor pruning
    );
    
    // Prune least important channels
    Tensor prune_channels(
        const Tensor& weight,
        const std::vector<int>& channels_to_keep
    );
    
    // Global pruning across all layers
    void global_prune(
        Model& model,
        float sparsity,
        const std::string& prune_scope = \"all\"  // \"all\", \"conv\", \"linear\"
    );
    
    // Gradual pruning schedule
    void iterative_prune(
        Model& model,
        DataLoader& train_data,
        int num_iterations = 10,
        int finetune_epochs_per_iteration = 5
    );
};

class TaylorPruner {
public:
    // Prune using first-order Taylor expansion: importance ≈ |w * grad(w)|
    std::vector<float> compute_taylor_importance(
        const Tensor& weight,
        const Tensor& weight_grad
    ) {
        std::vector<float> importance(weight.shape()[0], 0.0f);
        
        auto* w_data = weight.data<float>();
        auto* g_data = weight_grad.data<float>();
        
        int out_channels = weight.shape()[0];
        int elements_per_channel = weight.numel() / out_channels;
        
        for (int c = 0; c < out_channels; ++c) {
            float channel_importance = 0.0f;
            for (int i = 0; i < elements_per_channel; ++i) {
                int idx = c * elements_per_channel + i;
                channel_importance += std::abs(w_data[idx] * g_data[idx]);
            }
            importance[c] = channel_importance;
        }
        
        return importance;
    }
};
```

**Pruning workflow:**
```cpp
TEST(PruningTest, StructuredPruningWorkflow) {
    // Load pre-trained model
    auto model = load_resnet50();
    float acc_baseline = evaluate(model, test_data);  // 76.5%
    
    // Structured pruning (50% sparsity)
    StructuredPruner pruner(StructuredPruner::L1Norm, 0.5);
    pruner.iterative_prune(model, train_data, 10, 5);
    
    float acc_pruned = evaluate(model, test_data);  // Expected: >74.5% (< 2% drop)
    
    EXPECT_GT(acc_pruned, acc_baseline * 0.98);
    
    // Verify model size reduced
    size_t params_original = count_parameters(load_resnet50());
    size_t params_pruned = count_parameters(model);
    
    EXPECT_LT(params_pruned, params_original * 0.6);  // At least 40% reduction
}

BENCHMARK(BM_PrunedVsUnpruned) {
    auto model_full = load_resnet50();
    auto model_pruned = load_resnet50_pruned_50();
    
    Tensor input = Tensor::randn({1, 3, 224, 224});
    
    // Benchmark pruned model
    for (auto _ : state) {
        Tensor output = model_pruned.forward(input);
        benchmark::DoNotOptimize(output.data<float>());
    }
    
    // Expected: 1.5-2x speedup for 50% channel pruning
}
```

---

## TURN 5 — Knowledge Distillation with Soft Targets

**Instructions:**

Train a small student model to mimic a large teacher model.

**Background:** Knowledge distillation transfers knowledge from large (teacher) to small (student) models. Student learns from teacher's soft probabilities, not just hard labels. Can achieve better accuracy than training student alone.

**Requirements:**
- Temperature scaling for soft targets
- KL divergence loss between teacher and student
- Combined loss: αL_KD + (1-α)L_CE
- Student achieves \u003e95% of teacher accuracy

**Implement:**
```cpp
class DistillationLoss {
public:
    DistillationLoss(float temperature, float alpha)
        : temperature_(temperature), alpha_(alpha) {}
    
    Tensor compute_loss(
        const Tensor& student_logits,
        const Tensor& teacher_logits,
        const Tensor& labels
    ) {
        // Soft targets from teacher
        Tensor teacher_probs = softmax_with_temperature(teacher_logits, temperature_);
        Tensor student_log_probs = log_softmax_with_temperature(student_logits, temperature_);
        
        // KL divergence loss
        Tensor kl_loss = kl_divergence(student_log_probs, teacher_probs);
        kl_loss = kl_loss * (temperature_ * temperature_);  // Scale by T^2
        
        // Cross-entropy with hard labels
        Tensor ce_loss = cross_entropy(student_logits, labels);
        
        // Combined loss
        return alpha_ * kl_loss + (1.0f - alpha_) * ce_loss;
    }
    
private:
    float temperature_;
    float alpha_;
    
    Tensor softmax_with_temperature(const Tensor& logits, float T) {
        return softmax(logits / T);
    }
};

void train_with_distillation(
    Model& student,
    const Model& teacher,
    DataLoader& train_data,
    int num_epochs = 100,
    float temperature = 3.0f,
    float alpha = 0.7f
) {
    DistillationLoss criterion(temperature, alpha);
    SGDOptimizer optimizer(student.parameters(), /*lr=*/0.1);
    
    teacher.eval();
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto batch : train_data) {
            // Forward pass
            Tensor student_logits = student.forward(batch.inputs);
            Tensor teacher_logits = teacher.forward(batch.inputs);
            
            // Compute distillation loss
            Tensor loss = criterion.compute_loss(
                student_logits, teacher_logits, batch.labels
            );
            
            // Backward pass
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}
```

**Test distillation effectiveness:**
```cpp
TEST(DistillationTest, StudentMatchesTeacher) {
    // Teacher: ResNet50 (25M parameters, 76.5% accuracy)
    auto teacher = load_resnet50();
    
    // Student: ResNet18 (11M parameters)
    auto student = create_resnet18();
    
    // Train student with distillation
    train_with_distillation(student, teacher, train_data, 100);
    
    float teacher_acc = evaluate(teacher, test_data);  // 76.5%
    float student_acc = evaluate(student, test_data);  // Target: >72.7% (95% of teacher)
    
    EXPECT_GT(student_acc, 0.95 * teacher_acc);
    
    // Compare to training student alone (without distillation)
    auto student_baseline = create_resnet18();
    train_without_distillation(student_baseline, train_data, 100);
    float baseline_acc = evaluate(student_baseline, test_data);  // ~69%
    
    EXPECT_GT(student_acc, baseline_acc + 0.02);  // Distillation adds \u003e2% accuracy
}
```

---

## TURN 6 — Automatic Differentiation for Gradient Computation

**Instructions:**

Implement reverse-mode automatic differentiation to support training and fine-tuning.

**Background:** Need gradients for knowledge distillation and quantization-aware training. Build a computational graph and compute gradients via backpropagation.

**Implement:**
```cpp
class AutogradFunction {
public:
    virtual ~AutogradFunction() = default;
    
    virtual Tensor forward(const std::vector<Tensor>& inputs) = 0;
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
};

class AddBackward : public AutogradFunction {
public:
    Tensor forward(const std::vector<Tensor>& inputs) override {
        return inputs[0] + inputs[1];
    }
    
    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // d/da (a + b) = 1, d/db (a + b) = 1
        return {grad_output, grad_output};
    }
};

class MatMulBackward : public AutogradFunction {
public:
    Tensor forward(const std::vector<Tensor>& inputs) override {
        a_ = inputs[0];
        b_ = inputs[1];
        return ops::matmul(a_, b_);
    }
    
    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // d/dA (A @ B) = grad_output @ B^T
        // d/dB (A @ B) = A^T @ grad_output
        Tensor grad_a = ops::matmul(grad_output, b_.transpose(-1, -2));
        Tensor grad_b = ops::matmul(a_.transpose(-1, -2), grad_output);
        return {grad_a, grad_b};
    }
    
private:
    Tensor a_, b_;
};

// Computational graph node
struct GraphNode {
    Tensor value;
    std::shared_ptr<AutogradFunction> grad_fn;
    std::vector<std::shared_ptr<GraphNode>> inputs;
    bool requires_grad;
};

void backward(const Tensor& loss) {
    // Topological sort of computation graph
    std::vector<std::shared_ptr<GraphNode>> topo_order = topological_sort(loss.grad_fn_);
    
    // Initialize gradient
    loss.graph_node_->grad = Tensor::ones(loss.shape());
    
    // Reverse pass
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        auto node = *it;
        
        if (!node->requires_grad) continue;
        
        // Compute gradients for inputs
        auto grad_inputs = node->grad_fn->backward(node->grad);
        
        // Accumulate gradients
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            if (node->inputs[i]->requires_grad) {
                if (!node->inputs[i]->grad.defined()) {
                    node->inputs[i]->grad = grad_inputs[i];
                } else {
                    node->inputs[i]->grad = node->inputs[i]->grad + grad_inputs[i];
                }
            }
        }
    }
}
```

---

## TURN 7 — CUDA/cuDNN Integration for GPU Acceleration

**Instructions:**

Implement CUDA kernels for quantized inference and distillation training.

**Requirements:**
- Custom CUDA kernels for INT8 GEMM using Tensor Cores
- cuDNN integration for Conv2D
- Memory-efficient implementations (minimize D2H/H2D transfers)
- Kernel fusion (ReLU + Quantize in one kernel)

**Implement:**
```cpp
// CUDA kernel for INT8 GEMM using Tensor Cores (Turing+)
__global__ void gemm_int8_tensorcore(
    const int8_t* __restrict__ A,  // [M, K]
    const int8_t* __restrict__ B,  // [K, N]
    int32_t* __restrict__ C,       // [M, N]
    int M, int K, int N
) {
    // Use WMMA (Warp Matrix Multiply-Accumulate) API
    using namespace nvcuda::wmma;
    
    // Declare fragments
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0);
    
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y) * 16;
    
    // Accumulate across K dimension
    for (int k = 0; k < K; k += 16) {
        // Load matrices
        load_matrix_sync(a_frag, A + warp_m * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warp_n, N);
        
        // Tensor Core multiply-accumulate
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    store_matrix_sync(C + warp_m * N + warp_n, c_frag, N, mem_row_major);
}

// Fused ReLU + Quantize kernel
__global__ void fused_relu_quantize(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    int32_t zero_point,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // ReLU
        float val = fmaxf(input[idx], 0.0f);
        
        // Quantize
        int32_t q_val = __float2int_rn(val / scale) + zero_point;
        q_val = max(-128, min(127, q_val));  // Clamp to INT8 range
        
        output[idx] = static_cast<int8_t>(q_val);
    }
}
```

---

## TURN 8 — Force Failure: GPU Memory Overflow

**Ask the AI:**
> \"Your CUDA implementation works for small models but crashes with OOM (out-of-memory) on large models like BERT-Large. Show a test that demonstrates the failure and implement gradient checkpointing to trade compute for memory.\"

**Expected failure:**
```cpp
TEST(CUDATest, LargeModelOOM) {
    // BERT-Large: 24 layers, 1024 hidden dim, 340M parameters
    auto model = create_bert_large();
    model.to_device(Device::CUDA);
    
    // Batch size 32, sequence length 512
    Tensor input = Tensor::randn({32, 512, 1024}).to_device(Device::CUDA);
    
    // Forward pass
    Tensor output = model.forward(input);  // OOM: allocates ~24GB for activations
    
    // FAILURE: CUDA out of memory
}
```

**Fix: Gradient checkpointing:**
```cpp
class CheckpointedModel {
    // Don't store all activations - recompute during backward
    Tensor forward(const Tensor& x) {
        Tensor h = x;
        
        for (int i = 0; i < num_layers_; ++i) {
            if (i % checkpoint_every_ == 0) {
                // Checkpoint: save activation
                checkpoints_.push_back(h);
                h = layers_[i]->forward(h);
            } else {
                // No checkpoint: don't save activation
                h = layers_[i]->forward(h);
            }
        }
        
        return h;
    }
    
    void backward(const Tensor& grad_output) {
        // Recompute activations during backward
    }
};

---

---

## TURN 9 — INT4 Quantization with Weight Clustering

**Instructions:**

Implement aggressive INT4 quantization using k-means clustering for extremely low bit-width compression.

**Background:** INT4 provides 8x model compression compared to FP32 but significantly challenges accuracy. Weight clustering finds representative values, reducing unique weights to 16 clusters (4 bits). Combines with per-channel scales for better accuracy.

**Requirements:**
- K-means clustering to find 16 weight clusters per channel
- Lookup table (LUT) based INT4 operations
- Per-channel quantization scales
- Target: 8x compression with <3% accuracy loss

**Implement:**
```cpp
struct INT4Params {
    std::vector<float> cluster_centers;  // 16 centers per channel
    std::vector<int> cluster_assignments;  // Weight → cluster mapping
    float scale;
    int num_clusters = 16;
};

class INT4Quantizer {
public:
    // K-means clustering for weight quantization
    static INT4Params quantize_weights_kmeans(
        const Tensor& weights,  // [out_ch, in_ch, ...]
        int num_clusters = 16
    ) {
        INT4Params params;
        params.num_clusters = num_clusters;
        
        int out_channels = weights.shape()[0];
        int weights_per_channel = weights.numel() / out_channels;
        
        auto* data = weights.data<float>();
        
        // Per-channel clustering
        for (int ch = 0; ch < out_channels; ++ch) {
            std::vector<float> channel_weights(
                data + ch * weights_per_channel,
                data + (ch + 1) * weights_per_channel
            );
            
            // K-means clustering
            auto [centers, assignments] = kmeans_clustering(
                channel_weights, num_clusters
            );
            
            params.cluster_centers.insert(
                params.cluster_centers.end(),
                centers.begin(), centers.end()
            );
            params.cluster_assignments.insert(
                params.cluster_assignments.end(),
                assignments.begin(), assignments.end()
            );
        }
        
        return params;
    }
    
    // K-means algorithm
    static std::pair<std::vector<float>, std::vector<int>> kmeans_clustering(
        const std::vector<float>& data,
        int k,
        int max_iters = 100
   ) {
        std::vector<float> centers(k);
        std::vector<int> assignments(data.size());
        
        // Initialize centers using k-means++
        centers = kmeans_plus_plus_init(data, k);
        
        // Iterate until convergence
        for (int iter = 0; iter < max_iters; ++iter) {
            bool changed = false;
            
            // Assignment step
            for (size_t i = 0; i < data.size(); ++i) {
                int closest = 0;
                float min_dist = std::abs(data[i] - centers[0]);
                
                for (int j = 1; j < k; ++j) {
                    float dist = std::abs(data[i] - centers[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest = j;
                    }
                }
                
                if (assignments[i] != closest) {
                    assignments[i] = closest;
                    changed = true;
                }
            }
            
            if (!changed) break;
            
            // Update step
            std::vector<float> sum(k, 0.0f);
            std::vector<int> count(k, 0);
            
            for (size_t i = 0; i < data.size(); ++i) {
                int cluster = assignments[i];
                sum[cluster] += data[i];
                count[cluster]++;
            }
            
            for (int j = 0; j < k; ++j) {
                if (count[j] > 0) {
                    centers[j] = sum[j] / count[j];
                }
            }
        }
        
        return {centers, assignments};
    }
    
    // Encode weights to INT4 (packed)
    static std::vector<uint8_t> pack_int4(const std::vector<int>& indices) {
        // Pack two 4-bit indices into one byte
        std::vector<uint8_t> packed((indices.size() + 1) / 2);
        
        for (size_t i = 0; i < indices.size(); i += 2) {
            uint8_t low = indices[i] & 0x0F;
            uint8_t high = (i + 1 < indices.size()) ? (indices[i+1] & 0x0F) : 0;
            packed[i / 2] = (high << 4) | low;
        }
        
        return packed;
    }
};

// INT4 Linear layer
class INT4Linear {
public:
    INT4Linear(
        const Tensor& weight_fp32,
        const Tensor& bias,
        const INT4Params& params
    ) {
        // Store compressed weights
        packed_weights_ = INT4Quantizer::pack_int4(params.cluster_assignments);
        cluster_centers_ = params.cluster_centers;
        bias_ = bias;
        
        // Store dimensions
        out_features_ = weight_fp32.shape()[0];
        in_features_ = weight_fp32.shape()[1];
    }
    
    Tensor forward(const Tensor& input) {
        // Decompress weights
        Tensor weight = decompress_weights();
        
        // Standard matrix multiplication
        return ops::matmul(input, weight.transpose()) + bias_;
    }
    
private:
    Tensor decompress_weights() {
        Tensor weight({out_features_, in_features_}, DType::Float32);
        auto* w_data = weight.data<float>();
        
        int weights_per_channel = in_features_;
        
        for (int ch = 0; ch < out_features_; ++ch) {
            const float* centers = &cluster_centers_[ch * 16];
            
            for (int i = 0; i < weights_per_channel; ++i) {
                int global_idx = ch * weights_per_channel + i;
                
                // Unpack INT4
                uint8_t packed_byte = packed_weights_[global_idx / 2];
                int cluster_idx;
                if (global_idx % 2 == 0) {
                    cluster_idx = packed_byte & 0x0F;
                } else {
                    cluster_idx = (packed_byte >> 4) & 0x0F;
                }
                
                // Lookup cluster center
                w_data[ch * weights_per_channel + i] = centers[cluster_idx];
            }
        }
        
        return weight;
    }
    
    std::vector<uint8_t> packed_weights_;  // INT4 packed (2 per byte)
    std::vector<float> cluster_centers_;    // 16 centers per channel
    Tensor bias_;
    int out_features_;
    int in_features_;
};
```

**Test INT4 quantization:**
```cpp
TEST(INT4Test, WeightClustering) {
    // Create random weights
    Tensor weights = Tensor::randn({256, 256});
    
    // Quantize to INT4
    auto params = INT4Quantizer::quantize_weights_kmeans(weights, 16);
    
    // Verify: 16 unique clusters per channel
    EXPECT_EQ(params.cluster_centers.size(), 256 * 16);  // 256 channels × 16 centers
    
    // Reconstruct and measure error
    Tensor reconstructed = reconstruct_from_clusters(weights, params);
    
    float mse = compute_mse(weights, reconstructed);
    std::cout << "INT4 MSE: " << mse << std::endl;
    
    // Error should be reasonable
    EXPECT_LT(mse, 0.01);
}

BENCHMARK(BM_INT4_vs_INT8_vs_FP32) {
    Tensor input = Tensor::randn({32, 256});
    Tensor weight_fp32 = Tensor::randn({512, 256});
    
    // Baseline FP32
    auto time_fp32 = benchmark([&]() {
        return ops::matmul(input, weight_fp32.transpose());
    });
    
    // INT8
    auto params_int8 = Quantizer::calibrate_min_max(weight_fp32);
    auto weight_int8 = Quantizer::quantize(weight_fp32, params_int8);
    auto time_int8 = benchmark([&]() {
        // INT8 matmul
    });
    
    // INT4
    auto params_int4 = INT4Quantizer::quantize_weights_kmeans(weight_fp32);
    INT4Linear layer_int4(weight_fp32, Tensor::zeros({512}), params_int4);
    auto time_int4 = benchmark([&]() {
        return layer_int4.forward(input);
    });
    
    std::cout << "FP32: " << time_fp32 << "ms\n";
    std::cout << "INT8: " << time_int8 << "ms (" << time_fp32/time_int8 << "x)\n";
    std::cout << "INT4: " << time_int4 << "ms (" << time_fp32/time_int4 << "x)\n";
}
```

---

## TURN 10 — Dynamic Quantization for RNNs/LSTMs

**Instructions:**

Implement dynamic quantization for recurrent models where activations vary significantly across timesteps.

**Background:** Static quantization fails for RNNs because activation ranges change dramatically over sequence. Dynamic quantization computes scales per-timestep at runtime. Trade: slightly slower than static INT8, much faster than FP32.

**Requirements:**
- Per-timestep activation quantization
- Weight-only quantization (weights INT8, activations INT8 with dynamic scale)
- Hidden state quantization
- Target: 3x speedup over FP32 with <1% accuracy loss

**Implement:**
```cpp
class DynamicQuantizer {
public:
    // Compute quantization params on-the-fly
    static QuantParams compute_dynamic_params(const Tensor& x) {
        // Fast min/max reduction
        float x_min = compute_min(x);
        float x_max = compute_max(x);
        
        float scale = (x_max - x_min) / 255.0f;
        int zero_point = static_cast<int>(-x_min / scale);
        
        return {scale, zero_point, 0, 255};
    }
    
    // Dynamic quantization: quantize on the fly
    static Tensor dynamic_quantize_and_matmul(
        const Tensor& activation,      // FP32 [batch, in_features]
        const Tensor& weight_q,         // INT8 [out_features, in_features]
        const QuantParams& weight_params
    ) {
        // Dynamically quantize activation
        auto act_params = compute_dynamic_params(activation);
        Tensor activation_q = Quantizer::quantize(activation, act_params);
        
        // INT8 matmul
        Tensor output_q = Quantizer::quantized_matmul_int8(
            activation_q, weight_q,
            act_params, weight_params, /*output_params=*/{}
        );
        
        // Dequantize result
        return Quantizer::dequantize(output_q, /*computed output params*/);
    }
};

class DynamicQuantizedLSTM {
public:
    DynamicQuantizedLSTM(
        int input_size,
        int hidden_size,
        const Tensor& weight_ih_fp32,
        const Tensor& weight_hh_fp32
    ) : hidden_size_(hidden_size) {
        // Quantize weights statically
        weight_ih_params_ = Quantizer::calibrate_min_max(weight_ih_fp32);
        weight_ih_q_ = Quantizer::quantize(weight_ih_fp32, weight_ih_params_);
        
        weight_hh_params_ = Quantizer::calibrate_min_max(weight_hh_fp32);
        weight_hh_q_ = Quantizer::quantize(weight_hh_fp32, weight_hh_params_);
    }
    
    std::pair<Tensor, Tensor> forward(
        const Tensor& input,     // [seq_len, batch, input_size]
        const Tensor& h_0,       // [batch, hidden_size]
        const Tensor& c_0        // [batch, hidden_size]
    ) {
        int seq_len = input.shape()[0];
        int batch = input.shape()[1];
        
        Tensor h = h_0;
        Tensor c = c_0;
        
        std::vector<Tensor> outputs;
        
        for (int t = 0; t < seq_len; ++t) {
            Tensor x_t = input[t];  // [batch, input_size]
            
            // Input-hidden: x_t @ W_ih^T (dynamic quantization)
            Tensor gates_ih = DynamicQuantizer::dynamic_quantize_and_matmul(
                x_t, weight_ih_q_, weight_ih_params_
            );
            
            // Hidden-hidden: h @ W_hh^T (dynamic quantization)
            Tensor gates_hh = DynamicQuantizer::dynamic_quantize_and_matmul(
                h, weight_hh_q_, weight_hh_params_
            );
            
            // Combine gates
            Tensor gates = gates_ih + gates_hh;  // [batch, 4*hidden_size]
            
            // Split into i, f, g, o gates
            auto [i_gate, f_gate, g_gate, o_gate] = split_gates(gates);
            
            // LSTM computations (in FP32 for stability)
            i_gate = ops::sigmoid(i_gate);
            f_gate = ops::sigmoid(f_gate);
            g_gate = ops::tanh(g_gate);
            o_gate = ops::sigmoid(o_gate);
            
            // Update cell and hidden state
            c = f_gate * c + i_gate * g_gate;
            h = o_gate * ops::tanh(c);
            
            outputs.push_back(h);
        }
        
        // Stack outputs
        Tensor output = ops::stack(outputs, /*dim=*/0);
        
        return {output, h};
    }
    
private:
    int hidden_size_;
    Tensor weight_ih_q_;  // INT8
    Tensor weight_hh_q_;  // INT8
    QuantParams weight_ih_params_;
    QuantParams weight_hh_params_;
    
    std::tuple<Tensor, Tensor, Tensor, Tensor> split_gates(const Tensor& gates) {
        // Split [batch, 4*hidden] into 4 x [batch, hidden]
        return {
            gates.slice(1, 0, hidden_size_),
            gates.slice(1, hidden_size_, 2*hidden_size_),
            gates.slice(1, 2*hidden_size_, 3*hidden_size_),
            gates.slice(1, 3*hidden_size_, 4*hidden_size_),
        };
    }
};
```

**Test dynamic quantization:**
```cpp
TEST(DynamicQuantTest, LSTMAccuracy) {
    int seq_len = 50, batch = 32, input_size = 128, hidden_size = 256;
    
    // Create FP32 LSTM
    auto lstm_fp32 = create_lstm(input_size, hidden_size);
    
    // Create dynamically quantized LSTM
    DynamicQuantizedLSTM lstm_q(
        input_size, hidden_size,
        lstm_fp32.weight_ih, lstm_fp32.weight_hh
    );
    
    // Random input
    Tensor input = Tensor::randn({seq_len, batch, input_size});
    Tensor h_0 = Tensor::zeros({batch, hidden_size});
    Tensor c_0 = Tensor::zeros({batch, hidden_size});
    
    // Forward pass
    auto [output_fp32, h_fp32] = lstm_fp32.forward(input, h_0, c_0);
    auto [output_q, h_q] = lstm_q.forward(input, h_0, c_0);
    
    // Compare outputs
    float error = compute_relative_error(output_fp32, output_q);
    std::cout << "Dynamic quant error: " << error * 100 << "%\n";
    
    EXPECT_LT(error, 0.01);  // <1% error
}

BENCHMARK(BM_DynamicQuant_LSTM) {
    // Compare FP32 vs Dynamic INT8
    // Expected: 2-3x speedup
}
```

---

## TURN 11 — Quantization Error Analysis and Debugging Tools

**Instructions:**

Build comprehensive tools to analyze and debug quantization errors.

**Background:** When quantization fails, need tools to diagnose: which layers have high error? Is calibration data representative? Are there outliers?

**Tools to implement:**
- Layer-wise error attribution
- Activation histogram visualization
- Calibration data coverage analysis
- Outlier detection

**Implement:**
```cpp
class QuantizationAnalyzer {
public:
    struct LayerAnalysisResult {
        std::string layer_name;
        float weight_error;      // MSE between FP32 and quantized weights
        float activation_error;  // MSE between FP32 and quantized activations
        float output_error;      // Error in final layer output
        QuantParams weight_params;
        QuantParams activation_params;
        
        // Statistics
        float weight_min, weight_max;
        float activation_min, activation_max;
        int num_outliers;
    };
    
    // Analyze each layer's quantization error
    static std::vector<LayerAnalysisResult> analyze_model(
        const Model& model_fp32,
        const QuantizedModel& model_q,
        DataLoader& test_data,
        int num_batches = 10
    ) {
        std::vector<LayerAnalysisResult> results;
        
        // Hook to capture intermediate activations
        std::unordered_map<std::string, Tensor> fp32_activations;
        std::unordered_map<std::string, Tensor> q_activations;
        
        model_fp32.register_forward_hook([&](const std::string& name, const Tensor& act) {
            fp32_activations[name] = act;
        });
        
        model_q.register_forward_hook([&](const std::string& name, const Tensor& act) {
            q_activations[name] = act;
        });
        
        // Run inference
        for (int i = 0; i < num_batches; ++i) {
            auto batch = test_data.next();
            model_fp32.forward(batch.input);
            model_q.forward(batch.input);
        }
        
        // Analyze each layer
        for (const auto& [name, layer] : model_fp32.named_layers()) {
            LayerAnalysisResult result;
            result.layer_name = name;
            
            // Weight error
            auto weight_fp32 = layer->weight();
            auto weight_q = model_q.get_layer(name)->weight();
            result.weight_error = compute_mse(weight_fp32, weight_q);
            
            // Activation error
            if (fp32_activations.count(name) && q_activations.count(name)) {
                result.activation_error = compute_mse(
                    fp32_activations[name],
                    q_activations[name]
                );
            }
            
            // Statistics
            compute_statistics(weight_fp32, result);
            
            results.push_back(result);
        }
        
        return results;
    }
    
    // Generate detailed report
    static void generate_report(
        const std::vector<LayerAnalysisResult>& results,
        const std::string& output_path
    ) {
        std::ofstream report(output_path);
        
        report << "Quantization Analysis Report\n";
        report << "==============================\n\n";
        
        // Summary statistics
        float total_weight_error = 0.0f;
        float max_weight_error = 0.0f;
        std::string worst_layer;
        
        for (const auto& r : results) {
            total_weight_error += r.weight_error;
            if (r.weight_error > max_weight_error) {
                max_weight_error = r.weight_error;
                worst_layer = r.layer_name;
            }
        }
        
        report << "Summary:\n";
        report << "  Average weight MSE: " << total_weight_error / results.size() << "\n";
        report << "  Worst layer: " << worst_layer << " (MSE: " << max_weight_error << ")\n\n";
        
        // Per-layer details
        report << "Per-Layer Analysis:\n";
        report << "-------------------\n";
        
        for (const auto& r : results) {
            report << "\nLayer: " << r.layer_name << "\n";
            report << "  Weight Error (MSE): " << r.weight_error << "\n";
            report << "  Activation Error (MSE): " << r.activation_error << "\n";
            report << "  Weight Range: [" << r.weight_min << ", " << r.weight_max << "]\n";
            report << "  Activation Range: [" << r.activation_min << ", " << r.activation_max << "]\n";
            report << "  Outliers: " << r.num_outliers << "\n";
            report << "  Scale: " << r.weight_params.scale << "\n";
            report << "  Zero Point: " << r.weight_params.zero_point << "\n";
        }
        
        report.close();
    }
    
    // Visualize activation histograms
    static void plot_activation_histogram(
        const Tensor& activations,
        const QuantParams& params,
        const std::string& output_path
    ) {
        // Create histogram
        constexpr int num_bins = 100;
        std::vector<int> histogram(num_bins, 0);
        
        auto* data = activations.data<float>();
        float min_val = compute_min(activations);
        float max_val = compute_max(activations);
        float bin_width = (max_val - min_val) / num_bins;
        
        for (int i = 0; i < activations.numel(); ++i) {
            int bin = std::min(
                static_cast<int>((data[i] - min_val) / bin_width),
                num_bins - 1
            );
            histogram[bin]++;
        }
        
        // Save histogram (CSV for plotting)
        std::ofstream file(output_path);
        file << "bin,count,quantized_value\n";
        
        for (int i = 0; i < num_bins; ++i) {
            float bin_center = min_val + (i + 0.5f) * bin_width;
            float quant_value = std::round(bin_center / params.scale) * params.scale;
            file << i << "," << histogram[i] << "," << quant_value << "\n";
        }
        
        file.close();
    }
    
private:
    static void compute_statistics(const Tensor& tensor, LayerAnalysisResult& result) {
        auto* data = tensor.data<float>();
        int n = tensor.numel();
        
        result.weight_min = *std::min_element(data, data + n);
        result.weight_max = *std::max_element(data, data + n);
        
        // Count outliers (values >3 sigma from mean)
        float mean = std::accumulate(data, data + n, 0.0f) / n;
        float variance = 0.0f;
        for (int i = 0; i < n; ++i) {
            variance += (data[i] - mean) * (data[i] - mean);
        }
        float stddev = std::sqrt(variance / n);
        
        result.num_outliers = 0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(data[i] - mean) > 3 * stddev) {
                result.num_outliers++;
            }
        }
    }
};
```

---

## TURN 12 — Mixed-Precision Inference (INT8 + FP16)

**Instructions:**

Combine INT8 and FP16 for optimal accuracy/performance trade-off.

**Background:** Some layers are sensitive to INT8 quantization (e.g., first/last layers, attention). Run sensitive layers in FP16, rest in INT8 for best of both worlds.

**Strategy:**
- Automatic sensitivity analysis
- Selective layer quantization
- Efficient data type conversions

**Implement:**
```cpp
class MixedPrecisionModel {
public:
    enum class Precision {
        FP32,
        FP16,
        INT8,
    };
    
    struct LayerPrecisionConfig {
        std::unordered_map<std::string, Precision> layer_precisions;
    };
    
    // Automatically determine optimal precision per layer
    static LayerPrecisionConfig auto_mixed_precision(
        const Model& model,
        DataLoader& calibration_data,
        float accuracy_threshold = 0.01
    ) {
        LayerPrecisionConfig config;
        
        // Start: all INT8
        for (const auto& [name, layer] : model.named_layers()) {
            config.layer_precisions[name] = Precision::INT8;
        }
        
        // Iteratively upgrade sensitive layers
        float baseline_acc = evaluate(model, calibration_data);
        auto model_q = quantize_with_config(model, config);
        float quant_acc = evaluate(model_q, calibration_data);
        
        // If accuracy ok, done
        if (baseline_acc - quant_acc < accuracy_threshold) {
            return config;
        }
        
        // Otherwise, find sensitive layers
        auto sensitivity_scores = compute_layer_sensitivity(model, calibration_data);
        
        // Sort by sensitivity
        std::vector<std::pair<std::string, float>> sorted_layers(
            sensitivity_scores.begin(), sensitivity_scores.end()
        );
        std::sort(sorted_layers.begin(), sorted_layers.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Upgrade most sensitive layers to FP16
        for (const auto& [name, sensitivity] : sorted_layers) {
            config.layer_precisions[name] = Precision::FP16;
            
            model_q = quantize_with_config(model, config);
            quant_acc = evaluate(model_q, calibration_data);
            
            if (baseline_acc - quant_acc < accuracy_threshold) {
                break;  // Accuracy goal met
            }
        }
        
        return config;
    }
    
    // Forward pass with mixed precision
    Tensor forward(const Tensor& input) {
        Tensor x = input;
        
        for (const auto& [name, layer] : layers_) {
            Precision prec = config_.layer_precisions[name];
            
            if (prec == Precision::FP16) {
                // Convert to FP16
                x = x.to(DType::Float16);
                x = layer->forward(x);
                // Convert back to FP32
                x = x.to(DType::Float32);
            }
            else if (prec == Precision::INT8) {
                // Quantized forward
                x = quantized_layers_[name]->forward(x);
            }
            else {
                // FP32
                x = layer->forward(x);
            }
        }
        
        return x;
    }
    
private:
    std::unordered_map<std::string, std::shared_ptr<Layer>> layers_;
    std::unordered_map<std::string, std::shared_ptr<QuantizedLayer>> quantized_layers_;
    LayerPrecisionConfig config_;
};
```

---

## TURN 13 — Model Serialization and ONNX Export

**Instructions:**

Serialize quantized models to ONNX format for deployment.

**Background:** ONNX is standard for model interchange. Need to export quantized model while preserving quantization parameters (QDQ nodes).

**Requirements:**
- Export to ONNX with QuantizeLinear/DequantizeLinear nodes
- Preserve per-layer quantization params
- Support INT8 and INT4

**Implement:**
```cpp
#include <onnx/onnx_pb.h>

class ONNXExporter {
public:
    static void export_quantized_model(
        const QuantizedModel& model,
        const std::string& output_path
    ) {
        onnx::ModelProto onnx_model;
        
        // Set model metadata
        onnx_model.set_ir_version(8);
        onnx_model.set_producer_name("CompressEngine");
        
        auto* graph = onnx_model.mutable_graph();
        graph->set_name("quantized_model");
        
        // Add input
        add_input(graph, "input", {1, 3, 224, 224});
        
        // Convert each layer to ONNX nodes
        std::string prev_output = "input";
        int node_id = 0;
        
        for (const auto& [name, layer] : model.layers()) {
            std::string curr_output = export_layer(
                graph, layer, prev_output, node_id++
            );
            prev_output = curr_output;
        }
        
        // Add output
        add_output(graph, prev_output, {1, 1000});
        
        // Save to file
        std::ofstream output(output_path, std::ios::binary);
        onnx_model.SerializeToOstream(&output);
    }
    
private:
    static std::string export_layer(
        onnx::GraphProto* graph,
        const QuantizedLayer* layer,
        const std::string& input_name,
        int node_id
    ) {
        if (auto* qlinear = dynamic_cast<const QuantizedLinear*>(layer)) {
            return export_quantized_linear(graph, qlinear, input_name, node_id);
        }
        // Handle other layer types...
        
        return "";
    }
    
    static std::string export_quantized_linear(
        onnx::GraphProto* graph,
        const QuantizedLinear* layer,
        const std::string& input_name,
        int node_id
    ) {
        std::string node_name = "qlinear_" + std::to_string(node_id);
        
        // Add QuantizeLinear node (input quantization)
        auto* quant_node = graph->add_node();
        quant_node->set_op_type("QuantizeLinear");
        quant_node->set_name(node_name + "_quantize_input");
        quant_node->add_input(input_name);
        
        // Add scale and zero_point as initializers
        std::string scale_name = node_name + "_input_scale";
        std::string zp_name = node_name + "_input_zp";
        add_initializer(graph, scale_name, {layer->input_scale()});
        add_initializer(graph, zp_name, {layer->input_zero_point()});
        
        quant_node->add_input(scale_name);
        quant_node->add_input(zp_name);
        
        std::string quantized_input = node_name + "_input_quantized";
        quant_node->add_output(quantized_input);
        
        // Add QLinearMatMul node
        auto* matmul_node = graph->add_node();
        matmul_node->set_op_type("QLinearMatMul");
        matmul_node->set_name(node_name + "_matmul");
        
        // Inputs: quantized input, input scale/zp, weight (quantized), weight scale/zp, output scale/zp
        matmul_node->add_input(quantized_input);
        matmul_node->add_input(scale_name);
        matmul_node->add_input(zp_name);
        
        // Add weight as initializer
        std::string weight_name = node_name + "_weight";
        add_initializer(graph, weight_name, layer->quantized_weight());
        matmul_node->add_input(weight_name);
        
        std::string weight_scale_name = node_name + "_weight_scale";
        std::string weight_zp_name = node_name + "_weight_zp";
        add_initializer(graph, weight_scale_name, {layer->weight_scale()});
        add_initializer(graph, weight_zp_name, {layer->weight_zero_point()});
        matmul_node->add_input(weight_scale_name);
        matmul_node->add_input(weight_zp_name);
        
        // Output params (for next layer)
        std::string out_scale_name = node_name + "_output_scale";
        std::string out_zp_name = node_name + "_output_zp";
        add_initializer(graph, out_scale_name, {layer->output_scale()});
        add_initializer(graph, out_zp_name, {layer->output_zero_point()});
        matmul_node->add_input(out_scale_name);
        matmul_node->add_input(out_zp_name);
        
        std::string matmul_output = node_name + "_output";
        matmul_node->add_output(matmul_output);
        
        return matmul_output;
    }
    
    static void add_initializer(
        onnx::GraphProto* graph,
        const std::string& name,
        const std::vector<float>& data
    ) {
        auto* init = graph->add_initializer();
        init->set_name(name);
        init->set_data_type(onnx::TensorProto::FLOAT);
        
        for (float val : data) {
            init->add_float_data(val);
        }
    }
};
```

---

## TURN 14 — Production Deployment with TensorRT

**Instructions:**

Integrate with NVIDIA TensorRT for optimized GPU deployment.

**Background:** TensorRT is NVIDIA's inference optimizer. Provides INT8 support, kernel fusion, dynamic shapes. Can import ONNX and apply automatic optimizations.

**Requirements:**
- Convert model to TensorRT engine
- Enable INT8 precision
- Calibration for INT8
- Benchmark against ONNX Runtime

**Implement:**
```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTDeployer {
public:
    TensorRTDeployer() {
        logger_ = std::make_unique<Logger>();
        builder_ = nvinfer1::createInferBuilder(*logger_);
    }
    
    // Build TensorRT engine from ONNX
    void build_engine(
        const std::string& onnx_path,
        const std::string& engine_path,
        bool use_int8 = true,
        DataLoader* calibration_data = nullptr
    ) {
        // Create network
        auto network = builder_->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
            )
        );
        
        // Parse ONNX
        auto parser = nvonnxparser::createParser(*network, *logger_);
        parser->parseFromFile(
            onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
        );
        
        // Build config
        auto config = builder_->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);  // 1GB
        
        if (use_int8) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            
            // Set INT8 calibrator
            if (calibration_data) {
                auto calibrator = std::make_unique<Int8Calibrator>(
                    calibration_data, "calibration_cache.bin"
                );
                config->setInt8Calibrator(calibrator.get());
            }
        }
        
        // Build engine
        engine_ = builder_->buildEngineWithConfig(*network, *config);
        
        // Serialize and save
        auto serialized = engine_->serialize();
        std::ofstream engine_file(engine_path, std::ios::binary);
        engine_file.write(
            static_cast<const char*>(serialized->data()),
            serialized->size()
        );
    }
    
    // Run inference
    std::vector<float> infer(const std::vector<float>& input) {
        if (!context_) {
            context_ = engine_->createExecutionContext();
        }
        
        // Allocate GPU buffers
        void* buffers[2];  // input, output
        
        int input_index = engine_->getBindingIndex("input");
        int output_index = engine_->getBindingIndex("output");
        
        auto input_dims = engine_->getBindingDimensions(input_index);
        auto output_dims = engine_->getBindingDimensions(output_index);
        
        size_t input_size = volume(input_dims) * sizeof(float);
        size_t output_size = volume(output_dims) * sizeof(float);
        
        cudaMalloc(&buffers[input_index], input_size);
        cudaMalloc(&buffers[output_index], output_size);
        
        // Copy input to GPU
        cudaMemcpy(
            buffers[input_index], input.data(), input_size,
            cudaMemcpyHostToDevice
        );
        
        // Run inference
        context_->executeV2(buffers);
        
        // Copy output to CPU
        std::vector<float> output(volume(output_dims));
        cudaMemcpy(
            output.data(), buffers[output_index], output_size,
            cudaMemcpyDeviceToHost
        );
        
        // Cleanup
        cudaFree(buffers[input_index]);
        cudaFree(buffers[output_index]);
        
        return output;
    }
    
private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    };
    
    class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
    public:
        Int8Calibrator(DataLoader* data, const std::string& cache_file)
            : data_(data), cache_file_(cache_file) {}
        
        int getBatchSize() const noexcept override { return 32; }
        
        bool getBatch(
            void* bindings[], const char* names[], int nbBindings
        ) noexcept override {
            // Get next batch from data loader
            if (!data_->has_next()) return false;
            
            auto batch = data_->next();
            // Copy to GPU...
            return true;
        }
        
        const void* readCalibrationCache(size_t& length) noexcept override {
            // Read from cache file if exists
            return nullptr;
        }
        
        void writeCalibrationCache(const void* cache, size_t length) noexcept override {
            // Write to cache file
            std::ofstream file(cache_file_, std::ios::binary);
            file.write(static_cast<const char*>(cache), length);
        }
        
    private:
        DataLoader* data_;
        std::string cache_file_;
    };
    
    std::unique_ptr<Logger> logger_;
    nvinfer1::IBuilder* builder_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    size_t volume(const nvinfer1::Dims& dims) {
        size_t vol = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            vol *= dims.d[i];
        }
        return vol;
    }
};
```

---

## TURN 15 — End-to-End Benchmark (ResNet50 + BERT-base)

**Instructions:**

Comprehensive benchmark of all compression techniques on real models.

**Models:**
- ResNet50 (image classification)
- BERT-base (NLP)

**Metrics:**
- Accuracy
- Latency (ms)
- Throughput (samples/sec)
- Model size (MB)
- Memory usage (GB)

**Implement:**
```cpp
struct BenchmarkResult {
    std::string model_name;
    std::string precision;
    float accuracy;
    float latency_ms;
    float throughput;
    size_t model_size_mb;
    size_t memory_mb;
};

class EndToEndBenchmark {
public:
    static void run_all_benchmarks() {
        std::vector<BenchmarkResult> results;
        
        // ResNet50 benchmarks
        results.push_back(benchmark_resnet50_fp32());
        results.push_back(benchmark_resnet50_int8());
        results.push_back(benchmark_resnet50_int8_pruned());
        results.push_back(benchmark_resnet50_int4());
        results.push_back(benchmark_resnet50_mixed());
        
        // BERT benchmarks
        results.push_back(benchmark_bert_fp32());
        results.push_back(benchmark_bert_dynamic_int8());
        results.push_back(benchmark_bert_int8_distilled());
        
        // Print summary table
        print_results_table(results);
        
        // Generate charts
        generate_charts(results);
    }
    
private:
    static BenchmarkResult benchmark_resnet50_fp32() {
        auto model = load_resnet50();
        auto test_data = load_imagenet_val();
        
        // Accuracy
        float acc = evaluate_accuracy(model, test_data);
        
        // Latency (average over 1000 runs)
        auto latency = benchmark_latency(model, 1000);
        
        // Throughput
        float throughput = 1000.0f / latency;
        
        // Model size
        size_t size = compute_model_size(model);
        
        return {
            "ResNet50",
            "FP32",
            acc,
            latency,
            throughput,
            size / (1024 * 1024),  // MB
            measure_memory_usage(model)
        };
    }
    
    static BenchmarkResult benchmark_resnet50_int8() {
        auto model_fp32 = load_resnet50();
        
        // Quantize
        ModelQuantizer quantizer;
        quantizer.calibrate(model_fp32, calibration_data, 100);
        auto model_q = quantizer.quantize(model_fp32);
        
        // Evaluate
        auto test_data = load_imagenet_val();
        float acc = evaluate_accuracy(model_q, test_data);
        float latency = benchmark_latency(model_q, 1000);
        
        return {
            "ResNet50",
            "INT8 (PTQ)",
            acc,
            latency,
            1000.0f / latency,
            compute_model_size(model_q) / (1024 * 1024),
            measure_memory_usage(model_q)
        };
    }
    
    static BenchmarkResult benchmark_resnet50_int8_pruned() {
        auto model_fp32 = load_resnet50();
        
        // Prune
        StructuredPruner pruner(StructuredPruner::L1Norm, 0.5);
        pruner.iterative_prune(model_fp32, train_data, 10, 5);
        
        // Then quantize
        ModelQuantizer quantizer;
        quantizer.calibrate(model_fp32, calibration_data, 100);
        auto model_q = quantizer.quantize(model_fp32);
        
        // Evaluate
        auto test_data = load_imagenet_val();
        float acc = evaluate_accuracy(model_q, test_data);
        float latency = benchmark_latency(model_q, 1000);
        
        return {
            "ResNet50",
            "INT8 + 50% Pruned",
            acc,
            latency,
            1000.0f / latency,
            compute_model_size(model_q) / (1024 * 1024),
            measure_memory_usage(model_q)
        };
    }
    
    static void print_results_table(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=================================\n";
        std::cout << "End-to-End Benchmark Results\n";
        std::cout << "=================================\n\n";
        
        // Table header
        printf("%-20s %-20s %10s %12s %15s %10s %10s\n",
               "Model", "Precision", "Accuracy", "Latency(ms)",
               "Throughput", "Size(MB)", "Mem(MB)");
        printf("%s\n", std::string(110, '-').c_str());
        
        // Table rows
        for (const auto& r : results) {
            printf("%-20s %-20s %9.2f%% %11.2f %14.1f %9zu %9zu\n",
                   r.model_name.c_str(),
                   r.precision.c_str(),
                   r.accuracy * 100,
                   r.latency_ms,
                   r.throughput,
                   r.model_size_mb,
                   r.memory_mb);
        }
        
        std::cout << "\n";
        
        // Speedup analysis
        auto fp32_latency = results[0].latency_ms;
        for (size_t i = 1; i < results.size(); ++i) {
            if (results[i].model_name == results[0].model_name) {
                float speedup = fp32_latency / results[i].latency_ms;
                float compression = static_cast<float>(results[0].model_size_mb) /
                                  results[i].model_size_mb;
                printf("%s: %.2fx speedup, %.2fx compression\n",
                       results[i].precision.c_str(), speedup, compression);
            }
        }
    }
};
```

**Expected results:**
```
=================================
End-to-End Benchmark Results
=================================

Model                Precision            Accuracy  Latency(ms)     Throughput   Size(MB)   Mem(MB)
--------------------------------------------------------------------------------------------------------------
ResNet50             FP32                    76.13%        4.20            238.1        98        412
ResNet50             INT8 (PTQ)              75.87%        1.10            909.1        25        184
ResNet50             INT8 + 50% Pruned       74.92%        0.85           1176.5        13        142
ResNet50             INT4                    74.15%        0.78           1282.1         7         98
ResNet50             Mixed (INT8+FP16)       75.98%        1.25            800.0        31        198
BERT-base            FP32                    84.50%       12.30             81.3       438       3200
BERT-base            Dynamic INT8            84.21%        4.10            243.9       110       1100
BERT-base            INT8 Distilled          83.85%        2.95            339.0        85        890

INT8 (PTQ): 3.82x speedup, 3.92x compression
INT8 + 50% Pruned: 4.94x speedup, 7.54x compression
INT4: 5.38x speedup, 14.00x compression
Mixed (INT8+FP16): 3.36x speedup, 3.16x compression
```

---

**Final Deliverables:**
- ✅ Complete tensor library with CPU (AVX2/AVX-512) and GPU (CUDA) backends
- ✅ PTQ and QAT for INT8/INT4 quantization
- ✅ Structured and unstructured pruning
- ✅ Knowledge distillation framework
- ✅ Automatic differentiation engine
- ✅ Dynamic quantization for RNNs/LSTMs
- ✅ Quantization error analysis tools
- ✅ Mixed-precision inference (INT8 + FP16)
- ✅ ONNX export support
- ✅ TensorRT integration for production
- ✅ Comprehensive end-to-end benchmarks
- ✅ >300 comprehensive tests
- ✅ Performance matching or beating ONNX Runtime
- ✅ Production-ready code (no leaks, thread-safe, exception-safe)
```
