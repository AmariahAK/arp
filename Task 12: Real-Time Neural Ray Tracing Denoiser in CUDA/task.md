# Task: Real-Time Neural Ray Tracing Denoiser in CUDA/C++

## Overview

Build a production-grade neural network denoiser for real-time path-traced rendering. The system must denoise low sample-per-pixel (SPP) images in <16ms at 1080p while maintaining temporal stability and preserving fine details.

**Difficulty:** EXTREME  
**Estimated time:** 60-75 hours  
**Turns:** 18

---

## TURN 1 — G-Buffer Feature Extraction and Preprocessing

**Instructions:**

Implement efficient G-buffer preprocessing to extract features from auxiliary render targets (albedo, normals, depth, motion vectors) that will guide the neural denoiser.

**Background:** Path tracers output not just the noisy color image, but also geometric information (G-buffers). These guide the denoiser to preserve edges and details. Efficient preprocessing is critical as it runs every frame.

**Requirements:**
- Load and parse OpenEXR multi-channel images
- Extract and normalize features from G-buffers
- Implement edge-aware filtering for feature maps
- CUDA kernels for all preprocessing
- Target: <1ms preprocessing time at 1080p

**Implement:**

```cpp
// G-buffer structure
struct GBuffer {
    Tensor albedo;          // [H, W, 3] - surface color
    Tensor normal;          // [H, W, 3] - world-space normals
    Tensor depth;           // [H, W, 1] - linear depth
    Tensor motion_vectors;  // [H, W, 2] - screen-space motion
    Tensor noisy_color;     // [H, W, 3] - low SPP render (1-16 SPP)
};

class GBufferPreprocessor {
public:
    struct Features {
        Tensor color_variance;      // [H, W, 1]
        Tensor normal_gradients;    // [H, W, 2]
        Tensor depth_gradients;     // [H, W, 2]
        Tensor luminance;           // [H, W, 1]
    };
    
    // Extract denoising features from G-buffers
    Features extract_features(const GBuffer& gbuffer);
    
    // Normalize features to [0, 1] or [-1, 1] range
    void normalize_features(Features& features);
    
    // Compute local variance for adaptive filtering
    Tensor compute_local_variance(
        const Tensor& noisy_color,
        int kernel_size = 7
    );
};
```

**CUDA kernel for variance estimation:**
```cpp
__global__ void compute_variance_kernel(
    const float3* __restrict__ noisy_color,
    float* __restrict__ variance,
    int width, int height,
    int kernel_radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Compute local mean
    float3 mean = make_float3(0, 0, 0);
    int count = 0;
    
    for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
        for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
            int nx = clamp(x + dx, 0, width - 1);
            int ny = clamp(y + dy, 0, height - 1);
            
            mean = mean + noisy_color[ny * width + nx];
            count++;
        }
    }
    
    mean = mean / float(count);
    
    // Compute variance
    float var = 0.0f;
    for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
        for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
            int nx = clamp(x + dx, 0, width - 1);
            int ny = clamp(y + dy, 0, height - 1);
            
            float3 diff = noisy_color[ny * width + nx] - mean;
            var += dot(diff, diff);
        }
    }
    
    variance[y * width + x] = var / float(count);
}
```

**Tests:**
```cpp
TEST(GBufferTest, FeatureExtraction) {
    // Load test scene G-buffers
    auto gbuffer = load_test_gbuffer("cornell_box_4spp.exr");
    
    GBufferPreprocessor preprocessor;
    auto features = preprocessor.extract_features(gbuffer);
    
    // Validate feature dimensions
    EXPECT_EQ(features.color_variance.shape(), Shape{1080, 1920, 1});
    EXPECT_EQ(features.normal_gradients.shape(), Shape{1080, 1920, 2});
    
    // Validate feature ranges
    auto [min_var, max_var] = compute_min_max(features.color_variance);
    EXPECT_GE(min_var, 0.0f);  // Variance always non-negative
    
    // Benchmark preprocessing time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        features = preprocessor.extract_features(gbuffer);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    float avg_time_ms = std::chrono::duration<float, std::milli>(end - start).count() / 100.0f;
    std::cout << "Preprocessing time: " << avg_time_ms << "ms\n";
    
    EXPECT_LT(avg_time_ms, 1.0f);  // Target: <1ms
}
```

---

## TURN 2 — U-Net Architecture for Spatial Denoising

**Instructions:**

Implement a custom U-Net architecture optimized for real-time inference with Tensor Cores.

**Background:** U-Net is the de facto standard for image-to-image tasks. We need a variant that's fast enough for real-time while maintaining quality. Key: shallow network, wide channels, efficient skip connections.

**Requirements:**
- Encoder-decoder architecture with skip connections
- 4-5 downsampling levels maximum (for speed)
- Channel counts: 32 → 64 → 128 → 256 → 128 → 64 → 32
- All convolutions 3×3 (optimal for Tensor Cores)
- <30M parameters total
- FP16 inference support

**Implement:**

```cpp
class UNetDenoiser {
public:
    struct Config {
        int base_channels = 32;
        int num_levels = 4;
        bool use_attention = false;  // Attention is slow, optional
        ActivationType activation = ActivationType::ReLU;
    };
    
    UNetDenoiser(const Config& config);
    
    // Forward pass: noisy_image + features → denoised_image
    Tensor forward(
        const Tensor& noisy_image,     // [B, H, W, 3]
        const Tensor& albedo,           // [B, H, W, 3]
        const Tensor& normal,           // [B, H, W, 3]
        const Tensor& variance          // [B, H, W, 1]
    );
    
private:
    // Encoder block: Conv → ReLU → Conv → ReLU → Downsample
    struct EncoderBlock {
        Conv2d conv1, conv2;
        MaxPool2d pool;
        
        Tensor forward(const Tensor& x);
    };
    
    // Decoder block: Upsample → Concat(skip) → Conv → ReLU → Conv
    struct DecoderBlock {
        ConvTranspose2d upsample;
        Conv2d conv1, conv2;
        
        Tensor forward(const Tensor& x, const Tensor& skip);
    };
    
    std::vector<EncoderBlock> encoder;
    std::vector<DecoderBlock> decoder;
    Conv2d final_conv;  // Map to RGB output
};
```

**Efficient 3×3 convolution with Tensor Cores:**
```cpp
// CUDA kernel using Tensor Cores (WMMA API)
__global__ void conv2d_3x3_tensorcore(
    const half* __restrict__ input,    // [H, W, C_in]
    const half* __restrict__ weight,   // [3, 3, C_in, C_out]
    half* __restrict__ output,         // [H, W, C_out]
    int H, int W, int C_in, int C_out
) {
    using namespace nvcuda::wmma;
    
    // Tile size for Tensor Cores: 16x16x16
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 16;
    
    // Declare fragments
    fragment<matrix_a, TILE_M, TILE_N, TILE_K, half, row_major> a_frag;
    fragment<matrix_b, TILE_M, TILE_N, TILE_K, half, col_major> b_frag;
    fragment<accumulator, TILE_M, TILE_N, TILE_K, half> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, __float2half(0.0f));
    
    // Compute convolution using im2col + GEMM approach
    // (Details omitted for brevity - standard im2col transformation)
    
    // Tensor Core matrix multiply-accumulate
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result
    store_matrix_sync(output, c_frag, W, mem_row_major);
}
```

**Tests:**
```cpp
TEST(UNetTest, ForwardPass) {
    UNetDenoiser::Config config;
    config.base_channels = 32;
    config.num_levels = 4;
    
    UNetDenoiser model(config);
    
    // Create dummy input
    Tensor noisy = Tensor::randn({1, 1080, 1920, 3});
    Tensor albedo = Tensor::randn({1, 1080, 1920, 3});
    Tensor normal = Tensor::randn({1, 1080, 1920, 3});
    Tensor variance = Tensor::randn({1, 1080, 1920, 1});
    
    // Forward pass
    Tensor output = model.forward(noisy, albedo, normal, variance);
    
    // Validate output shape
    EXPECT_EQ(output.shape(), Shape{1, 1080, 1920, 3});
    
    // Validate output range (should be similar to input)
    auto [min_val, max_val] = compute_min_max(output);
    EXPECT_GT(min_val, -1.0f);
    EXPECT_LT(max_val, 2.0f);
}

BENCHMARK(BM_UNet_Inference_1080p) {
    UNetDenoiser model(UNetDenoiser::Config{});
    Tensor input = Tensor::randn({1, 1080, 1920, 10});  // Concatenated features
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        model.forward(input);
    }
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        model.forward(input);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    std::cout << "Avg inference time: " << ms / 100.0f << "ms\n";
    
    // Target: <15ms for spatial denoising only
    EXPECT_LT(ms / 100.0f, 15.0f);
}
```

---

## TURN 3 — Training Dataset Generation from Path Tracer

**Instructions:**

Generate a large-scale training dataset by rendering scenes at both low SPP (noisy) and high SPP (ground truth).

**Background:** Need 50k+ image pairs for training. Must cover diverse scenes, materials, lighting. Use OptiX or Embree for fast rendering.

**Requirements:**
- Render 100+ diverse scenes
- Generate pairs: (1-16 SPP noisy, 1000+ SPP reference)
- Include all G-buffers (albedo, normal, depth, motion)
- Data augmentation: crops, flips, rotations
- Store in efficient format (HDF5 or custom binary)

**Implement:**

```python
# Training data generator (Python + OptiX)
import pyoptix as optix
import numpy as np
import h5py

class DatasetGenerator:
    def __init__(self, scene_dir: str, output_dir: str):
        self.scene_dir = scene_dir
        self.output_dir = output_dir
        self.optix_context = optix.DeviceContext()
        
    def render_scene(
        self,
        scene_path: str,
        spp: int,
        resolution: tuple[int, int] = (1920, 1080)
    ) -> dict:
        """Render scene with OptiX path tracer."""
        # Load scene
        scene = optix.load_scene(scene_path)
        
        # Configure renderer
        renderer = optix.PathTracer(
            self.optix_context,
            max_depth=8,
            samples_per_pixel=spp
        )
        
        # Render
        result = renderer.render(scene, resolution)
        
        return {
            'color': result.color,           # [H, W, 3] HDR
            'albedo': result.albedo,         # [H, W, 3]
            'normal': result.normal,         # [H, W, 3]
            'depth': result.depth,           # [H, W, 1]
            'motion': result.motion_vectors  # [H, W, 2]
        }
    
    def generate_training_pair(self, scene_path: str) -> tuple:
        """Generate (noisy, reference) pair."""
        # Random low SPP (1, 2, 4, 8, or 16)
        low_spp = np.random.choice([1, 2, 4, 8, 16])
        high_spp = 2048  # Ground truth
        
        # Render both
        noisy = self.render_scene(scene_path, low_spp)
        reference = self.render_scene(scene_path, high_spp)
        
        return noisy, reference
    
    def generate_dataset(self, num_samples: int = 50000):
        """Generate full training dataset."""
        # Load all scenes
        scenes = glob.glob(f"{self.scene_dir}/**/*.obj", recursive=True)
        
        with h5py.File(f"{self.output_dir}/training_data.h5", 'w') as f:
            # Create datasets
            f.create_dataset('noisy_color', shape=(num_samples, 1080, 1920, 3), dtype='float32')
            f.create_dataset('reference_color', shape=(num_samples, 1080, 1920, 3), dtype='float32')
            f.create_dataset('albedo', shape=(num_samples, 1080, 1920, 3), dtype='float32')
            f.create_dataset('normal', shape=(num_samples, 1080, 1920, 3), dtype='float32')
            # ... other G-buffers
            
            for i in tqdm(range(num_samples)):
                # Random scene
                scene = np.random.choice(scenes)
                
                # Generate pair
                noisy, reference = self.generate_training_pair(scene)
                
                # Store
                f['noisy_color'][i] = noisy['color']
                f['reference_color'][i] = reference['color']
                f['albedo'][i] = noisy['albedo']
                f['normal'][i] = noisy['normal']
                # ...
                
                if i % 1000 == 0:
                    print(f"Generated {i}/{num_samples} samples")
```

---

## TURN 4 — Loss Functions for Denoising

**Instructions:**

Implement specialized loss functions that balance perceptual quality, detail preservation, and numerical accuracy.

**Background:** Simple L2 loss produces blurry results. Need perceptual losses (LPIPS, SSIM) and edge-aware losses.

**Requirements:**
- L1 + L2 reconstruction loss
- SSIM (structural similarity) loss
- Perceptual loss (VGG features)
- Edge-aware loss (preserve high-frequency details)
- Temporal loss (for video sequences)

**Implement:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingLoss(nn.Module):
    def __init__(self, weights: dict = None):
        super().__init__()
        
        # Default weights
        self.weights = weights or {
            'l1': 1.0,
            'l2': 0.5,
            'ssim': 0.3,
            'perceptual': 0.2,
            'edge': 0.1
        }
        
        # Perceptual loss network (VGG16)
        self.vgg = VGG16FeatureExtractor()
        
    def forward(
        self,
        pred: torch.Tensor,      # [B, 3, H, W]
        target: torch.Tensor,    # [B, 3, H, W]
        albedo: torch.Tensor = None
    ) -> dict:
        """Compute combined loss."""
        losses = {}
        
        # L1 loss
        losses['l1'] = F.l1_loss(pred, target)
        
        # L2 loss
        losses['l2'] = F.mse_loss(pred, target)
        
        # SSIM loss
        losses['ssim'] = 1.0 - ssim(pred, target)
        
        # Perceptual loss (VGG features)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        losses['perceptual'] = sum([
            F.mse_loss(pf, tf)
            for pf, tf in zip(pred_features, target_features)
        ])
        
        # Edge-aware loss (preserve high-frequency details)
        pred_edges = compute_sobel_edges(pred)
        target_edges = compute_sobel_edges(target)
        losses['edge'] = F.l1_loss(pred_edges, target_edges)
        
        # Weighted sum
        total_loss = sum(self.weights[k] * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return losses

def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2
) -> torch.Tensor:
    """Compute SSIM (Structural Similarity Index)."""
    # Create Gaussian window
    window = create_gaussian_window(window_size, img1.shape[1])
    window = window.to(img1.device)
    
    # Compute local means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()
```

---

## TURN 5 — Training Loop with Mixed Precision

**Instructions:**

Implement the training loop in PyTorch with automatic mixed precision (AMP) for faster training.

**Background:** Training on 50k images takes days. AMP uses FP16 for speed while maintaining FP32 master weights for stability.

**Requirements:**
- PyTorch training loop with gradient accumulation
- AMP with dynamic loss scaling
- Learning rate scheduling (cosine decay with warmup)
- Validation every N steps
- Checkpoint saving and resuming
- Target: <24 hours training on 4× RTX 4090

**Implement:**

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict
    ):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['epochs'] // 3,
            T_mult=2
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Loss function
        self.criterion = DenoisingLoss()
        
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to GPU
            noisy = batch['noisy'].cuda()
            reference = batch['reference'].cuda()
            albedo = batch['albedo'].cuda()
            normal = batch['normal'].cuda()
            variance = batch['variance'].cuda()
            
            # Forward pass with AMP
            with autocast():
                pred = self.model(noisy, albedo, normal, variance)
                losses = self.criterion(pred, reference, albedo)
                loss = losses['total']
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                noisy = batch['noisy'].cuda()
                reference = batch['reference'].cuda()
                albedo = batch['albedo'].cuda()
                normal = batch['normal'].cuda()
                variance = batch['variance'].cuda()
                
                with autocast():
                    pred = self.model(noisy, albedo, normal, variance)
                
                # Compute metrics
                psnr = compute_psnr(pred, reference)
                ssim_val = compute_ssim(pred, reference)
                
                total_psnr += psnr
                total_ssim += ssim_val
        
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        print(f"Validation - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
        return avg_psnr, avg_ssim
    
    def train(self, num_epochs: int):
        best_psnr = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                psnr, ssim = self.validate()
                
                # Save best model
                if psnr > best_psnr:
                    best_psnr = psnr
                    self.save_checkpoint(f"best_model_psnr_{psnr:.2f}.pth")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
```

---

## TURN 6 — Temporal Reprojection for Video Sequences

**Instructions:**

Implement temporal reprojection to accumulate information across frames for stable video denoising.

**Background:** Single-frame denoising flickers in video. Temporal reprojection uses motion vectors to reuse previous frames' information, dramatically improving stability.

**Requirements:**
- Motion vector-based reprojection
- Disocclusion detection (newly visible pixels)
- Adaptive blending based on confidence
- Handle camera and object motion
- Target: <0.5% temporal flicker

**Implement:**

```cpp
class TemporalAccumulator {
public:
    struct Config {
        float alpha = 0.2f;           // Blend factor for new frame
        float disocclusion_threshold = 0.1f;
        bool use_variance_clipping = true;
    };
    
    TemporalAccumulator(int width, int height, const Config& config);
    
    // Accumulate current frame with history
    Tensor accumulate(
        const Tensor& current_denoised,    // [H, W, 3] - current frame output
        const Tensor& motion_vectors,       // [H, W, 2] - screen-space motion
        const Tensor& depth_current,        // [H, W, 1] - current depth
        const Tensor& depth_previous        // [H, W, 1] - previous depth
    );
    
private:
    // Reproject previous frame using motion vectors
    Tensor reproject_frame(
        const Tensor& prev_frame,
        const Tensor& motion_vectors
    );
    
    // Detect disocclusions (newly visible regions)
    Tensor detect_disocclusions(
        const Tensor& depth_current,
        const Tensor& depth_reprojected,
        const Tensor& motion_vectors
    );
    
    // Adaptive blending based on confidence
    Tensor adaptive_blend(
        const Tensor& current,
        const Tensor& reprojected,
        const Tensor& disocclusion_mask,
        const Tensor& variance
    );
    
    Tensor history_frame_;      // Previous accumulated frame
    Tensor history_variance_;   // Variance estimate
    Config config_;
};
```

**CUDA kernel for bilinear reprojection:**

```cpp
__global__ void reproject_kernel(
    const float3* __restrict__ prev_frame,
    const float2* __restrict__ motion_vectors,
    float3* __restrict__ reprojected,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Read motion vector
    float2 mv = motion_vectors[y * width + x];
    
    // Compute source position
    float src_x = float(x) + mv.x;
    float src_y = float(y) + mv.y;
    
    // Check bounds
    if (src_x < 0 || src_x >= width - 1 || src_y < 0 || src_y >= height - 1) {
        reprojected[y * width + x] = make_float3(0, 0, 0);  // Disoccluded
        return;
    }
    
    // Bilinear interpolation
    int x0 = int(src_x);
    int y0 = int(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    float fx = src_x - x0;
    float fy = src_y - y0;
    
    float3 c00 = prev_frame[y0 * width + x0];
    float3 c10 = prev_frame[y0 * width + x1];
    float3 c01 = prev_frame[y1 * width + x0];
    float3 c11 = prev_frame[y1 * width + x1];
    
    // Bilinear blend
    float3 c0 = lerp(c00, c10, fx);
    float3 c1 = lerp(c01, c11, fx);
    float3 result = lerp(c0, c1, fy);
    
    reprojected[y * width + x] = result;
}
```

**Tests:**

```cpp
TEST(TemporalTest, ReprojectionAccuracy) {
    // Create synthetic motion (horizontal pan)
    Tensor frame1 = load_image("frame_001.exr");
    Tensor frame2 = load_image("frame_002.exr");  // Shifted by 10 pixels
    
    // Known motion vector
    Tensor motion = Tensor::constant({1080, 1920, 2}, {10.0f, 0.0f});
    
    TemporalAccumulator accumulator(1920, 1080, {});
    
    // Reproject frame1 using motion
    Tensor reprojected = accumulator.reproject_frame(frame1, motion);
    
    // Should match frame2 (minus edge pixels)
    Tensor diff = (reprojected - frame2).abs();
    float avg_error = diff.mean();
    
    EXPECT_LT(avg_error, 0.01f);  // Very low error for perfect motion
}
```

---

## TURN 7 — Force Failure: Temporal Ghosting Artifacts

**Ask the AI:**

> "Your temporal accumulator works well for static scenes, but produces severe ghosting artifacts when objects move quickly or when there are disocclusions. Show a test that demonstrates this failure and implement a robust solution using variance-based confidence weighting."

**Expected failure:**

```cpp
TEST(TemporalTest, FastMotionGhosting) {
    // Scene with fast-moving object
    auto sequence = load_video_sequence("fast_car.exr", num_frames=60);
    
    TemporalAccumulator accumulator(1920, 1080, {});
    
    std::vector<Tensor> denoised_frames;
    for (const auto& frame : sequence) {
        Tensor denoised = denoise_spatial(frame.noisy);
        Tensor accumulated = accumulator.accumulate(
            denoised, frame.motion, frame.depth_current, frame.depth_prev
        );
        denoised_frames.push_back(accumulated);
    }
    
    // Measure ghosting (temporal variance in moving regions)
    float ghosting_score = measure_ghosting(denoised_frames, sequence.motion_masks);
    
    std::cout << "Ghosting score: " << ghosting_score << std::endl;
    
    // FAILURE: High ghosting (>5% variance in moving regions)
    EXPECT_LT(ghosting_score, 0.01);  // FAILS with score ~0.08
}
```

**Root cause:** Naive blending doesn't account for motion confidence. Fast-moving objects leave trails.

**Fix: Variance-based confidence weighting:**

```cpp
class ImprovedTemporalAccumulator {
public:
    Tensor accumulate(
        const Tensor& current_denoised,
        const Tensor& motion_vectors,
        const Tensor& depth_current,
        const Tensor& depth_previous,
        const Tensor& variance_current  // NEW: variance estimate
    ) {
        // Reproject previous frame
        Tensor reprojected = reproject_frame(history_frame_, motion_vectors);
        Tensor reprojected_variance = reproject_frame(history_variance_, motion_vectors);
        
        // Detect disocclusions
        Tensor disocclusion_mask = detect_disocclusions(
            depth_current, depth_previous, motion_vectors
        );
        
        // Compute confidence based on:
        // 1. Disocclusion (low confidence)
        // 2. High variance (low confidence - noisy region)
        // 3. Large motion (low confidence - potential error)
        Tensor confidence = compute_confidence(
            disocclusion_mask,
            variance_current,
            reprojected_variance,
            motion_vectors
        );
        
        // Adaptive blending: high confidence → more history, low confidence → more current
        Tensor alpha = 0.2f * (1.0f - confidence) + 0.8f * confidence;
        
        Tensor accumulated = alpha * reprojected + (1.0f - alpha) * current_denoised;
        
        // Variance clipping (reject outliers)
        if (config_.use_variance_clipping) {
            Tensor variance_diff = (reprojected_variance - variance_current).abs();
            Tensor outlier_mask = variance_diff > 3.0f * variance_current;
            accumulated = torch::where(outlier_mask, current_denoised, accumulated);
        }
        
        // Update history
        history_frame_ = accumulated;
        history_variance_ = variance_current;
        
        return accumulated;
    }
    
private:
    Tensor compute_confidence(
        const Tensor& disocclusion_mask,
        const Tensor& variance_current,
        const Tensor& variance_reprojected,
        const Tensor& motion_vectors
    ) {
        // Start with full confidence
        Tensor confidence = Tensor::ones_like(disocclusion_mask);
        
        // Reduce confidence for disocclusions
        confidence = confidence * (1.0f - disocclusion_mask);
        
        // Reduce confidence for high variance (noisy regions)
        Tensor variance_factor = torch::exp(-variance_current / 0.1f);
        confidence = confidence * variance_factor;
        
        // Reduce confidence for large motion (potential reprojection error)
        Tensor motion_magnitude = torch::sqrt(
            motion_vectors[..., 0] * motion_vectors[..., 0] +
            motion_vectors[..., 1] * motion_vectors[..., 1]
        );
        Tensor motion_factor = torch::exp(-motion_magnitude / 10.0f);
        confidence = confidence * motion_factor;
        
        return confidence;
    }
};
```

**After fix:**

```cpp
TEST(TemporalTest, FastMotionGhostingFixed) {
    auto sequence = load_video_sequence("fast_car.exr", num_frames=60);
    
    ImprovedTemporalAccumulator accumulator(1920, 1080, {});
    
    std::vector<Tensor> denoised_frames;
    for (const auto& frame : sequence) {
        Tensor denoised = denoise_spatial(frame.noisy);
        Tensor accumulated = accumulator.accumulate(
            denoised, frame.motion, frame.depth_current, frame.depth_prev, frame.variance
        );
        denoised_frames.push_back(accumulated);
    }
    
    float ghosting_score = measure_ghosting(denoised_frames, sequence.motion_masks);
    
    std::cout << "Ghosting score (fixed): " << ghosting_score << std::endl;
    
    EXPECT_LT(ghosting_score, 0.01);  // ✅ PASSES with score ~0.004
}
```

---

## TURN 8 — Model Export from PyTorch to C++/CUDA

**Instructions:**

Export the trained PyTorch model to a format loadable in C++/CUDA for production inference.

**Background:** PyTorch models can't be used directly in C++ renderers. Need to export weights and rebuild network in C++.

**Requirements:**
- Export PyTorch weights to binary format
- Implement identical network architecture in C++/CUDA
- Verify numerical equivalence (output matches PyTorch)
- Support FP16 weights for memory efficiency

**Implement:**

```python
# Export script (Python)
import torch
import struct

def export_model_weights(model: nn.Module, output_path: str):
    """Export PyTorch model weights to binary format."""
    model.eval()
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', 0x4E4E4D44))  # Magic: "NNMD"
        f.write(struct.pack('I', 1))            # Version
        
        # Write each layer's weights
        for name, param in model.named_parameters():
            # Write layer name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write shape
            shape = list(param.shape)
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))
            
            # Write data (convert to FP16)
            data_fp16 = param.detach().cpu().half().numpy()
            f.write(data_fp16.tobytes())
    
    print(f"Exported model to {output_path}")

# Usage
model = UNetDenoiser()
model.load_state_dict(torch.load('best_model.pth'))
export_model_weights(model, 'denoiser_weights.bin')
```

**C++ weight loader:**

```cpp
class ModelWeightLoader {
public:
    struct WeightInfo {
        std::string name;
        std::vector<int> shape;
        std::vector<half> data;  // FP16 weights
    };
    
    static std::vector<WeightInfo> load_weights(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open weights file");
        }
        
        // Read header
        uint32_t magic, version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        
        if (magic != 0x4E4E4D44) {
            throw std::runtime_error("Invalid magic number");
        }
        
        std::vector<WeightInfo> weights;
        
        // Read each layer
        while (file.peek() != EOF) {
            WeightInfo info;
            
            // Read name
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            info.name.resize(name_len);
            file.read(&info.name[0], name_len);
            
            // Read shape
            uint32_t num_dims;
            file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
            info.shape.resize(num_dims);
            for (uint32_t i = 0; i < num_dims; ++i) {
                file.read(reinterpret_cast<char*>(&info.shape[i]), sizeof(int));
            }
            
            // Read data
            size_t num_elements = 1;
            for (int dim : info.shape) {
                num_elements *= dim;
            }
            info.data.resize(num_elements);
            file.read(reinterpret_cast<char*>(info.data.data()), 
                     num_elements * sizeof(half));
            
            weights.push_back(std::move(info));
        }
        
        return weights;
    }
};

// Load weights into C++ model
void UNetDenoiser::load_weights(const std::string& weights_path) {
    auto weights = ModelWeightLoader::load_weights(weights_path);
    
    for (const auto& weight_info : weights) {
        // Find corresponding layer
        if (weight_info.name.find("encoder.0.conv1.weight") != std::string::npos) {
            encoder[0].conv1.set_weights(weight_info.data, weight_info.shape);
        }
        // ... map all layers
    }
    
    std::cout << "Loaded " << weights.size() << " weight tensors\n";
}
```

**Numerical equivalence test:**

```cpp
TEST(ExportTest, PyTorchCppEquivalence) {
    // Load same weights in PyTorch and C++
    // (Assume we have Python bindings or save PyTorch output)
    
    Tensor input = load_test_image("test_noisy.exr");
    Tensor albedo = load_test_image("test_albedo.exr");
    Tensor normal = load_test_image("test_normal.exr");
    Tensor variance = compute_variance(input);
    
    // C++ inference
    UNetDenoiser cpp_model;
    cpp_model.load_weights("denoiser_weights.bin");
    Tensor cpp_output = cpp_model.forward(input, albedo, normal, variance);
    
    // PyTorch inference (load reference output)
    Tensor pytorch_output = load_tensor("pytorch_output.bin");
    
    // Compare
    Tensor diff = (cpp_output - pytorch_output).abs();
    float max_error = diff.max();
    float mean_error = diff.mean();
    
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Mean error: " << mean_error << std::endl;
    
    // Allow small numerical differences due to FP16
    EXPECT_LT(max_error, 1e-3);
    EXPECT_LT(mean_error, 1e-4);
}
```

---

## TURN 9 — CUDA Kernel Optimization and Profiling

**Instructions:**

Optimize CUDA kernels for maximum throughput using profiling tools (Nsight Compute, nvprof).

**Background:** Initial CUDA implementation may be memory-bound or have low occupancy. Profiling reveals bottlenecks.

**Requirements:**
- Profile all kernels with Nsight Compute
- Optimize memory access patterns (coalescing)
- Maximize occupancy (registers, shared memory)
- Kernel fusion where beneficial
- Target: >80% of theoretical peak bandwidth

**Implement:**

```cpp
// Optimized convolution with shared memory tiling
template<int TILE_SIZE, int KERNEL_SIZE>
__global__ void conv2d_optimized_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int H, int W, int C_in, int C_out
) {
    // Shared memory for input tile (with halo for kernel)
    __shared__ half tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    
    // Cooperative loading of tile into shared memory
    for (int dy = ty; dy < TILE_SIZE + KERNEL_SIZE - 1; dy += blockDim.y) {
        for (int dx = tx; dx < TILE_SIZE + KERNEL_SIZE - 1; dx += blockDim.x) {
            int gx = bx + dx - KERNEL_SIZE/2;
            int gy = by + dy - KERNEL_SIZE/2;
            
            if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                tile[dy][dx] = input[gy * W + gx];
            } else {
                tile[dy][dx] = __float2half(0.0f);
            }
        }
    }
    
    __syncthreads();
    
    // Compute convolution using shared memory
    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        half sum = __float2half(0.0f);
        
        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                half input_val = tile[ty + ky][tx + kx];
                half weight_val = weight[ky * KERNEL_SIZE + kx];
                sum = __hadd(sum, __hmul(input_val, weight_val));
            }
        }
        
        int out_x = bx + tx;
        int out_y = by + ty;
        if (out_x < W && out_y < H) {
            output[out_y * W + out_x] = sum;
        }
    }
}
```

**Profiling script:**

```bash
#!/bin/bash
# Profile denoiser with Nsight Compute

# Compile with profiling info
nvcc -O3 -lineinfo -arch=sm_86 denoiser.cu -o denoiser

# Run profiler
ncu --set full \
    --target-processes all \
    --kernel-name ".*conv2d.*" \
    --launch-skip 10 \
    --launch-count 100 \
    ./denoiser test_input.exr

# Key metrics to check:
# - Memory throughput (should be >70% of peak)
# - Occupancy (should be >50%)
# - Warp execution efficiency (should be >90%)
# - Shared memory bank conflicts (should be minimal)
```

**Optimization checklist:**

```cpp
// Performance optimization checklist
class KernelOptimizer {
public:
    struct OptimizationReport {
        float memory_throughput_percent;
        float occupancy_percent;
        float warp_efficiency_percent;
        int shared_memory_bank_conflicts;
        
        bool is_optimized() const {
            return memory_throughput_percent > 70.0f &&
                   occupancy_percent > 50.0f &&
                   warp_efficiency_percent > 90.0f &&
                   shared_memory_bank_conflicts < 100;
        }
    };
    
    static OptimizationReport profile_kernel(const std::string& kernel_name);
};

TEST(OptimizationTest, KernelPerformance) {
    auto report = KernelOptimizer::profile_kernel("conv2d_3x3_tensorcore");
    
    std::cout << "Memory throughput: " << report.memory_throughput_percent << "%\n";
    std::cout << "Occupancy: " << report.occupancy_percent << "%\n";
    std::cout << "Warp efficiency: " << report.warp_efficiency_percent << "%\n";
    
    EXPECT_TRUE(report.is_optimized());
}
```

---

## TURN 10 — Kernel Fusion for End-to-End Pipeline

**Instructions:**

Fuse preprocessing, inference, and postprocessing into minimal kernel launches to reduce overhead.

**Background:** Each kernel launch has ~5μs overhead. Fusing operations reduces latency and memory traffic.

**Requirements:**
- Fuse G-buffer preprocessing + first conv layer
- Fuse final conv layer + tone mapping + output
- Minimize intermediate memory allocations
- Target: <10 total kernel launches per frame

**Implement:**

```cpp
// Fused preprocessing + first convolution
__global__ void fused_preprocess_and_conv_kernel(
    // Inputs (G-buffers)
    const float3* __restrict__ noisy_color,
    const float3* __restrict__ albedo,
    const float3* __restrict__ normal,
    const float* __restrict__ depth,
    // Weights
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    // Output
    half* __restrict__ conv_output,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // === PREPROCESSING (inline) ===
    
    // Compute variance (local 7x7 window)
    float variance = 0.0f;
    float3 mean = make_float3(0, 0, 0);
    
    #pragma unroll
    for (int dy = -3; dy <= 3; ++dy) {
        #pragma unroll
        for (int dx = -3; dx <= 3; ++dx) {
            int nx = clamp(x + dx, 0, width - 1);
            int ny = clamp(y + dy, 0, height - 1);
            mean = mean + noisy_color[ny * width + nx];
        }
    }
    mean = mean / 49.0f;
    
    #pragma unroll
    for (int dy = -3; dy <= 3; ++dy) {
        #pragma unroll
        for (int dx = -3; dx <= 3; ++dx) {
            int nx = clamp(x + dx, 0, width - 1);
            int ny = clamp(y + dy, 0, height - 1);
            float3 diff = noisy_color[ny * width + nx] - mean;
            variance += dot(diff, diff);
        }
    }
    variance /= 49.0f;
    
    // Normalize features
    float3 norm_color = noisy_color[idx] / 10.0f;  // Assume HDR range [0, 10]
    float3 norm_albedo = albedo[idx];
    float3 norm_normal = normal[idx] * 0.5f + 0.5f;  // [-1,1] -> [0,1]
    float norm_variance = sqrtf(variance);
    
    // === FIRST CONVOLUTION (inline) ===
    
    // Pack features into input channels [10 channels total]
    half input_features[10];
    input_features[0] = __float2half(norm_color.x);
    input_features[1] = __float2half(norm_color.y);
    input_features[2] = __float2half(norm_color.z);
    input_features[3] = __float2half(norm_albedo.x);
    input_features[4] = __float2half(norm_albedo.y);
    input_features[5] = __float2half(norm_albedo.z);
    input_features[6] = __float2half(norm_normal.x);
    input_features[7] = __float2half(norm_normal.y);
    input_features[8] = __float2half(norm_normal.z);
    input_features[9] = __float2half(norm_variance);
    
    // 1x1 convolution (channel mixing)
    constexpr int OUT_CHANNELS = 32;
    for (int c_out = 0; c_out < OUT_CHANNELS; ++c_out) {
        half sum = conv_bias[c_out];
        
        #pragma unroll
        for (int c_in = 0; c_in < 10; ++c_in) {
            half w = conv_weight[c_out * 10 + c_in];
            sum = __hadd(sum, __hmul(input_features[c_in], w));
        }
        
        // ReLU activation
        sum = __hmax(sum, __float2half(0.0f));
        
        conv_output[idx * OUT_CHANNELS + c_out] = sum;
    }
}
```

**Benchmark fusion benefits:**

```cpp
BENCHMARK(BM_Fused_vs_Separate) {
    Tensor noisy = Tensor::randn({1080, 1920, 3});
    Tensor albedo = Tensor::randn({1080, 1920, 3});
    Tensor normal = Tensor::randn({1080, 1920, 3});
    
    // Separate kernels
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto variance = compute_variance(noisy);
        auto features = normalize_features(noisy, albedo, normal, variance);
        auto conv_out = first_conv(features);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    float time_separate = std::chrono::duration<float, std::milli>(end - start).count() / 100.0f;
    
    // Fused kernel
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto conv_out = fused_preprocess_and_conv(noisy, albedo, normal);
        cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    float time_fused = std::chrono::duration<float, std::milli>(end - start).count() / 100.0f;
    
    std::cout << "Separate kernels: " << time_separate << "ms\n";
    std::cout << "Fused kernel: " << time_fused << "ms\n";
    std::cout << "Speedup: " << time_separate / time_fused << "x\n";
    
    // Expected: 1.5-2x speedup from fusion
    EXPECT_GT(time_separate / time_fused, 1.3f);
}
```

---

## TURN 11 — TensorRT Integration for Maximum Performance

**Instructions:**

Integrate with NVIDIA TensorRT for automatic kernel optimization and INT8 quantization.

**Background:** TensorRT can automatically optimize networks, fuse layers, and use INT8 Tensor Cores for even faster inference.

**Requirements:**
- Export model to ONNX format
- Build TensorRT engine with FP16 and INT8 modes
- Calibration for INT8 quantization
- Compare performance: PyTorch vs C++ vs TensorRT

**Implement:**

```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTDenoiser {
public:
    TensorRTDenoiser(const std::string& onnx_path, bool use_int8 = false);
    
    Tensor denoise(
        const Tensor& noisy,
        const Tensor& albedo,
        const Tensor& normal,
        const Tensor& variance
    );
    
private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    };
    
    class INT8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
    public:
        INT8Calibrator(const std::vector<Tensor>& calibration_data);
        
        int getBatchSize() const noexcept override { return 1; }
        
        bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
        
        const void* readCalibrationCache(size_t& length) noexcept override;
        void writeCalibrationCache(const void* cache, size_t length) noexcept override;
        
    private:
        std::vector<Tensor> calibration_data_;
        int current_index_ = 0;
        std::vector<char> calibration_cache_;
    };
    
    void build_engine(const std::string& onnx_path, bool use_int8);
    
    Logger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
};

void TensorRTDenoiser::build_engine(const std::string& onnx_path, bool use_int8) {
    auto builder = nvinfer1::createInferBuilder(logger_);
    auto network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    );
    
    // Parse ONNX
    auto parser = nvonnxparser::createParser(*network, logger_);
    parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    
    // Build config
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1ULL << 30);  // 1GB
    
    // Enable FP16
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    // Enable INT8 (if requested)
    if (use_int8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        
        // Load calibration data
        std::vector<Tensor> calib_data = load_calibration_dataset(100);
        auto calibrator = std::make_unique<INT8Calibrator>(calib_data);
        config->setInt8Calibrator(calibrator.get());
    }
    
    // Build engine
    engine_ = builder->buildEngineWithConfig(*network, *config);
    
    // Create execution context
    context_ = engine_->createExecutionContext();
    
    std::cout << "TensorRT engine built successfully\n";
}
```

**Benchmark TensorRT vs custom implementation:**

```cpp
TEST(TensorRTTest, PerformanceComparison) {
    // Load test image
    Tensor noisy = load_image("test_4spp.exr");
    Tensor albedo = load_image("test_albedo.exr");
    Tensor normal = load_image("test_normal.exr");
    Tensor variance = compute_variance(noisy);
    
    // Custom C++/CUDA implementation
    UNetDenoiser custom_model;
    custom_model.load_weights("denoiser_weights.bin");
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto output = custom_model.forward(noisy, albedo, normal, variance);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    float time_custom = std::chrono::duration<float, std::milli>(end - start).count() / 100.0f;
    
    // TensorRT FP16
    TensorRTDenoiser trt_fp16("denoiser.onnx", false);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto output = trt_fp16.denoise(noisy, albedo, normal, variance);
        cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    float time_trt_fp16 = std::chrono::duration<float, std::milli>(end - start).count() / 100.0f;
    
    // TensorRT INT8
    TensorRTDenoiser trt_int8("denoiser.onnx", true);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto output = trt_int8.denoise(noisy, albedo, normal, variance);
        cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    float time_trt_int8 = std::chrono::duration<float, std::milli>(end - start).count() / 100.0f;
    
    std::cout << "Custom C++/CUDA: " << time_custom << "ms\n";
    std::cout << "TensorRT FP16: " << time_trt_fp16 << "ms (" << time_custom/time_trt_fp16 << "x)\n";
    std::cout << "TensorRT INT8: " << time_trt_int8 << "ms (" << time_custom/time_trt_int8 << "x)\n";
    
    // TensorRT should be faster
    EXPECT_LT(time_trt_fp16, time_custom);
    EXPECT_LT(time_trt_int8, time_trt_fp16);
}
```

---

## TURN 12 — OptiX Integration as Denoiser Callback

**Instructions:**

Integrate the denoiser into NVIDIA OptiX as a custom denoiser callback for seamless use in path tracers.

**Background:** OptiX 7+ supports custom denoisers. Integration allows drop-in replacement for OptiX AI Denoiser.

**Requirements:**
- Implement OptiX denoiser API
- Handle OptiX buffer formats
- Support streaming (multiple frames in flight)
- Match or beat OptiX AI Denoiser performance

**Implement:**

```cpp
#include <optix.h>
#include <optix_stubs.h>

class OptiXCustomDenoiser {
public:
    OptiXCustomDenoiser(OptixDeviceContext context);
    
    // OptiX denoiser API
    OptixResult setup(
        unsigned int width,
        unsigned int height,
        OptixDenoiserSizes* sizes
    );
    
    OptixResult invoke(
        CUstream stream,
        const OptixDenoiserParams* params,
        const OptixDenoiserGuideLayer* guide_layer,
        const OptixDenoiserLayer* layers,
        unsigned int num_layers,
        unsigned int input_offset_x,
        unsigned int input_offset_y,
        void* scratch,
        size_t scratch_size
    );
    
private:
    OptixDeviceContext context_;
    UNetDenoiser denoiser_;
    TemporalAccumulator temporal_;
    
    // Convert OptiX buffers to our tensor format
    Tensor convert_optix_buffer(const OptixImage2D& image);
};

OptixResult OptiXCustomDenoiser::invoke(
    CUstream stream,
    const OptixDenoiserParams* params,
    const OptixDenoiserGuideLayer* guide_layer,
    const OptixDenoiserLayer* layers,
    unsigned int num_layers,
    unsigned int input_offset_x,
    unsigned int input_offset_y,
    void* scratch,
    size_t scratch_size
) {
    // Extract inputs from OptiX buffers
    Tensor noisy = convert_optix_buffer(layers[0].input);
    Tensor albedo = convert_optix_buffer(guide_layer->albedo);
    Tensor normal = convert_optix_buffer(guide_layer->normal);
    Tensor motion = convert_optix_buffer(guide_layer->flow);
    
    // Compute variance
    Tensor variance = compute_variance_gpu(noisy);
    
    // Spatial denoising
    Tensor denoised = denoiser_.forward(noisy, albedo, normal, variance);
    
    // Temporal accumulation (if motion vectors available)
    if (motion.defined()) {
        denoised = temporal_.accumulate(denoised, motion, /*depth*/{}, /*depth_prev*/{});
    }
    
    // Write output back to OptiX buffer
    write_to_optix_buffer(denoised, layers[0].output);
    
    return OPTIX_SUCCESS;
}
```

**Usage in OptiX renderer:**

```cpp
// In OptiX path tracer
void render_with_custom_denoiser() {
    // Create custom denoiser
    OptiXCustomDenoiser denoiser(optix_context);
    
    // Setup
    OptixDenoiserSizes sizes;
    denoiser.setup(width, height, &sizes);
    
    // Allocate scratch memory
    CUdeviceptr scratch;
    cudaMalloc(&scratch, sizes.withoutOverlapScratchSizeInBytes);
    
    // Render loop
    for (int frame = 0; frame < num_frames; ++frame) {
        // Path trace (low SPP)
        path_trace(4 /* SPP */);
        
        // Denoise
        OptixDenoiserParams params = {};
        params.blendFactor = 0.0f;  // No blending (we handle temporal)
        
        OptixDenoiserGuideLayer guide_layer = {};
        guide_layer.albedo = albedo_buffer;
        guide_layer.normal = normal_buffer;
        guide_layer.flow = motion_buffer;
        
        OptixDenoiserLayer layer = {};
        layer.input = noisy_buffer;
        layer.output = denoised_buffer;
        
        denoiser.invoke(
            cuda_stream,
            &params,
            &guide_layer,
            &layer,
            1,
            0, 0,
            (void*)scratch,
            sizes.withoutOverlapScratchSizeInBytes
        );
        
        // Display denoised result
        display(denoised_buffer);
    }
}
```

---

## TURN 13 — Comprehensive Benchmarking Suite

**Instructions:**

Create a comprehensive benchmark suite comparing against industry baselines across diverse scenes.

**Background:** Need rigorous validation that the denoiser works across different content types and SPP levels.

**Requirements:**
- Test on 20+ diverse scenes (indoor, outdoor, materials, lighting)
- Test SPP levels: 1, 2, 4, 8, 16
- Metrics: PSNR, SSIM, LPIPS, temporal stability
- Compare: OptiX AI Denoiser, Intel OIDN, NVIDIA NRD
- Generate comparison report with visualizations

**Implement:**

```cpp
class DenoisingBenchmark {
public:
    struct SceneConfig {
        std::string name;
        std::string path;
        int spp_low;
        int spp_reference;
    };
    
    struct BenchmarkResult {
        std::string scene_name;
        std::string denoiser_name;
        int spp;
        float psnr;
        float ssim;
        float lpips;
        float latency_ms;
        float temporal_variance;
    };
    
    static std::vector<BenchmarkResult> run_benchmark(
        const std::vector<SceneConfig>& scenes,
        const std::vector<std::string>& denoisers
    );
    
    static void generate_report(
        const std::vector<BenchmarkResult>& results,
        const std::string& output_path
    );
};

std::vector<DenoisingBenchmark::BenchmarkResult> DenoisingBenchmark::run_benchmark(
    const std::vector<SceneConfig>& scenes,
    const std::vector<std::string>& denoisers
) {
    std::vector<BenchmarkResult> results;
    
    for (const auto& scene : scenes) {
        std::cout << "\\nBenchmarking scene: " << scene.name << "\\n";
        
        // Load reference (high SPP)
        Tensor reference = render_scene(scene.path, scene.spp_reference);
        
        // Test each SPP level
        for (int spp : {1, 2, 4, 8, 16}) {
            if (spp >= scene.spp_reference) continue;
            
            // Render noisy
            Tensor noisy = render_scene(scene.path, spp);
            Tensor albedo = render_gbuffer(scene.path, "albedo");
            Tensor normal = render_gbuffer(scene.path, "normal");
            
            // Test each denoiser
            for (const auto& denoiser_name : denoisers) {
                BenchmarkResult result;
                result.scene_name = scene.name;
                result.denoiser_name = denoiser_name;
                result.spp = spp;
                
                // Denoise
                auto start = std::chrono::high_resolution_clock::now();
                Tensor denoised = run_denoiser(denoiser_name, noisy, albedo, normal);
                auto end = std::chrono::high_resolution_clock::now();
                
                result.latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
                
                // Compute metrics
                result.psnr = compute_psnr(denoised, reference);
                result.ssim = compute_ssim(denoised, reference);
                result.lpips = compute_lpips(denoised, reference);
                
                results.push_back(result);
                
                std::cout << "  " << denoiser_name << " @ " << spp << " SPP: "
                          << "PSNR=" << result.psnr << " dB, "
                          << "SSIM=" << result.ssim << ", "
                          << "Latency=" << result.latency_ms << " ms\\n";
            }
        }
    }
    
    return results;
}
```

**Test scenes:**

```cpp
std::vector<DenoisingBenchmark::SceneConfig> create_test_scenes() {
    return {
        {"Cornell Box", "scenes/cornell_box.obj", 4, 2048},
        {"Living Room", "scenes/living_room.obj", 4, 2048},
        {"Outdoor Forest", "scenes/forest.obj", 8, 4096},
        {"Car Interior", "scenes/car_interior.obj", 4, 2048},
        {"Glossy Materials", "scenes/glossy_spheres.obj", 8, 4096},
        {"Caustics", "scenes/caustics.obj", 16, 8192},
        {"Subsurface Scattering", "scenes/sss_dragon.obj", 8, 4096},
        {"Volumetric Fog", "scenes/fog.obj", 16, 8192},
        // ... 12 more scenes
    };
}
```

---

## TURN 14 — Quality Metrics and Ablation Studies

**Instructions:**

Implement comprehensive quality metrics and perform ablation studies to validate design choices.

**Background:** Need to quantify impact of each component (temporal, perceptual loss, etc.) on final quality.

**Requirements:**
- Implement LPIPS (perceptual similarity)
- Temporal stability metric (flicker detection)
- Ablation: remove each component, measure impact
- Statistical significance testing

**Implement:**

```python
# LPIPS implementation (perceptual loss)
import lpips

class PerceptualMetrics:
    def __init__(self):
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute LPIPS (lower is better, range [0, 1])."""
        # Normalize to [-1, 1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        
        with torch.no_grad():
            distance = self.lpips_model(img1, img2)
        
        return distance.item()
    
    def compute_temporal_stability(
        self,
        frames: List[torch.Tensor],
        motion_masks: List[torch.Tensor]
    ) -> float:
        """Compute temporal variance in static regions (lower is better)."""
        total_variance = 0.0
        count = 0
        
        for i in range(len(frames) - 1):
            # Static regions (low motion)
            static_mask = (motion_masks[i] < 0.1).float()
            
            # Compute variance between consecutive frames
            diff = (frames[i+1] - frames[i]).pow(2)
            variance = (diff * static_mask).sum() / (static_mask.sum() + 1e-6)
            
            total_variance += variance.item()
            count += 1
        
        return total_variance / count
```

**Ablation study:**

```cpp
struct AblationConfig {
    bool use_temporal = true;
    bool use_perceptual_loss = true;
    bool use_edge_loss = true;
    bool use_variance_weighting = true;
};

class AblationStudy {
public:
    static void run_ablation() {
        // Baseline: all features enabled
        AblationConfig baseline;
        float psnr_baseline = train_and_evaluate(baseline);
        
        std::cout << "Baseline PSNR: " << psnr_baseline << " dB\\n\\n";
        
        // Ablate each component
        {
            AblationConfig config = baseline;
            config.use_temporal = false;
            float psnr = train_and_evaluate(config);
            std::cout << "Without temporal: " << psnr << " dB "
                      << "(Δ = " << psnr - psnr_baseline << ")\\n";
        }
        
        {
            AblationConfig config = baseline;
            config.use_perceptual_loss = false;
            float psnr = train_and_evaluate(config);
            std::cout << "Without perceptual loss: " << psnr << " dB "
                      << "(Δ = " << psnr - psnr_baseline << ")\\n";
        }
        
        {
            AblationConfig config = baseline;
            config.use_edge_loss = false;
            float psnr = train_and_evaluate(config);
            std::cout << "Without edge loss: " << psnr << " dB "
                      << "(Δ = " << psnr - psnr_baseline << ")\\n";
        }
        
        {
            AblationConfig config = baseline;
            config.use_variance_weighting = false;
            float psnr = train_and_evaluate(config);
            std::cout << "Without variance weighting: " << psnr << " dB "
                      << "(Δ = " << psnr - psnr_baseline << ")\\n";
        }
    }
};
```

---

## TURN 15 — Memory Optimization for 4K Resolution

**Instructions:**

Optimize memory usage to support 4K (3840×2160) denoising within 8GB VRAM budget.

**Background:** 4K frames are 4x larger than 1080p. Need aggressive memory optimization.

**Requirements:**
- Tile-based processing for 4K
- Streaming inference (process tiles sequentially)
- Overlap tiles to avoid seams
- Target: <8GB VRAM for 4K

**Implement:**

```cpp
class TiledDenoiser {
public:
    struct Config {
        int tile_size = 512;      // Process 512×512 tiles
        int overlap = 32;         // 32-pixel overlap to avoid seams
        int num_streams = 4;      // Concurrent CUDA streams
    };
    
    TiledDenoiser(const Config& config);
    
    Tensor denoise_4k(
        const Tensor& noisy_4k,      // [2160, 3840, 3]
        const Tensor& albedo_4k,
        const Tensor& normal_4k
    );
    
private:
    Tensor process_tile(
        const Tensor& tile_noisy,
        const Tensor& tile_albedo,
        const Tensor& tile_normal,
        cudaStream_t stream
    );
    
    void blend_overlap(
        Tensor& output,
        const Tensor& tile,
        int tile_x, int tile_y
    );
    
    Config config_;
    UNetDenoiser denoiser_;
    std::vector<cudaStream_t> streams_;
};

Tensor TiledDenoiser::denoise_4k(
    const Tensor& noisy_4k,
    const Tensor& albedo_4k,
    const Tensor& normal_4k
) {
    int height = 2160;
    int width = 3840;
    int tile_size = config_.tile_size;
    int overlap = config_.overlap;
    
    // Output buffer
    Tensor output = Tensor::zeros({height, width, 3});
    
    // Process tiles
    int num_tiles_y = (height + tile_size - overlap - 1) / (tile_size - overlap);
    int num_tiles_x = (width + tile_size - overlap - 1) / (tile_size - overlap);
    
    int stream_idx = 0;
    
    for (int ty = 0; ty < num_tiles_y; ++ty) {
        for (int tx = 0; tx < num_tiles_x; ++tx) {
            // Compute tile bounds
            int y_start = ty * (tile_size - overlap);
            int x_start = tx * (tile_size - overlap);
            int y_end = std::min(y_start + tile_size, height);
            int x_end = std::min(x_start + tile_size, width);
            
            // Extract tile
            Tensor tile_noisy = noisy_4k.slice(0, y_start, y_end).slice(1, x_start, x_end);
            Tensor tile_albedo = albedo_4k.slice(0, y_start, y_end).slice(1, x_start, x_end);
            Tensor tile_normal = normal_4k.slice(0, y_start, y_end).slice(1, x_start, x_end);
            
            // Process tile (async on stream)
            cudaStream_t stream = streams_[stream_idx];
            Tensor tile_denoised = process_tile(tile_noisy, tile_albedo, tile_normal, stream);
            
            // Blend into output (handle overlap)
            blend_overlap(output, tile_denoised, x_start, y_start);
            
            // Cycle through streams
            stream_idx = (stream_idx + 1) % config_.num_streams;
        }
    }
    
    // Synchronize all streams
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
    
    return output;
}

void TiledDenoiser::blend_overlap(
    Tensor& output,
    const Tensor& tile,
    int tile_x, int tile_y
) {
    int tile_h = tile.shape()[0];
    int tile_w = tile.shape()[1];
    int overlap = config_.overlap;
    
    for (int y = 0; y < tile_h; ++y) {
        for (int x = 0; x < tile_w; ++x) {
            int out_y = tile_y + y;
            int out_x = tile_x + x;
            
            // Compute blend weight (feather at edges)
            float weight_y = 1.0f;
            float weight_x = 1.0f;
            
            if (y < overlap) {
                weight_y = float(y) / overlap;
            } else if (y >= tile_h - overlap) {
                weight_y = float(tile_h - y) / overlap;
            }
            
            if (x < overlap) {
                weight_x = float(x) / overlap;
            } else if (x >= tile_w - overlap) {
                weight_x = float(tile_w - x) / overlap;
            }
            
            float weight = weight_x * weight_y;
            
            // Blend
            output[out_y][out_x] = output[out_y][out_x] * (1.0f - weight) + 
                                   tile[y][x] * weight;
        }
    }
}
```

---

## TURN 16 — Production Deployment and API Design

**Instructions:**

Design a clean, production-ready API for integration into rendering pipelines.

**Background:** Final denoiser must be easy to integrate, well-documented, and robust.

**Requirements:**
- C API for maximum compatibility
- Python bindings for research
- Thread-safe, exception-safe
- Comprehensive error handling
- Detailed documentation

**Implement:**

```cpp
// C API (denoiser.h)
#ifdef __cplusplus
extern "C" {
#endif

typedef struct DenoiserHandle_t* DenoiserHandle;

typedef enum {
    DENOISER_SUCCESS = 0,
    DENOISER_ERROR_INVALID_ARGUMENT = 1,
    DENOISER_ERROR_OUT_OF_MEMORY = 2,
    DENOISER_ERROR_CUDA_ERROR = 3,
    DENOISER_ERROR_MODEL_LOAD_FAILED = 4
} DenoiserStatus;

typedef struct {
    int width;
    int height;
    int channels;
    void* data;  // Device pointer
} DenoiserImage;

/**
 * Create a denoiser instance.
 * 
 * @param handle Output handle
 * @param model_path Path to trained model weights
 * @param use_temporal Enable temporal accumulation
 * @return Status code
 */
DenoiserStatus denoiser_create(
    DenoiserHandle* handle,
    const char* model_path,
    bool use_temporal
);

/**
 * Denoise a single frame.
 * 
 * @param handle Denoiser handle
 * @param noisy Noisy input image (device pointer)
 * @param albedo Albedo G-buffer (device pointer)
 * @param normal Normal G-buffer (device pointer)
 * @param motion Motion vectors (device pointer, can be NULL)
 * @param output Denoised output (device pointer)
 * @return Status code
 */
DenoiserStatus denoiser_execute(
    DenoiserHandle handle,
    const DenoiserImage* noisy,
    const DenoiserImage* albedo,
    const DenoiserImage* normal,
    const DenoiserImage* motion,
    DenoiserImage* output
);

/**
 * Destroy denoiser instance.
 */
void denoiser_destroy(DenoiserHandle handle);

#ifdef __cplusplus
}
#endif
```

**Python bindings (using pybind11):**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PyDenoiser {
public:
    PyDenoiser(const std::string& model_path, bool use_temporal) {
        DenoiserStatus status = denoiser_create(&handle_, model_path.c_str(), use_temporal);
        if (status != DENOISER_SUCCESS) {
            throw std::runtime_error("Failed to create denoiser");
        }
    }
    
    ~PyDenoiser() {
        denoiser_destroy(handle_);
    }
    
    py::array_t<float> denoise(
        py::array_t<float> noisy,
        py::array_t<float> albedo,
        py::array_t<float> normal,
        py::array_t<float> motion = py::array_t<float>()
    ) {
        // Convert numpy arrays to DenoiserImage
        // Execute denoising
        // Return result as numpy array
    }
    
private:
    DenoiserHandle handle_;
};

PYBIND11_MODULE(denoiser, m) {
    py::class_<PyDenoiser>(m, "Denoiser")
        .def(py::init<const std::string&, bool>(),
             py::arg("model_path"),
             py::arg("use_temporal") = true)
        .def("denoise", &PyDenoiser::denoise,
             py::arg("noisy"),
             py::arg("albedo"),
             py::arg("normal"),
             py::arg("motion") = py::array_t<float>());
}
```

---

## TURN 17 — Continuous Integration and Testing

**Instructions:**

Set up comprehensive CI/CD pipeline with automated testing and performance regression detection.

**Background:** Production code needs robust testing to catch regressions.

**Requirements:**
- Unit tests (>200 tests)
- Integration tests (end-to-end)
- Performance benchmarks (regression detection)
- Memory leak detection
- Code coverage >90%

**Implement:**

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake g++ libopenexr-dev
      
      - name: Build
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON ..
          make -j$(nproc)
      
      - name: Run tests
        run: |
          cd build
          ctest --output-on-failure
      
      - name: Check coverage
        run: |
          cd build
          gcov -r ..
          # Upload to codecov
  
  test-gpu:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      
      - name: Build with CUDA
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON ..
          make -j$(nproc)
      
      - name: Run GPU tests
        run: |
          cd build
          ./test_cuda
      
      - name: Run benchmarks
        run: |
          cd build
          ./benchmark_denoiser --scenes=test_scenes/ --output=results.json
      
      - name: Check performance regression
        run: |
          python scripts/check_regression.py \
            --current=results.json \
            --baseline=baseline_results.json \
            --threshold=0.05  # 5% regression threshold
```

**Performance regression detection:**

```python
# scripts/check_regression.py
import json
import sys

def check_regression(current_file, baseline_file, threshold=0.05):
    with open(current_file) as f:
        current = json.load(f)
    
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    regressions = []
    
    for scene in current['scenes']:
        scene_name = scene['name']
        current_latency = scene['latency_ms']
        
        # Find baseline
        baseline_scene = next((s for s in baseline['scenes'] if s['name'] == scene_name), None)
        if not baseline_scene:
            continue
        
        baseline_latency = baseline_scene['latency_ms']
        
        # Check regression
        if current_latency > baseline_latency * (1 + threshold):
            regression_pct = (current_latency / baseline_latency - 1) * 100
            regressions.append({
                'scene': scene_name,
                'baseline': baseline_latency,
                'current': current_latency,
                'regression_pct': regression_pct
            })
    
    if regressions:
        print("Performance regressions detected:")
        for r in regressions:
            print(f"  {r['scene']}: {r['baseline']:.2f}ms -> {r['current']:.2f}ms "
                  f"({r['regression_pct']:.1f}% slower)")
        sys.exit(1)
    else:
        print("No performance regressions detected")
        sys.exit(0)
```

---

## TURN 18 — End-to-End Validation and Documentation

**Instructions:**

Final validation across all requirements and comprehensive documentation.

**Background:** Ensure all success criteria are met and system is production-ready.

**Deliverables:**
- Final benchmark report
- User guide
- API documentation
- Performance tuning guide
- Example integrations

**Final validation checklist:**

```cpp
class FinalValidation {
public:
    static void run_all_validations() {
        std::cout << "=== FINAL VALIDATION ===\\n\\n";
        
        // 1. Performance requirements
        validate_performance();
        
        // 2. Quality requirements
        validate_quality();
        
        // 3. Memory requirements
        validate_memory();
        
        // 4. Robustness
        validate_robustness();
        
        // 5. Integration
        validate_integration();
        
        std::cout << "\\n=== ALL VALIDATIONS PASSED ===\\n";
    }
    
private:
    static void validate_performance() {
        std::cout << "Validating performance...\\n";
        
        // 1080p @ 60 FPS
        auto latency_1080p = benchmark_latency(1920, 1080);
        std::cout << "  1080p latency: " << latency_1080p << "ms\\n";
        assert(latency_1080p < 16.0f && "Must be <16ms for 60 FPS");
        
        // 4K @ 30 FPS
        auto latency_4k = benchmark_latency(3840, 2160);
        std::cout << "  4K latency: " << latency_4k << "ms\\n";
        assert(latency_4k < 33.0f && "Must be <33ms for 30 FPS");
        
        std::cout << "  ✓ Performance requirements met\\n\\n";
    }
    
    static void validate_quality() {
        std::cout << "Validating quality...\\n";
        
        auto scenes = load_test_scenes();
        float avg_psnr = 0.0f;
        float avg_ssim = 0.0f;
        
        for (const auto& scene : scenes) {
            auto metrics = evaluate_scene(scene);
            avg_psnr += metrics.psnr;
            avg_ssim += metrics.ssim;
        }
        
        avg_psnr /= scenes.size();
        avg_ssim /= scenes.size();
        
        std::cout << "  Average PSNR: " << avg_psnr << " dB\\n";
        std::cout << "  Average SSIM: " << avg_ssim << "\\n";
        
        assert(avg_psnr > 40.0f && "PSNR must be >40 dB");
        assert(avg_ssim > 0.95f && "SSIM must be >0.95");
        
        std::cout << "  ✓ Quality requirements met\\n\\n";
    }
    
    static void validate_memory() {
        std::cout << "Validating memory usage...\\n";
        
        size_t mem_1080p = measure_memory_usage(1920, 1080);
        size_t mem_4k = measure_memory_usage(3840, 2160);
        
        std::cout << "  1080p memory: " << mem_1080p / (1024*1024) << " MB\\n";
        std::cout << "  4K memory: " << mem_4k / (1024*1024) << " MB\\n";
        
        assert(mem_1080p < 500 * 1024 * 1024 && "1080p must use <500MB");
        assert(mem_4k < 8 * 1024 * 1024 * 1024 && "4K must use <8GB");
        
        std::cout << "  ✓ Memory requirements met\\n\\n";
    }
};
```

**Final benchmark report:**

```
=================================
FINAL BENCHMARK REPORT
=================================

Performance (NVIDIA RTX 4090):
  1080p (1920×1080):
    - Latency: 7.2ms (138 FPS) ✓
    - Throughput: 138 frames/sec
    - Memory: 287 MB
  
  4K (3840×2160):
    - Latency: 24.1ms (41 FPS) ✓
    - Throughput: 41 frames/sec
    - Memory: 6.8 GB

Quality (average across 20 test scenes):
  PSNR: 41.3 dB ✓ (target: >40 dB)
  SSIM: 0.967 ✓ (target: >0.95)
  LPIPS: 0.042 ✓ (lower is better)
  Temporal stability: 0.003 ✓ (target: <0.01)

Comparison with baselines (1080p, 4 SPP):
                    PSNR    SSIM    Latency   Memory
  Ours              41.3    0.967   7.2ms     287MB  ✓
  OptiX AI Denoiser 40.8    0.961   6.8ms     312MB
  Intel OIDN        39.2    0.952   9.1ms     198MB
  NVIDIA NRD        41.1    0.964   7.5ms     301MB

→ Competitive with state-of-the-art ✓

Training:
  Dataset: 52,000 image pairs
  Training time: 18.3 hours (4× RTX 4090)
  Final validation loss: 0.0042

Model:
  Parameters: 28.7M
  Size (FP16): 57.4 MB
  Architecture: Custom U-Net with temporal

ALL SUCCESS CRITERIA MET ✓
=================================
```

---

**Final Deliverables:**

1. ✅ Trained neural denoiser model (28.7M parameters)
2. ✅ C++/CUDA inference engine (<16ms @ 1080p)
3. ✅ Temporal reprojection system (<0.5% flicker)
4. ✅ OptiX integration (drop-in replacement)
5. ✅ TensorRT optimization (INT8 support)
6. ✅ Python training pipeline
7. ✅ Comprehensive test suite (>200 tests)
8. ✅ Benchmark suite (20+ scenes)
9. ✅ Production API (C + Python bindings)
10. ✅ Complete documentation

**Estimated completion time:** 60-75 hours for expert CUDA/ML engineer

**Difficulty:** EXTREME - requires mastery of deep learning, CUDA programming, computer graphics, and production software engineering.
