# Task: Build a Production-Grade Real-Time SIMD Image Processor

## Overview
Implement a zero-allocation, cache-optimized image processing library in Zig that uses SIMD intrinsics (AVX2/AVX-512 or NEON) to achieve real-time performance on high-resolution images. The system must process 4K video at 60 FPS, handle various pixel formats without runtime overhead, implement numerically stable color space conversions, and achieve >80% memory bandwidth utilization.

**Key Challenge:** You CANNOT use OpenCV, IPP, or vendor libraries. Everything must be implemented from scratch using SIMD intrinsics, achieving performance competitive with hand-tuned assembly.

---

## TURN 1 — SIMD Fundamentals: Vectorized Pixel Operations

**Role:** You are a performance engineer who has optimized computer vision pipelines for real-time applications (autonomous vehicles, video encoding, medical imaging). You understand cache hierarchies, memory bandwidth limitations, and can write SIMD code that matches assembly-level performance.

**Background:** Image processing is memory-bandwidth bound. Naive scalar code wastes 90% of CPU capability. SIMD allows processing 16-32 pixels simultaneously. First step: basic vectorized operations.

**Reference:** Study:
- Intel Intrinsics Guide (AVX2/AVX-512)
- ARM NEON intrinsics
- Zig's `@Vector` builtin and inline assembly
- Cache-oblivious algorithms for image processing

**VERY IMPORTANT:**
- All data must be aligned to SIMD width (16/32/64 bytes)
- No scalar fallback in hot paths
- Zero dynamic allocation during processing
- Must handle edge cases (partial vectors at image boundaries)
- Numerical accuracy: <0.01 error in color conversions
- Cache-friendly access patterns (no random access)

**Goal:** Implement basic SIMD image operations with perfect memory access patterns.

**Instructions:**

1. **Define image structure:**
```zig
const std = @import("std");
const builtin = @import("builtin");

pub const PixelFormat = enum {
    rgb8,       // 8-bit RGB
    rgba8,      // 8-bit RGBA
    rgb16,      // 16-bit RGB
    gray8,      // 8-bit grayscale
    gray16,     // 16-bit grayscale
    yuv420,     // YUV 4:2:0
};

pub const Image = struct {
    width: usize,
    height: usize,
    stride: usize, // Bytes per row (may include padding for alignment)
    format: PixelFormat,
    data: []align(32) u8, // 32-byte aligned for AVX2
    
    pub fn init(allocator: std.mem.Allocator, width: usize, height: usize, format: PixelFormat) !Image {
        const bytes_per_pixel = format.bytesPerPixel();
        const row_bytes = width * bytes_per_pixel;
        
        // Align stride to 32 bytes for SIMD
        const stride = (row_bytes + 31) & ~@as(usize, 31);
        const total_bytes = stride * height;
        
        const data = try allocator.alignedAlloc(u8, 32, total_bytes);
        
        return Image{
            .width = width,
            .height = height,
            .stride = stride,
            .format = format,
            .data = data,
        };
    }
    
    pub fn pixel_ptr(self: *Image, x: usize, y: usize) [*]u8 {
        return self.data.ptr + y * self.stride + x * self.format.bytesPerPixel();
    }
};
```

2. **SIMD wrapper for portability:**
```zig
// Abstraction over AVX2/AVX-512/NEON
pub const SimdOps = switch (builtin.cpu.arch) {
    .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2))
        Avx2Ops
    else
        SseOps,
    .aarch64 => NeonOps,
    else => @compileError("Unsupported architecture"),
};

const Avx2Ops = struct {
    const Vec8x32 = @Vector(32, u8);  // 32 bytes = 32 pixels
    const Vec16x16 = @Vector(16, i16);
    const Vec8x32f = @Vector(8, f32);
    
    pub inline fn load_aligned(ptr: [*]const u8) Vec8x32 {
        return @as(*const Vec8x32, @ptrCast(@alignCast(ptr))).*;
    }
    
    pub inline fn store_aligned(ptr: [*]u8, vec: Vec8x32) void {
        @as(*Vec8x32, @ptrCast(@alignCast(ptr))).* = vec;
    }
    
    pub inline fn add_saturate_u8(a: Vec8x32, b: Vec8x32) Vec8x32 {
        // Saturating add using intrinsic
        return asm volatile ("vpaddsb %[b], %[a], %[out]"
            : [out] "=x" (-> Vec8x32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    }
    
    pub inline fn mul_u8(a: Vec8x32, scalar: u8) Vec8x32 {
        const scalar_vec = @splat(32, scalar);
        // Multiply requires unpacking to 16-bit
        const a_lo = @shuffle(i16, @bitCast(@Vector(16, i16), a), undefined, [_]i32{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
        const a_hi = @shuffle(i16, @bitCast(@Vector(16, i16), a), undefined, [_]i32{16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31});
        
        const result_lo = a_lo * scalar_vec;
        const result_hi = a_hi * scalar_vec;
        
        // Pack back to 8-bit with saturation
        return @bitCast(Vec8x32, pack_saturate_i16_to_u8(result_lo, result_hi));
    }
    
    pub inline fn blend(a: Vec8x32, b: Vec8x32, mask: Vec8x32) Vec8x32 {
        return asm volatile ("vpblendvb %[mask], %[b], %[a], %[out]"
            : [out] "=x" (-> Vec8x32),
            : [mask] "x" (mask),
              [a] "x" (a),
              [b] "x" (b),
        );
    }
};
```

3. **Implementation: Brightness adjustment (vectorized):**
```zig
pub fn adjust_brightness(img: *Image, delta: i16) !void {
    if (img.format != .rgb8) return error.UnsupportedFormat;
    
    const delta_vec = @splat(32, @intCast(i8, std.math.clamp(delta, -127, 127)));
    
    var y: usize = 0;
    while (y < img.height) : (y += 1) {
        const row_ptr = img.data.ptr + y * img.stride;
        
        var x: usize = 0;
        // Process 32 bytes (32 pixels in grayscale, or 10 RGB pixels) at a time
        while (x + 32 <= img.width * 3) : (x += 32) {
            const pixels = SimdOps.load_aligned(row_ptr + x);
            const adjusted = SimdOps.add_saturate_u8(pixels, delta_vec);
            SimdOps.store_aligned(row_ptr + x, adjusted);
        }
        
        // Handle remaining pixels (scalar fallback for edge)
        while (x < img.width * 3) : (x += 1) {
            const val = row_ptr[x];
            const new_val = std.math.clamp(@as(i16, val) + delta, 0, 255);
            row_ptr[x] = @intCast(u8, new_val);
        }
    }
}
```

4. **Tests:**
```zig
test "brightness adjustment correctness" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 640, 480, .rgb8);
    defer img.deinit(allocator);
    
    // Fill with gray (128)
    @memset(img.data, 128);
    
    // Increase brightness by 50
    try adjust_brightness(&img, 50);
    
    // Verify all pixels are 178 (128 + 50)
    for (img.data) |pixel| {
        try std.testing.expectEqual(@as(u8, 178), pixel);
    }
}

test "brightness saturation at boundaries" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 100, 100, .rgb8);
    defer img.deinit(allocator);
    
    // Fill with near-max (250)
    @memset(img.data, 250);
    
    // Increase by 100 (should saturate at 255)
    try adjust_brightness(&img, 100);
    
    for (img.data) |pixel| {
        try std.testing.expectEqual(@as(u8, 255), pixel);
    }
}

test "SIMD alignment validation" {
    const allocator = std.testing.allocator;
    
    var img = try Image.init(allocator, 1920, 1080, .rgb8);
    defer img.deinit(allocator);
    
    // Verify data pointer is 32-byte aligned
    const addr = @ptrToInt(img.data.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % 32);
    
    // Verify stride is 32-byte aligned
    try std.testing.expectEqual(@as(usize, 0), img.stride % 32);
}
```

**Deliverables:**
- SIMD abstraction layer (AVX2/NEON)
- Basic image operations (brightness, contrast)
- Aligned memory allocation
- Tests for correctness and alignment

---

## TURN 2 — Convolution with Separable Filters

**Instructions:**

Implement 2D convolution using separable filters (blur, sharpen, edge detection).

**Background:** 2D convolution is O(n² × k²) where k is kernel size. Separable filters reduce to O(n² × k) by splitting into horizontal and vertical passes.

**Implement:**
```zig
pub fn gaussian_blur(img: *const Image, output: *Image, radius: usize) !void {
    // 1D Gaussian kernel (separable)
    const kernel = try compute_gaussian_kernel(radius);
    defer allocator.free(kernel);
    
    // Temporary buffer for horizontal pass
    const temp = try Image.init(allocator, img.width, img.height, img.format);
    defer temp.deinit(allocator);
    
    // Horizontal pass (vectorized)
    try convolve_horizontal_simd(img, &temp, kernel);
    
    // Vertical pass (vectorized, transposed access)
    try convolve_vertical_simd(&temp, output, kernel);
}

fn convolve_horizontal_simd(src: *const Image, dst: *Image, kernel: []const f32) !void {
    const radius = kernel.len / 2;
    
    var y: usize = 0;
    while (y < src.height) : (y += 1) {
        const src_row = src.data.ptr + y * src.stride;
        const dst_row = dst.data.ptr + y * dst.stride;
        
        var x: usize = 0;
        // Process 8 pixels at once (AVX2: 8x f32)
        while (x + 8 <= src.width) : (x += 8) {
            var accum = @splat(8, @as(f32, 0.0));
            
            // Convolve with kernel
            for (kernel) |coeff, k| {
                const offset = @intCast(isize, k) - @intCast(isize, radius);
                const sample_x = @intCast(usize, @max(0, @min(@intCast(isize, src.width) - 1, @intCast(isize, x) + offset)));
                
                // Load 8 pixels, convert to float, multiply by coefficient
                const pixels_u8 = SimdOps.load_aligned(src_row + sample_x * 3);
                const pixels_f32 = u8_to_f32_simd(pixels_u8);
                accum += pixels_f32 * @splat(8, coeff);
            }
            
            // Convert back to u8 and store
            const result_u8 = f32_to_u8_simd(accum);
            SimdOps.store_aligned(dst_row + x * 3, result_u8);
        }
        
        // Scalar cleanup
        while (x < src.width) : (x += 1) {
            // ...
        }
    }
}
```

**Optimization: Cache blocking for vertical pass:**
```zig
fn convolve_vertical_simd_blocked(src: *const Image, dst: *Image, kernel: []const f32) !void {
    // Process image in tiles to fit in L2 cache
    const tile_height = 64; // Tuned for cache size
    
    var tile_y: usize = 0;
    while (tile_y < src.height) : (tile_y += tile_height) {
        const tile_end = @min(tile_y + tile_height, src.height);
        
        // Process this tile
        var y = tile_y;
        while (y < tile_end) : (y += 1) {
            // Vertical convolution...
        }
    }
}
```

**Benchmark:**
```zig
test "gaussian blur performance" {
    // 1920x1080 image, radius=5
    // Naive scalar: ~500ms
    // SIMD horizontal + vertical: ~15ms (33x faster)
    // SIMD + cache blocking: ~8ms (62x faster)
}
```

---

## TURN 3 — Force Failure: Unaligned Memory Access Crash

**Instructions:**

Introduce a bug where SIMD loads from unaligned addresses.

**Ask the AI:**
> "Your SIMD code assumes all row pointers are 32-byte aligned, but when image width is not a multiple of 32, rows may be unaligned. Show a test where this causes a SIGSEGV on AVX2 aligned load."

**Expected failure:**
```zig
test "unaligned access crash" {
    // Create image with width=1921 (not multiple of 32)
    var img = try Image.init(allocator, 1921, 1080, .rgb8);
    defer img.deinit(allocator);
    
    // Try to process with aligned loads
    try adjust_brightness(&img, 10);
    // CRASH: SIGSEGV on row 1 (pointer not 32-byte aligned)
}
```

**Fix:** Use unaligned loads or ensure stride is always aligned.

---

## TURN 4 — Color Space Conversion: RGB ↔ YUV

**Instructions:**

Implement numerically stable RGB ↔ YUV color space conversion.

**Challenge:** Standard conversion has rounding errors. Must maintain <0.01 error and be reversible.

**Implement:**
```zig
pub fn rgb_to_yuv(img: *const Image, output: *Image) !void {
    if (img.format != .rgb8 or output.format != .yuv420) return error.FormatMismatch;
    
    // BT.601 coefficients (scaled for integer math)
    // Y =  0.299*R + 0.587*G + 0.114*B
    // U = -0.169*R - 0.331*G + 0.500*B + 128
    // V =  0.500*R - 0.419*G - 0.081*B + 128
    
    // Use fixed-point arithmetic: multiply by 2^16 for precision
    const kr = 19595;  // 0.299 * 65536
    const kg = 38470;  // 0.587 * 65536
    const kb = 7471;   // 0.114 * 65536
    
    var y: usize = 0;
    while (y < img.height) : (y += 1) {
        const rgb_row = img.data.ptr + y * img.stride;
        const y_row = output.data.ptr + y * output.stride;
        
        var x: usize = 0;
        // Process 8 RGB pixels at once = 24 bytes
        while (x + 8 <= img.width) : (x += 8) {
            // Load RGB triplets
            const rgb = SimdOps.load_aligned(rgb_row + x * 3);
            
            // Deinterleave RGB into separate R, G, B vectors
            const r_vec = deinterleave_r(rgb);
            const g_vec = deinterleave_g(rgb);
            const b_vec = deinterleave_b(rgb);
            
            // Compute Y = 0.299*R + 0.587*G + 0.114*B (using fixed-point)
            const r_scaled = @intCast(@Vector(8, i32), r_vec) * @splat(8, kr);
            const g_scaled = @intCast(@Vector(8, i32), g_vec) * @splat(8, kg);
            const b_scaled = @intCast(@Vector(8, i32), b_vec) * @splat(8, kb);
            
            const y_sum = r_scaled + g_scaled + b_scaled;
            const y_vec = @intCast(@Vector(8, u8), y_sum >> 16); // Divide by 65536
            
            SimdOps.store_aligned(y_row + x, y_vec);
        }
        
        // Scalar cleanup
        // ...
    }
    
    // Subsample U and V (4:2:0)
    subsample_uv(img, output);
}

fn subsample_uv(img: *const Image, output: *Image) !void {
    // U and V are sampled at half resolution horizontally and vertically
    // Average 2x2 blocks of UV values
    
    // This is complex but critical for video encoding
    // ...
}
```

**Numerical accuracy test:**
```zig
test "rgb to yuv reversibility" {
    const rgb_original = [_]u8{255, 128, 64};
    
    // Convert RGB → YUV → RGB
    const yuv = rgb_to_yuv_pixel(rgb_original);
    const rgb_reconverted = yuv_to_rgb_pixel(yuv);
    
    // Check error < 1 (due to quantization)
    for (rgb_original) |orig, i| {
        const diff = @abs(@as(i16, orig) - @as(i16, rgb_reconverted[i]));
        try std.testing.expect(diff <= 1);
    }
}
```

---

## TURN 5 — Resize with High-Quality Interpolation

**Instructions:**

Implement image resize using Lanczos3 interpolation (better quality than bilinear).

**Lanczos3:** High-quality resampling using sinc function with 3-lobe windowing.

**Implement:**
```zig
pub fn resize_lanczos3(src: *const Image, dst: *Image) !void {
    // Precompute filter weights
    const weights = try compute_lanczos3_weights(src.width, dst.width, src.height, dst.height);
    defer weights.deinit();
    
    // Horizontal resize first
    const temp = try Image.init(allocator, dst.width, src.height, src.format);
    defer temp.deinit(allocator);
    
    try resize_horizontal_lanczos3(src, &temp, weights.horizontal);
    try resize_vertical_lanczos3(&temp, dst, weights.vertical);
}

fn resize_horizontal_lanczos3(src: *const Image, dst: *Image, weights: []const ResampleWeights) !void {
    var y: usize = 0;
    while (y < src.height) : (y += 1) {
        const src_row = src.data.ptr + y * src.stride;
        const dst_row = dst.data.ptr + y * dst.stride;
        
        for (dst_row[0..dst.width]) |*dst_pixel, x| {
            const weight_set = &weights[x];
            
            var accum = @splat(4, @as(f32, 0.0)); // RGBA
            
            // Lanczos3 has up to 6 taps per output pixel
            for (weight_set.coeffs) |coeff, i| {
                const src_x = weight_set.indices[i];
                const src_pixel = get_pixel_f32(src_row, src_x);
                accum += src_pixel * @splat(4, coeff);
            }
            
            // Clamp and convert to u8
            const clamped = @max(@splat(4, @as(f32, 0)), @min(@splat(4, @as(f32, 255)), accum));
            store_pixel_u8(dst_pixel, @floatToInt(@Vector(4, u8), clamped));
        }
    }
}

fn lanczos3_kernel(x: f32) f32 {
    if (@fabs(x) >= 3.0) return 0.0;
    if (x == 0.0) return 1.0;
    
    const pi_x = std.math.pi * x;
    return (3.0 * @sin(pi_x) * @sin(pi_x / 3.0)) / (pi_x * pi_x);
}
```

**Vectorize weight application:**
```zig
// Apply 6 Lanczos3 taps using SIMD
fn apply_lanczos3_taps_simd(src_row: [*]const u8, weight_set: *const ResampleWeights) @Vector(4, f32) {
    var accum = @splat(4, @as(f32, 0.0));
    
    // Load 6 source pixels and weights as vectors
    const pixels = load_6_pixels_simd(src_row, weight_set.indices);
    const weights_vec = @Vector(6, f32){
        weight_set.coeffs[0], weight_set.coeffs[1], weight_set.coeffs[2],
        weight_set.coeffs[3], weight_set.coeffs[4], weight_set.coeffs[5],
    };
    
    // Multiply and accumulate using FMA
    accum = fma_simd(pixels, weights_vec);
    
    return accum;
}
```

**Quality comparison:**
```zig
test "resize quality metrics" {
    // Downscale 1920x1080 → 1280x720
    // Measure PSNR (Peak Signal-to-Noise Ratio)
    
    // Nearest neighbor: PSNR = 28 dB
    // Bilinear: PSNR = 32 dB
    // Lanczos3: PSNR = 38 dB (best quality)
}
```

---

## TURN 6 — Edge Detection with Sobel Operator

**Instructions:**

Implement Sobel edge detection using SIMD.

**Sobel:** Computes gradient magnitude and direction using 3x3 kernels.

**Implement:**
```zig
pub fn sobel_edge_detection(img: *const Image, output: *Image) !void {
    // Sobel kernels
    // Gx = [-1 0 1]     Gy = [-1 -2 -1]
    //      [-2 0 2]          [ 0  0  0]
    //      [-1 0 1]          [ 1  2  1]
    
    var y: usize = 1;
    while (y < img.height - 1) : (y += 1) {
        const row_m1 = img.data.ptr + (y-1) * img.stride;
        const row_0 = img.data.ptr + y * img.stride;
        const row_p1 = img.data.ptr + (y+1) * img.stride;
        const dst_row = output.data.ptr + y * output.stride;
        
        var x: usize = 1;
        // Process 16 pixels at once
        while (x + 16 <= img.width - 1) : (x += 16) {
            // Load 3x3 neighborhoods for 16 pixels
            // This requires loading 18 pixels per row (16 + overlap)
            
            // Row -1
            const p00 = load_u8_vec(row_m1 + x - 1);
            const p01 = load_u8_vec(row_m1 + x);
            const p02 = load_u8_vec(row_m1 + x + 1);
            
            // Row 0
            const p10 = load_u8_vec(row_0 + x - 1);
            const p12 = load_u8_vec(row_0 + x + 1);
            
            // Row +1
            const p20 = load_u8_vec(row_p1 + x - 1);
            const p21 = load_u8_vec(row_p1 + x);
            const p22 = load_u8_vec(row_p1 + x + 1);
            
            // Compute Gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22
            var gx = sub_i16(p02, p00);
            gx = add_i16(gx, mul_i16(sub_i16(p12, p10), @splat(16, @as(i16, 2))));
            gx = add_i16(gx, sub_i16(p22, p20));
            
            // Compute Gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22
            var gy = sub_i16(p20, p00);
            gy = add_i16(gy, mul_i16(sub_i16(p21, p01), @splat(16, @as(i16, 2))));
            gy = add_i16(gy, sub_i16(p22, p02));
            
            // Magnitude = sqrt(Gx^2 + Gy^2) ≈ abs(Gx) + abs(Gy) (faster approximation)
            const mag = add_i16(abs_i16(gx), abs_i16(gy));
            
            // Clamp to [0, 255]
            const result = clamp_and_pack_u8(mag);
            
            store_aligned(dst_row + x, result);
        }
        
        // Scalar cleanup
        // ...
    }
}
```

**SIMD optimizations:**
```zig
// Use AVX2 absolute value (faster than branch)
inline fn abs_i16(vec: @Vector(16, i16)) @Vector(16, i16) {
    return asm volatile ("vpabsw %[vec], %[out]"
        : [out] "=x" (-> @Vector(16, i16)),
        : [vec] "x" (vec),
    );
}

// Saturating pack 16-bit to 8-bit
inline fn clamp_and_pack_u8(vec: @Vector(16, i16)) @Vector(16, u8) {
    const zero = @splat(16, @as(i16, 0));
    const max_val = @splat(16, @as(i16, 255));
    const clamped = @max(zero, @min(max_val, vec));
    
    // Pack two 16-bit vectors into one 8-bit vector
    return asm volatile ("vpackuswb %[vec], %[vec], %[out]"
        : [out] "=x" (-> @Vector(16, u8)),
        : [vec] "x" (clamped),
    );
}
```

---

## TURN 7 — Force Failure: Numerical Overflow in Convolution

**Ask the AI:**
> "Your convolution code multiplies u8 values by float coefficients and accumulates in f32. For very large kernels (radius=50), accumulated rounding errors cause visible banding artifacts. Show a test demonstrating this and fix using higher-precision arithmetic."

**Expected failure:**
```zig
test "large kernel rounding errors" {
    // Gaussian blur with radius=50
    var img = try create_test_image(1920, 1080);
    var output = try Image.init(allocator, 1920, 1080, .rgb8);
    
    try gaussian_blur(&img, &output, 50);
    
    // Check for banding: adjacent pixels should differ smoothly
    for (output.data[0..output.width]) |pixel, i| {
        if (i > 0) {
            const diff = @abs(@as(i16, pixel) - @as(i16, output.data[i-1]));
            try std.testing.expect(diff <= 1); // FAILS: banding visible (diff up to 3-4)
        }
    }
}
```

**Fix:** Use double-precision or compensated summation (Kahan).

---

## TURN 8 — Histogram Equalization with SIMD

**Instructions:**

Implement histogram equalization for contrast enhancement.

**Algorithm:**
1. Compute histogram (256 bins)
2. Compute CDF (cumulative distribution function)
3. Map each pixel value using CDF

**Implement:**
```zig
pub fn histogram_equalization(img: *Image) !void {
    // Step 1: Compute histogram using SIMD gather
    var histogram = [_]u32{0} ** 256;
    compute_histogram_simd(img, &histogram);
    
    // Step 2: Compute CDF
    var cdf = [_]u32{0} ** 256;
    cdf[0] = histogram[0];
    for (histogram[1..]) |count, i| {
        cdf[i+1] = cdf[i] + count;
    }
    
    // Step 3: Normalize CDF and create lookup table
    const total_pixels = img.width * img.height;
    var lut = [_]u8{0} ** 256;
    for (cdf) |cumsum, i| {
        lut[i] = @intCast(u8, (cumsum * 255) / total_pixels);
    }
    
    // Step 4: Apply LUT using SIMD shuffle
    apply_lut_simd(img, &lut);
}

fn compute_histogram_simd(img: *const Image, histogram: *[256]u32) void {
    // Fast histogram using AVX2
    var local_hist = [_]@Vector(8, u32){@splat(8, @as(u32, 0))} ** 32; // 256 bins / 8 = 32 vectors
    
    for (img.data) |pixel_batch, batch_idx| {
        if (batch_idx % 32 != 0) continue;
        
        // Load 32 pixels
        const pixels = SimdOps.load_aligned(img.data.ptr + batch_idx);
        
        // Increment histogram bins (complex SIMD gather/scatter)
        for (pixels) |pixel| {
            const bin = pixel;
            local_hist[bin / 8][bin % 8] += 1;
        }
    }
    
    // Reduce local histogram to global
    for (local_hist) |vec, i| {
        for (vec) |count, j| {
            histogram[i * 8 + j] += count;
        }
    }
}

fn apply_lut_simd(img: *Image, lut: *const [256]u8) void {
    // Use VPSHUFB for fast table lookup (16 bytes at a time)
    
    var offset: usize = 0;
    while (offset + 16 <= img.data.len) : (offset += 16) {
        const pixels = SimdOps.load_aligned(img.data.ptr + offset);
        
        // Split into low and high nibbles for two-stage lookup
        const result = vpshufb_lut(pixels, lut);
        
        SimdOps.store_aligned(img.data.ptr + offset, result);
    }
}
```

---

## TURN 9 — Real-Time Video Processing Pipeline

**Instructions:**

Create end-to-end video processing pipeline hitting 60 FPS on 4K.

**Pipeline stages:**
1. Decode frame (from video file)
2. Denoise (bilateral filter)
3. Color correction (LUT)
4. Edge enhance (unsharp mask)
5. Encode frame (to video file)

**Implement:**
```zig
pub const VideoPipeline = struct {
    decoder: VideoDecoder,
    encoder: VideoEncoder,
    
    // Preallocated buffers (zero allocation during processing)
    frame_buffer: Image,
    temp_buffer: Image,
    
    // Processing stages
    pub fn process_frame(&mut self, frame_in: *const Image, frame_out: *Image) !void {
        const start = std.time.nanoTimestamp();
        
        // Stage 1: Denoise (bilateral filter)
        try bilateral_filter(frame_in, &self.temp_buffer, sigma_spatial=3.0, sigma_range=50.0);
        
        // Stage 2: Color correction
        const lut = color_correction_lut();
        try apply_lut_simd(&self.temp_buffer, lut);
        
        // Stage 3: Unsharp mask
        try unsharp_mask(&self.temp_buffer, frame_out, amount=1.5, radius=2);
        
        const end = std.time.nanoTimestamp();
        const elapsed_ms = @intToFloat(f64, end - start) / 1_000_000.0;
        
        // Check real-time constraint
        const target_ms = 1000.0 / 60.0; // 16.67ms for 60 FPS
        if (elapsed_ms > target_ms) {
            std.debug.print("WARNING: Frame processing took {d:.2}ms (target: {d:.2}ms)\n", .{elapsed_ms, target_ms});
        }
    }
    
    pub fn process_video(&mut self, input_path: []const u8, output_path: []const u8) !void {
        try self.decoder.open(input_path);
        try self.encoder.open(output_path, self.decoder.width, self.decoder.height);
        
        var frame_count: usize = 0;
        const total_start = std.time.nanoTimestamp();
        
        while (try self.decoder.read_frame(&self.frame_buffer)) {
            try self.process_frame(&self.frame_buffer, &self.temp_buffer);
            try self.encoder.write_frame(&self.temp_buffer);
            frame_count += 1;
        }
        
        const total_end = std.time.nanoTimestamp();
        const total_sec = @intToFloat(f64, total_end - total_start) / 1_000_000_000.0;
        const fps = @intToFloat(f64, frame_count) / total_sec;
        
        std.debug.print("Processed {d} frames in {d:.2}s ({d:.1} FPS)\n", .{frame_count, total_sec, fps});
    }
};
```

**Performance benchmarks:**
```
Input: 4K (3840x2160) 60 FPS video, 1000 frames
Target: Real-time (60 FPS) = 16.67ms per frame

Results:
- Bilateral filter: 8ms
- LUT application: 0.5ms
- Unsharp mask: 6ms
Total: 14.5ms per frame ✅ (69 FPS)

Memory bandwidth utilization: 87% of theoretical peak
```

---

## TURN 10 — Advanced: Optical Flow (Motion Estimation)

**Instructions:**

Implement Lucas-Kanade optical flow for motion estimation.

**Application:** Video stabilization, object tracking, motion blur.

**Implement:**
```zig
pub fn lucas_kanade_optical_flow(
    img1: *const Image,
    img2: *const Image,
    flow_x: *Image,
    flow_y: *Image,
    window_size: usize,
) !void {
    // Compute image gradients
    var Ix = try Image.init(allocator, img1.width, img1.height, .gray16);
    var Iy = try Image.init(allocator, img1.width, img1.height, .gray16);
    var It = try Image.init(allocator, img1.width, img1.height, .gray16);
    defer {
        Ix.deinit(allocator);
        Iy.deinit(allocator);
        It.deinit(allocator);
    }
    
    try compute_spatial_gradient(img1, &Ix, &Iy);
    try compute_temporal_gradient(img1, img2, &It);
    
    // For each pixel, solve 2x2 system using least squares
    const half_window = window_size / 2;
    
    var y: usize = half_window;
    while (y < img1.height - half_window) : (y += 1) {
        var x: usize = half_window;
        
        // Process 4 pixels at once using SIMD
        while (x + 4 <= img1.width - half_window) : (x += 4) {
            // Accumulate Structure Tensor components
            var sum_IxIx = @splat(4, @as(f32, 0));
            var sum_IxIy = @splat(4, @as(f32, 0));
            var sum_IyIy = @splat(4, @as(f32, 0));
            var sum_IxIt = @splat(4, @as(f32, 0));
            var sum_IyIt = @splat(4, @as(f32, 0));
            
            // Sum over window
            var wy: usize = 0;
            while (wy < window_size) : (wy += 1) {
                var wx: usize = 0;
                while (wx < window_size) : (wx += 1) {
                    const py = y + wy - half_window;
                    const px_base = x + wx - half_window;
                    
                    // Load gradients for 4 adjacent pixels
                    const ix_vals = load_4_i16(&Ix, px_base, py);
                    const iy_vals = load_4_i16(&Iy, px_base, py);
                    const it_vals = load_4_i16(&It, px_base, py);
                    
                    sum_IxIx += ix_vals * ix_vals;
                    sum_IxIy += ix_vals * iy_vals;
                    sum_IyIy += iy_vals * iy_vals;
                    sum_IxIt += ix_vals * it_vals;
                    sum_IyIt += iy_vals * it_vals;
                }
            }
            
            // Solve 2x2 system: [IxIx IxIy] [u] = -[IxIt]
            //                   [IxIy IyIy] [v]    [IyIt]
            const det = sum_IxIx * sum_IyIy - sum_IxIy * sum_IxIy;
            const u = (sum_IxIy * sum_IyIt - sum_IyIy * sum_IxIt) / det;
            const v = (sum_IxIy * sum_IxIt - sum_IxIx * sum_IyIt) / det;
            
            // Store flow vectors
            store_4_f32(&flow_x, x, y, u);
            store_4_f32(&flow_y, x, y, v);
        }
    }
}
```

---

## TURN 11 — Final Integration: Image Processing Library

**Instructions:**

Create complete image processing library with full API.

**Features:**
- Load/save: JPEG, PNG, BMP
- Color spaces: RGB, YUV, HSV, LAB
- Filters: Gaussian, bilateral, median, Sobel
- Transforms: Resize, rotate, affine
- Effects: Histogram eq, color correction, sharpening
- Advanced: Optical flow, feature detection (SIFT/SURF)
- Video: Read/write, real-time processing

**Public API:**
```zig
pub const ImageProc = struct {
    // Core operations
    pub fn load(path: []const u8) !Image;
    pub fn save(img: *const Image, path: []const u8) !void;
    
    // Filters
    pub fn gaussian_blur(img: *Image, radius: f32) !void;
    pub fn bilateral_filter(img: *Image, sigma_spatial: f32, sigma_range: f32) !void;
    pub fn median_filter(img: *Image, radius: usize) !void;
    
    // Edge detection
    pub fn sobel(img: *const Image) !Image;
    pub fn canny(img: *const Image, low: u8, high: u8) !Image;
    
    // Transforms
    pub fn resize(img: *const Image, width: usize, height: usize, method: ResizeMethod) !Image;
    pub fn rotate(img: *const Image, angle: f32) !Image;
    
    // Color
    pub fn rgb_to_yuv(img: *Image) !Image;
    pub fn adjust_hue(img: *Image, delta: f32) !void;
    pub fn histogram_equalization(img: *Image) !void;
    
    // Video
    pub fn process_video(input: []const u8, output: []const u8, processor: *const ProcessorFn) !void;
};
```

**Final benchmarks:**
| Operation | Image Size | Our Impl | OpenCV | Speedup |
|-----------|-----------|----------|--------|---------|
| Gaussian Blur | 4K | 8ms | 12ms | 1.5x |
| Resize (Lanczos) | 4K→1080p | 15ms | 22ms | 1.47x |
| Sobel | 4K | 6ms | 10ms | 1.67x |
| Bilateral Filter | 4K | 12ms | 18ms | 1.5x |
| Full Pipeline | 4K @60fps | 14.5ms | 25ms | 1.72x |

**Deliverables:**
- Complete image processing library
- SIMD optimizations for all hot paths
- Zero-allocation real-time processing
- Comprehensive benchmarks vs OpenCV
- Full documentation and examples
