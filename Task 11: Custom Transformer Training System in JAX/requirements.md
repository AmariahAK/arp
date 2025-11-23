# Requirements

**Task Created:** November 23, 2025  
**Language:** Python with JAX/Flax  
**Repository:** Custom implementation from scratch

> **Note:** This task requires implementing a production-grade transformer training system from scratch using JAX for automatic differentiation and XLA compilation. The implementation must scale to multi-GPU/TPU setups and achieve competitive performance with established frameworks.

## Prerequisites
- Expert-level Python programming
- Deep understanding of transformer architectures (attention, positional encodings, layer norm)
- JAX and XLA compilation model
- Distributed training (data parallelism, model parallelism, pipeline parallelism)
- Mixed-precision training (FP16/BF16)
- Optimization algorithms (Adam, AdamW, LAMB)
- Understanding of numerical stability in gradients
- Experience with large-scale ML training

## Initial Setup
The developer should provide:
1. Python 3.10 or higher
2. JAX 0.4.20+ with CUDA support (for GPU) or TPU support
3. Flax 0.7.0+ (neural network library for JAX)
4. Optax 0.1.7+ (gradient processing and optimization)
5. Access to GPU (A100/H100) or TPU v3/v4 pods
6. Ability to run distributed training (multi-GPU/TPU)
7. Weights & Biases or TensorBoard for experiment tracking

## Dependencies
- JAX 0.4.20+ with GPU/TPU support
- Flax 0.7.0+ (neural networks in JAX)
- Optax 0.1.7+ (optimizers)
- Chex (testing JAX code)
- Einops (tensor operations)
- Hugging Face Datasets (data loading)
- Weights & Biases (experiment tracking)
- No PyTorch or TensorFlow allowed - pure JAX implementation

## Testing Environment
- Minimum 32GB RAM (64GB recommended)
- NVIDIA A100 40GB GPU (or TPU v3-8)
- Multi-GPU setup for distributed training tests (2-8 GPUs)
- SSD storage for dataset checkpoints
- Linux preferred (Ubuntu 22.04+)

## Performance Requirements
- **Training Speed:** Match or exceed PyTorch Flash Attention on same hardware
- **Memory Efficiency:** Train GPT-2 Medium (345M params) on single A100 40GB
- **Throughput:** \u003e1M tokens/second on 8x A100 for GPT-2 (125M)
- **Scaling Efficiency:** \u003e85% linear scaling from 1 to 8 GPUs
- **Mixed Precision:** BF16 training with \u003c0.1% accuracy loss vs FP32
- **Gradient Checkpointing:** Enable 2x larger models with \u003c30% slowdown
- **Compile Time:** XLA compilation \u003c60 seconds for GPT-2 architecture

## Code Quality Requirements
- Type hints for all functions (checked with mypy)
- Docstrings following NumPy style guide
- Unit tests with \u003e90% coverage (using Chex)
- Deterministic tests (JAX PRNG keys properly managed)
- Proper checkpoint/resume functionality
- Experiment reproducibility (fixed random seeds)
- Clean separation: model / training / data / utils
