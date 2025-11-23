# Task: Build a Production-Grade Transformer Training System from Scratch

**Language:** Python with JAX/Flax  
**Created:** November 23, 2025  
**Difficulty:** EXTREME

## Overview
Implement a complete transformer training framework in JAX that supports GPT-style autoregressive models, BERT-style masked language models, and vision transformers. The system must scale efficiently to multi-GPU/TPU setups, support advanced optimizations (Flash Attention, gradient checkpointing, mixed precision), achieve competitive performance with PyTorch/Hugging Face, and handle models up to 1B+ parameters.

**Key Challenge:** Build everything from scratch using JAX's functional paradigm - no PyTorch/TensorFlow. Must master pure functions, automatic differentiation, XLA compilation, and distributed training in JAX's ecosystem. Production-quality with proper checkpointing, logging, and reproducibility.

---

## TURN 1 — Core Attention Mechanism with Numerical Stability

**Role:** You are a transformer架构 expert who has implemented attention from the 2017 "Attention Is All You Need" paper. You understand the numerical pitfalls (softmax overflow, NaN gradients) and can optimize for both speed and stability.

**Background:** Scaled dot-product attention is the foundation of transformers. Naive implementations suffer from numerical instability (exp overflow) and poor performance (quadratic complexity). Must implement correctly before adding optimizations.

**Reference:** Study:
- "Attention Is All You Need" (Vaswani et al., 2017)
- JAX documentation on automatic differentiation
- Flax NNX module system
- Online Softmax trick for numerical stability

**VERY IMPORTANT:**
- Use safe softmax (subtract max before exp)
- Gradient clipping for stability
- Proper masking for causal/padding attention
- Deterministic behavior (JAX PRNG keys)
- No hidden state mutations (pure functional)
- Type annotations for all functions

**Goal:** Implement numerically stable scaled dot-product attention in JAX.

**Instructions:**

1. **Define attention function:**
```python
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from flax import nnx
from typing import Optional, Tuple
import einops

def scaled_dot_product_attention(
    query: jax.Array,  # [batch, seq_len_q, d_k]
    key: jax.Array,    # [batch, seq_len_k, d_k]
    value: jax.Array,  # [batch, seq_len_v, d_v]
    mask: Optional[jax.Array] = None,  # [batch, seq_len_q, seq_len_k]
    dropout_rate: float = 0.0,
    is_training: bool = True,
    rng: Optional[jax.Array] = None,
) -> jax.Array:
    \"\"\"
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: Query tensor [batch, seq_len_q, d_k]
        key: Key tensor [batch, seq_len_k, d_k]
        value: Value tensor [batch, seq_len_v, d_v]
        mask: Optional attention mask (additive, -inf for masked positions)
        dropout_rate: Dropout probability
        is_training: Whether in training mode
        rng: Random key for dropout
    
    Returns:
        Attention output [batch, seq_len_q, d_v]
    \"\"\"
    d_k = query.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = jnp.einsum('bqd,bkd->bqk', query, key)
    scores = scores / jnp.sqrt(d_k)
    
    # Apply mask (additive)
    if mask is not None:
        scores = scores + mask
    
    # Numerically stable softmax
    # Subtract max for numerical stability (prevents exp overflow)
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores_exp = jnp.exp(scores - scores_max)
    attention_weights = scores_exp / jnp.sum(scores_exp, axis=-1, keepdims=True)
    
    # Check for NaN (can happen with all -inf mask)
    attention_weights = jnp.where(
        jnp.isnan(attention_weights),
        jnp.zeros_like(attention_weights),
        attention_weights
    )
    
    # Apply dropout
    if is_training and dropout_rate > 0.0:
        assert rng is not None, \"RNG key required for dropout\"
        keep_prob = 1.0 - dropout_rate
        mask_dropout = jax.random.bernoulli(rng, keep_prob, attention_weights.shape)
        attention_weights = jnp.where(mask_dropout, attention_weights / keep_prob, 0.0)
    
    # Compute attention output
    output = jnp.einsum('bqk,bkv->bqv', attention_weights, value)
    
    return output


def create_causal_mask(seq_len: int) -> jax.Array:
    \"\"\"Create causal mask for autoregressive attention.\"\"\"
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    # Convert to additive mask (-inf for masked positions)
    mask = jnp.where(mask == 0, -1e9, 0.0)
    return mask


def create_padding_mask(
    lengths: jax.Array,  # [batch]
    max_len: int
) -> jax.Array:
    \"\"\"Create padding mask from sequence lengths.\"\"\"
    # [batch, max_len]
    positions = jnp.arange(max_len)[None, :]
    lengths = lengths[:, None]
    mask = positions < lengths  # True for valid positions
    
    # Convert to additive mask
    mask = jnp.where(mask, 0.0, -1e9)
    return mask[:, None, :]  # [batch, 1, max_len] for broadcasting
```

2. **Multi-head attention module:**
```python
class MultiHeadAttention(nnx.Module):
    \"\"\"Multi-head attention layer.\"\"\"
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        assert d_model % num_heads == 0, \"d_model must be divisible by num_heads\"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate
        
        # Linear projections for Q, K, V
        self.w_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_v = nnx.Linear(d_model, d_model, rngs=rngs)
        
        # Output projection
        self.w_o = nnx.Linear(d_model, d_model, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    
    def __call__(
        self,
        x: jax.Array,  # [batch, seq_len, d_model]
        mask: Optional[jax.Array] = None,
        is_training: bool = True,
    ) -> jax.Array:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and split into heads
        q = self.w_q(x)  # [batch, seq_len, d_model]
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Reshape to [batch, num_heads, seq_len, d_k]
        q = einops.rearrange(
            q, 'b s (h d) -> b h s d', h=self.num_heads
        )
        k = einops.rearrange(
            k, 'b s (h d) -> b h s d', h=self.num_heads
        )
        v = einops.rearrange(
            v, 'b s (h d) -> b h s d', h=self.num_heads
        )
        
        # Apply attention for each head
        attention_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=mask,
            dropout_rate=self.dropout_rate,
            is_training=is_training,
            rng=self.dropout.rngs.dropout() if is_training else None,
        )
        
        # Concatenate heads
        # [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        output = einops.rearrange(
            attention_output, 'b h s d -> b s (h d)'
        )
        
        # Final linear projection
        output = self.w_o(output)
        
        return output
```

3. **Test numerical stability:**
```python
import chex
import pytest

def test_attention_numerical_stability():
    \"\"\"Test attention doesn't produce NaN/Inf with extreme inputs.\"\"\"
    key = jax.random.PRNGKey(0)
    
    # Create inputs with extreme values
    batch, seq_len, d_k = 2, 128, 64
    
    # Very large values (test exp overflow)
    query = jax.random.normal(key, (batch, seq_len, d_k)) * 100
    key_tensor = jax.random.normal(key, (batch, seq_len, d_k)) * 100
    value = jax.random.normal(key, (batch, seq_len, d_k))
    
    output = scaled_dot_product_attention(query, key_tensor, value)
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(output)), \"Output contains NaN\"
    assert not jnp.any(jnp.isinf(output)), \"Output contains Inf\"
    
    # Test with all-masked sequence
    mask = jnp.full((batch, seq_len, seq_len), -1e9)
    output_masked = scaled_dot_product_attention(
        query, key_tensor, value, mask=mask
    )
    
    # Should produce zeros for all-masked (not NaN)
    assert not jnp.any(jnp.isnan(output_masked)), \"Masked output contains NaN\"


def test_causal_masking():
    \"\"\"Test causal mask prevents attending to future positions.\"\"\"
    key = jax.random.PRNGKey(42)
    
    batch, seq_len, d_k = 1, 4, 8
    query = jax.random.normal(key, (batch, seq_len, d_k))
    key_tensor = jax.random.normal(key, (batch, seq_len, d_k))
    value = jnp.eye(seq_len)[None, :, :]  # Identity to see attention pattern
    
    mask = create_causal_mask(seq_len)
    mask = mask[None, :, :]  # Add batch dimension
    
    output = scaled_dot_product_attention(query, key_tensor, value, mask=mask)
    
    # With identity value and causal mask, output[i] should not depend on value[j] for j > i
    # This means output should be lower-triangular weighted sum
    
    # Verify shape
    assert output.shape == (batch, seq_len, seq_len)


def test_gradient_flow():
    \"\"\"Test gradients flow correctly through attention.\"\"\"
    key = jax.random.PRNGKey(123)
    
    batch, seq_len, d_k = 2, 16, 32
    query = jax.random.normal(key, (batch, seq_len, d_k))
    key_tensor = jax.random.normal(key, (batch, seq_len, d_k))
    value = jax.random.normal(key, (batch, seq_len, d_k))
    
    def loss_fn(q, k, v):
        output = scaled_dot_product_attention(q, k, v)
        return jnp.sum(output ** 2)
    
    # Compute gradients
    grad_fn = grad(loss_fn, argnums=(0, 1, 2))
    grads = grad_fn(query, key_tensor, value)
    
    # Check gradients exist and are not NaN
    for g in grads:
        assert g.shape == query.shape or g.shape == key_tensor.shape or g.shape == value.shape
        assert not jnp.any(jnp.isnan(g)), \"Gradient contains NaN\"
        assert not jnp.any(jnp.isinf(g)), \"Gradient contains Inf\"


@pytest.mark.parametrize(\"num_heads\", [1, 4, 8])
def test_multi_head_attention(num_heads):
    \"\"\"Test multi-head attention module.\"\"\"
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(0)
    
    batch, seq_len, d_model = 2, 32, 512
    
    model = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout_rate=0.1,
        rngs=rngs,
    )
    
    x = jax.random.normal(key, (batch, seq_len, d_model))
    
    # Forward pass
    output = model(x, is_training=True)
    
    assert output.shape == (batch, seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))
    
    # Test deterministic mode
    output_eval = model(x, is_training=False)
    # Without dropout, should be deterministic
    # (Note: need to reset RNG to test this properly)
```

4. **Benchmark against JAX reference:**
```python
from flax.linen import attention as flax_attention
import time

def benchmark_attention():
    \"\"\"Compare custom attention with Flax reference.\"\"\"
    key = jax.random.PRNGKey(0)
    
    batch, seq_len, d_model, num_heads = 32, 512, 512, 8
    
    # Custom implementation
    rngs_custom = nnx.Rngs(0)
    model_custom = MultiHeadAttention(d_model, num_heads, rngs=rngs_custom)
    
    x = jax.random.normal(key, (batch, seq_len, d_model))
    
    # Warmup + JIT compilation
    _ = model_custom(x, is_training=False)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = model_custom(x, is_training=False)
    jax.block_until_ready(_)  # Wait for computation
    elapsed_custom = time.perf_counter() - start
    
    print(f\"Custom attention: {elapsed_custom:.3f}s for 100 iterations\")
    print(f\"Throughput: {100 * batch * seq_len / elapsed_custom:.0f} tokens/sec\")
    
    # Expected: Within 2x of optimized Flax implementation
```

**Deliverables:**
- Numerically stable attention implementation
- Multi-head attention module
- Comprehensive tests (stability, masking, gradients)
- Performance benchmark

---

## TURN 2 — Transformer Encoder Block with Layer Normalization

**Instructions:**

Build complete transformer encoder block with residual connections and layer normalization.

**Background:** Transformer blocks stack attention + FFN with residuals and LayerNorm. Order matters: Pre-LN (LN before attention) is now preferred over Post-LN for stability.

**Components:**
- Multi-head self-attention
- Position-wise feed-forward network (2 linear layers + activation)
- Residual connections
- Layer normalization (Pre-LN variant)

**Implement:**
```python
class FeedForward(nnx.Module):
    \"\"\"Position-wise feed-forward network.\"\"\"
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.linear2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    
    def __call__(self, x: jax.Array, is_training: bool = True) -> jax.Array:
        # x: [batch, seq_len, d_model]
        h = self.linear1(x)
        h = jax.nn.gelu(h)  # GELU activation (modern transformers)
        h = self.dropout(h, deterministic=not is_training)
        h = self.linear2(h)
        h = self.dropout(h, deterministic=not is_training)
        return h


class TransformerEncoderBlock(nnx.Module):
    \"\"\"Single transformer encoder block.\"\"\"
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        # Pre-LN: LayerNorm before attention and FFN
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attention = MultiHeadAttention(
            d_model, num_heads, dropout_rate, rngs=rngs
        )
        
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        is_training: bool = True,
    ) -> jax.Array:
        # Pre-LN attention block
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask=mask, is_training=is_training)
        x = self.dropout(x, deterministic=not is_training)
        x = x + residual  # Residual connection
        
        # Pre-LN FFN block
        residual = x
        x = self.ln2(x)
        x = self.ffn(x, is_training=is_training)
        x = x + residual
        
        return x


class TransformerEncoder(nnx.Module):
    \"\"\"Stack of transformer encoder blocks.\"\"\"
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = [
            TransformerEncoderBlock(
                d_model, num_heads, d_ff, dropout_rate, rngs=rngs
            )
            for _ in range(num_layers)
        ]
        
        self.final_ln = nnx.LayerNorm(d_model, rngs=rngs)
    
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        is_training: bool = True,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(x, mask=mask, is_training=is_training)
        
        x = self.final_ln(x)
        return x
```

**Test gradient flow through deep stack:**
```python
def test_deep_transformer_gradients():
    \"\"\"Test gradients flow correctly through 24-layer transformer.\"\"\"
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(0)
    
    # Deep transformer (24 layers like GPT-2 Medium)
    model = TransformerEncoder(
        num_layers=24,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1,
        rngs=rngs,
    )
    
    batch, seq_len, d_model = 2, 128, 512
    x = jax.random.normal(key, (batch, seq_len, d_model))
    
    def loss_fn(model, x):
        output = model(x, is_training=True)
        return jnp.mean(output ** 2)
    
    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model, x)
    
    # Check no gradient explosion/vanishing
    grad_norms = jax.tree_map(lambda g: jnp.linalg.norm(g), grads)
    
    # Print gradient norms for analysis
    print(f\"Loss: {loss:.4f}\")
    # All gradient norms should be reasonable (not 1e-10 or 1e10)
    
    assert jnp.isfinite(loss), \"Loss is not finite\"
```

---

## TURN 3 — Positional Encodings and Embeddings

**Instructions:**

Implement positional encodings (sinusoidal and learned) and token embeddings.

**Background:** Transformers have no notion of position. Must inject positional information. Original paper uses sinusoidal; GPT/BERT use learned. Also need efficient token embeddings.

**Strategies:**
- Sinusoidal positional encoding (Attention is All You Need)
- Learned absolute positional embeddings (GPT-2, BERT)
- Rotary positional embeddings (RoPE) - modern approach
- ALiBi positional biases (attention bias based on distance)

**Implement:**
```python
def sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
) -> jax.Array:
    \"\"\"Generate sinusoidal positional encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    \"\"\"
    position = jnp.arange(seq_len)[:, None]  # [seq_len, 1]
    div_term = jnp.exp(
        jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model)
    )  # [d_model//2]
    
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    
    return pe


class LearnedPositionalEncoding(nnx.Module):
    \"\"\"Learned positional embeddings (GPT-2 style).\"\"\"
    
    def __init__(
        self,
        max_len: int,
        d_model: int,
        *,
        rngs: nnx.Rngs,
    ):
        # Learnable position embeddings
        self.pos_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (max_len, d_model)) * 0.02
        )
        self.max_len = max_len
    
    def __call__(self, seq_len: int) -> jax.Array:
        assert seq_len <= self.max_len, f\"Sequence length {seq_len} exceeds max {self.max_len}\"
        return self.pos_embedding.value[:seq_len, :]


class TokenEmbedding(nnx.Module):
    \"\"\"Token embedding layer with weight tying support.\"\"\"
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        rngs: nnx.Rngs,
    ):
        # Token embeddings
        # Initialize with small values (Xavier/He initialization)
        scale = jnp.sqrt(1.0 / d_model)
        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (vocab_size, d_model)) * scale
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def __call__(self, token_ids: jax.Array) -> jax.Array:
        \"\"\"Look up token embeddings.
        
        Args:
            token_ids: [batch, seq_len] integer token IDs
        
        Returns:
            embeddings: [batch, seq_len, d_model]
        \"\"\"
        return self.embedding.value[token_ids]
    
    def decode(self, hidden_states: jax.Array) -> jax.Array:
        \"\"\"Decode hidden states to logits (weight tying).
        
        Args:
            hidden_states: [batch, seq_len, d_model]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        \"\"\"
        # Logits = hidden @ embedding^T
        return hidden_states @ self.embedding.value.T
```

**Test positional encoding properties:**
```python
def test_sinusoidal_position_encoding():
    \"\"\"Test sinusoidal PE has correct properties.\"\"\"
    seq_len, d_model = 100, 512
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    
    assert pe.shape == (seq_len, d_model)
    
    # Test periodic properties
    # PE should capture relative positions via dot products
    # positions i and j with fixed offset should have similar PE dot product
    
    offset = 5
    similarities = []
    for i in range(seq_len - offset):
        sim = jnp.dot(pe[i], pe[i + offset])
        similarities.append(sim)
    
    similarities = jnp.array(similarities)
    
    # Similarity should be relatively stable across positions
    # (this is a key property of sinusoidal PE)
    std_similarity = jnp.std(similarities)
    print(f\"Std of PE similarities for offset {offset}: {std_similarity:.4f}\")


def test_learned_positional_robustness():
    \"\"\"Test learned PE can extrapolate to longer sequences.\"\"\"
    rngs = nnx.Rngs(42)
    
    max_len, d_model = 512, 256
    pe = LearnedPositionalEncoding(max_len, d_model, rngs=rngs)
    
    # Should work for sequences up to max_len
    encoding_short = pe(128)
    assert encoding_short.shape == (128, d_model)
    
    encoding_max = pe(512)
    assert encoding_max.shape == (512, d_model)
    
    # Should fail for longer sequences
    with pytest.raises(AssertionError):
        _ = pe(513)
```

---

## TURN 4 — Complete GPT Model Architecture

**Instructions:**

Assemble complete GPT-style autoregressive language model.

**Components:**
- Token embeddings + positional encodings
- Transformer encoder stack (decoder-only for GPT)
- Causal masking
- Tied embeddings (share token embedding weights with output projection)

**Implement:**
```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    \"\"\"GPT model configuration.\"\"\"
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 1024
    d_model: int = 768  # GPT-2 Small
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072  # 4 * d_model
    dropout_rate: float = 0.1
    use_tied_embeddings: bool = True


class GPTModel(nnx.Module):
    \"\"\"GPT-style autoregressive transformer.\"\"\"
    
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config
        
        # Token + position embeddings
        self.token_embedding = TokenEmbedding(
            config.vocab_size, config.d_model, rngs=rngs
        )
        self.pos_encoding = LearnedPositionalEncoding(
            config.max_seq_len, config.d_model, rngs=rngs
        )
        
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        
        # Transformer blocks
        self.transformer = TransformerEncoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        
        # Output projection (optionally tied with embeddings)
        if config.use_tied_embeddings:
            # Use token embedding weights for output projection
            self.lm_head = None
        else:
            self.lm_head = nnx.Linear(
                config.d_model, config.vocab_size, rngs=rngs
            )
    
    def __call__(
        self,
        input_ids: jax.Array,  # [batch, seq_len]
        is_training: bool = True,
    ) -> jax.Array:
        \"\"\"Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            is_training: Whether in training mode
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        \"\"\"
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Add positional encodings
        pos_enc = self.pos_encoding(seq_len)  # [seq_len, d_model]
        x = x + pos_enc[None, :, :]  # Broadcast over batch
        
        x = self.dropout(x, deterministic=not is_training)
        
        # Causal mask for autoregressive modeling
        causal_mask = create_causal_mask(seq_len)
        causal_mask = causal_mask[None, :, :]  # Add batch dim
        
        # Transformer forward
        x = self.transformer(x, mask=causal_mask, is_training=is_training)
        
        # Output projection to vocabulary
        if self.config.use_tied_embeddings:
            logits = self.token_embedding.decode(x)
        else:
            logits = self.lm_head(x)
        
        return logits
    
    def generate(
        self,
        input_ids: jax.Array,  # [batch, seq_len]
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng: jax.Array = None,
    ) -> jax.Array:
        \"\"\"Autoregressive generation.
        
        Args:
            input_ids: Prompt tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)
            rng: Random key for sampling
        
        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        \"\"\"
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self(input_ids, is_training=False)
            next_logits = logits[:, -1, :] / temperature  # [batch, vocab_size]
            
            if top_k is not None:
                # Top-k sampling
                top_k_logits, top_k_indices = jax.lax.top_k(next_logits, top_k)
                rng, sample_rng = jax.random.split(rng)
                samples = jax.random.categorical(sample_rng, top_k_logits, axis=-1)
                next_token = top_k_indices[jnp.arange(input_ids.shape[0]), samples]
            else:
                # Greedy decoding
                next_token = jnp.argmax(next_logits, axis=-1)
            
            # Append to sequence
            input_ids = jnp.concatenate([
                input_ids,
                next_token[:, None]
            ], axis=1)
        
        return input_ids
```

**Test GPT model:**
```python
def test_gpt_forward():
    \"\"\"Test GPT model forward pass.\"\"\"
    config = GPTConfig(
        vocab_size=50257,
        max_seq_len=512,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
    )
    
    rngs = nnx.Rngs(0)
    model = GPTModel(config, rngs=rngs)
    
    batch, seq_len = 4, 128
    input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (batch, seq_len), 0, config.vocab_size
    )
    
    logits = model(input_ids, is_training=True)
    
    assert logits.shape == (batch, seq_len, config.vocab_size)
    assert jnp.isfinite(logits).all()


def test_gpt_generation():
    \"\"\"Test autoregressive generation.\"\"\"
    config = GPTConfig(d_model=128, num_layers=2, num_heads=4)
    rngs = nnx.Rngs(42)
    model = GPTModel(config, rngs=rngs)
    
    # Prompt
    prompt = jnp.array([[1, 2, 3, 4]])  # [1, 4]
    
    # Generate 10 new tokens
    rng = jax.random.PRNGKey(0)
    generated = model.generate(
        prompt, max_new_tokens=10, temperature=0.8, top_k=40, rng=rng
    )
    
    assert generated.shape == (1, 14)  # 4 + 10
    assert jnp.all(generated[:, :4] == prompt)  # Prompt unchanged
```

---

---

## TURN 5 — Force Failure: Gradient Exploding in Deep Models

**Instructions:**

Deliberately expose gradient exploding problem in very deep transformers without proper initialization/scaling.

**Ask the AI:**
> "Your 48-layer transformer is experiencing gradient explosion during training. Show a test that demonstrates the failure (gradient norms growing exponentially), explain why Pre-LN helps but isn't sufficient, and implement gradient clipping + proper initialization to fix it."

**Expected failure:**
```python
def test_deep_model_gradient_explosion():
    """Test: Train very deep model without gradient clipping."""
    config = GPTConfig(
        d_model=512,
        num_layers=48,  # Very deep
        num_heads=8,
        d_ff=2048,
    )
    
    rngs = nnx.Rngs(0)
    model = GPTModel(config, rngs=rngs)
    
    # Training data
    batch_size, seq_len = 4, 128
    input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_len), 0, config.vocab_size
    )
    
    # Training loop
    optimizer = optax.adam(1e-3)  # Simple Adam without clipping
    opt_state = optimizer.init(nnx.state(model))
    
    grad_norms = []
    for step in range(100):
        def loss_fn(model):
            logits = model(input_ids, is_training=True)
            # Cross-entropy loss
            labels = jnp.roll(input_ids, -1, axis=1)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1].reshape(-1, config.vocab_size),
                labels[:, :-1].reshape(-1)
            )
            return jnp.mean(loss)
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        
        # Compute global gradient norm
        grad_norm = optax.global_norm(grads)
        grad_norms.append(grad_norm)
        
        # Apply gradients
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)
        
        print(f"Step {step}: Loss={loss:.4f}, Grad Norm={grad_norm:.4f}")
    
    # FAILURE: Gradient norms explode
    # Expected pattern: norms grow from ~1.0 to >1000 within 50 steps
    assert grad_norms[-1] < 100, f"Gradient explosion: norm={grad_norms[-1]:.2f}"
```

**Explanation:**
- Each layer multiplies gradients by weight matrices
- For 48 layers, gradient ≈ W^48 in backprop
- If ||W|| > 1, gradients explode exponentially
- Pre-LN helps (reduces to W^24) but not enough

**Fix 1: Gradient clipping:**
```python
def create_optimizer_with_clipping(
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
) -> optax.GradientTransformation:
    """Create optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),  # Clip gradients
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
```

**Fix 2: Proper initialization (Scaled initialization):**
```python
class TransformerEncoderBlock(nnx.Module):
    """Transformer block with proper scaling."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, *, rngs):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate, rngs=rngs)
        
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        
        # Learnable residual scaling (reduces effective depth)
        self.layer_scale_1 = nnx.Param(jnp.ones(d_model) * 0.1)  # Small init
        self.layer_scale_2 = nnx.Param(jnp.ones(d_model) * 0.1)
    
    def __call__(self, x, mask=None, is_training=True):
        # Scaled residual for attention
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask=mask, is_training=is_training)
        x = self.dropout(x, deterministic=not is_training)
        x = x * self.layer_scale_1.value  # Scale before residual
        x = x + residual
        
        # Scaled residual for FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x, is_training=is_training)
        x = x * self.layer_scale_2.value
        x = x + residual
        
        return x
```

**Test fix:**
```python
def test_gradient_clipping_stabilizes_training():
    """Test gradient clipping prevents explosion."""
    config = GPTConfig(num_layers=48)
    rngs = nnx.Rngs(42)
    model = GPTModel(config, rngs=rngs)
    
    optimizer = create_optimizer_with_clipping(max_grad_norm=1.0)
    opt_state = optimizer.init(nnx.state(model))
    
    grad_norms = []
    for step in range(100):
        # ... training code ...
        grad_norm = optax.global_norm(grads)
        grad_norms.append(grad_norm)
    
    # With clipping, norms should stay bounded
    assert max(grad_norms) < 2.0, f"Clipping failed: max norm={max(grad_norms)}"
    assert np.std(grad_norms) < 0.5, "Grad norms too unstable"
```

---

## TURN 6 — Training Loop with Mixed Precision (BF16)

**Instructions:**

Implement complete training loop with BF16 mixed precision for 2x speedup.

**Background:** BF16 (Brain Float 16) has same exponent range as FP32 but less precision. Faster than FP32 (2x on modern GPUs), more stable than FP16 (no loss scaling needed). JAX makes this easy with `jax.jit` and dtype casting.

**Requirements:**
- Train with BF16 activations and gradients
- Keep master weights in FP32
- No accuracy loss (<0.1% vs FP32)
- 2x+ throughput improvement

**Implement:**
```python
from functools import partial
import optax

def create_train_state(
    model: GPTModel,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
) -> dict:
    """Create training state with optimizer."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    
    opt_state = optimizer.init(nnx.state(model))
    
    return {
        'model': model,
        'optimizer': optimizer,
        'opt_state': opt_state,
        'step': 0,
    }


def compute_loss(
    model: GPTModel,
    input_ids: jax.Array,
    use_bf16: bool = False,
) -> jax.Array:
    """Compute cross-entropy loss."""
    # Cast inputs to BF16 if needed
    if use_bf16:
        # Note: we'll cast inside model forward pass
        pass
    
    # Forward pass
    logits = model(input_ids, is_training=True)  # [batch, seq_len, vocab]
    
    # Shift labels for autoregressive modeling
    labels = jnp.roll(input_ids, -1, axis=1)
    
    # Compute loss (ignore last logit, first label)
    logits_flat = logits[:, :-1].reshape(-1, model.config.vocab_size)
    labels_flat = labels[:, :-1].reshape(-1)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_flat, labels_flat
    )
    
    return jnp.mean(loss)


@partial(jax.jit, static_argnums=(2,))  # JIT compile for speed
def train_step(
    model: GPTModel,
    opt_state: dict,
    use_bf16: bool,
    input_ids: jax.Array,
) -> tuple:
    """Single training step with optional mixed precision."""
    
    # Cast to BF16 if needed
    if use_bf16:
        input_ids = input_ids.astype(jnp.bfloat16)
        # Note: model parameters stay FP32, activations are BF16
    
    def loss_fn(model):
        return compute_loss(model, input_ids, use_bf16)
    
    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Cast gradients back to FP32 for optimizer
    if use_bf16:
        grads = jax.tree_map(lambda g: g.astype(jnp.float32), grads)
    
    # Update parameters
    updates, new_opt_state = model.optimizer.update(grads, opt_state, nnx.state(model))
    nnx.update(model, updates)
    
    return loss, new_opt_state


def train_epoch(
    model: GPTModel,
    opt_state: dict,
    dataloader: Iterator,
    num_steps: int,
    use_bf16: bool = False,
) -> dict:
    """Train for one epoch."""
    total_loss = 0.0
    
    for step in range(num_steps):
        batch = next(dataloader)
        input_ids = batch['input_ids']
        
        loss, opt_state = train_step(model, opt_state, use_bf16, input_ids)
        total_loss += loss
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")
    
    avg_loss = total_loss / num_steps
    return {'loss': avg_loss, 'opt_state': opt_state}
```

**Mixed precision wrapper:**
```python
class MixedPrecisionWrapper:
    """Wrapper to handle BF16 mixed precision training."""
    
    def __init__(self, model: GPTModel):
        self.model = model
        self._param_dtype = jnp.float32  # Master weights in FP32
        self._compute_dtype = jnp.bfloat16  # Activations in BF16
    
    def __call__(self, input_ids, is_training=True):
        # Cast inputs to BF16
        input_ids_bf16 = input_ids.astype(self._compute_dtype)
        
        # Forward pass with BF16 activations
        # (model parameters automatically cast during computation)
        logits = self.model(input_ids_bf16, is_training=is_training)
        
        # Cast output back to FP32
        return logits.astype(jnp.float32)


# Alternative: JAX's built-in mixed precision
from jax.experimental import multihost_utils

def train_step_with_jax_mp(model, opt_state, input_ids):
    """Use JAX's automatic mixed precision."""
    
    # Enable BF16 computation policy
    policy = jax.experimental.compute_dtype_policy.ComputeDtypePolicy(
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )
    
    with jax.experimental.compute_dtype(policy):
        def loss_fn(model):
            logits = model(input_ids, is_training=True)
            # ... compute loss ...
            return loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        # ... update ...
    
    return loss, new_opt_state
```

**Benchmark BF16 vs FP32:**
```python
import time

def benchmark_mixed_precision():
    """Compare FP32 vs BF16 training speed."""
    config = GPTConfig(d_model=768, num_layers=12)
    
    # FP32 model
    rngs_fp32 = nnx.Rngs(0)
    model_fp32 = GPTModel(config, rngs=rngs_fp32)
    
    # BF16 model
    rngs_bf16 = nnx.Rngs(0)
    model_bf16 = MixedPrecisionWrapper(GPTModel(config, rngs=rngs_bf16))
    
    # Dummy data
    batch_size, seq_len = 32, 512
    input_ids = jax.random.randint(
        jax.random.PRNGKey(42), (batch_size, seq_len), 0, config.vocab_size
    )
    
    # Warmup
    for _ in range(10):
        _ = train_step(model_fp32, {}, False, input_ids)
        _ = train_step(model_bf16, {}, True, input_ids)
    
    # Benchmark FP32
    start = time.perf_counter()
    for _ in range(100):
        loss, _ = train_step(model_fp32, {}, False, input_ids)
    jax.block_until_ready(loss)
    time_fp32 = time.perf_counter() - start
    
    # Benchmark BF16
    start = time.perf_counter()
    for _ in range(100):
        loss, _ = train_step(model_bf16, {}, True, input_ids)
    jax.block_until_ready(loss)
    time_bf16 = time.perf_counter() - start
    
    speedup = time_fp32 / time_bf16
    print(f"FP32: {time_fp32:.3f}s")
    print(f"BF16: {time_bf16:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Expected: 1.8-2.2x speedup on A100
    assert speedup > 1.5, f"BF16 speedup too low: {speedup:.2f}x"
```

---

## TURN 7 — Distributed Data Parallel Training (Multi-GPU)

**Instructions:**

Implement data parallelism across multiple GPUs/TPUs using JAX's `pmap`.

**Background:** Data parallelism replicates model on each device, splits batch across devices. Each device computes gradients on its portion, then gradients are averaged (all-reduce). JAX uses `pmap` for single-program-multiple-data (SPMD) execution.

**Requirements:**
- Support 2, 4, 8 GPUs
- Efficient gradient synchronization (all-reduce)
- >85% linear scaling efficiency
- Proper device placement and sharding

**Implement:**
```python
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

def get_device_mesh(num_devices: int = None) -> Mesh:
    """Create device mesh for distributed training."""
    if num_devices is None:
        num_devices = jax.local_device_count()
    
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=('data',))
    
    return mesh


def shard_batch(batch: dict, mesh: Mesh) -> dict:
    """Shard batch across data parallel devices."""
    sharding = NamedSharding(mesh, P('data'))
    
    # Shard each tensor in batch
    sharded_batch = {}
    for key, value in batch.items():
        # Split first dimension across devices
        sharded_batch[key] = jax.device_put(value, sharding)
    
    return sharded_batch


@partial(jax.pmap, axis_name='data')
def distributed_train_step(
    model_state: dict,
    batch: dict,
) -> tuple:
    """Training step with data parallelism."""
    
    def loss_fn(params):
        logits = model.apply(params, batch['input_ids'])
        loss = compute_loss(logits, batch['labels'])
        return loss
    
    # Compute gradients on each device
    loss, grads = jax.value_and_grad(loss_fn)(model_state['params'])
    
    # Average gradients across devices (all-reduce)
    grads = jax.lax.pmean(grads, axis_name='data')
    
    # Update parameters
    updates, new_opt_state = optimizer.update(
        grads, model_state['opt_state'], model_state['params']
    )
    new_params = optax.apply_updates(model_state['params'], updates)
    
    new_model_state = {
        'params': new_params,
        'opt_state': new_opt_state,
    }
    
    # Average loss for logging
    loss = jax.lax.pmean(loss, axis_name='data')
    
    return new_model_state, loss


def train_distributed(
    model: GPTModel,
    train_dataloader: Iterator,
    num_epochs: int = 10,
    num_devices: int = None,
) -> GPTModel:
    """Train model with data parallelism."""
    
    # Create device mesh
    mesh = get_device_mesh(num_devices)
    num_devices = len(mesh.devices)
    
    print(f"Training on {num_devices} devices")
    
    # Replicate model on all devices
    model_state = {
        'params': nnx.state(model),
        'opt_state': create_optimizer().init(nnx.state(model)),
    }
    
   # Replicate state across devices
    model_state = jax.device_put_replicated(model_state, mesh.devices)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0
        
        for batch in train_dataloader:
            # Shard batch across devices
            sharded_batch = shard_batch(batch, mesh)
            
            # Distributed training step
            model_state, loss = distributed_train_step(model_state, sharded_batch)
            
            # Loss is already averaged across devices
            epoch_loss += loss[0]  # Take from first device (all same)
            num_steps += 1
            
            if num_steps % 100 == 0:
                print(f"Epoch {epoch}, Step {num_steps}: Loss = {loss[0]:.4f}")
        
        avg_loss = epoch_loss / num_steps
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
    
    # Get parameters from first device
    final_params = jax.tree_map(lambda x: x[0], model_state['params'])
    nnx.update(model, final_params)
    
    return model
```

**Scaling efficiency test:**
```python
def test_scaling_efficiency():
    """Test linear scaling across multiple GPUs."""
    config = GPTConfig(d_model=512, num_layers=12)
    
    # Dummy dataloader
    def create_dataloader(batch_size, num_batches=100):
        for _ in range(num_batches):
            yield {
                'input_ids': jax.random.randint(
                    jax.random.PRNGKey(_), (batch_size, 512), 0, config.vocab_size
                ),
            }
    
    throughputs = {}
    
    for num_devices in [1, 2, 4, 8]:
        if jax.local_device_count() < num_devices:
            continue
        
        rngs = nnx.Rngs(0)
        model = GPTModel(config, rngs=rngs)
        
        # Batch size per device
        batch_per_device = 16
        global_batch = batch_per_device * num_devices
        
        dataloader = create_dataloader(global_batch, num_batches=50)
        
        start = time.perf_counter()
        train_distributed(model, dataloader, num_epochs=1, num_devices=num_devices)
        elapsed = time.perf_counter() - start
        
        # Tokens per second
        tokens_per_step = global_batch * 512
        steps = 50
        throughput = (tokens_per_step * steps) / elapsed
        
        throughputs[num_devices] = throughput
        print(f"{num_devices} GPU(s): {throughput:.0f} tokens/sec")
    
    # Check scaling efficiency
    if len(throughputs) > 1:
        baseline = throughputs[1]
        for n, tput in throughputs.items():
            efficiency = (tput / baseline) / n
            print(f"{n} GPU scaling efficiency: {efficiency*100:.1f}%")
            
            # Should be >85% efficient
            assert efficiency > 0.85, f"Poor scaling: {efficiency*100:.1f}%"
```

---

## TURN 8 — Flash Attention for Memory Efficiency

**Instructions:**

Implement Flash Attention algorithm for 2x speedup and 50% memory reduction.

**Background:** Standard attention has O(N²) memory (stores full attention matrix). Flash Attention uses tiling and recomputation to achieve O(N) memory with same output. Critical for long sequences.

**Algorithm:**
1. Partition Q, K, V into blocks
2. Compute attention for each block pair in SRAM (on-chip)
3. Never materialize full N×N attention matrix
4. Recompute attention during backward pass

**Reference:** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)

**Implement:**
```python
def flash_attention(
    query: jax.Array,  # [batch, num_heads, seq_len, head_dim]
    key: jax.Array,
    value: jax.Array,
    block_size: int = 128,
) -> jax.Array:
    """
    Flash Attention implementation.
    
    Memory usage: O(N) instead of O(N²)
    Speed: 2-4x faster than standard attention
    """
    batch, num_heads, seq_len, head_dim = query.shape
    
    # Number of blocks
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Output accumulator
    output = jnp.zeros_like(query)
    
    # Normalization terms (for online softmax)
    row_max = jnp.full((batch, num_heads, seq_len), -jnp.inf)
    row_sum = jnp.zeros((batch, num_heads, seq_len))
    
    # Iterate over key/value blocks (outer loop)
    for j in range(num_blocks):
        kj_start = j * block_size
        kj_end = min((j + 1) * block_size, seq_len)
        
        # Load K, V blocks
        k_block = key[:, :, kj_start:kj_end, :]  # [batch, heads, block, head_dim]
        v_block = value[:, :, kj_start:kj_end, :]
        
        # Iterate over query blocks (inner loop)
        for i in range(num_blocks):
            qi_start = i * block_size
            qi_end = min((i + 1) * block_size, seq_len)
            
            # Load Q block
            q_block = query[:, :, qi_start:qi_end, :]  # [batch, heads, block, head_dim]
            
            # Compute attention scores for this block pair
            # qk = Q_i @ K_j^T / sqrt(d_k)
            scores = jnp.einsum('bhqd,bhkd->bhqk', q_block, k_block)
            scores = scores / jnp.sqrt(head_dim)
            
            # Online softmax: update running max and sum
            # (avoids storing full attention matrix)
            block_max = jnp.max(scores, axis=-1)  # [batch, heads, block_i]
            old_max = row_max[:, :, qi_start:qi_end]
            
            new_max = jnp.maximum(old_max, block_max)
            
            # Rescale previous output and sum
            exp_old = jnp.exp(old_max - new_max)
            exp_new = jnp.exp(block_max - new_max)
            
            # Update row sum
            old_sum = row_sum[:, :, qi_start:qi_end]
            new_sum_block = jnp.sum(jnp.exp(scores - new_max[..., None]), axis=-1)
            new_sum = exp_old * old_sum + new_sum_block
            
            # Update output (weighted average)
            # Rescale old output
            output_block = output[:, :, qi_start:qi_end, :]
            output_block = output_block * exp_old[..., None]
            
            # Add new contribution
            attention_weights = jnp.exp(scores - new_max[..., None])
            new_output = jnp.einsum('bhqk,bhkd->bhqd', attention_weights, v_block)
            
            output_block = output_block + new_output
            output = output.at[:, :, qi_start:qi_end, :].set(output_block)
            
            # Update running max and sum
            row_max = row_max.at[:, :, qi_start:qi_end].set(new_max)
            row_sum = row_sum.at[:, :, qi_start:qi_end].set(new_sum)
    
    # Final normalization
    output = output / row_sum[..., None]
    
    return output


class FlashMultiHeadAttention(nnx.Module):
    """Multi-head attention with Flash Attention."""
    
    def __init__(self, d_model, num_heads, dropout_rate=0.0, *, rngs):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_o = nnx.Linear(d_model, d_model, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    
    def __call__(self, x, mask=None, is_training=True):
        batch, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # Flash Attention
        output = flash_attention(q, k, v, block_size=128)
        
        # Reshape back
        output = einops.rearrange(output, 'b h s d -> b s (h d)')
        
        # Output projection
        output = self.w_o(output)
        
        return output
```

**Benchmark Flash vs Standard:**
```python
def benchmark_flash_attention():
    """Compare Flash Attention with standard attention."""
    batch, num_heads, seq_len, head_dim = 16, 12, 2048, 64
    
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, num_heads, seq_len, head_dim))
    k = jax.random.normal(key, (batch, num_heads, seq_len, head_dim))
    v = jax.random.normal(key, (batch, num_heads, seq_len, head_dim))
    
    # Standard attention
    def standard_attention_fn():
        # Materisl full attention matrix
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(head_dim)
        attn = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        return output
    
    # Flash attention
    def flash_attention_fn():
        return flash_attention(q, k, v, block_size=128)
    
    # Warmup
    _ = standard_attention_fn()
    _ = flash_attention_fn()
    
    # Benchmark standard
    start = time.perf_counter()
    for _ in range(20):
        out_std = standard_attention_fn()
    jax.block_until_ready(out_std)
    time_std = time.perf_counter() - start
    
    # Benchmark flash
    start = time.perf_counter()
    for _ in range(20):
        out_flash = flash_attention_fn()
    jax.block_until_ready(out_flash)
    time_flash = time.perf_counter() - start
    
    # Check correctness
    error = jnp.max(jnp.abs(out_std - out_flash))
    print(f"Max error: {error:.6f}")
    assert error < 1e-4, f"Flash Attention error too large: {error}"
    
    # Check speedup
    speedup = time_std / time_flash
    print(f"Standard: {time_std:.3f}s")
    print(f"Flash: {time_flash:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Expected: 2-3x speedup for long sequences
    assert speedup > 1.5, f"Flash Attention speedup too low: {speedup:.2f}x"
```

---

## TURN 9 — Gradient Checkpointing for Large Models

**Instructions:**

Implement gradient checkpointing to trade computation for memory (train 2x larger models).

**Background:** Gradient checkpointing doesn't store all intermediate activations during forward pass. Instead, it recomputes them during backward pass. Trades ~30% more compute for 50-80% less memory.

**Strategy:**
- Checkpoint every N layers (e.g., every 2-4 layers)
- Save only checkpointed activations
- Recompute other activations during backward

**Implement:**
```python
from jax.experimental import checkpoint as jax_checkpoint

class GradientCheckpointedTransformerEncoder(nnx.Module):
    """Transformer with gradient checkpointing."""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.0,
        checkpoint_every: int = 2,  # Checkpoint every N layers
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = [
            TransformerEncoderBlock(
                d_model, num_heads, d_ff, dropout_rate, rngs=rngs
            )
            for _ in range(num_layers)
        ]
        self.checkpoint_every = checkpoint_every
        self.final_ln = nnx.LayerNorm(d_model, rngs=rngs)
    
    def __call__(self, x, mask=None, is_training=True):
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every == 0:
                # Checkpoint this layer - recompute during backward
                def layer_fn(x_input):
                    return layer(x_input, mask=mask, is_training=is_training)
                
                x = jax_checkpoint.checkpoint(layer_fn)(x)
            else:
                # Normal forward pass
                x = layer(x, mask=mask, is_training=is_training)
        
        x = self.final_ln(x)
        return x


# Alternative: Checkpoint entire blocks
def create_checkpointed_layer_fn(layer, mask):
    """Create checkpointed version of layer forward."""
    @jax_checkpoint.checkpoint
    def checkpointed_fn(x):
        return layer(x, mask=mask, is_training=True)
    
    return checkpointed_fn
```

**Memory profiling:**
```python
import tracemalloc

def profile_memory_usage():
    """Compare memory usage with/without checkpointing."""
    config = GPTConfig(
        d_model=1024,
        num_layers=48,
        num_heads=16,
        d_ff=4096,
    )
    
    batch, seq_len = 8, 1024
    input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (batch, seq_len), 0, config.vocab_size
    )
    
    # Model without checkpointing
    rngs_normal = nnx.Rngs(0)
    model_normal = GPTModel(config, rngs=rngs_normal)
    
    tracemalloc.start()
    def loss_fn_normal(model):
        logits = model(input_ids, is_training=True)
        return jnp.mean(logits ** 2)
    
    try:
        loss, grads = nnx.value_and_grad(loss_fn_normal)(model_normal)
        current, peak = tracemalloc.get_traced_memory()
        mem_normal = peak / 1024**3  # GB
        print(f"Without checkpointing: {mem_normal:.2f} GB")
    except Exception as e:
        print(f"Without checkpointing: OOM ({e})")
        mem_normal = float('inf')
    tracemalloc.stop()
    
    # Model with checkpointing
    rngs_ckpt = nnx.Rngs(0)
    model_ckpt = GPTModel(
        config,
        rngs=rngs_ckpt,
        use_checkpointing=True,
        checkpoint_every=2,
    )
    
    tracemalloc.start()
    def loss_fn_ckpt(model):
        logits = model(input_ids, is_training=True)
        return jnp.mean(logits ** 2)
    
    loss, grads = nnx.value_and_grad(loss_fn_ckpt)(model_ckpt)
    current, peak = tracemalloc.get_traced_memory()
    mem_ckpt = peak / 1024**3  # GB
    print(f"With checkpointing: {mem_ckpt:.2f} GB")
    tracemalloc.stop()
    
    if mem_normal != float('inf'):
        reduction = (mem_normal - mem_ckpt) / mem_normal * 100
        print(f"Memory reduction: {reduction:.1f}%")
        assert reduction > 40, f"Checkpointing not effective: {reduction:.1f}%"
```

## TURN 10 — Advanced Optimizers (AdamW, LAMB, Lion)

**Instructions:**

Implement and compare advanced optimizers for transformer training.

**Background:** Standard Adam has weight decay issues. AdamW decouples weight decay from gradient updates. LAMB enables large batch training. Lion is a new optimizer (2023) showing promise.

**Optimizers to implement:**
- AdamW: Decoupled weight decay
- LAMB: Layer-wise adaptive large batch training
- Lion: Sign-based optimizer with momentum

**Implement:**
```python
import optax
from typing import NamedTuple, Any

class OptimizerFactory:
    """Factory for creating different optimizers."""
    
    @staticmethod
    def create_adamw(
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
    ) -> optax.GradientTransformation:
        """Create AdamW optimizer with gradient clipping."""
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(
                learning_rate=learning_rate,
                b1=beta1,
                b2=beta2,
                eps=epsilon,
                weight_decay=weight_decay,
            ),
        )
    
    @staticmethod
    def create_lamb(
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-6,
        max_grad_norm: float = 1.0,
    ) -> optax.GradientTransformation:
        """Create LAMB optimizer for large batch training."""
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.lamb(
                learning_rate=learning_rate,
                b1=beta1,
                b2=beta2,
                eps=epsilon,
                weight_decay=weight_decay,
            ),
        )
    
    @staticmethod
    def create_lion(
        learning_rate: float = 1e-4,  # Lion uses smaller LR
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        max_grad_norm: float = 1.0,
    ) -> optax.GradientTransformation:
        """Create Lion optimizer."""
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.lion(
                learning_rate=learning_rate,
                b1=beta1,
                b2=beta2,
                weight_decay=weight_decay,
            ),
        )


# Custom implementation of Lion for educational purposes
class LionState(NamedTuple):
    """State for Lion optimizer."""
    count: jax.Array
    momentum: Any  # Pytree of momentum


def lion_optimizer(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """
    Lion optimizer implementation.
    
    Lion uses sign of interpolated gradient for updates.
    Update = sign(beta1 * m + (1-beta1) * g)
    """
    
    def init_fn(params):
        momentum = jax.tree_map(jnp.zeros_like, params)
        return LionState(count=jnp.zeros([], jnp.int32), momentum=momentum)
    
    def update_fn(updates, state, params=None):
        # Interpolate: c = beta1 * m + (1 - beta1) * g
        interpolated = jax.tree_map(
            lambda m, g: beta1 * m + (1 - beta1) * g,
            state.momentum,
            updates,
        )
        
        # Update = -lr * sign(c)
        new_updates = jax.tree_map(
            lambda c: -learning_rate * jnp.sign(c),
            interpolated,
        )
        
        # Add weight decay if params provided
        if params is not None and weight_decay > 0:
            new_updates = jax.tree_map(
                lambda u, p: u - learning_rate * weight_decay * p,
                new_updates,
                params,
            )
        
        # Update momentum: m' = beta2 * m + (1 - beta2) * g
        new_momentum = jax.tree_map(
            lambda m, g: beta2 * m + (1 - beta2) * g,
            state.momentum,
            updates,
        )
        
        new_state = LionState(
            count=state.count + 1,
            momentum=new_momentum,
        )
        
        return new_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)
```

**Optimizer comparison experiment:**
```python
def compare_optimizers():
    """Compare different optimizers on same task."""
    config = GPTConfig(
        vocab_size=10000,  # Smaller for faster testing
        d_model=512,
        num_layers=6,
        num_heads=8,
    )
    
    optimizers = {
        'AdamW': OptimizerFactory.create_adamw(learning_rate=1e-3),
        'LAMB': OptimizerFactory.create_lamb(learning_rate=1e-3),
        'Lion': OptimizerFactory.create_lion(learning_rate=1e-4),
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\n=== Testing {name} ===")
        
        # Fresh model for each optimizer
        rngs = nnx.Rngs(42)  # Same seed for fair comparison
        model = GPTModel(config, rngs=rngs)
        
        # Train for fixed steps
        losses = train_with_optimizer(
            model,
            optimizer,
            num_steps=1000,
            batch_size=32,
            seq_len=128,
        )
        
        results[name] = {
            'final_loss': losses[-1],
            'convergence_speed': np.argmin(losses),  # Step where minimum reached
            'stability': np.std(losses[-100:]),  # Variance in final 100 steps
        }
        
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Convergence at step: {results[name]['convergence_speed']}")
        print(f"Stability (std): {results[name]['stability']:.4f}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    for name in optimizers:
        plt.plot(results[name]['losses'], label=name)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Optimizer Comparison')
    plt.savefig('optimizer_comparison.png')
    
    return results
```

**Expected results:**
- AdamW: Most stable, good baseline
- LAMB: Better for large batches (batch ≥ 1024)
- Lion: Faster convergence, lower memory (no second moment)

---

## TURN 11 — Learning Rate Scheduling and Warmup

**Instructions:**

Implement learning rate schedules with warmup for stable training.

**Background:** Learning rate is critical for transformer training. Too high → divergence, too low → slow convergence. Warmup prevents early instability. Cosine decay maintains performance.

**Schedules to implement:**
- Linear warmup
- Cosine decay with warmup
- Inverse square root decay
- Constant with warmup

**Implement:**
```python
def create_learning_rate_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    schedule_type: str = 'cosine',
    end_lr_factor: float = 0.1,
) -> optax.Schedule:
    """
    Create learning rate schedule with warmup.
    
    Args:
        peak_lr: Maximum learning rate after warmup
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        schedule_type: 'cosine', 'linear', 'inverse_sqrt', 'constant'
        end_lr_factor: Final LR as fraction of peak_lr (for cosine)
    
    Returns:
        Learning rate schedule function
    """
    
    if schedule_type == 'cosine':
        # Cosine decay after warmup
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=peak_lr * end_lr_factor,
        )
    
    elif schedule_type == 'linear':
        # Linear warmup + linear decay
        schedule = optax.piecewise_constant_schedule(
            init_value=0.0,
            boundaries_and_scales={
                warmup_steps: peak_lr / (warmup_steps or 1),
                total_steps: -peak_lr / (total_steps - warmup_steps),
            }
        )
    
    elif schedule_type == 'inverse_sqrt':
        # Transformer paper schedule: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        def schedule_fn(step):
            step = jnp.maximum(step, 1)  # Avoid division by zero
            warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
            decay_factor = jnp.sqrt(warmup_steps / jnp.maximum(step, warmup_steps))
            return peak_lr * warmup_factor * decay_factor
        
        schedule = schedule_fn
    
    elif schedule_type == 'constant':
        # Linear warmup then constant
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=peak_lr,
                    transition_steps=warmup_steps,
                ),
                optax.constant_schedule(peak_lr),
            ],
            boundaries=[warmup_steps],
        )
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return schedule


# Visualization
def visualize_schedules():
    """Plot different learning rate schedules."""
    peak_lr = 1e-3
    warmup_steps = 1000
    total_steps = 10000
    
    schedules = {
        'Cosine': create_learning_rate_schedule(peak_lr, warmup_steps, total_steps, 'cosine'),
        'Linear': create_learning_rate_schedule(peak_lr, warmup_steps, total_steps, 'linear'),
        'InvSqrt': create_learning_rate_schedule(peak_lr, warmup_steps, total_steps, 'inverse_sqrt'),
        'Constant': create_learning_rate_schedule(peak_lr, warmup_steps, total_steps, 'constant'),
    }
    
    import matplotlib.pyplot as plt
    
    steps = np.arange(total_steps)
    
    plt.figure(figsize=(12, 6))
    for name, schedule_fn in schedules.items():
        lrs = [schedule_fn(s) for s in steps]
        plt.plot(steps, lrs, label=name)
    
    plt.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5, label='End of Warmup')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Schedules')
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_schedules.png')
```

**Integration with optimizer:**
```python
def create_optimizer_with_schedule(
    peak_lr: float = 1e-3,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    schedule_type: str = 'cosine',
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Create optimizer with LR schedule."""
    
    lr_schedule = create_learning_rate_schedule(
        peak_lr, warmup_steps, total_steps, schedule_type
    )
    
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
        ),
    )
```

**Test warmup importance:**
```python
def test_warmup_necessity():
    """Show that warmup prevents early training failure."""
    config = GPTConfig(num_layers=12)
    
    results = {}
    
    for warmup_steps in [0, 100, 500, 1000]:
        rngs = nnx.Rngs(42)
        model = GPTModel(config, rngs=rngs)
        
        optimizer = create_optimizer_with_schedule(
            peak_lr=1e-3,
            warmup_steps=warmup_steps,
            total_steps=5000,
        )
        
        losses = train_model(model, optimizer, num_steps=5000)
        
        results[warmup_steps] = {
            'min_loss': np.min(losses),
            'diverged': np.isnan(losses).any() or np.max(losses) > 100,
        }
        
        print(f"Warmup={warmup_steps}: Min Loss={results[warmup_steps]['min_loss']:.4f}, "
              f"Diverged={results[warmup_steps]['diverged']}")
    
    # Expected: warmup_steps=0 might diverge or converge slowly
    # warmup_steps ≥ 500 should converge reliably
```

---

## TURN 12 — Gradient Accumulation for Large Batches

**Instructions:**

Implement gradient accumulation to simulate large batch sizes on limited memory.

**Background:** Large batches (2048-4096) improve training stability and throughput. But memory limits batch size. Gradient accumulation accumulates gradients over N micro-batches before updating.

**Effective batch = micro_batch_size × accumulation_steps × num_gpus**

**Implement:**
```python
@jax.jit
def train_step_with_accumulation(
    model: GPTModel,
    opt_state: dict,
    batches: list,  # List of micro-batches
    accumulation_steps: int,
) -> tuple:
    """Training step with gradient accumulation."""
    
    def compute_loss_and_grads(model, batch):
        def loss_fn(model):
            logits = model(batch['input_ids'], is_training=True)
            return compute_loss(logits, batch)
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        return loss, grads
    
    # Accumulate gradients over micro-batches
    total_loss = 0.0
    accumulated_grads = None
    
    for i in range(accumulation_steps):
        loss, grads = compute_loss_and_grads(model, batches[i])
        
        # Scale gradients by accumulation steps
        grads = jax.tree_map(lambda g: g / accumulation_steps, grads)
        
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree_map(
                lambda acc, g: acc + g,
                accumulated_grads,
                grads,
            )
        
        total_loss += loss / accumulation_steps
    
    # Single optimizer step with accumulated gradients
    updates, new_opt_state = model.optimizer.update(
        accumulated_grads, opt_state, nnx.state(model)
    )
    nnx.update(model, updates)
    
    return total_loss, new_opt_state


# Training loop with accumulation
def train_with_gradient_accumulation(
    model: GPTModel,
    dataloader: Iterator,
    optimizer: optax.GradientTransformation,
    accumulation_steps: int = 4,
    num_steps: int = 1000,
):
    """Train with gradient accumulation."""
    
    opt_state = optimizer.init(nnx.state(model))
    
    losses = []
    
    for step in range(num_steps):
        # Collect micro-batches
        micro_batches = []
        for _ in range(accumulation_steps):
            batch = next(dataloader)
            micro_batches.append(batch)
        
        # Accumulated gradient step
        loss, opt_state = train_step_with_accumulation(
            model, opt_state, micro_batches, accumulation_steps
        )
        
        losses.append(float(loss))
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")
    
    return losses
```

**Memory-efficient implementation:**
```python
def train_step_efficient_accumulation(
    model, opt_state, dataloader, accumulation_steps
):
    """Memory-efficient gradient accumulation using scan."""
    
    def accumulate_fn(accumulated_grads, batch):
        """Accumulate gradients for one micro-batch."""
        def loss_fn(model):
            logits = model(batch['input_ids'], is_training=True)
            return compute_loss(logits, batch)
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        
        # Scale and accumulate
        grads = jax.tree_map(lambda g: g / accumulation_steps, grads)
        
        if accumulated_grads is None:
            return grads, loss
        else:
            new_grads = jax.tree_map(
                lambda a, g: a + g, accumulated_grads, grads
            )
            return new_grads, loss
    
    # Use lax.scan for efficient accumulation
    batches = [next(dataloader) for _ in range(accumulation_steps)]
    
    final_grads, losses = jax.lax.scan(
        accumulate_fn,
        init=None,
        xs=batches,
    )
    
    # Apply accumulated gradients
    updates, new_opt_state = model.optimizer.update(
        final_grads, opt_state, nnx.state(model)
    )
    nnx.update(model, updates)
    
    avg_loss = jnp.mean(jnp.array(losses))
    
    return avg_loss, new_opt_state
```

**Test gradient accumulation equivalence:**
```python
def test_gradient_accumulation_equivalence():
    """Verify gradient accumulation equals large batch."""
    config = GPTConfig(d_model=256, num_layers=4)
    
    # Model 1: Large batch (if memory allows)
    rngs1 = nnx.Rngs(0)
    model1 = GPTModel(config, rngs=rngs1)
    
    # Model 2: Small batch with accumulation
    rngs2 = nnx.Rngs(0)  # Same seed
    model2 = GPTModel(config, rngs=rngs2)
    
    # Same data
    large_batch_size = 64
    small_batch_size = 16
    accumulation_steps = large_batch_size // small_batch_size
    
    key = jax.random.PRNGKey(42)
    large_batch = jax.random.randint(
        key, (large_batch_size, 128), 0, config.vocab_size
    )
    
    # Split into micro-batches
    micro_batches = [
        large_batch[i*small_batch_size:(i+1)*small_batch_size]
        for i in range(accumulation_steps)
    ]
    
    # Train model1 with large batch
    def loss_fn1(model):
        logits = model(large_batch, is_training=False)  # deterministic
        return jnp.mean(logits ** 2)
    
    loss1, grads1 = nnx.value_and_grad(loss_fn1)(model1)
    
    # Train model2 with accumulated micro-batches
    accumulated_grads = None
    for micro_batch in micro_batches:
        def loss_fn2(model):
            logits = model(micro_batch, is_training=False)
            return jnp.mean(logits ** 2)
        
        _, grads = nnx.value_and_grad(loss_fn2)(model2)
        grads = jax.tree_map(lambda g: g / accumulation_steps, grads)
        
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree_map(
                lambda a, g: a + g, accumulated_grads, grads
            )
    
    # Compare gradients
    grad_diff = jax.tree_map(
        lambda g1, g2: jnp.max(jnp.abs(g1 - g2)),
        grads1,
        accumulated_grads,
    )
    
    max_diff = max(jax.tree_leaves(grad_diff))
    print(f"Max gradient difference: {max_diff:.8f}")
    
    # Should be nearly identical (within numerical precision)
    assert max_diff < 1e-5, f"Accumulation not equivalent: diff={max_diff}"
```

---

## TURN 13 — Efficient Data Loading Pipeline

**Instructions:**

Build high-throughput data pipeline that doesn't bottleneck training.

**Background:** Data loading can be a bottleneck (CPU-bound). Need prefetching, batching, and efficient tokenization. For transformers, must handle variable-length sequences.

**Requirements:**
- Prefetch batches to overlap with training
- Efficient tokenization (use pre-tokenized data)
- Handle variable lengths with padding
- Shuffle for better generalization
- Multi-worker data loading

**Implement:**
```python
from datasets import load_dataset
import threading
from queue import Queue
from typing import Iterator, Dict

class DataLoader:
    """Efficient data loader for transformer training."""
    
    def __init__(
        self,
        dataset_name: str,
        split: str,
        batch_size: int,
        seq_len: int,
        tokenizer=None,
        shuffle: bool = True,
        prefetch_size: int = 2,
        num_workers: int = 4,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, split=split)
        
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42)
        
        self.tokenizer = tokenizer
        
        # Prefetch queue
        self.prefetch_size = prefetch_size
        self.batch_queue = Queue(maxsize=prefetch_size)
        
        # Start prefetching thread
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
        )
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Background thread to prefetch batches."""
        batch = []
        
        for example in self.dataset:
            # Tokenize if needed
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(example['text'])
            else:
                tokens = example['input_ids']  # Pre-tokenized
            
            # Truncate or pad to seq_len
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            else:
                tokens = tokens + [0] * (self.seq_len - len(tokens))
            
            batch.append(tokens)
            
            if len(batch) == self.batch_size:
                # Convert to JAX array
                batch_array = jnp.array(batch, dtype=jnp.int32)
                self.batch_queue.put({'input_ids': batch_array})
                batch = []
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict[str, jax.Array]:
        batch = self.batch_queue.get()
        if batch is None:
            raise StopIteration
        return batch


# More efficient: use JAX's built-in data loading
class JAXDataLoader:
    """JAX-native data loader with vectorized operations."""
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Pre-process entire dataset into JAX arrays
        print("Pre-processing dataset...")
        all_tokens = []
        for example in dataset:
            tokens = example['input_ids'][:seq_len]
            if len(tokens) < seq_len:
                tokens = tokens + [0] * (seq_len - len(tokens))
            all_tokens.append(tokens)
        
        self.data = jnp.array(all_tokens, dtype=jnp.int32)
        print(f"Dataset loaded: {self.data.shape}")
        
        self.num_samples = len(self.data)
        self.num_batches = self.num_samples // batch_size
        
        if self.drop_last:
            self.data = self.data[:self.num_batches * batch_size]
    
    def __iter__(self):
        # Shuffle if requested
        if self.shuffle:
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
            perm = jax.random.permutation(key, self.num_samples)
            data = self.data[perm]
        else:
            data = self.data
        
        # Yield batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            if len(batch) == self.batch_size:  # Drop last if smaller
                yield {'input_ids': batch}
    
    def __len__(self):
        return self.num_batches


# Prefetching wrapper
class PrefetchDataLoader:
    """Wrapper to prefetch batches on CPU while GPU trains."""
    
    def __init__(self, dataloader: Iterator, prefetch_size: int = 2):
        self.dataloader = dataloader
        self.prefetch_size = prefetch_size
    
    def __iter__(self):
        # Use JAX's prefetch mechanism
        return jax.device_put_prefetch(
            iter(self.dataloader),
            size=self.prefetch_size,
        )
```

**Benchmark data loading:**
```python
import time

def benchmark_data_loading():
    """Benchmark different data loading strategies."""
    
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    batch_size = 32
    seq_len = 512
    num_batches = 100
    
    # Strategy 1: Naive (no prefetching)
    loader1 = JAXDataLoader(dataset, batch_size, seq_len, shuffle=False)
    
    start = time.perf_counter()
    for i, batch in enumerate(loader1):
        if i >= num_batches:
            break
        _ = batch['input_ids']  # Just access
    time1 = time.perf_counter() - start
    
    # Strategy 2: With prefetching
    loader2 = PrefetchDataLoader(
        JAXDataLoader(dataset, batch_size, seq_len, shuffle=False),
        prefetch_size=4,
    )
    
    start = time.perf_counter()
    for i, batch in enumerate(loader2):
        if i >= num_batches:
            break
        _ = batch['input_ids']
    time2 = time.perf_counter() - start
    
    print(f"Without prefetch: {time1:.3f}s ({num_batches/time1:.1f} batches/sec)")
    print(f"With prefetch: {time2:.3f}s ({num_batches/time2:.1f} batches/sec)")
    print(f"Speedup: {time1/time2:.2f}x")
    
    # Expected: 1.5-2x speedup with prefetching
```

---

## TURN 14 — Checkpointing and Model Resume

**Instructions:**

Implement robust checkpointing to save/resume training.

**Background:** Training can be interrupted (OOM, hardware failure, preemption). Must save model weights, optimizer state, training step, RNG state. Resume exactly where left off.

**Requirements:**
- Save full training state periodically
- Async checkpoint writing (don't block training)
- Keep N most recent checkpoints
- Atomic writes (don't corrupt on crash)
- Resume from checkpoint seamlessly

**Implement:**
```python
import orbax.checkpoint as ocp
from pathlib import Path
import json

class CheckpointManager:
    """Manage model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 3,
        save_interval_steps: int = 1000,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.save_interval_steps = save_interval_steps
        
        # Orbax checkpoint manager
        self.checkpointer = ocp.PyTreeCheckpointer()
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
        )
        
        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            self.checkpointer,
            self.options,
        )
    
    def save(
        self,
        step: int,
        model: GPTModel,
        optimizer_state: dict,
        rng_state: jax.Array,
        metrics: dict = None,
    ) -> None:
        """Save checkpoint."""
        
        checkpoint = {
            'step': step,
            'model': nnx.state(model),
            'optimizer': optimizer_state,
            'rng': rng_state,
            'config': model.config.__dict__,
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Async save (non-blocking)
        self.manager.save(step, checkpoint)
        
        print(f"Saved checkpoint at step {step}")
    
    def restore(
        self,
        step: int = None,
    ) -> dict:
        """
        Restore checkpoint.
        
        Args:
            step: Specific step to restore. If None, restores latest.
        
        Returns:
            Checkpoint dictionary
        """
        if step is None:
            step = self.manager.latest_step()
        
        if step is None:
            raise ValueError("No checkpoints found")
        
        checkpoint = self.manager.restore(step)
        print(f"Restored checkpoint from step {step}")
        
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return self.manager.all_steps()


# Integration with training loop
def train_with_checkpointing(
    model: GPTModel,
    dataloader: Iterator,
    num_steps: int = 100000,
    checkpoint_dir: str = './checkpoints',
    resume_from: str = None,
):
    """Train with automatic checkpointing and resume."""
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir,
        max_to_keep=3,
        save_interval_steps=1000,
    )
    
    # Create optimizer
    optimizer = create_optimizer_with_schedule(
        peak_lr=1e-3,
        warmup_steps=1000,
        total_steps=num_steps,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from is not None:
        checkpoint = ckpt_manager.restore()
        
        # Restore model state
        nnx.update(model, checkpoint['model'])
        
        # Restore optimizer state
        opt_state = checkpoint['optimizer']
        
        # Restore RNG
        rng = checkpoint['rng']
        
        # Resume from checkpoint step
        start_step = checkpoint['step'] + 1
        
        print(f"Resuming training from step {start_step}")
    else:
        opt_state = optimizer.init(nnx.state(model))
        rng = jax.random.PRNGKey(42)
    
    # Training loop
    losses = []
    
    for step in range(start_step, num_steps):
        # Get batch
        batch = next(dataloader)
        
        # Training step
        rng, step_rng = jax.random.split(rng)
        loss, opt_state = train_step(
            model, opt_state, batch, step_rng
        )
        
        losses.append(float(loss))
        
        # Logging
        if step % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            print(f"Step {step}/{num_steps}: Loss = {avg_loss:.4f}")
        
        # Checkpointing
        if step % ckpt_manager.save_interval_steps == 0:
            metrics = {
                'loss': float(loss),
                'avg_loss_100': np.mean(losses[-100:]) if len(losses) >= 100 else float(loss),
            }
            
            ckpt_manager.save(
                step=step,
                model=model,
                optimizer_state=opt_state,
                rng_state=rng,
                metrics=metrics,
            )
    
    # Final checkpoint
    ckpt_manager.save(
        step=num_steps,
        model=model,
        optimizer_state=opt_state,
        rng_state=rng,
        metrics={'final_loss': float(loss)},
    )
    
    return model, losses


# Test resume functionality
def test_checkpoint_resume():
    """Test that training resumes correctly."""
    config = GPTConfig(d_model=256, num_layers=4)
    checkpoint_dir = '/tmp/test_checkpoints'
    
    # Train for 500 steps
    rngs1 = nnx.Rngs(0)
    model1 = GPTModel(config, rngs=rngs1)
    
    dataloader1 = create_dummy_dataloader(batch_size=16)
    
    model1, losses1 = train_with_checkpointing(
        model1,
        dataloader1,
        num_steps=500,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Resume and train for 500 more steps
    rngs2 = nnx.Rngs(0)  # Same seed (will be overwritten by checkpoint)
    model2 = GPTModel(config, rngs=rngs2)
    
    dataloader2 = create_dummy_dataloader(batch_size=16)
    
    model2, losses2 = train_with_checkpointing(
        model2,
        dataloader2,
        num_steps=1000,
        checkpoint_dir=checkpoint_dir,
        resume_from=checkpoint_dir,
    )
    
    # Verify losses match
    # (losses2 should start from where losses1 ended)
    print(f"Original loss at step 500: {losses1[-1]:.4f}")
    print(f"Resumed loss at step 501: {losses2[501]:.4f}")
```

---

## TURN 15 — Monitoring, Logging, and Experiment Tracking

**Instructions:**

Integrate Weights & Biases (W&B) for experiment tracking and visualization.

**Background:** Track metrics, hyperparameters, model artifacts. Compare experiments. Visualize training dynamics. Essential for research and production.

**Integrate:**
- W&B logging
- TensorBoard (alternative)
- Custom metrics tracking
- Hyperparameter logging
- Model artifact saving

**Implement:**
```python
import wandb
from typing import Optional

class ExperimentTracker:
    """Wrapper for experiment tracking (W&B or TensorBoard)."""
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: dict,
        backend: str = 'wandb',  # 'wandb' or 'tensorboard'
    ):
        self.backend = backend
        
        if backend == 'wandb':
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
            )
            self.logger = wandb
        
        elif backend == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(log_dir=f'./runs/{experiment_name}')
            self.step = 0
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics."""
        if self.backend == 'wandb':
            wandb.log(metrics, step=step)
        
        elif self.backend == 'tensorboard':
            if step is None:
                step = self.step
                self.step += 1
            
            for key, value in metrics.items():
                self.logger.add_scalar(key, value, step)
    
    def log_histogram(self, name: str, values: jax.Array, step: int):
        """Log histogram of values."""
        if self.backend == 'wandb':
            wandb.log({name: wandb.Histogram(np.array(values))}, step=step)
        
        elif self.backend == 'tensorboard':
            self.logger.add_histogram(name, np.array(values), step)
    
    def log_text(self, name: str, text: str, step: int):
        """Log generated text."""
        if self.backend == 'wandb':
            wandb.log({name: wandb.Html(f'<pre>{text}</pre>')}, step=step)
        
        elif self.backend == 'tensorboard':
            self.logger.add_text(name, text, step)
    
    def save_model(self, model: GPTModel, name: str):
        """Save model as artifact."""
        if self.backend == 'wandb':
            # Save model checkpoint
            artifact = wandb.Artifact(name, type='model')
            
            # Save to temp file then upload
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
                # Save model state
                import pickle
                pickle.dump(nnx.state(model), f)
                f.flush()
                
                artifact.add_file(f.name)
            
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish logging."""
        if self.backend == 'wandb':
            wandb.finish()
        elif self.backend == 'tensorboard':
            self.logger.close()


# Training with logging
def train_with_logging(
    model: GPTModel,
    dataloader: Iterator,
    num_steps: int = 10000,
    project_name: str = 'gpt-training',
    experiment_name: str = 'gpt2-small',
):
    """Train with comprehensive logging."""
    
    # Initialize tracker
    tracker = ExperimentTracker(
        project_name=project_name,
        experiment_name=experiment_name,
        config={
            'model': model.config.__dict__,
            'batch_size': 32,
            'seq_len': 1024,
            'num_steps': num_steps,
            'optimizer': 'adamw',
            'lr': 1e-3,
        },
        backend='wandb',
    )
    
    # Create optimizer
    optimizer = create_optimizer_with_schedule(
        peak_lr=1e-3,
        warmup_steps=1000,
        total_steps=num_steps,
    )
    opt_state = optimizer.init(nnx.state(model))
    
    # Training loop
    for step in range(num_steps):
        batch = next(dataloader)
        
        # Training step
        loss, opt_state = train_step(model, opt_state, batch)
        
        # Log metrics
        metrics = {'train/loss': float(loss)}
        
        # Log learning rate
        if hasattr(optimizer, '_learning_rate'):
            lr = optimizer._learning_rate(step)
            metrics['train/learning_rate'] = float(lr)
        
        tracker.log(metrics, step=step)
        
        # Periodic detailed logging
        if step % 500 == 0:
            # Gradient norms
            def get_grads(model):
                def loss_fn(model):
                    logits = model(batch['input_ids'], is_training=True)
                    return compute_loss(logits, batch)
                _, grads = nnx.value_and_grad(loss_fn)(model)
                return grads
            
            grads = get_grads(model)
            grad_norm = optax.global_norm(grads)
            
            tracker.log({
                'train/grad_norm': float(grad_norm),
            }, step=step)
            
            # Parameter norms
            param_norm = jnp.sqrt(sum(
                jnp.sum(p ** 2)
                for p in jax.tree_leaves(nnx.state(model))
            ))
            
            tracker.log({
                'model/param_norm': float(param_norm),
            }, step=step)
        
        # Generate sample text
        if step % 1000 == 0:
            sample_text = generate_sample(model, tokenizer, max_length=50)
            tracker.log_text('samples/generated_text', sample_text, step)
            print(f"\nStep {step} sample:\n{sample_text}\n")
        
        # Log parameter histograms
        if step % 5000 == 0:
            for name, param in nnx.state(model).items():
                if isinstance(param, jax.Array):
                    tracker.log_histogram(
                        f'params/{name}',
                        param.flatten(),
                        step,
                    )
        
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}: Loss = {loss:.4f}")
    
    # Save final model
    tracker.save_model(model, f'{experiment_name}_final')
    
    # Finish
    tracker.finish()
    
    return model
```

---

## TURN 16 — Evaluation Metrics and Perplexity

**Instructions:**

Implement proper evaluation metrics for language models.

**Background:** Perplexity is standard metric for LMs. Lower = better. perplexity = exp(cross_entropy_loss). Also track other metrics: accuracy, token-level metrics.

**Metrics to implement:**
- Perplexity
- Top-k accuracy
- Bits per character/byte
- Token-level accuracy

**Implement:**
```python
def compute_perplexity(
    model: GPTModel,
    eval_dataloader: Iterator,
    num_batches: int = None,
) -> dict:
    """
    Compute perplexity on evaluation set.
    
    Perplexity = exp(average cross-entropy loss)
    """
    
    total_loss = 0.0
    total_tokens = 0
    num_evaluated = 0
    
    for i, batch in enumerate(eval_dataloader):
        if num_batches is not None and i >= num_batches:
            break
        
        # Forward pass (evaluation mode)
        logits = model(batch['input_ids'], is_training=False)
        
        # Compute loss
        labels = jnp.roll(batch['input_ids'], -1, axis=1)
        
        # Mask padding tokens (assume 0 is padding)
        mask = (batch['input_ids'] != 0).astype(jnp.float32)
        
        # Cross-entropy
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1].reshape(-1, model.config.vocab_size),
            labels[:, :-1].reshape(-1),
        )
        
        # Apply mask
        loss = loss * mask[:, :-1].reshape(-1)
        
        # Accumulate
        batch_tokens = jnp.sum(mask[:, :-1])
        total_loss += jnp.sum(loss)
        total_tokens += batch_tokens
        num_evaluated += 1
    
    # Average loss
    avg_loss = total_loss / total_tokens
    
    # Perplexity
    perplexity = jnp.exp(avg_loss)
    
    # Bits per token
    bits_per_token = avg_loss / jnp.log(2)
    
    return {
        'perplexity': float(perplexity),
        'loss': float(avg_loss),
        'bits_per_token': float(bits_per_token),
        'num_tokens': int(total_tokens),
        'num_batches': num_evaluated,
    }


def compute_token_accuracy(
    model: GPTModel,
    eval_dataloader: Iterator,
    num_batches: int = None,
    top_k: int = 1,
) -> dict:
    """Compute top-k token prediction accuracy."""
    
    total_correct = 0
    total_tokens = 0
    
    for i, batch in enumerate(eval_dataloader):
        if num_batches is not None and i >= num_batches:
            break
        
        logits = model(batch['input_ids'], is_training=False)
        
        # Get predictions
        if top_k == 1:
            predictions = jnp.argmax(logits, axis=-1)
        else:
            # Top-k predictions
            top_k_preds = jax.lax.top_k(logits, top_k)[1]
        
        # Ground truth (shifted)
        labels = jnp.roll(batch['input_ids'], -1, axis=1)
        
        # Mask padding
        mask = (batch['input_ids'] != 0).astype(jnp.float32)
        
        # Compute accuracy
        if top_k == 1:
            correct = (predictions[:, :-1] == labels[:, :-1]).astype(jnp.float32)
        else:
            # Check if true label in top-k
            correct = jnp.any(
                top_k_preds[:, :-1, :] == labels[:, :-1, None],
                axis=-1,
            ).astype(jnp.float32)
        
        correct = correct * mask[:, :-1]
        
        total_correct += jnp.sum(correct)
        total_tokens += jnp.sum(mask[:, :-1])
    
    accuracy = total_correct / total_tokens
    
    return {
        f'top_{top_k}_accuracy': float(accuracy),
        'num_tokens': int(total_tokens),
    }


# Evaluation during training
def evaluate_model(
    model: GPTModel,
    eval_dataloader: Iterator,
    step: int,
    tracker: ExperimentTracker,
):
    """Run full evaluation and log results."""
    
    print(f"\n=== Evaluating at step {step} ===")
    
    # Perplexity
    metrics = compute_perplexity(model, eval_dataloader, num_batches=100)
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Loss: {metrics['loss']:.4f}")
    
    # Top-k accuracy
    for k in [1, 5, 10]:
        acc_metrics = compute_token_accuracy(
            model, eval_dataloader, num_batches=100, top_k=k
        )
        metrics.update(acc_metrics)
        print(f"Top-{k} Accuracy: {acc_metrics[f'top_{k}_accuracy']*100:.2f}%")
    
    # Log to tracker
    eval_metrics = {f'eval/{k}': v for k, v in metrics.items()}
    tracker.log(eval_metrics, step=step)
    
    return metrics
```

---

## TURN 17 — End-to-End Training (GPT-2 on WikiText-103)

**Instructions:**

Put everything together to train GPT-2 Small on WikiText-103 dataset.

**Goal:** Train GPT-2 (117M parameters) to competitive perplexity in <24 hours on 8x A100.

**Target metrics:**
- Perplexity <20 on WikiText-103 validation
- Train for 100k steps
- Throughput >800k tokens/sec on 8 GPUs

**Full training script:**
```python
#!/usr/bin/env python3
"""
Train GPT-2 Small on WikiText-103
Target: <20 perplexity in <24 hours on 8x A100
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from datasets import load_dataset
import numpy as np
from pathlib import Path
import time

# Import all our implementations
from models.gpt import GPTModel, GPTConfig
from training.optimizers import OptimizerFactory, create_optimizer_with_schedule
from training.trainer import train_step, evaluate_model
from data.dataloaders import JAXDataLoader, PrefetchDataLoader
from utils.logging import ExperimentTracker
from utils.checkpointing import CheckpointManager


def main():
    # Configuration
    config = GPTConfig(
        vocab_size=50257,
        max_seq_len=1024,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        dropout_rate=0.1,
        use_tied_embeddings=True,
    )
    
    # Training hyperparameters
    batch_size = 32  # Per device
    seq_len = 1024
    num_steps = 100000
    warmup_steps = 2000
    peak_lr = 6e-4
    weight_decay = 0.1
    
    # Multi-GPU setup
    num_devices = jax.local_device_count()
    global_batch_size = batch_size * num_devices
    
    print(f"Training GPT-2 Small ({sum(p.size for p in jax.tree_leaves(nnx.state(model)))/1e6:.1f}M params)")
    print(f"Devices: {num_devices}")
    print(f"Batch size: {batch_size} per device, {global_batch_size} global")
    print(f"Sequence length: {seq_len}")
    print(f"Total steps: {num_steps}")
    
    # Initialize model
    rngs = nnx.Rngs(42)
    model = GPTModel(config, rngs=rngs)
    
    # Create optimizer with schedule
    optimizer = create_optimizer_with_schedule(
        peak_lr=peak_lr,
        warmup_steps=warmup_steps,
        total_steps=num_steps,
        schedule_type='cosine',
        weight_decay=weight_decay,
    )
    
    # Load data
    print("Loading WikiText-103...")
    train_dataset = load_dataset(
        'wikitext',
        'wikitext-103-raw-v1',
        split='train',
    )
    valid_dataset = load_dataset(
        'wikitext',
        'wikitext-103-raw-v1',
        split='validation',
    )
    
    # Create dataloaders
    train_loader = PrefetchDataLoader(
        JAXDataLoader(
            train_dataset,
            batch_size=global_batch_size,
            seq_len=seq_len,
            shuffle=True,
        ),
        prefetch_size=4,
    )
    
    valid_loader = JAXDataLoader(
        valid_dataset,
        batch_size=global_batch_size,
        seq_len=seq_len,
        shuffle=False,
    )
    
    # Initialize experiment tracking
    tracker = ExperimentTracker(
        project_name='gpt2-training',
        experiment_name=f'gpt2-small-wikitext103-{int(time.time())}',
        config={
            **config.__dict__,
            'batch_size': batch_size,
            'global_batch_size': global_batch_size,
            'num_steps': num_steps,
            'peak_lr': peak_lr,
            'weight_decay': weight_decay,
        },
        backend='wandb',
    )
    
    # Initialize checkpointing
    checkpoint_dir = Path('./checkpoints') / tracker.experiment_name
    ckpt_manager = CheckpointManager(
        str(checkpoint_dir),
        max_to_keep=3,
        save_interval_steps=5000,
    )
    
    # Training loop
    opt_state = optimizer.init(nnx.state(model))
    start_time = time.time()
    tokens_processed = 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    for step in range(num_steps):
        # Get batch
        batch = next(train_loader)
        
        # Training step
        loss, opt_state = train_step(model, opt_state, batch)
        
        # Update counters
        tokens_processed += global_batch_size * seq_len
        
        # Logging
        if step % 50 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            
            metrics = {
                'train/loss': float(loss),
                'train/tokens_per_sec': tokens_per_sec,
                'train/step': step,
            }
            
            tracker.log(metrics, step=step)
            
            print(f"Step {step:6d} | Loss: {loss:.4f} | "
                  f"Throughput: {tokens_per_sec/1000:.1f}k tok/s | "
                  f"Elapsed: {elapsed/60:.1f}m")
        
        # Evaluation
        if step % 2000 == 0 and step > 0:
            eval_metrics = evaluate_model(
                model, valid_loader, step, tracker
            )
            print(f"Validation Perplexity: {eval_metrics['perplexity']:.2f}")
        
        # Checkpointing
        if step % 5000 == 0 and step > 0:
            ckpt_manager.save(
                step=step,
                model=model,
                optimizer_state=opt_state,
                rng_state=rngs.default(),
                metrics={'loss': float(loss)},
            )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training complete! Running final evaluation...")
    print("="*60 + "\n")
    
    final_metrics = evaluate_model(model, valid_loader, num_steps, tracker)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Perplexity: {final_metrics['perplexity']:.2f}")
    print(f"Loss: {final_metrics['loss']:.4f}")
    print(f"Training time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Tokens processed: {tokens_processed/1e9:.2f}B")
    print(f"Average throughput: {tokens_processed/(time.time()-start_time)/1000:.1f}k tok/s")
    
    # Save final checkpoint
    ckpt_manager.save(
        step=num_steps,
        model=model,
        optimizer_state=opt_state,
        rng_state=rngs.default(),
        metrics=final_metrics,
    )
    
    # Finish tracking
    tracker.finish()
    
    print("\nTraining finished successfully! 🎉")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
```

**Expected results:**
```
Training GPT-2 Small (117M params)
Devices: 8
Batch size: 32 per device, 256 global
Sequence length: 1024
Total steps: 100000

Step      0 | Loss: 10.8234 | Throughput: 0.0k tok/s | Elapsed: 0.0m
Step     50 | Loss: 8.9234 | Throughput: 832.1k tok/s | Elapsed: 0.8m
Step    100 | Loss: 7.2341 | Throughput:: 891.3k tok/s | Elapsed: 1.5m
...
Step   2000 | Loss: 4.1234 | Throughput: 887.2k tok/s | Elapsed: 30.1m
Validation Perplexity: 32.45
...
Step 100000 | Loss: 2.9124 | Throughput: 893.4k tok/s | Elapsed: 1430.2m

============================================================
FINAL RESULTS
============================================================
Perplexity: 18.73
Loss: 2.93
Training time: 23.8 hours
Tokens processed: 26.2B
Average throughput: 887.3k tok/s

Training finished successfully! 🎉
```

**Success criteria:**
✅ Perplexity <20 (achieved 18.73)
✅ Training time <24 hours (23.8 hours)
✅ Throughput >800k tok/s (887k tok/s)
✅ Model converges smoothly
✅ No NaN/divergence

---

**Final Deliverables:**
- ✅ Complete GPT implementation in JAX/Flax
- ✅ Multi-GPU/TPU distributed training (Turn 7)
- ✅ Mixed-precision training BF16 (Turn 6)
- ✅ Flash Attention (Turn 8) and gradient checkpointing (Turn 9)
- ✅ Advanced optimizers: AdamW, LAMB, Lion (Turn 10)
- ✅ Learning rate schedules with warmup (Turn 11)
- ✅ Gradient accumulation (Turn 12)
- ✅ Efficient data loading (Turn 13)
- ✅ Robust checkpointing and resume (Turn 14)
- ✅ W&B integration and monitoring (Turn 15)
- ✅ Comprehensive metrics (Turn 16)
- ✅ End-to-end training achieving competitive results (Turn 17)
- ✅ >250 unit tests with >90% coverage
- ✅ Performance: Train GPT-2 (125M) to <20 perplexity in <24 hours on 8x A100
