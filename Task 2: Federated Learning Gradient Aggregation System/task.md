# Task: Build a Production-Grade Federated Learning Gradient Aggregation System

## Overview
Implement a distributed gradient aggregation system for federated learning that supports secure multi-party computation, differential privacy, Byzantine fault tolerance, and extreme communication efficiency. This system must handle heterogeneous workers (different compute speeds, network conditions), maintain numerical stability across millions of gradient updates, and provide cryptographic guarantees about privacy.

---

## TURN 1 — Core Gradient Aggregation with Numerical Stability

**Role:** You are a senior ML systems engineer who has built gradient aggregation pipelines at scale (think Google's Federated Learning infrastructure or OpenAI's distributed training systems). You deeply understand floating-point arithmetic pitfalls and have debugged convergence failures due to numerical drift.

**Background:** We need to aggregate gradients from 10-1000 workers training on private data shards. The naive approach (sum all gradients, divide by N) causes catastrophic numerical errors when gradients have vastly different magnitudes or when accumulated over thousands of rounds.

**Reference:** Study Google's "Towards Federated Learning at Scale" paper and Kahan summation algorithm for numerically stable accumulation.

**VERY IMPORTANT:**
- Gradients must be aggregated in a numerically stable way (no loss of precision)
- Must handle gradients with vastly different scales (1e-8 to 1e8 range)
- Zero memory leaks when aggregating 100M+ parameters
- Bit-exact reproducibility across runs (same inputs → same outputs)
- Must detect and reject NaN/Inf gradients without crashing

**Goal:** Implement core FedAvg (Federated Averaging) algorithm with production-grade numerical stability.

**Instructions:**

1. **Design the aggregation architecture** covering:
   - How to accumulate gradients from heterogeneous workers
   - Numerical stability strategy (Kahan summation? Pairwise summation? Mixed precision?)
   - Memory-efficient storage (don't materialize all worker gradients simultaneously)
   - How to handle stragglers (slow workers)
   - Gradient clipping strategy (per-layer vs global norm)

2. **Implement core structure:**
```python
from typing import Dict, List, Optional
import torch
from dataclasses import dataclass

@dataclass
class WorkerGradient:
    """Gradient from a single worker"""
    worker_id: str
    gradients: Dict[str, torch.Tensor]  # param_name -> gradient
    num_samples: int  # number of samples this worker trained on
    timestamp: float
    metadata: Optional[Dict] = None

class GradientAggregator:
    """Production-grade gradient aggregator with numerical stability"""
    
    def __init__(
        self,
        aggregation_method: str = "fedavg",  # "fedavg", "fedprox", "scaffold"
        clip_norm: Optional[float] = None,
        compensation_enabled: bool = True,  # Kahan summation
    ):
        self.aggregation_method = aggregation_method
        self.clip_norm = clip_norm
        self.compensation_enabled = compensation_enabled
        self._compensation_buffers = {}  # For Kahan summation
        
    def aggregate(
        self,
        worker_gradients: List[WorkerGradient],
        server_model_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients from multiple workers.
        
        Must guarantee:
        - Numerical stability even with 1000+ workers
        - Weighted averaging based on num_samples
        - Gradient clipping applied per-worker before aggregation
        - NaN/Inf detection and rejection
        - Bit-exact reproducibility
        
        Returns:
            Aggregated gradients ready to apply to server model
        """
        # TODO: Implement with Kahan summation for stability
        pass
    
    def _kahan_sum(
        self,
        tensors: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Numerically stable summation using Kahan algorithm.
        Maintains O(ε) error instead of O(Nε) for N tensors.
        """
        pass
    
    def _clip_gradient(self, gradient: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Clip gradient to max_norm without losing numerical precision"""
        pass
    
    def validate_gradient(self, gradient: WorkerGradient) -> bool:
        """
        Validate gradient for:
        - No NaN/Inf values
        - Reasonable magnitude (not 1e100)
        - Correct shapes matching server model
        """
        pass
```

3. **Write comprehensive tests:**
```python
def test_numerical_stability_large_scale():
    """
    Test: Aggregate 1000 workers with gradients ranging from 1e-8 to 1e8.
    Naive summation would lose precision for small gradients.
    
    Expected: Kahan summation maintains <1e-6 relative error.
    """
    pass

def test_gradient_clipping():
    """
    Test: One worker has exploding gradients (norm=1e10).
    Expected: Clipped to max_norm before aggregation, doesn't corrupt other workers.
    """
    pass

def test_nan_rejection():
    """
    Test: Worker 5 sends NaN gradients.
    Expected: Worker 5 rejected, aggregation continues with remaining workers.
    """
    pass

def test_reproducibility():
    """
    Test: Run aggregation twice with same inputs (fixed random seed).
    Expected: Bit-exact identical outputs.
    """
    pass

def test_weighted_averaging():
    """
    Test: Worker A trained on 1000 samples, Worker B on 100 samples.
    Expected: Worker A's gradients weighted 10x more in final average.
    """
    pass
```

4. **Provide a minimal working example:**
```python
# Simulate 3 workers training on different data shards
model = SimpleNN()  # Simple 2-layer network for testing
aggregator = GradientAggregator(clip_norm=1.0)

workers = []
for i in range(3):
    worker_grad = compute_worker_gradient(model, data_shard=i)
    workers.append(worker_grad)

# Aggregate and update server model
aggregated = aggregator.aggregate(workers, model.state_dict())
apply_gradients(model, aggregated)

# Verify convergence
assert loss_decreased(model)
```

**Deliverables:**
- Full implementation with Kahan summation
- All tests passing with numerical stability guarantees
- Benchmarks showing performance (gradients/sec, memory usage)
- Documentation explaining numerical stability choices

---

## TURN 2 — Secure Multi-Party Aggregation with Shamir Secret Sharing

**Instructions:**

Workers don't trust the central server. Implement secure aggregation where the server learns only the sum, never individual gradients.

**Background:** Use Shamir's (t, n) threshold secret sharing. Each worker splits their gradient into n shares, sends shares to n aggregators. Server reconstructs sum only if t+ aggregators cooperate.

**Requirements:**
- Server cannot learn any individual worker's gradient
- Robust to up to t-1 colluding aggregators
- Zero performance degradation vs. plaintext (use vectorized operations)
- Must work with PyTorch autograd (backward pass → encrypted shares → aggregation)

**Implement:**
```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
import secrets

class SecureAggregator(GradientAggregator):
    """Secure aggregation using Shamir Secret Sharing"""
    
    def __init__(
        self,
        num_aggregators: int = 5,
        threshold: int = 3,
        field_size: int = 2**127 - 1,  # Mersenne prime for fast modular arithmetic
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_aggregators = num_aggregators
        self.threshold = threshold
        self.field_size = field_size
        self._aggregator_shares = {}
        
    def worker_split_gradient(
        self,
        gradient: WorkerGradient,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Split each gradient tensor into n shares using Shamir Secret Sharing.
        
        Returns:
            List of n share dictionaries (one per aggregator)
        """
        pass
    
    def aggregator_combine_shares(
        self,
        shares_from_all_workers: List[Dict[str, torch.Tensor]],
        aggregator_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Combine shares from t+ aggregators to reconstruct sum.
        
        Must guarantee:
        - Cannot reconstruct with <t shares
        - Reconstructed sum equals plaintext sum (mathematical correctness)
        - Constant-time operations (no timing side-channels)
        """
        pass
    
    def _shamir_share(
        self,
        secret: torch.Tensor,
        threshold: int,
        num_shares: int,
    ) -> List[torch.Tensor]:
        """Generate Shamir secret shares"""
        # Use random polynomial of degree t-1
        # Evaluate at points 1, 2, ..., n
        pass
    
    def _shamir_reconstruct(
        self,
        shares: List[torch.Tensor],
        indices: List[int],
    ) -> torch.Tensor:
        """Reconstruct secret from t shares using Lagrange interpolation"""
        pass
```

**Tests:**
```python
def test_secure_aggregation_correctness():
    """
    Test: Secure aggregation result matches plaintext aggregation.
    Expected: Bit-exact match (no information loss from secret sharing).
    """
    pass

def test_privacy_guarantee():
    """
    Test: Given t-1 shares, cannot reconstruct original gradient.
    Expected: Reconstructed value is uniformly random in field.
    """
    pass

def test_aggregator_collusion():
    """
    Test: Simulate t-1 malicious aggregators colluding.
    Expected: Cannot learn individual worker gradient, only sum.
    """
    pass

def test_performance_overhead():
    """
    Test: Benchmark secure vs. plaintext aggregation.
    Expected: <10x slowdown (vectorized secret sharing is fast).
    """
    pass
```

**Challenge:** Shamir Secret Sharing works over finite fields, but gradients are floats. How do you handle the quantization error? Implement fixed-point encoding with <1e-6 error.

---

## TURN 3 — Differential Privacy with Adaptive Noise Calibration

**Instructions:**

Add differential privacy guarantees to protect individual training examples.

**Requirements:**
- (ε, δ)-differential privacy with ε ≤ 1.0, δ = 10⁻⁵
- Use Gaussian mechanism with noise scaled to sensitivity
- Adaptive noise calibration (reduce noise as training converges)
- Privacy budget accounting across multiple rounds (Rényi DP)
- Must maintain model accuracy within 2% of non-private baseline

**Implement:**
```python
from typing import Tuple
import math

class DPGradientAggregator(SecureAggregator):
    """Differentially private gradient aggregation"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,  # Sensitivity bound
        noise_multiplier: Optional[float] = None,
        adaptive_clipping: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.adaptive_clipping = adaptive_clipping
        
        if noise_multiplier is None:
            # Compute noise_multiplier from (ε, δ) budget
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier
            
        self._privacy_accountant = RenyiPrivacyAccountant()
        self._clipping_quantiles = []  # For adaptive clipping
        
    def aggregate_with_privacy(
        self,
        worker_gradients: List[WorkerGradient],
        server_model_state: Dict[str, torch.Tensor],
        round_num: int,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Aggregate with DP guarantees.
        
        Returns:
            (aggregated_gradients, privacy_spent)
        """
        # 1. Clip each worker's gradient to max_grad_norm (sensitivity bound)
        # 2. Aggregate clipped gradients
        # 3. Add Gaussian noise: N(0, (σ · S)²) where S = max_grad_norm
        # 4. Update privacy accountant
        pass
    
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier σ such that mechanism is (ε, δ)-DP.
        Uses moment accountant or analytic formula.
        """
        # For Gaussian mechanism: σ ≥ √(2 ln(1.25/δ)) / ε
        pass
    
    def _adaptive_clip_norm(self, worker_gradients: List[WorkerGradient]) -> float:
        """
        Adaptively set clipping norm based on gradient distribution.
        Target: Clip k% of gradients (e.g., k=10) to minimize utility loss.
        """
        norms = [compute_gradient_norm(wg.gradients) for wg in worker_gradients]
        quantile = torch.quantile(torch.tensor(norms), 0.9)
        return quantile.item()
    
    def _add_gaussian_noise(
        self,
        gradient: torch.Tensor,
        sensitivity: float,
        noise_multiplier: float,
    ) -> torch.Tensor:
        """Add calibrated Gaussian noise for DP"""
        std = sensitivity * noise_multiplier
        noise = torch.randn_like(gradient) * std
        return gradient + noise


class RenyiPrivacyAccountant:
    """Track cumulative privacy loss using Rényi Differential Privacy"""
    
    def __init__(self, orders: List[float] = None):
        self.orders = orders or [1 + x / 10.0 for x in range(10, 100)]
        self._rdp_budget = {order: 0.0 for order in self.orders}
        
    def accumulate(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int = 1,
    ):
        """Accumulate privacy loss for given noise and sampling"""
        for order in self.orders:
            rdp = self._compute_rdp(order, noise_multiplier, sampling_rate)
            self._rdp_budget[order] += steps * rdp
    
    def get_epsilon(self, delta: float) -> float:
        """Convert RDP to (ε, δ)-DP"""
        epsilons = [
            rdp + math.log(1 / delta) / (order - 1)
            for order, rdp in self._rdp_budget.items()
        ]
        return min(epsilons)
    
    def _compute_rdp(self, order, noise_multiplier, sampling_rate):
        """Compute Rényi divergence for given parameters"""
        # Implement RDP formula for subsampled Gaussian mechanism
        pass
```

**Tests:**
```python
def test_privacy_guarantee():
    """
    Test: Run aggregation for 1000 rounds with ε=1.0 budget.
    Expected: Final ε ≤ 1.0 (privacy accountant working correctly).
    """
    pass

def test_noise_magnitude():
    """
    Test: Check added noise has correct standard deviation.
    Expected: σ = noise_multiplier × sensitivity.
    """
    pass

def test_adaptive_clipping():
    """
    Test: Clipping norm adapts to gradient distribution over time.
    Expected: Early rounds: high clip norm. Later rounds: lower clip norm.
    """
    pass

def test_accuracy_degradation():
    """
    Test: Train model with DP (ε=1.0) vs. without DP.
    Expected: Accuracy loss <2% on test set.
    """
    pass

def test_privacy_composition():
    """
    Test: Sequential composition of privacy loss.
    Expected: ε_total ≤ sum of per-round ε (advanced composition gives tighter bound).
    """
    pass
```

**Challenge:** Differential privacy and secure aggregation are in tension. DP adds noise after aggregation, but secure aggregation hides individual gradients. How do you add noise without breaking security? Implement noise addition in the encrypted domain.

---

## TURN 4 — Force Failure: Numerical Instability in Long Training Runs

**Instructions:**

Deliberately introduce a subtle numerical instability that only manifests after 500+ aggregation rounds.

**Ask the AI:**
> "Your current implementation accumulates gradients in float32. What happens after 1000 rounds of aggregation when some parameters have been updated 1000 times? Show the exact failure mode with a test that demonstrates precision loss causing model divergence."

**Expected failure:**
- Small gradients (1e-7) get lost in accumulation
- Large gradients cause overflow after many rounds
- Model oscillates instead of converging
- Final loss is 10x higher than expected

**Test:**
```python
def test_long_training_divergence():
    """
    Test: Train for 1000 federated rounds with tiny learning rate.
    Accumulate gradients in float32 (buggy implementation).
    
    Expected failure:
    - Small gradient components accumulate incorrectly
    - Model converges after 100 rounds, then diverges after 500 rounds
    - Loss increases from round 500 to 1000
    """
    model = LargeNN(num_params=10_000_000)
    aggregator = GradientAggregator()  # Buggy: uses float32
    
    losses = []
    for round_num in range(1000):
        worker_grads = simulate_workers(model, num_workers=10)
        aggregated = aggregator.aggregate(worker_grads, model.state_dict())
        update_model(model, aggregated, lr=1e-4)
        losses.append(eval_loss(model))
    
    # Expected: losses should monotonically decrease
    # Actual (with bug): losses increase after round 500
    assert all(losses[i+1] <= losses[i] for i in range(len(losses)-1)), \
        "Model diverged due to numerical instability!"
```

**Fix required:** 
1. Use mixed precision (float32 for computation, float64 for accumulation)
2. Implement periodic rescaling of gradients
3. Add gradient norm monitoring to detect divergence early

---

## TURN 5 — Byzantine Fault Tolerance with Krum and Multi-Krum

**Instructions:**

Defend against malicious workers sending poisoned gradients to corrupt the model.

**Background:** Up to 30% of workers may be Byzantine (malicious). They could send:
- Scaled-up gradients (gradient × 1000) to cause divergence
- Inverted gradients (gradient × -1) to move model in wrong direction
- Coordinated attacks (multiple workers send correlated poison)

**Strategy:** Implement Krum and Multi-Krum aggregation rules that are robust to Byzantine workers.

**Implement:**
```python
class ByzantineRobustAggregator(DPGradientAggregator):
    """Byzantine-robust gradient aggregation"""
    
    def __init__(
        self,
        aggregation_rule: str = "krum",  # "krum", "multi-krum", "trimmed-mean", "median"
        num_byzantine: int = None,  # If None, auto-detect
        **kwargs
    ):
        super().__init__(**kwargs)
        self.aggregation_rule = aggregation_rule
        self.num_byzantine = num_byzantine
        
    def aggregate_byzantine_robust(
        self,
        worker_gradients: List[WorkerGradient],
        server_model_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients with Byzantine fault tolerance.
        
        Must guarantee:
        - If ≤f workers are Byzantine, honest workers' aggregation is correct
        - No single Byzantine worker can corrupt the model
        - Performance degradation <20% vs. non-robust aggregation
        """
        if self.aggregation_rule == "krum":
            return self._krum_aggregate(worker_gradients)
        elif self.aggregation_rule == "multi-krum":
            return self._multi_krum_aggregate(worker_gradients)
        else:
            raise ValueError(f"Unknown rule: {self.aggregation_rule}")
    
    def _krum_aggregate(
        self,
        worker_gradients: List[WorkerGradient],
    ) -> Dict[str, torch.Tensor]:
        """
        Krum: Select single gradient with smallest distance to neighbors.
        
        Algorithm:
        1. Compute pairwise distances between all worker gradients
        2. For each worker i, sum distances to closest n-f-2 workers
        3. Select worker with smallest sum (most "central" gradient)
        4. Return that single gradient (not average)
        
        Guarantees: Robust to f Byzantine workers if n ≥ 2f + 3
        """
        n = len(worker_gradients)
        f = self.num_byzantine or (n // 3)
        
        # Compute distance matrix
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i+1, n):
                dist = self._gradient_distance(
                    worker_gradients[i].gradients,
                    worker_gradients[j].gradients,
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each worker, sum distances to closest n-f-2 neighbors
        scores = []
        for i in range(n):
            closest_k = n - f - 2
            sorted_distances = torch.sort(distances[i])[0]
            score = torch.sum(sorted_distances[1:closest_k+1])  # Exclude distance to self (0)
            scores.append(score)
        
        # Select worker with minimum score
        selected_idx = torch.argmin(torch.tensor(scores))
        return worker_gradients[selected_idx].gradients
    
    def _multi_krum_aggregate(
        self,
        worker_gradients: List[WorkerGradient],
        m: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-Krum: Select top-m workers and average their gradients.
        More robust than single Krum, less variance.
        """
        if m is None:
            f = self.num_byzantine or (len(worker_gradients) // 3)
            m = len(worker_gradients) - f
        
        # Get Krum scores for all workers
        scores = self._compute_krum_scores(worker_gradients)
        
        # Select top-m workers (lowest scores = most central)
        top_m_indices = torch.topk(torch.tensor(scores), m, largest=False).indices
        
        # Average selected workers
        selected_grads = [worker_gradients[i] for i in top_m_indices]
        return self._average_gradients(selected_grads)
    
    def _gradient_distance(
        self,
        grad1: Dict[str, torch.Tensor],
        grad2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute L2 distance between two gradient dictionaries"""
        total_dist_sq = 0.0
        for key in grad1.keys():
            diff = grad1[key] - grad2[key]
            total_dist_sq += torch.sum(diff ** 2).item()
        return math.sqrt(total_dist_sq)
```

**Tests:**
```python
def test_single_byzantine_worker():
    """
    Test: 10 honest workers + 1 malicious worker sending gradient × 100.
    Expected: Krum selects an honest worker, model converges.
    """
    pass

def test_coordinated_byzantine_attack():
    """
    Test: 3 out of 10 workers collude and send correlated poison gradients.
    Expected: Multi-Krum excludes all 3, uses remaining 7.
    """
    pass

def test_krum_convergence_rate():
    """
    Test: Compare convergence with/without Byzantine workers.
    Expected: With Krum, convergence only 10-20% slower.
    """
    pass

def test_byzantine_upper_bound():
    """
    Test: n=10 workers, f=4 Byzantine (exceeds n/3 bound).
    Expected: Krum may fail, but fails gracefully (doesn't crash).
    """
    pass
```

**Challenge:** Krum is computationally expensive (O(n²) distance computations). Optimize it to O(n log n) using approximate nearest neighbors or clustering.

---

## TURN 6 — Communication Compression with Gradient Sparsification

**Instructions:**

Reduce communication cost by 10-100x using gradient sparsification and quantization.

**Requirements:**
- Send only top-k gradients by magnitude (k = 1% to 10% of parameters)
- Accumulate unsent gradients locally (error feedback mechanism)
- Maintain model accuracy within 1% of dense baseline
- Support multiple compression schemes (top-k, random-k, threshold)

**Implement:**
```python
class CompressedGradientAggregator(ByzantineRobustAggregator):
    """Gradient compression for communication efficiency"""
    
    def __init__(
        self,
        compression_ratio: float = 0.01,  # Send 1% of gradients
        compression_method: str = "topk",  # "topk", "randomk", "threshold"
        error_feedback: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.compression_ratio = compression_ratio
        self.compression_method = compression_method
        self.error_feedback = error_feedback
        self._error_accumulator = {}  # Per-worker error feedback buffers
        
    def compress_gradient(
        self,
        gradient: Dict[str, torch.Tensor],
        worker_id: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Compress gradient before sending to server.
        
        Returns:
            (compressed_gradient, compression_metadata)
        """
        compressed = {}
        metadata = {"indices": {}, "shapes": {}}
        
        for param_name, grad_tensor in gradient.items():
            if self.error_feedback and worker_id in self._error_accumulator:
                # Add accumulated error from previous round
                grad_tensor = grad_tensor + self._error_accumulator[worker_id].get(param_name, 0)
            
            # Compress
            if self.compression_method == "topk":
                compressed_grad, indices = self._topk_compress(grad_tensor)
            elif self.compression_method == "randomk":
                compressed_grad, indices = self._randomk_compress(grad_tensor)
            else:
                raise ValueError(f"Unknown method: {self.compression_method}")
            
            compressed[param_name] = compressed_grad
            metadata["indices"][param_name] = indices
            metadata["shapes"][param_name] = grad_tensor.shape
            
            # Store error for next round
            if self.error_feedback:
                error = grad_tensor.clone()
                error.view(-1)[indices] = 0  # Zero out sent values
                self._error_accumulator.setdefault(worker_id, {})[param_name] = error
        
        return compressed, metadata
    
    def _topk_compress(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k values by absolute magnitude.
        
        Returns:
            (values, indices) where values has shape (k,) and indices has shape (k,)
        """
        flat = tensor.view(-1)
        k = max(1, int(flat.numel() * self.compression_ratio))
        
        # Get top-k by magnitude
        _, indices = torch.topk(torch.abs(flat), k)
        values = flat[indices]
        
        return values, indices
    
    def decompress_gradient(
        self,
        compressed_gradient: Dict[str, torch.Tensor],
        metadata: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Decompress gradient on server side"""
        decompressed = {}
        
        for param_name, values in compressed_gradient.items():
            indices = metadata["indices"][param_name]
            shape = metadata["shapes"][param_name]
            
            # Reconstruct sparse tensor
            full_tensor = torch.zeros(shape).view(-1)
            full_tensor[indices] = values
            full_tensor = full_tensor.view(shape)
            
            decompressed[param_name] = full_tensor
        
        return decompressed
```

**Tests:**
```python
def test_compression_ratio():
    """
    Test: Compress 10M parameter gradient with ratio=0.01.
    Expected: Only 100k values transmitted, 100x reduction.
    """
    pass

def test_error_feedback_convergence():
    """
    Test: Train with compression (1%) + error feedback vs. without error feedback.
    Expected: With error feedback, convergence only 5% slower than dense.
    """
    pass

def test_topk_stability():
    """
    Test: Top-k indices should be deterministic for same input.
    Expected: Bit-exact reproducibility across runs.
    """
    pass

def test_communication_cost():
    """
    Test: Measure actual bytes sent over network.
    Expected: Compressed = dense_size × compression_ratio.
    """
    pass
```

**Challenge:** Top-k compression breaks differential privacy (attacker can infer which gradients were largest). Implement DP-compatible compression using randomized response or private top-k selection.

TURN 7 — Asynchronous Aggregation with Staleness Handling
Instructions:
Real federated learning has stragglers (slow workers due to weak devices, poor network). Implement asynchronous aggregation where server doesn't wait for all workers.
Background: Synchronous aggregation waits for all N workers before updating model. This is slow (blocked by slowest worker). Asynchronous aggregation updates model as soon as k workers arrive, but creates "staleness" problem: late workers computed gradients on outdated model.
Requirements:

Server updates model after receiving k out of N workers (e.g., k=N/2)
Handle gradient staleness (gradients computed on model from τ rounds ago)
Adaptive staleness tolerance (reject gradients >τ_max rounds old)
Maintain convergence guarantees despite asynchrony
Support straggler mitigation strategies

Implement:
pythonfrom collections import deque
from threading import Lock
import time

class AsyncGradientAggregator(CompressedGradientAggregator):
    """Asynchronous gradient aggregation with staleness handling"""
    
    def __init__(
        self,
        min_workers_per_round: int = 5,  # Update after receiving k workers
        max_staleness: int = 10,  # Reject gradients >10 rounds old
        staleness_weighting: str = "polynomial",  # "polynomial", "constant", "adaptive"
        staleness_exponent: float = 0.5,  # For polynomial: weight = 1 / (1 + τ)^α
        enable_stragglers_mitigation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_workers_per_round = min_workers_per_round
        self.max_staleness = max_staleness
        self.staleness_weighting = staleness_weighting
        self.staleness_exponent = staleness_exponent
        self.enable_stragglers_mitigation = enable_stragglers_mitigation
        
        self._current_round = 0
        self._model_version_history = deque(maxlen=max_staleness + 1)
        self._pending_gradients = []
        self._gradient_buffer_lock = Lock()
        self._worker_latency_history = {}  # Track worker speeds
        
    def submit_gradient_async(
        self,
        worker_gradient: WorkerGradient,
        model_version: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Worker submits gradient asynchronously.
        
        Args:
            worker_gradient: Gradient from worker
            model_version: Model version worker used for computation
            
        Returns:
            Updated model if aggregation triggered, else None
        """
        staleness = self._current_round - model_version
        
        # Reject stale gradients
        if staleness > self.max_staleness:
            self._log_rejected_gradient(worker_gradient.worker_id, staleness)
            return None
        
        # Add staleness metadata
        worker_gradient.metadata = worker_gradient.metadata or {}
        worker_gradient.metadata["staleness"] = staleness
        worker_gradient.metadata["model_version"] = model_version
        
        # Track worker latency
        self._update_worker_latency(worker_gradient)
        
        with self._gradient_buffer_lock:
            self._pending_gradients.append(worker_gradient)
            
            # Trigger aggregation if enough workers arrived
            if len(self._pending_gradients) >= self.min_workers_per_round:
                aggregated = self._aggregate_and_update()
                self._pending_gradients.clear()
                self._current_round += 1
                return aggregated
        
        return None
    
    def _aggregate_with_staleness_weighting(
        self,
        worker_gradients: List[WorkerGradient],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients with staleness-aware weighting.
        
        Fresh gradients (staleness=0) get full weight.
        Stale gradients (staleness>0) get reduced weight.
        """
        weighted_grads = {}
        total_weight = 0.0
        
        for wg in worker_gradients:
            staleness = wg.metadata.get("staleness", 0)
            weight = self._compute_staleness_weight(staleness)
            
            # Also weight by num_samples
            weight *= wg.num_samples
            total_weight += weight
            
            for param_name, grad in wg.gradients.items():
                if param_name not in weighted_grads:
                    weighted_grads[param_name] = torch.zeros_like(grad)
                weighted_grads[param_name] += weight * grad
        
        # Normalize
        for param_name in weighted_grads:
            weighted_grads[param_name] /= total_weight
        
        return weighted_grads
    
    def _compute_staleness_weight(self, staleness: int) -> float:
        """Compute weight for gradient with given staleness"""
        if self.staleness_weighting == "constant":
            return 1.0
        elif self.staleness_weighting == "polynomial":
            return 1.0 / ((1 + staleness) ** self.staleness_exponent)
        elif self.staleness_weighting == "adaptive":
            # Adaptive: reduce weight more aggressively as training progresses
            progress = self._current_round / 1000  # Assume 1000 total rounds
            alpha = 0.5 + 1.5 * progress  # α increases from 0.5 to 2.0
            return 1.0 / ((1 + staleness) ** alpha)
        else:
            raise ValueError(f"Unknown weighting: {self.staleness_weighting}")
    
    def _update_worker_latency(self, worker_gradient: WorkerGradient):
        """Track worker latency for straggler detection"""
        worker_id = worker_gradient.worker_id
        latency = time.time() - worker_gradient.timestamp
        
        if worker_id not in self._worker_latency_history:
            self._worker_latency_history[worker_id] = deque(maxlen=10)
        
        self._worker_latency_history[worker_id].append(latency)
    
    def get_worker_selection_probabilities(self) -> Dict[str, float]:
        """
        Compute worker selection probabilities for next round.
        Fast workers get higher probability (straggler mitigation).
        """
        if not self.enable_stragglers_mitigation:
            return {}
        
        # Compute average latency per worker
        avg_latencies = {}
        for worker_id, latencies in self._worker_latency_history.items():
            if len(latencies) > 0:
                avg_latencies[worker_id] = sum(latencies) / len(latencies)
        
        if not avg_latencies:
            return {}
        
        # Convert latencies to selection probabilities (inverse relationship)
        # Fast workers (low latency) get high probability
        max_latency = max(avg_latencies.values())
        probabilities = {}
        
        for worker_id, latency in avg_latencies.items():
            # Inverse latency, normalized
            probabilities[worker_id] = (max_latency - latency + 0.1) / max_latency
        
        # Normalize to sum to 1
        total_prob = sum(probabilities.values())
        for worker_id in probabilities:
            probabilities[worker_id] /= total_prob
        
        return probabilities
    
    def _store_model_version(self, model_state: Dict[str, torch.Tensor]):
        """Store model version for staleness correction (if needed)"""
        self._model_version_history.append({
            "round": self._current_round,
            "state": {k: v.clone() for k, v in model_state.items()},
        })
Tests:
pythondef test_staleness_rejection():
    """
    Test: Worker computes gradient on model from 15 rounds ago (staleness=15).
    Config: max_staleness=10.
    Expected: Gradient rejected, not included in aggregation.
    """
    pass

def test_staleness_weighting():
    """
    Test: Two workers, one fresh (staleness=0), one stale (staleness=5).
    Expected: Fresh gradient weighted more heavily in aggregation.
    """
    pass

def test_async_convergence():
    """
    Test: Train with async aggregation (min_workers=5 out of 10) vs. sync.
    Expected: Async converges to similar loss, but 2x faster wall-clock time.
    """
    pass

def test_straggler_mitigation():
    """
    Test: 10 workers, 2 are consistently slow (10x latency).
    Expected: After 50 rounds, slow workers selected less frequently.
    """
    pass

def test_thread_safety():
    """
    Test: 100 workers submitting gradients concurrently.
    Expected: No race conditions, all gradients processed exactly once.
    """
    pass
Challenge: Asynchronous aggregation can cause model divergence if staleness weighting is too aggressive. Implement adaptive staleness tolerance that tightens as training progresses.

TURN 8 — Force Failure: Gradient Staleness Causes Divergence
Instructions:
Deliberately configure the system to demonstrate how excessive staleness breaks convergence.
Ask the AI:

"Your async aggregator allows staleness up to τ_max=10 rounds. What happens when the learning rate is too high AND workers are very slow (staleness often 8-10 rounds)? Show the exact divergence pattern with a test that plots loss over time."

Expected failure:

Model converges for first 100 rounds
Around round 150, loss starts oscillating
By round 300, loss explodes to infinity
Root cause: Stale gradients pointing in outdated directions

Test:
pythondef test_staleness_divergence():
    """
    Test: Async training with high staleness (τ_max=10) and high learning rate (0.1).
    Simulate 20 workers where 50% have random staleness in [5, 10].
    
    Expected failure:
    - Rounds 0-100: Loss decreases normally
    - Rounds 100-300: Loss oscillates with increasing amplitude
    - Round 300+: Loss > 1e6 (divergence)
    """
    model = SimpleNN()
    aggregator = AsyncGradientAggregator(
        min_workers_per_round=10,
        max_staleness=10,
        staleness_weighting="constant",  # Bug: should use polynomial
    )
    
    losses = []
    learning_rate = 0.1  # Too high for async
    
    for round_num in range(500):
        # Simulate workers with varying staleness
        workers = []
        for i in range(20):
            if i < 10:
                staleness = 0  # Fast workers
            else:
                staleness = np.random.randint(5, 11)  # Slow workers
            
            # Worker computes gradient on stale model
            stale_model_version = round_num - staleness
            wg = compute_gradient_on_version(model, stale_model_version)
            wg.metadata = {"staleness": staleness}
            workers.append(wg)
        
        # Aggregate first 10 workers that arrive
        aggregated = aggregator.submit_gradient_async(workers[0], round_num)
        if aggregated:
            update_model(model, aggregated, lr=learning_rate)
            loss = eval_loss(model)
            losses.append(loss)
            
            if round_num > 300:
                assert loss < 1e3, f"Model diverged at round {round_num}! Loss={loss}"
    
    # Plot losses to visualize divergence
    plt.plot(losses)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Divergence due to Staleness")
    plt.savefig("staleness_divergence.png")
Fix required:

Reduce learning rate as staleness increases: lr_effective = lr_base / (1 + avg_staleness)
Use polynomial staleness weighting with α ≥ 1.0
Implement momentum-based correction for stale gradients
Add divergence detection (if loss increases 2x, halt training)


TURN 9 — Gradient Compression + DP + Byzantine Robustness: Integration
Instructions:
All previous features work independently. Now integrate them into a single pipeline where they must work together without conflicts.
Challenge: These techniques have subtle interactions:

Compression + DP: Compressed gradients change sensitivity, affecting DP noise calibration
Compression + Byzantine: Attacker can exploit sparsity to amplify attack
DP + Byzantine: DP noise can mask Byzantine detection
Async + Byzantine: Stale Byzantine gradients harder to detect

Requirements:

All features enabled simultaneously: compression (1%), DP (ε=1.0), Byzantine-robust (Krum), async (τ_max=5)
Model must still converge to within 5% of baseline accuracy
Privacy, security, and efficiency guarantees all maintained
No catastrophic interactions between components

Implement:
pythonclass IntegratedFederatedAggregator:
    """Full production system with all features integrated"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        
        # Initialize all components
        self.compressor = GradientCompressor(config.compression)
        self.dp_mechanism = DPMechanism(config.privacy)
        self.byzantine_detector = ByzantineDetector(config.security)
        self.async_manager = AsyncManager(config.async_config)
        
        # Integration-specific components
        self.sensitivity_calibrator = SensitivityCalibrator()
        self.attack_detector = IntegratedAttackDetector()
        
    def process_worker_gradient(
        self,
        worker_gradient: WorkerGradient,
        model_version: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Full pipeline:
        1. Decompress gradient
        2. Check staleness (async)
        3. Detect Byzantine behavior
        4. Aggregate with Krum/Multi-Krum
        5. Add DP noise with corrected sensitivity
        6. Update model
        """
        
        # Step 1: Decompress
        if worker_gradient.compressed:
            gradient = self.compressor.decompress(
                worker_gradient.compressed_data,
                worker_gradient.metadata,
            )
        else:
            gradient = worker_gradient.gradients
        
        # Step 2: Staleness check
        staleness = self._current_round - model_version
        if staleness > self.config.async_config.max_staleness:
            return None
        
        # Step 3: Byzantine detection with compression-aware threshold
        is_byzantine = self.byzantine_detector.detect(
            gradient,
            worker_gradient.worker_id,
            compression_ratio=self.config.compression.ratio,
        )
        if is_byzantine:
            self._handle_byzantine_worker(worker_gradient.worker_id)
            return None
        
        # Add to buffer for aggregation
        self.async_manager.add_gradient(worker_gradient, staleness)
        
        # Step 4: Check if ready to aggregate
        if not self.async_manager.ready_to_aggregate():
            return None
        
        # Step 5: Krum aggregation
        pending = self.async_manager.get_pending_gradients()
        aggregated = self.byzantine_detector.krum_aggregate(pending)
        
        # Step 6: DP noise with CORRECTED sensitivity
        # CRITICAL: Compression changes gradient bounds!
        sensitivity = self.sensitivity_calibrator.compute_sensitivity(
            compression_ratio=self.config.compression.ratio,
            clip_norm=self.config.privacy.max_grad_norm,
        )
        
        aggregated_with_dp = self.dp_mechanism.add_noise(
            aggregated,
            sensitivity=sensitivity,
        )
        
        return aggregated_with_dp
    
    def _handle_byzantine_worker(self, worker_id: str):
        """Quarantine Byzantine worker, alert monitoring system"""
        self._quarantined_workers.add(worker_id)
        self._emit_security_alert(worker_id)


class SensitivityCalibrator:
    """Compute correct DP sensitivity when compression is enabled"""
    
    def compute_sensitivity(
        self,
        compression_ratio: float,
        clip_norm: float,
    ) -> float:
        """
        Sensitivity with compression:
        
        Without compression: S = clip_norm
        With compression (top-k): S = clip_norm / sqrt(k/d)
        
        Intuition: Only k out of d gradients are sent, so L2 norm is smaller.
        But worst-case adversary could send k largest gradients.
        """
        # Pessimistic: Assume adversary maximizes sent gradient norm
        # Top-k of clip_norm gradient has norm approximately clip_norm * sqrt(compression_ratio)
        return clip_norm * math.sqrt(compression_ratio)


class IntegratedAttackDetector:
    """Detect attacks that exploit interactions between features"""
    
    def detect_compression_exploitation(
        self,
        worker_gradient: WorkerGradient,
        compression_metadata: Dict,
    ) -> bool:
        """
        Attack: Malicious worker sends only large positive gradients in top-k.
        Effect: Skews model even under Krum if multiple workers collude.
        
        Detection: Check if top-k indices are suspiciously clustered.
        """
        indices = compression_metadata.get("indices", {})
        
        for param_name, param_indices in indices.items():
            # Check if indices are clustered (not uniformly distributed)
            # Honest compression should have somewhat uniform distribution
            if self._is_clustered(param_indices):
                return True
        
        return False
    
    def detect_dp_masking_attack(
        self,
        gradients: List[WorkerGradient],
        aggregated: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Attack: Byzantine workers send gradients designed to look normal
        after DP noise is added.
        
        Detection: Check if post-DP aggregated gradient has anomalous structure.
        """
        # Compute expected noise magnitude
        expected_noise_std = self._compute_expected_dp_noise()
        
        # Check if aggregated gradient has suspiciously high variance
        for param_name, param_grad in aggregated.items():
            actual_std = torch.std(param_grad).item()
            if actual_std > 3 * expected_noise_std:
                return True
        
        return False
Tests:
pythondef test_integrated_convergence():
    """
    Test: Full system with all features enabled.
    Config: compression=1%, DP ε=1.0, Byzantine f=3, async τ_max=5
    Expected: Converges to within 5% of baseline accuracy.
    """
    pass

def test_compression_dp_interaction():
    """
    Test: DP noise calibration with compressed gradients.
    Expected: Privacy budget accurately tracked, no over-spending.
    """
    pass

def test_byzantine_attack_under_compression():
    """
    Test: Byzantine workers exploit compression by sending only large gradients.
    Expected: Attack detected and mitigated by integrated attack detector.
    """
    pass

def test_async_byzantine_interaction():
    """
    Test: Byzantine workers send stale poisoned gradients.
    Expected: Staleness check rejects old gradients before Byzantine detection runs.
    """
    pass

def test_performance_overhead():
    """
    Test: Benchmark integrated system vs. baseline.
    Expected: <50% slowdown despite all features enabled.
    """
    pass
Challenge: The sensitivity calibration for DP + compression is subtle. If calibrated incorrectly, either privacy is violated or utility is destroyed. Implement formal proof that integrated system maintains (ε, δ)-DP.

TURN 10 — Cross-Device Federated Learning: Mobile Deployment
Instructions:
Deploy the system for real cross-device federated learning (millions of phones, not servers).
New challenges:

Extreme heterogeneity: Devices range from iPhone 15 to old Android phones
Limited compute: Some devices have <1GB RAM, slow CPUs
Intermittent connectivity: Devices go offline mid-training
Battery constraints: Can't use 100% CPU for long
Diverse data distributions: Each device has very different data

Requirements:

Client-side gradient computation must run in <500MB RAM
Support model checkpointing (resume training after device restarts)
Adaptive batch size based on device capability
Power-aware training (throttle when battery <20%)
Implement device selection policy (don't always pick fast devices)

Implement:
pythonclass MobileClient:
    """On-device client for federated learning"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device_profile: DeviceProfile,
    ):
        self.model = model
        self.device_profile = device_profile
        
        # Adaptive configuration based on device
        self.batch_size = self._compute_optimal_batch_size()
        self.num_local_epochs = self._compute_local_epochs()
        self.gradient_accumulation_steps = self._compute_accumulation_steps()
        
        # Power management
        self.power_monitor = PowerMonitor()
        self.training_paused = False
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager()
        
    def train_local_model(
        self,
        local_data: DataLoader,
        server_model_version: int,
    ) -> Optional[WorkerGradient]:
        """
        Train model on local data with adaptive resource management.
        
        Returns gradient if training completes, None if interrupted.
        """
        # Check if we can train
        if not self._can_start_training():
            return None
        
        # Load checkpoint if resuming
        start_step = 0
        if self.checkpoint_manager.has_checkpoint():
            checkpoint = self.checkpoint_manager.load()
            self.model.load_state_dict(checkpoint["model_state"])
            start_step = checkpoint["step"]
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        gradient_accumulator = {}
        
        for epoch in range(self.num_local_epochs):
            for step, (data, target) in enumerate(local_data):
                # Skip if resuming from checkpoint
                if epoch == 0 and step < start_step:
                    continue
                
                # Power check
                if not self.power_monitor.can_continue_training():
                    # Save checkpoint and pause
                    self.checkpoint_manager.save({
                        "model_state": self.model.state_dict(),
                        "step": step,
                        "epoch": epoch,
                    })
                    return None
                
                # Forward pass
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation (for memory efficiency)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Accumulate gradients
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if name not in gradient_accumulator:
                                gradient_accumulator[name] = torch.zeros_like(param.grad)
                            gradient_accumulator[name] += param.grad.detach()
                    
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Normalize accumulated gradients
        total_steps = len(local_data) * self.num_local_epochs
        for name in gradient_accumulator:
            gradient_accumulator[name] /= total_steps
        
        return WorkerGradient(
            worker_id=self.device_profile.device_id,
            gradients=gradient_accumulator,
            num_samples=len(local_data.dataset),
            timestamp=time.time(),
            metadata={
                "device_type": self.device_profile.device_type,
                "training_time": self._get_training_time(),
            },
        )
    
    def _compute_optimal_batch_size(self) -> int:
        """Compute batch size based on available RAM"""
        available_memory_mb = self.device_profile.available_memory_mb
        
        # Heuristic: 1MB per sample in batch (very rough)
        max_batch_size = max(1, available_memory_mb // 10)
        
        # Clamp to reasonable range
        return min(max_batch_size, 64)
    
    def _compute_local_epochs(self) -> int:
        """More epochs for faster devices"""
        if self.device_profile.device_type == "high_end":
            return 5
        elif self.device_profile.device_type == "mid_range":
            return 3
        else:  # low_end
            return 1
    
    def _can_start_training(self) -> bool:
        """Check if device can start training"""
        # Battery check
        if self.power_monitor.battery_level < 0.2:
            return False
        
        # Connectivity check
        if not self.power_monitor.is_connected_to_wifi():
            return False
        
        # Not actively being used
        if self.power_monitor.is_device_active():
            return False
        
        return True


class DeviceProfile:
    """Profile of device capabilities"""
    device_id: str
    device_type: str  # "high_end", "mid_range", "low_end"
    available_memory_mb: int
    cpu_cores: int
    has_gpu: bool


class PowerMonitor:
    """Monitor device power state"""
    
    @property
    def battery_level(self) -> float:
        """Battery level in [0, 1]"""
        # In real implementation, query OS API
        return 0.8
    
    def is_connected_to_wifi(self) -> bool:
        """Check if device is on WiFi (not cellular)"""
        return True
    
    def is_device_active(self) -> bool:
        """Check if user is actively using device"""
        return False
    
    def can_continue_training(self) -> bool:
        """Check if training should continue"""
        if self.battery_level < 0.15:  # Stop if battery drops too low
            return False
        
        if self.is_device_active():  # Pause if user starts using device
            return False
        
        return True


class ServerDeviceSelector:
    """Server-side device selection for next training round"""
    
    def __init__(self, selection_strategy: str = "fair"):
        self.selection_strategy = selection_strategy
        self._device_participation_history = {}
        
    def select_devices(
        self,
        available_devices: List[str],
        target_num_devices: int,
        device_profiles: Dict[str, DeviceProfile],
    ) -> List[str]:
        """
        Select devices for next training round.
        
        Strategies:
        - "fair": Ensure all devices participate roughly equally
        - "fast": Prefer fast devices for quick convergence
        - "diverse": Select devices with diverse data distributions
        """
        if self.selection_strategy == "fair":
            return self._fair_selection(available_devices, target_num_devices)
        elif self.selection_strategy == "fast":
            return self._fast_selection(available_devices, target_num_devices, device_profiles)
        elif self.selection_strategy == "diverse":
            return self._diverse_selection(available_devices, target_num_devices, device_profiles)
        else:
            raise ValueError(f"Unknown strategy: {self.selection_strategy}")
    
    def _fair_selection(
        self,
        available_devices: List[str],
        target_num_devices: int,
    ) -> List[str]:
        """Select devices that have participated least recently"""
        # Sort by participation count (ascending)
        sorted_devices = sorted(
            available_devices,
            key=lambda d: self._device_participation_history.get(d, 0),
        )
        
        selected = sorted_devices[:target_num_devices]
        
        # Update history
        for device_id in selected:
            self._device_participation_history[device_id] = \
                self._device_participation_history.get(device_id, 0) + 1
        
        return selected
Tests:
pythondef test_adaptive_batch_size():
    """
    Test: Simulate devices with 512MB, 2GB, 8GB RAM.
    Expected: Batch sizes scale appropriately (8, 32, 64).
    """
    pass

def test_training_interruption_resume():
    """
    Test: Start training, interrupt at 50% progress, resume.
    Expected: Training completes from checkpoint, correct gradient computed.
    """
    pass

def test_power_aware_training():
    """
    Test: Battery drops from 50% to 15% during training.
    Expected: Training pauses at 15%, checkpoint saved.
    """
    pass

def test_fair_device_selection():
    """
    Test: 1000 devices, select 100 per round for 20 rounds.
    Expected: All devices participate at least once, variance <10%.
    """
    pass

def test_heterogeneous_convergence():
    """
    Test: Mix of high-end (30%), mid-range (50%), low-end (20%) devices.
    Expected: Model converges despite heterogeneity, <10% accuracy loss vs. homogeneous.
    """
    pass

TURN 11 — Personalization: Local Fine-Tuning with Meta-Learning
Instructions:
Not all users want the same global model. Implement personalized federated learning where each device fine-tunes the global model to its local data.
Background: Standard federated learning learns a single global model. But users have diverse preferences (e.g., keyboard autocorrect should learn your writing style). Solution: Meta-learning (MAML, Reptile) + local fine-tuning.
Requirements:

Server learns meta-parameters that are good starting point for local adaptation
Each device fine-tunes with 1-5 gradient steps on local data
Aggregation preserves meta-learning property (generalization to new devices)
Support both task-agnostic (MAML) and task-specific personalization

Implement:
pythonclass PersonalizedFederatedAggregator(IntegratedFederatedAggregator):
    """Personalized FL with meta-learning"""
    
    def __init__(
        self,
        meta_learning_algorithm: str = "maml",  # "maml", "reptile", "per-fedavg"
        personalization_steps: int = 5,
        personalization_lr: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.meta_learning_algorithm = meta_learning_algorithm
        self.personalization_steps = personalization_steps
        self.personalization_lr = personalization_lr
        
    def aggregate_with_personalization(
        self,
        worker_gradients: List[WorkerGradient],
        server_model_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients with meta-learning objective.
        
        MAML: Aggregate gradients computed AFTER local fine-tuning.
        Reptile: Aggregate difference between initial and fine-tuned models.
        """
        if self.meta_learning_algorithm == "maml":
            return self._maml_aggregate(worker_gradients, server_model_state)
        elif self.meta_learning_algorithm == "reptile":
            return self._reptile_aggregate(worker_gradients, server_model_state)
        else:
            return super().aggregate(worker_gradients, server_model_state)
    
    def _maml_aggregate(
        self,
        worker_gradients
        : List[WorkerGradient],
server_model_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
"""
MAML aggregation: Meta-gradient through local adaptation.
    Algorithm:
    1. Each worker fine-tunes global model on local data (inner loop)
    2. Compute gradient of fine-tuned model on validation set (outer loop)
    3. Aggregate meta-gradients to update global model
    
    This makes global model a good initialization for local adaptation.
    """
    meta_gradients = {}
    
    for wg in worker_gradients:
        # Worker should provide TWO sets of gradients:
        # 1. Inner gradients (from local fine-tuning)
        # 2. Meta-gradient (from validation after fine-tuning)
        
        if "meta_gradient" not in wg.metadata:
            raise ValueError(f"Worker {wg.worker_id} did not provide meta-gradient")
        
        meta_grad = wg.metadata["meta_gradient"]
        
        # Accumulate meta-gradients
        for param_name, grad in meta_grad.items():
            if param_name not in meta_gradients:
                meta_gradients[param_name] = torch.zeros_like(grad)
            meta_gradients[param_name] += grad * wg.num_samples
    
    # Normalize by total samples
    total_samples = sum(wg.num_samples for wg in worker_gradients)
    for param_name in meta_gradients:
        meta_gradients[param_name] /= total_samples
    
    return meta_gradients

def _reptile_aggregate(
    self,
    worker_gradients: List[WorkerGradient],
    server_model_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Reptile aggregation: Aggregate model differences after local training.
    
    Algorithm:
    1. Each worker fine-tunes global model on local data
    2. Compute difference: Δθ = θ_fine_tuned - θ_global
    3. Aggregate differences: θ_new = θ_global + lr * avg(Δθ)
    
    Simpler than MAML (no second-order derivatives), but similar performance.
    """
    model_deltas = {}
    
    for wg in worker_gradients:
        if "fine_tuned_model" not in wg.metadata:
            raise ValueError(f"Worker {wg.worker_id} did not provide fine-tuned model")
        
        fine_tuned_state = wg.metadata["fine_tuned_model"]
        
        # Compute difference
        for param_name, global_param in server_model_state.items():
            fine_tuned_param = fine_tuned_state[param_name]
            delta = fine_tuned_param - global_param
            
            if param_name not in model_deltas:
                model_deltas[param_name] = torch.zeros_like(delta)
            model_deltas[param_name] += delta * wg.num_samples
    
    # Normalize and convert to gradient (negative delta)
    total_samples = sum(wg.num_samples for wg in worker_gradients)
    gradients = {}
    for param_name in model_deltas:
        avg_delta = model_deltas[param_name] / total_samples
        # Reptile uses positive delta (move toward fine-tuned models)
        # But our interface expects gradients (negative direction)
        gradients[param_name] = -avg_delta
    
    return gradients
class PersonalizedMobileClient(MobileClient):
"""Mobile client with personalized model"""
def __init__(
    self,
    global_model: torch.nn.Module,
    device_profile: DeviceProfile,
    personalization_config: PersonalizationConfig,
):
    super().__init__(global_model, device_profile)
    self.personalization_config = personalization_config
    
    # Store personalized model locally
    self.personalized_model = copy.deepcopy(global_model)
    
    # Split local data into train/val for meta-learning
    self.local_train_data = None
    self.local_val_data = None
    
def train_with_personalization(
    self,
    local_data: DataLoader,
    global_model_state: Dict[str, torch.Tensor],
    algorithm: str = "maml",
) -> WorkerGradient:
    """
    Train personalized model and compute meta-gradient.
    
    Returns:
        WorkerGradient with meta-gradient or fine-tuned model
    """
    # Split data
    self._split_train_val_data(local_data)
    
    if algorithm == "maml":
        return self._train_maml()
    elif algorithm == "reptile":
        return self._train_reptile(global_model_state)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def _train_maml(self) -> WorkerGradient:
    """
    MAML training with double backward pass.
    
    Inner loop: Fine-tune on local train data
    Outer loop: Compute meta-gradient on local val data
    """
    # Start from global model
    self.model.load_state_dict(self.personalized_model.state_dict())
    
    # Inner loop: Local fine-tuning
    inner_optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.personalization_config.inner_lr,
    )
    
    # Store initial model for meta-gradient computation
    initial_params = {
        name: param.clone()
        for name, param in self.model.named_parameters()
    }
    
    # Fine-tune for K steps
    for step in range(self.personalization_config.inner_steps):
        for data, target in self.local_train_data:
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            
            if step >= self.personalization_config.inner_steps:
                break
    
    # Outer loop: Compute meta-gradient on validation data
    meta_gradients = {}
    
    for data, target in self.local_val_data:
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        
        # Compute gradient w.r.t. fine-tuned parameters
        loss.backward()
        
        # Accumulate meta-gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in meta_gradients:
                    meta_gradients[name] = torch.zeros_like(param.grad)
                meta_gradients[name] += param.grad.detach()
    
    # Normalize
    for name in meta_gradients:
        meta_gradients[name] /= len(self.local_val_data)
    
    return WorkerGradient(
        worker_id=self.device_profile.device_id,
        gradients={},  # Regular gradients not used in MAML
        num_samples=len(self.local_train_data.dataset),
        timestamp=time.time(),
        metadata={
            "meta_gradient": meta_gradients,
            "algorithm": "maml",
        },
    )

def _train_reptile(
    self,
    global_model_state: Dict[str, torch.Tensor],
) -> WorkerGradient:
    """
    Reptile training: Simple fine-tuning + model difference.
    """
    # Start from global model
    self.model.load_state_dict(global_model_state)
    
    # Fine-tune on all local data
    optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.personalization_config.inner_lr,
    )
    
    for step in range(self.personalization_config.inner_steps):
        for data, target in self.local_train_data:
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Store fine-tuned model
    fine_tuned_state = {
        name: param.clone()
        for name, param in self.model.named_parameters()
    }
    
    # Update local personalized model
    self.personalized_model.load_state_dict(fine_tuned_state)
    
    return WorkerGradient(
        worker_id=self.device_profile.device_id,
        gradients={},  # Not used in Reptile
        num_samples=len(self.local_train_data.dataset),
        timestamp=time.time(),
        metadata={
            "fine_tuned_model": fine_tuned_state,
            "algorithm": "reptile",
        },
    )

def infer_with_personalized_model(
    self,
    input_data: torch.Tensor,
) -> torch.Tensor:
    """Run inference using personalized model"""
    self.personalized_model.eval()
    with torch.no_grad():
        output = self.personalized_model(input_data)
    return output

def _split_train_val_data(self, local_data: DataLoader):
    """Split local data into train (80%) and val (20%)"""
    dataset = local_data.dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )
    
    self.local_train_data = DataLoader(
        train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
    )
    self.local_val_data = DataLoader(
        val_dataset,
        batch_size=self.batch_size,
        shuffle=False,
    )
@dataclass
class PersonalizationConfig:
"""Configuration for personalized federated learning"""
inner_steps: int = 5  # Number of local fine-tuning steps
inner_lr: float = 0.01  # Learning rate for local fine-tuning
meta_lr: float = 0.001  # Learning rate for meta-updates
val_split: float = 0.2  # Fraction of local data for validation
class PersonalizationEvaluator:
"""Evaluate personalization performance"""
def evaluate_personalization_gain(
    self,
    global_model: torch.nn.Module,
    personalized_models: Dict[str, torch.nn.Module],
    test_data_per_device: Dict[str, DataLoader],
) -> Dict[str, float]:
    """
    Compare personalized model performance vs. global model.
    
    Returns:
        {
            "global_accuracy": X,
            "personalized_accuracy": Y,
            "personalization_gain": Y - X,
        }
    """
    global_accuracies = []
    personalized_accuracies = []
    
    for device_id, test_data in test_data_per_device.items():
        # Evaluate global model
        global_acc = self._evaluate_model(global_model, test_data)
        global_accuracies.append(global_acc)
        
        # Evaluate personalized model
        personalized_model = personalized_models[device_id]
        personalized_acc = self._evaluate_model(personalized_model, test_data)
        personalized_accuracies.append(personalized_acc)
    
    avg_global = np.mean(global_accuracies)
    avg_personalized = np.mean(personalized_accuracies)
    
    return {
        "global_accuracy": avg_global,
        "personalized_accuracy": avg_personalized,
        "personalization_gain": avg_personalized - avg_global,
        "per_device_accuracies": {
            device_id: (global_accuracies[i], personalized_accuracies[i])
            for i, device_id in enumerate(test_data_per_device.keys())
        },
    }

def _evaluate_model(
    self,
    model: torch.nn.Module,
    test_data: DataLoader,
) -> float:
    """Compute accuracy on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_data:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0.0

**Tests:**
```python
def test_maml_personalization():
    """
    Test: Train 100 devices with diverse local data using MAML.
    Expected: Personalized models outperform global model by >5% accuracy.
    """
    pass

def test_reptile_personalization():
    """
    Test: Compare MAML vs. Reptile personalization.
    Expected: Similar accuracy gains, Reptile 2x faster (no double backward).
    """
    pass

def test_few_shot_adaptation():
    """
    Test: New device with only 10 training samples fine-tunes global model.
    Expected: After 5 gradient steps, accuracy improves from 60% to 75%.
    """
    pass

def test_personalization_forgetting():
    """
    Test: Device fine-tunes, then receives new global model. How much does it forget?
    Expected: Personalized model retains >80% of local adaptations.
    """
    pass

def test_meta_learning_convergence():
    """
    Test: Train global model with MAML for 500 rounds.
    Expected: New devices can adapt in fewer steps as training progresses.
    """
    pass

def test_heterogeneous_personalization():
    """
    Test: Devices with very different data distributions (e.g., English vs. Chinese text).
    Expected: Personalized models specialize correctly, global model maintains both.
    """
    pass
```

---

## TURN 12 — Production Deployment: Full System Integration & Monitoring

**Instructions:**

Deploy the complete federated learning system to production with comprehensive monitoring, alerting, and operational tooling.

**Requirements:**
- Complete deployment on Kubernetes cluster (10+ server nodes, 1000+ simulated mobile clients)
- Full observability stack (metrics, logs, traces, dashboards)
- Automated anomaly detection and alerting
- A/B testing framework for algorithm variants
- Model versioning and rollback capability
- Compliance reporting (GDPR, privacy audit logs)

**Implement:**
```python
from dataclasses import dataclass
from typing import Optional, List, Dict
import prometheus_client as prom
from opentelemetry import trace, metrics
import logging
import json

# Prometheus metrics
GRADIENT_AGGREGATION_LATENCY = prom.Histogram(
    'federated_gradient_aggregation_seconds',
    'Time spent aggregating gradients',
    ['algorithm', 'num_workers'],
)

PRIVACY_BUDGET_REMAINING = prom.Gauge(
    'federated_privacy_budget_remaining',
    'Remaining privacy budget (epsilon)',
    ['model_version'],
)

BYZANTINE_WORKERS_DETECTED = prom.Counter(
    'federated_byzantine_workers_total',
    'Number of Byzantine workers detected',
    ['detection_method'],
)

DEVICE_PARTICIPATION = prom.Counter(
    'federated_device_participation_total',
    'Number of times each device participated',
    ['device_id', 'device_type'],
)

MODEL_ACCURACY = prom.Gauge(
    'federated_model_accuracy',
    'Current model accuracy on validation set',
    ['model_version', 'metric_type'],
)


class ProductionFederatedServer:
    """Production-ready federated learning server"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        
        # Core components
        self.aggregator = self._create_aggregator()
        self.model_registry = ModelRegistry(config.model_registry)
        self.device_manager = DeviceManager(config.device_manager)
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(config.alerting)
        
        # A/B testing
        self.experiment_manager = ExperimentManager()
        
        # Compliance
        self.audit_logger = AuditLogger(config.compliance)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def run_training_round(self, round_num: int) -> TrainingRoundResult:
        """Execute one training round with full observability"""
        
        with trace.get_tracer(__name__).start_as_current_span("training_round") as span:
            span.set_attribute("round_num", round_num)
            
            try:
                # 1. Select devices
                selected_devices = self._select_devices_for_round(round_num)
                span.set_attribute("num_devices", len(selected_devices))
                
                # 2. Send global model to devices
                global_model = self.model_registry.get_latest_model()
                self._distribute_model(global_model, selected_devices)
                
                # 3. Collect gradients from devices
                worker_gradients = self._collect_gradients(
                    selected_devices,
                    timeout=self.config.gradient_timeout,
                )
                
                # 4. Aggregate with monitoring
                with GRADIENT_AGGREGATION_LATENCY.labels(
                    algorithm=self.config.aggregation_algorithm,
                    num_workers=len(worker_gradients),
                ).time():
                    aggregated = self.aggregator.aggregate(
                        worker_gradients,
                        global_model.state_dict(),
                    )
                
                # 5. Update model
                updated_model = self._update_model(global_model, aggregated)
                
                # 6. Evaluate and log metrics
                metrics = self._evaluate_model(updated_model, round_num)
                self._log_metrics(metrics, round_num)
                
                # 7. Detect anomalies
                anomalies = self.anomaly_detector.detect(metrics, worker_gradients)
                if anomalies:
                    self._handle_anomalies(anomalies, round_num)
                
                # 8. Update privacy budget
                self._update_privacy_accounting(round_num)
                
                # 9. Audit logging
                self.audit_logger.log_round(
                    round_num=round_num,
                    participating_devices=selected_devices,
                    metrics=metrics,
                )
                
                # 10. Model versioning
                self.model_registry.save_model(
                    updated_model,
                    version=f"round_{round_num}",
                    metrics=metrics,
                )
                
                return TrainingRoundResult(
                    success=True,
                    round_num=round_num,
                    metrics=metrics,
                    num_participants=len(worker_gradients),
                )
                
            except Exception as e:
                self.logger.error(f"Training round {round_num} failed: {e}")
                span.record_exception(e)
                self.alert_manager.send_alert(
                    severity="critical",
                    message=f"Training round {round_num} failed: {e}",
                )
                return TrainingRoundResult(
                    success=False,
                    round_num=round_num,
                    error=str(e),
                )
    
    def _select_devices_for_round(self, round_num: int) -> List[str]:
        """Select devices with A/B testing support"""
        
        # Get available devices
        available = self.device_manager.get_available_devices()
        
        # A/B testing: Assign devices to experimental groups
        if self.experiment_manager.has_active_experiments():
            return self.experiment_manager.assign_devices_to_groups(
                available,
                round_num,
            )
        
        # Standard device selection
        return self.device_manager.select_devices(
            available,
            target_num=self.config.devices_per_round,
        )
    
    def _evaluate_model(
        self,
        model: torch.nn.Module,
        round_num: int,
    ) -> Dict[str, float]:
        """Evaluate model and update Prometheus metrics"""
        
        # Standard metrics
        val_accuracy = self._compute_accuracy(model, self.config.val_dataset)
        val_loss = self._compute_loss(model, self.config.val_dataset)
        
        # Update Prometheus
        MODEL_ACCURACY.labels(
            model_version=f"round_{round_num}",
            metric_type="accuracy",
        ).set(val_accuracy)
        
        MODEL_ACCURACY.labels(
            model_version=f"round_{round_num}",
            metric_type="loss",
        ).set(val_loss)
        
        # Fairness metrics (if applicable)
        fairness_metrics = self._compute_fairness_metrics(model)
        
        return {
            "accuracy": val_accuracy,
            "loss": val_loss,
            **fairness_metrics,
        }
    
    def _handle_anomalies(self, anomalies: List[Anomaly], round_num: int):
        """Handle detected anomalies"""
        
        for anomaly in anomalies:
            self.logger.warning(f"Anomaly detected in round {round_num}: {anomaly}")
            
            if anomaly.severity == "critical":
                # Halt training
                self.alert_manager.send_alert(
                    severity="critical",
                    message=f"Critical anomaly: {anomaly.description}",
                )
                raise RuntimeError(f"Critical anomaly detected: {anomaly}")
            
            elif anomaly.severity == "high":
                # Rollback model
                self.logger.info("Rolling back to previous model version")
                previous_model = self.model_registry.get_model(version="previous")
                self.model_registry.set_latest_model(previous_model)
                
            elif anomaly.severity == "medium":
                # Adjust hyperparameters
                self._adjust_training_parameters(anomaly)
    
    def _update_privacy_accounting(self, round_num: int):
        """Update privacy budget tracking"""
        
        if not self.config.differential_privacy_enabled:
            return
        
        # Get current privacy spent
        epsilon = self.aggregator.get_privacy_spent()
        
        # Update metric
        PRIVACY_BUDGET_REMAINING.labels(
            model_version=f"round_{round_num}",
        ).set(self.config.privacy_budget - epsilon)
        
        # Alert if budget running low
        if epsilon > 0.9 * self.config.privacy_budget:
            self.alert_manager.send_alert(
                severity="warning",
                message=f"Privacy budget 90% consumed: ε={epsilon:.2f}/{self.config.privacy_budget}",
            )


class AnomalyDetector:
    """Detect anomalies in training metrics"""
    
    def __init__(self):
        self.metric_history = []
        self.gradient_norms_history = []
        
    def detect(
        self,
        current_metrics: Dict[str, float],
        worker_gradients: List[WorkerGradient],
    ) -> List[Anomaly]:
        """Detect anomalies using statistical methods"""
        
        anomalies = []
        
        # 1. Check for sudden accuracy drop
        if len(self.metric_history) > 0:
            prev_accuracy = self.metric_history[-1].get("accuracy", 0)
            curr_accuracy = current_metrics.get("accuracy", 0)
            
            if curr_accuracy < prev_accuracy - 0.1:  # 10% drop
                anomalies.append(Anomaly(
                    type="accuracy_drop",
                    severity="high",
                    description=f"Accuracy dropped from {prev_accuracy:.3f} to {curr_accuracy:.3f}",
                ))
        
        # 2. Check for gradient explosion
        gradient_norms = [
            self._compute_gradient_norm(wg.gradients)
            for wg in worker_gradients
        ]
        
        median_norm = np.median(gradient_norms)
        for i, norm in enumerate(gradient_norms):
            if norm > 10 * median_norm:  # Outlier
                anomalies.append(Anomaly(
                    type="gradient_explosion",
                    severity="medium",
                    description=f"Worker {worker_gradients[i].worker_id} has exploding gradient: {norm:.2e}",
                ))
        
        # 3. Check for coordinated Byzantine attack
        if self._detect_coordinated_attack(worker_gradients):
            anomalies.append(Anomaly(
                type="coordinated_byzantine_attack",
                severity="critical",
                description="Multiple workers sending correlated malicious gradients",
            ))
        
        # Store history
        self.metric_history.append(current_metrics)
        self.gradient_norms_history.append(gradient_norms)
        
        return anomalies
    
    def _detect_coordinated_attack(
        self,
        worker_gradients: List[WorkerGradient],
    ) -> bool:
        """Detect if multiple workers are sending correlated poison"""
        
        if len(worker_gradients) < 3:
            return False
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(worker_gradients)):
            for j in range(i + 1, len(worker_gradients)):
                corr = self._gradient_correlation(
                    worker_gradients[i].gradients,
                    worker_gradients[j].gradients,
                )
                correlations.append(corr)
        
        # If many pairs have very high correlation, suspicious
        high_corr_count = sum(1 for c in correlations if c > 0.95)
        
        return high_corr_count > len(worker_gradients) // 2
    
    def _gradient_correlation(
        self,
        grad1: Dict[str, torch.Tensor],
        grad2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute correlation between two gradients"""
        # Flatten and compute Pearson correlation
        flat1 = torch.cat([g.flatten() for g in grad1.values()])
        flat2 = torch.cat([g.flatten() for g in grad2.values()])
        
        return torch.corrcoef(torch.stack([flat1, flat2]))[0, 1].item()


class ExperimentManager:
    """A/B testing for federated learning algorithms"""
    
    def __init__(self):
        self.experiments = {}
        self.device_assignments = {}
        
    def create_experiment(
        self,
        experiment_name: str,
        variants: List[str],
        traffic_split: List[float],
    ):
        """Create A/B test experiment"""
        assert len(variants) == len(traffic_split)
        assert sum(traffic_split) == 1.0
        
        self.experiments[experiment_name] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "metrics": {v: [] for v in variants},
        }
    
    def assign_devices_to_groups(
        self,
        devices: List[str],
        round_num: int,
    ) -> Dict[str, List[str]]:
        """Assign devices to experimental groups"""
        
        # For simplicity, assume one active experiment
        experiment = list(self.experiments.values())[0]
        variants = experiment["variants"]
        split = experiment["traffic_split"]
        
        # Randomly assign based on split
        assignments = {v: [] for v in variants}
        
        for device_id in devices:
            # Deterministic assignment based on device_id hash
            device_hash = hash(device_id + str(round_num))
            variant_idx = 0
            cumulative = 0.0
            
            for i, prob in enumerate(split):
                cumulative += prob
                if (device_hash % 1000) / 1000.0 < cumulative:
                    variant_idx = i
                    break
            
            assignments[variants[variant_idx]].append(device_id)
        
        return assignments


@dataclass
class Anomaly:
    type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ProductionConfig:
    aggregation_algorithm: str
    devices_per_round: int
    gradient_timeout: float
    privacy_budget: float
    differential_privacy_enabled: bool
    model_registry: Dict
    device_manager: Dict
    alerting: Dict
    compliance: Dict
    val_dataset: Optional[Any] = None


@dataclass
class TrainingRoundResult:
    success: bool
    round_num: int
    metrics: Dict[str, float] = None
    num_participants: int = 0
    error: Optional[str] = None
```

**Deployment artifacts:**
```yaml
# kubernetes/federated-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: federated-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: federated-server
  template:
    metadata:
      labels:
        app: federated-server
    spec:
      containers:
      - name: server
        image: federated-learning:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: REDIS_ADDR
          value: "redis-service:6379"
        - name: PROMETHEUS_PORT
          value: "9090"
        ports:
        - containerPort: 8080
          name: grpc
        - containerPort: 9090
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: federated-server
spec:
  selector:
    app: federated-server
  ports:
  - name: grpc
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

**Tests:**
```python
def test_end_to_end_production_training():
    """
    Test: Run 100 training rounds with full production setup.
    - 10 server replicas
    - 1000 simulated mobile clients
    - All features enabled (compression, DP, Byzantine, async, personalization)
    
    Expected:
    - Model converges to >85% accuracy
    - Zero crashes or data loss
    - All metrics tracked correctly
    - Privacy budget not exceeded
    - No Byzantine workers compromise model
    """
    pass

def test_server_failover():
    """
    Test: Kill primary server mid-round.
    Expected: Backup server takes over, round completes successfully.
    """
    pass

def test_model_rollback():
    """
    Test: Model accuracy drops by 15% in one round (simulate bug).
    Expected: Anomaly detector triggers, model rolls back to previous version.
    """
    pass

def test_privacy_budget_enforcement():
    """
    Test: Train with DP (ε=1.0) for 1000 rounds.
    Expected: Training stops when privacy budget exhausted.
    """
    pass

def test_ab_testing_convergence():
    """
    Test: Run A/B test comparing FedAvg vs. FedProx.
    Expected: Both converge, FedProx 5% better with heterogeneous data.
    """
    pass

def test_compliance_audit_log():
    """
    Test: Verify all training rounds are logged for compliance.
    Expected: Audit log contains all device IDs, timestamps, gradients metadata.
    """
    pass

def test_scaling_to_10k_devices():
    """
    Test: Simulate 10,000 devices, 1000 selected per round.
    Expected: Server handles load, latency <10s per round.
    """
    passDeliverables:

Full production-grade federated learning system
Kubernetes deployment manifests
Monitoring dashboards (Grafana/Prometheus)
Complete observability stack (metrics, logs, traces)
A/B testing framework
Compliance and audit logging
Comprehensive documentation (deployment guide, troubleshooting, API docs)
All integration tests passing