# Task: Contribute ML-Based I/O Scheduler to Linux Kernel

**Repository:** https://github.com/torvalds/linux  
**Version:** Linux Kernel 6.11+  
**Created:** November 22, 2025  
**Language:** C  
**Difficulty:** EXTREME

## Overview
Implement a production-ready I/O scheduler for the Linux kernel that uses lightweight machine learning to predict I/O access patterns and optimize scheduling decisions. This contribution must integrate with the kernel's multiqueue block layer, handle all I/O types (sequential, random, mixed), provide better tail latency than existing schedulers, and be suitable for upstream merge. Requires deep kernel expertise, understanding of storage devices, and ability to implement ML inference in constrained kernel environment.

**Key Challenge:** This must be production-quality kernel code with zero bugs. You cannot use floating-point in the kernel, must handle all error paths, avoid deadlocks, and pass extensive testing.

---

## TURN 1 — Understanding Linux Block Layer Architecture

**Role:** You are a Linux kernel developer who understands the block layer, I/O schedulers, and storage stack inside-out. You've read `Documentation/block/` and can explain how requests flow from filesystems to devices.

**Background:** Linux uses a multi-queue (blk-mq) architecture where I/O requests are queued and dispatched to devices. Schedulers determine the order of dispatch to optimize for throughput, latency, or fairness. Existing schedulers: mq-deadline (deadline-based), BFQ (bandwidth fairness), Kyber (latency-focused), none (no scheduling).

**Reference:** Study:
- `Documentation/block/blk-mq.rst` - Multi-queue architecture
- `block/mq-deadline.c` - Deadline scheduler implementation
- `block/kyber-iosched.c` - Kyber scheduler
- `block/blk-mq.c` - Core MQ infrastructure

**VERY IMPORTANT:**
- No floating-point arithmetic in kernel (use fixed-point)
- All memory allocations must be GFP_KERNEL or GFP_ATOMIC
- Must handle allocation failures gracefully
- No unbounded loops (risk of hanging kernel)
- Must acquire locks in correct order (avoid deadlocks)
- Cannot call schedule() while holding spinlocks
- Must be fully preemptible (for PREEMPT_RT)

**Goal:** Understand block layer, implement basic I/O scheduler skeleton.

**Instructions:**

1. **Study the block layer:**

Understand how I/O requests flow:
```
VFS → Block Layer → I/O Scheduler → Device Driver → NVMe/SSD

Key structures:
- struct request: Represents an I/O request
- struct request_queue: Queue for a block device
- struct blk_mq_hw_ctx: Per-CPU hardware queue context
- struct elevator_type: I/O scheduler interface
```

2. **Define scheduler data structures:**

```c
// block/ml-iosched.c

/*
 * ML-based I/O Scheduler
 * 
 * Uses lightweight online learning to predict I/O access patterns
 * and optimize scheduling decisions.
 */

#include <linux/blkdev.h>
#include <linux/elevator.h>
#include <linux/bio.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/compiler.h>
#include <linux/rbtree.h>

/*
 * Request metadata for ML prediction
 */
struct ml_request {
    struct request *rq;
    struct rb_node rb_node;
    
    // Features for ML model
    sector_t sector;
    unsigned int data_len;
    int priority;
    u64 submit_time;       // ktime in ns
    
    // Prediction
    u32 predicted_latency; // In microseconds (fixed-point)
    u16 predicted_type;    // 0=sequential, 1=random
};

/*
 * Per-queue scheduler data
 */
struct ml_sched_data {
    struct rb_root sorted_requests;   // Sorted by sector number
    spinlock_t lock;
    
    // Statistics for ML model
    u64 total_requests;
    u64 seq_requests;
    u64 rand_requests;
    sector_t last_sector;
    
    // Simple ML model (linear regression weights)
    s32 weights[8];        // Fixed-point weights (16.16 format)
    u32 bias;
    
    // Performance counters
    u64 total_latency_us;
    u64 dispatched_requests;
    
    // Tunables
    unsigned int batch_size;
    unsigned int max_latency_target_us;
};

/*
 * Per-hardware-queue context
 */
struct ml_hctx_data {
    struct list_head rq_list;
    spinlock_t lock;
};

static struct kmem_cache *ml_request_cache;
```

3. **Implement elevator interface:**

```c
/*
 * Initialize scheduler for a request queue
 */
static int ml_init_queue(struct request_queue *q, struct elevator_type *e)
{
    struct elevator_queue *eq;
    struct ml_sched_data *md;
    
    eq = elevator_alloc(q, e);
    if (!eq)
        return -ENOMEM;
    
    md = kzalloc(sizeof(*md), GFP_KERNEL);
    if (!md) {
        elevator_free(eq);
        return -ENOMEM;
    }
    
    md->sorted_requests = RB_ROOT;
    spin_lock_init(&md->lock);
    
    // Initialize ML model with default weights
    ml_init_model(md);
    
    md->batch_size = 32;  // Configurable via sysfs
    md->max_latency_target_us = 10000;  // 10ms
    
    eq->elevator_data = md;
    q->elevator = eq;
    
    return 0;
}

/*
 * Clean up scheduler
 */
static void ml_exit_queue(struct elevator_queue *e)
{
    struct ml_sched_data *md = e->elevator_data;
    
    // Free all pending requests
    ml_free_all_requests(md);
    
    kfree(md);
}

/*
 * Insert request into scheduler
 */
static void ml_insert_requests(struct blk_mq_hw_ctx *hctx,
                                 struct list_head *list, bool at_head)
{
    struct request_queue *q = hctx->queue;
    struct ml_sched_data *md = q->elevator->elevator_data;
    struct request *rq;
    
    spin_lock(&md->lock);
    
    while (!list_empty(list)) {
        rq = list_first_entry(list, struct request, queuelist);
        list_del_init(&rq->queuelist);
        
        ml_add_request(md, rq);
    }
    
    spin_unlock(&md->lock);
}

/*
 * Dispatch next request to device
 */
static struct request *ml_dispatch_request(struct blk_mq_hw_ctx *hctx)
{
    struct request_queue *q = hctx->queue;
    struct ml_sched_data *md = q->elevator->elevator_data;
    struct request *rq;
    
    spin_lock(&md->lock);
    rq = ml_select_best_request(md);
    if (rq) {
        ml_remove_request(md, rq);
        md->dispatched_requests++;
    }
    spin_unlock(&md->lock);
    
    return rq;
}

static struct elevator_type ml_sched = {
    .ops = {
        .init_sched= ml_init_queue,
        .exit_sched= ml_exit_queue,
        .insert_requests= ml_insert_requests,
        .dispatch_request= ml_dispatch_request,
        .has_work= ml_has_work,
        .completed_request= ml_completed_request,
    },
    .elevator_name = "ml-sched",
    .elevator_owner = THIS_MODULE,
};

static int __init ml_init(void)
{
    ml_request_cache = kmem_cache_create("ml_request",
                                          sizeof(struct ml_request),
                                          0, 0, NULL);
    if (!ml_request_cache)
        return -ENOMEM;
    
    return elv_register(&ml_sched);
}

static void __exit ml_exit(void)
{
    elv_unregister(&ml_sched);
    kmem_cache_destroy(ml_request_cache);
}

module_init(ml_init);
module_exit(ml_exit);

MODULE_AUTHOR("Your Name");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("ML-based I/O Scheduler");
```

4. **Test basic functionality:**

```bash
# Compile kernel module
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules

# Load module
sudo insmod ml-iosched.ko

# Switch scheduler for device
echo ml-sched | sudo tee /sys/block/nvme0n1/queue/scheduler

# Run basic I/O test
sudo fio --name=test --ioengine=libaio --direct=1 --bs=4k --iodepth=32 \
         --rw=randread --size=1G --filename=/dev/nvme0n1

# Check stats
cat /sys/block/nvme0n1/queue/iosched/stats
```

**Deliverables:**
- Working I/O scheduler skeleton
- Loads as kernel module
- Can be selected for a block device
- Handles I/O without crashes

---

## TURN 2 — Lightweight ML Model for Pattern Prediction

**Instructions:**

Implement simple ML model that predicts whether next I/O will be sequential or random.

**Challenge:** No floating-point in kernel! Use fixed-point arithmetic (16.16 format).

**Implement:**

```c
/*
 * Fixed-point arithmetic for kernel ML
 * Format: 16-bit integer, 16-bit fractional (Q16.16)
 */
typedef s32 fixed_t;

#define FIXED_ONE (1 << 16)
#define INT_TO_FIXED(x) ((x) << 16)
#define FIXED_TO_INT(x) ((x) >> 16)

static inline fixed_t fixed_mul(fixed_t a, fixed_t b)
{
    s64 result = (s64)a * (s64)b;
    return (fixed_t)(result >> 16);
}

static inline fixed_t fixed_div(fixed_t a, fixed_t b)
{
    s64 result = ((s64)a << 16) / b;
    return (fixed_t)result;
}

/*
 * Extract features from I/O request for ML prediction
 */
static void ml_extract_features(struct ml_sched_data *md,
                                  struct request *rq,
                                  fixed_t *features)
{
    sector_t sector = blk_rq_pos(rq);
    unsigned int len = blk_rq_bytes(rq);
    
    // Feature 0: Distance from last sector (normalized)
    s64 distance = abs(sector - md->last_sector);
    features[0] = INT_TO_FIXED(min_t(s64, distance, 1024)) / 1024;
    
    // Feature 1: Request size (normalized 0-1)
    features[1] = INT_TO_FIXED(min(len, 1024*1024)) / (1024*1024);
    
    // Feature 2: Sequential likelihood (running average)
    u64 seq_ratio = (md->seq_requests * 100) / max(md->total_requests, 1UL);
    features[2] = INT_TO_FIXED(seq_ratio) / 100;
    
    // Feature 3: Time since last request
    // ... more features ...
}

/*
 * Simple linear model: y = w0*x0 + w1*x1 + ... + bias
 */
static u16 ml_predict_type(struct ml_sched_data *md, fixed_t *features)
{
    fixed_t sum = md->bias;
    int i;
    
    for (i = 0; i < 8; i++) {
        sum += fixed_mul(md->weights[i], features[i]);
    }
    
    // Threshold at 0.5 (fixed-point)
    return (sum > (FIXED_ONE / 2)) ? 1 : 0;  // 1=sequential, 0=random
}

/*
 * Update ML model weights using stochastic gradient descent
 */
static void ml_update_model(struct ml_sched_data *md,
                              fixed_t *features,
                              u16 actual_type)
{
    u16 predicted = ml_predict_type(md, features);
    
    if (predicted == actual_type)
        return;  // Correct prediction, no update needed
    
    // Simple gradient descent update
    fixed_t learning_rate = FIXED_ONE / 100;  // 0.01
    fixed_t error = INT_TO_FIXED(actual_type - predicted);
    
    int i;
    for (i = 0; i < 8; i++) {
        fixed_t gradient = fixed_mul(error, features[i]);
        md->weights[i] += fixed_mul(learning_rate, gradient);
    }
    
    md->bias += fixed_mul(learning_rate, error);
}

/*
 * Determine if I/O is sequential (for training)
 */
static bool is_sequential_io(struct ml_sched_data *md, struct request *rq)
{
    sector_t sector = blk_rq_pos(rq);
    sector_t distance = abs(sector - md->last_sector);
    
    // Sequential if within 256 sectors (~128KB for 512B sectors)
    return (distance < 256);
}

/*
 * Train model on completed request
 */
static void ml_train_on_completion(struct ml_sched_data *md,
                                     struct ml_request *mlrq)
{
    fixed_t features[8];
    u16 actual_type;
    
    ml_extract_features(md, mlrq->rq, features);
    actual_type = mlrq->predicted_type;  // Actual observed pattern
    
    ml_update_model(md, features, actual_type);
}
```

**Testing:**

```c
// Verify fixed-point arithmetic
static int __init test_fixed_point(void)
{
    fixed_t a = INT_TO_FIXED(3);      // 3.0
    fixed_t b = INT_TO_FIXED(2);      // 2.0
    fixed_t c = fixed_mul(a, b);      // 6.0
    
    BUG_ON(FIXED_TO_INT(c) != 6);
    
    fixed_t d = fixed_div(a, b);      // 1.5
    BUG_ON(FIXED_TO_INT(d) != 1);     // Integer part
    
    return 0;
}
```

---

## TURN 3 — Force Failure: Deadlock from Incorrect Locking

**Instructions:**

Introduce a subtle deadlock where lock acquisition order is wrong.

**Ask the AI:**
> "Your scheduler acquires md->lock in both ml_insert_requests and ml_dispatch_request. What happens when a CPU is dispatching (holds md->lock) and another CPU tries to insert while also being interrupted to complete a request (which also needs md->lock)? Show the exact deadlock scenario and how to fix it with proper lock ordering or lockless techniques."

**Expected failure:**

```
CPU0: ml_dispatch_request() holds md->lock
      → calls device driver
      → device interrupts immediately
      → IRQ handler calls ml_completed_request()
      → tries to acquire md->lock → DEADLOCK (same CPU)
```

**Fix:** Use separate locks or lockless structures (per-cpu queues, RCU).

---

## TURN 4 — Latency Prediction Using Historical Data

**Instructions:**

Extend ML model to predict I/O latency, enabling latency-aware scheduling.

**Implement:**

```c
/*
 * Latency histogram for training data
 */
struct latency_histogram {
    u32 buckets[16];      // Latency buckets (exponential)
    u64 total_samples;
};

/*
 * Predict latency for request based on features
 */
static u32 ml_predict_latency_us(struct ml_sched_data *md,
                                   struct request *rq)
{
    fixed_t features[8];
    fixed_t predicted;
    
    ml_extract_features(md, rq, features);
    
    // Linear regression: latency = w · x + b
    predicted = md->bias;
    for (int i = 0; i < 8; i++)
        predicted += fixed_mul(md->weights[i], features[i]);
    
    // Convert fixed-point to microseconds
    return max(FIXED_TO_INT(predicted), 0U);
}

/*
 * Request scheduler uses predicted latency for earliest-deadline-first
 */
static struct request *ml_select_best_request(struct ml_sched_data *md)
{
    struct rb_node *node;
    struct ml_request *mlrq, *best = NULL;
    u32 min_latency = U32_MAX;
    u64 now = ktime_get_ns();
    
    // Simple heuristic: prioritize requests with lowest predicted latency
    // to meet latency targets
    
    for (node = rb_first(&md->sorted_requests); node; node = rb_next(node)) {
        mlrq = rb_entry(node, struct ml_request, rb_node);
        
        // Check if request is aging (submitted long ago)
        u64 age_ns = now - mlrq->submit_time;
        if (age_ns > md->max_latency_target_us * 1000) {
            // Priority to aged requests to avoid starvation
            return mlrq->rq;
        }
        
        if (mlrq->predicted_latency < min_latency) {
            min_latency = mlrq->predicted_latency;
            best = mlrq;
        }
    }
    
    return best ? best->rq : NULL;
}

/*
 * Update model on request completion
 */
static void ml_completed_request(struct request *rq)
{
    struct ml_sched_data *md = rq->q->elevator->elevator_data;
    struct ml_request *mlrq = rq->elv.priv[0];
    u64 actual_latency_ns;
    u32 actual_latency_us;
    
    actual_latency_ns = ktime_get_ns() - mlrq->submit_time;
    actual_latency_us = actual_latency_ns / 1000;
    
    // Update statistics
    md->total_latency_us += actual_latency_us;
    
    // Train model with actual latency
    ml_train_on_latency(md, mlrq, actual_latency_us);
    
    kmem_cache_free(ml_request_cache, mlrq);
}
```

---

## TURN 5 — Batch Processing for Throughput

**Instructions:**

Implement batched dispatch to improve throughput while maintaining latency targets.

**Implement:**

```c
/*
 * Dispatch multiple requests in a batch
 */
static int ml_dispatch_batch(struct blk_mq_hw_ctx *hctx,
                               struct list_head *list)
{
    struct request_queue *q = hctx->queue;
    struct ml_sched_data *md = q->elevator->elevator_data;
    struct request *rq;
    int dispatched = 0;
    int batch_size = md->batch_size;
    
    spin_lock(&md->lock);
    
    while (dispatched < batch_size) {
        rq = ml_select_best_request(md);
        if (!rq)
            break;
        
        ml_remove_request(md, rq);
        list_add_tail(&rq->queuelist, list);
        dispatched++;
    }
    
    spin_unlock(&md->lock);
    
    return dispatched;
}

/*
 * Adaptive batch sizing based on queue depth
 */
static void ml_adjust_batch_size(struct ml_sched_data *md)
{
    unsigned int queue_depth = rb_tree_count(&md->sorted_requests);
    
    if (queue_depth > 1000) {
        md->batch_size = 64;  // Large queue → larger batches for throughput
    } else if (queue_depth > 100) {
        md->batch_size = 32;
    } else {
        md->batch_size = 8;   // Small queue → small batches for latency
    }
}
```

---

## TURN 6 — Sysfs Interface for Runtime Configuration

**Instructions:**

Add sysfs interface for tuning scheduler parameters.

**Implement:**

```c
/*
 * Sysfs attributes for ML scheduler
 */

static ssize_t
ml_batch_size_show(struct elevator_queue *e, char *page)
{
    struct ml_sched_data *md = e->elevator_data;
    return sprintf(page, "%u\n", md->batch_size);
}

static ssize_t
ml_batch_size_store(struct elevator_queue *e, const char *page, size_t count)
{
    struct ml_sched_data *md = e->elevator_data;
    unsigned int val;
    
    if (kstrtouint(page, 10, &val) || val < 1 || val > 256)
        return -EINVAL;
    
    spin_lock(&md->lock);
    md->batch_size = val;
    spin_unlock(&md->lock);
    
    return count;
}

static ssize_t
ml_stats_show(struct elevator_queue *e, char *page)
{
    struct ml_sched_data *md = e->elevator_data;
    u64 avg_latency = 0;
    
    if (md->dispatched_requests > 0)
        avg_latency = md->total_latency_us / md->dispatched_requests;
    
    return sprintf(page,
                   "total_requests: %llu\n"
                   "seq_requests: %llu\n"
                   "rand_requests: %llu\n"
                   "dispatched: %llu\n"
                   "avg_latency_us: %llu\n",
                   md->total_requests,
                   md->seq_requests,
                   md->rand_requests,
                   md->dispatched_requests,
                   avg_latency);
}

#define ML_ATTR(name) \
    __ATTR(name, 0644, ml_##name##_show, ml_##name##_store)

#define ML_ATTR_RO(name) \
    __ATTR(name, 0444, ml_##name##_show, NULL)

static struct elv_fs_entry ml_attrs[] = {
    ML_ATTR(batch_size),
    ML_ATTR(max_latency_target_us),
    ML_ATTR_RO(stats),
    __ATTR_NULL
};

static struct elevator_type ml_sched = {
    .ops = {
        // ... ops ...
    },
    .elevator_attrs = ml_attrs,
    .elevator_name = "ml-sched",
    .elevator_owner = THIS_MODULE,
};
```

**Usage:**

```bash
# View stats
cat /sys/block/nvme0n1/queue/iosched/stats

# Tune batch size
echo 64 > /sys/block/nvme0n1/queue/iosched/batch_size

# Set latency target
echo 5000 > /sys/block/nvme0n1/queue/iosched/max_latency_target_us
```

---

## TURN 7 — Priority and Cgroup Integration

**Instructions:**

Integrate with Linux cgroups to provide I/O priority and isolation per container/process.

**Implement:**

```c
/*
 * Get I/O priority from cgroup or request
 */
static int ml_get_request_priority(struct request *rq)
{
    struct bio *bio = rq->bio;
    
    if (!bio)
        return IOPRIO_NORM;
    
    // Extract I/O priority from bio
    return IOPRIO_PRIO_VALUE(bio->bi_ioprio);
}

/*
 * Schedule requests considering priority
 */
static struct request *ml_select_request_with_priority(struct ml_sched_data *md)
{
    struct rb_node *node;
    struct ml_request *mlrq, *best_high = NULL, *best_norm = NULL;
    
    for (node = rb_first(&md->sorted_requests); node; node = rb_next(node)) {
        mlrq = rb_entry(node, struct ml_request, rb_node);
        
        if (mlrq->priority >= IOPRIO_CLASS_RT) {
            // Real-time I/O - dispatch immediately
            if (!best_high || mlrq->submit_time < best_high->submit_time)
                best_high = mlrq;
        } else {
            // Normal I/O - use ML prediction
            if (!best_norm || mlrq->predicted_latency < best_norm->predicted_latency)
                best_norm = mlrq;
        }
    }
    
    // Prefer high-priority if available
    return best_high ? best_high->rq : (best_norm ? best_norm->rq : NULL);
}
```

---

## TURN 8 — Comprehensive Testing with blktests

**Instructions:**

Write tests using the kernel's blktests framework.

**Tests:**

```bash
# Install blktests
git clone https://github.com/osandov/blktests.git
cd blktests

# Create test for ML scheduler
cat > tests/block/035 << 'EOF'
#!/bin/bash
# Test ML scheduler under various workloads

. tests/block/rc

DESCRIPTION="test ML I/O scheduler"

requires() {
    _have_module ml-iosched
}

test() {
    echo "Testing ML scheduler"
    
    local dev=/dev/nvme0n1
    
    # Switch to ML scheduler
    echo ml-sched > /sys/block/$(basename $dev)/queue/scheduler
    
    # Test 1: Sequential read
    fio --name=seqread --ioengine=libaio --direct=1 --bs=128k \
        --iodepth=32 --rw=read --size=1G --filename=$dev \
        --output-format=json > /tmp/seqread.json
    
    # Test 2: Random read
    fio --name=randread --ioengine=libaio --direct=1 --bs=4k \
        --iodepth=64 --rw=randread --size=1G --filename=$dev \
        --output-format=json > /tmp/randread.json
    
    # Test 3: Mixed workload
    fio --name=mixed --ioengine=libaio --direct=1 --bs=4k \
        --iodepth=32 --rw=randrw --size=1G --filename=$dev \
        --output-format=json > /tmp/mixed.json
    
    # Verify no errors
    grep -q "error" /tmp/*.json && return 1
    
    # Check stats
    cat /sys/block/$(basename $dev)/queue/iosched/stats
    
    echo "PASSED"
    return 0
}
EOF

chmod +x tests/block/035

# Run test
sudo ./check block/035
```

---

## TURN 9 — Performance Benchmarking vs Existing Schedulers

**Instructions:**

Benchmark ML scheduler against all existing schedulers (none, mq-deadline, BFQ, Kyber).

**Benchmark script:**

```bash
#!/bin/bash
# compare_schedulers.sh

DEVICE=/dev/nvme0n1
SCHEDULERS="none mq-deadline bfq kyber ml-sched"
WORKLOADS="seqread seqwrite randread randwrite randrw"

for sched in $SCHEDULERS; do
    echo "Testing scheduler: $sched"
    echo $sched > /sys/block/$(basename $DEVICE)/queue/scheduler
    
    for workload in $WORKLOADS; do
        echo "  Workload: $workload"
        
        fio --name=$workload \
            --filename=$DEVICE \
            --ioengine=libaio \
            --direct=1 \
            --bs=4k \
            --iodepth=32 \
            --rw=$workload \
            --runtime=60 \
            --time_based \
            --output-format=json \
            --output=results/${sched}_${workload}.json
    done
done

# Parse results
python3 parse_results.py results/*.json > comparison.csv
```

**Expected results:**

```
Scheduler  | SeqRead | RandRead | RandWrite | Avg Latency
-----------|---------|----------|-----------|-------------
none       | 3.2GB/s | 580MB/s  | 520MB/s   | 45μs
mq-deadline| 3.1GB/s | 620MB/s  | 550MB/s   | 42μs
BFQ        | 2.9GB/s | 590MB/s  | 530MB/s   | 48μs
Kyber      | 3.2GB/s | 650MB/s  | 570MB/s   | 38μs
ml-sched   | 3.2GB/s | 680MB/s  | 590MB/s   | 35μs ✅

Goal: Match or exceed best schedulers ✅
```

---

## TURN 10 — Documentation and Upstream Submission

**Instructions:**

Write comprehensive documentation in kernel-doc format and prepare for upstream submission.

**Documentation:**

```c
/**
 * DOC: ML-based I/O Scheduler
 *
 * This scheduler uses lightweight machine learning to predict I/O
 * access patterns and latency, enabling better scheduling decisions
 * for mixed workloads.
 *
 * Features:
 * - Online learning (updates model during operation)
 * - Fixed-point arithmetic (no floating-point in kernel)
 * - Latency prediction for deadline-based scheduling
 * - Pattern prediction (sequential vs random)
 * - Cgroup integration for I/O priority
 *
 * Algorithm:
 * 1. Extract features from I/O request (sector, size, history)
 * 2. Predict latency and pattern using linear model
 * 3. Schedule requests to minimize tail latency
 * 4. Update model weights on request completion
 *
 * Tunable parameters (via sysfs):
 * - batch_size: Number of requests to dispatch per batch
 * - max_latency_target_us: Maximum acceptable latency target
 *
 * See Documentation/block/ml-iosched.rst for details.
 */
```

**Patch submission:**

```
From: Your Name <your.email@example.com>
Date: Fri, 22 Nov 2025 08:17:14 +0300
Subject: [PATCH v1] block: Add ML-based I/O scheduler

This patch introduces a new I/O scheduler that uses lightweight
machine learning to predict I/O access patterns and optimize
scheduling decisions.

The scheduler uses a simple linear regression model with fixed-point
arithmetic (no floating-point) to predict request latency and
pattern (sequential vs random). The model is trained online as
requests complete.

Performance testing shows 15-20% improvement in tail latency for
mixed workloads compared to existing schedulers.

Signed-off-by: Your Name <your.email@example.com>
---
 block/Kconfig.iosched  |   11 +
 block/Makefile         |    1 +
 block/ml-iosched.c     | 1247 ++++++++++++++++++++++++++++++++++++++
 3 files changed, 1259 insertions(+)
 create mode 100644 block/ml-iosched.c
```

---

## TURN 11 — Stress Testing and Bug Fixes

**Instructions:**

Perform extensive stress testing to find and fix bugs.

**Stress tests:**

```c
// Test concurrent access from multiple CPUs
static int stress_test_concurrency(void)
{
    // Spawn 16 threads, each submitting I/O
    // Check for races, deadlocks, crashes
}

// Test memory allocation failures
static int stress_test_oom(void)
{
    // Inject allocation failures
    // Verify graceful handling
}

// Test device errors
static int stress_test_device_errors(void)
{
    // Simulate device timeouts, errors
    // Verify recovery without kernel panic
}
```

**Known issues to fix:**

1. **Race condition in request removal** - Fixed with RCU
2. **Memory leak on scheduler exit** - Fixed by freeing all requests
3. **Potential overflow in latency calculation** - Fixed with u64
4. **Spinlock held too long** - Fixed with finer-grained locking

**Final deliverables:**
- Production-ready kernel module
- All blktests passing
- Performance benchmarks showing improvement
- Full documentation
- Patch series ready for LKML submission
