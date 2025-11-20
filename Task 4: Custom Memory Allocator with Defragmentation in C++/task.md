# Task: Build a Production-Grade Custom Memory Allocator with Defragmentation

## Overview
Implement a zero-overhead custom memory allocator in C++20 that supports real-time defragmentation, thread-safe operations, and memory compaction under fragmentation pressure. This allocator must outperform `malloc`/`new` for real-world workloads, handle pathological fragmentation patterns, guarantee deterministic performance, and work correctly under extreme concurrency with zero memory leaks or corruption.

**Key Challenge:** You CANNOT use existing allocators (tcmalloc, jemalloc, mimalloc). Everything must be built from scratch using only `mmap`/`munmap`.

---

## TURN 1 — Core Allocator Architecture with Segregated Free Lists

**Role:** You are a systems programmer who has built memory allocators for game engines, databases, or embedded systems. You understand cache-line alignment, false sharing, memory fragmentation, and the trade-offs between speed and memory efficiency.

**Background:** We need a general-purpose allocator that handles small (8-256 bytes), medium (256B-64KB), and large (>64KB) allocations efficiently. Naive bump allocators fragment. Naive free-list allocators are slow. You must design a hybrid approach.

**Reference:** Study:
- Doug Lea's malloc algorithm (dlmalloc)
- TCMalloc's thread-local caching strategy
- Hoard allocator's per-thread heaps
- Linux kernel's SLUB allocator

**VERY IMPORTANT:**
- All allocations must be 16-byte aligned minimum (SSE/NEON requirements)
- Zero internal fragmentation for power-of-2 sizes
- All operations must be thread-safe with lock-free fast paths where possible
- No memory leaks even under OOM conditions
- Must pass Valgrind and AddressSanitizer checks with zero errors

**Goal:** Implement core allocator with segregated free lists for different size classes.

**Instructions:**

1. **Design the allocator architecture** covering:
   - Memory layout strategy (how to organize virtual address space)
   - Size class design (which sizes get dedicated free lists)
   - Metadata storage (where to store allocation headers without overhead)
   - How to request memory from OS (mmap flags, huge pages)
   - Thread-local vs. global allocations strategy
   - Coalescing strategy for adjacent free blocks

2. **Implement core structure:**
```cpp
#include <cstddef>
#include <atomic>
#include <mutex>
#include <sys/mman.h>

namespace custom_alloc {

// Size classes: 16, 32, 48, 64, 96, 128, 192, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K
constexpr size_t NUM_SIZE_CLASSES = 16;
constexpr size_t MAX_SMALL_SIZE = 65536;
constexpr size_t ALIGNMENT = 16;
constexpr size_t CACHE_LINE_SIZE = 64;

// Block header (stored before each allocation)
struct BlockHeader {
    size_t size;           // Includes header size
    bool is_free;
    BlockHeader* next;     // For free list
    BlockHeader* prev;     // For coalescing
    uint32_t magic;        // For corruption detection
    uint32_t thread_id;    // Which thread allocated this
} __attribute__((aligned(ALIGNMENT)));

static_assert(sizeof(BlockHeader) == 32, "BlockHeader must be 32 bytes");

// Per-thread cache to avoid lock contention
struct ThreadCache {
    BlockHeader* free_lists[NUM_SIZE_CLASSES];
    size_t cached_bytes;
    static constexpr size_t MAX_CACHE_BYTES = 2 * 1024 * 1024; // 2MB
    
    ThreadCache();
    ~ThreadCache();
};

// Global allocator
class MemoryAllocator {
public:
    static MemoryAllocator& instance();
    
    void* allocate(size_t size, size_t alignment = ALIGNMENT);
    void deallocate(void* ptr);
    size_t get_allocation_size(void* ptr) const;
    
    // Statistics
    struct Stats {
        std::atomic<size_t> total_allocated{0};
        std::atomic<size_t> total_freed{0};
        std::atomic<size_t> active_allocations{0};
        std::atomic<size_t> os_memory_mapped{0};
        std::atomic<size_t> fragmentation_bytes{0};
    };
    
    const Stats& get_stats() const { return stats_; }
    
private:
    MemoryAllocator();
    ~MemoryAllocator();
    
    // Disable copy/move
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    
    size_t size_class_index(size_t size) const;
    void* allocate_small(size_t size_class_idx);
    void* allocate_large(size_t size);
    
    void* request_from_os(size_t size);
    void return_to_os(void* ptr, size_t size);
    
    BlockHeader* coalesce(BlockHeader* block);
    void split_block(BlockHeader* block, size_t required_size);
    
    // Per-size-class free lists (global)
    std::mutex size_class_locks_[NUM_SIZE_CLASSES];
    BlockHeader* size_class_free_lists_[NUM_SIZE_CLASSES];
    
    // Large allocation tracking
    std::mutex large_alloc_lock_;
    std::unordered_map<void*, size_t> large_allocations_;
    
    Stats stats_;
    
    static thread_local ThreadCache thread_cache_;
};

} // namespace custom_alloc

// Global operators (optional, for drop-in replacement)
void* operator new(size_t size);
void operator delete(void* ptr) noexcept;
void* operator new[](size_t size);
void operator delete[](void* ptr) noexcept;
```

3. **Implement key algorithms:**

```cpp
// Map size to size class index
size_t MemoryAllocator::size_class_index(size_t size) const {
    // Round up to nearest size class
    // Example: 20 bytes -> 32 byte class (index 1)
    if (size <= 16) return 0;
    if (size <= 32) return 1;
    // ... implement full mapping
    
    // Use bit manipulation for fast size class lookup
    // Or use lookup table
    throw std::runtime_error("Implement me");
}

void* MemoryAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    if (size > MAX_SMALL_SIZE) {
        return allocate_large(size);
    }
    
    // Try thread-local cache first (lock-free fast path)
    size_t sc_idx = size_class_index(size);
    ThreadCache& tc = thread_cache_;
    
    if (tc.free_lists[sc_idx] != nullptr) {
        BlockHeader* block = tc.free_lists[sc_idx];
        tc.free_lists[sc_idx] = block->next;
        block->is_free = false;
        tc.cached_bytes -= block->size;
        stats_.active_allocations.fetch_add(1, std::memory_order_relaxed);
        return reinterpret_cast<char*>(block) + sizeof(BlockHeader);
    }
    
    // Cache miss - get from global free list (slow path with lock)
    return allocate_small(sc_idx);
}

void* MemoryAllocator::allocate_small(size_t size_class_idx) {
    std::lock_guard<std::mutex> lock(size_class_locks_[size_class_idx]);
    
    BlockHeader* block = size_class_free_lists_[size_class_idx];
    if (block != nullptr) {
        // Found free block
        size_class_free_lists_[size_class_idx] = block->next;
        block->is_free = false;
        block->next = nullptr;
        stats_.active_allocations.fetch_add(1, std::memory_order_relaxed);
        return reinterpret_cast<char*>(block) + sizeof(BlockHeader);
    }
    
    // No free blocks - allocate new span from OS
    // Allocate 64KB span and split into blocks of this size class
    void* span = request_from_os(65536);
    // ... split span into blocks and add to free list
    // ... retry allocation
    
    throw std::runtime_error("Implement me");
}

void MemoryAllocator::deallocate(void* ptr) {
    if (ptr == nullptr) return;
    
    BlockHeader* block = reinterpret_cast<BlockHeader*>(
        reinterpret_cast<char*>(ptr) - sizeof(BlockHeader)
    );
    
    // Validate magic number (detect corruption)
    if (block->magic != 0xDEADBEEF) {
        throw std::runtime_error("Heap corruption detected!");
    }
    
    if (block->is_free) {
        throw std::runtime_error("Double free detected!");
    }
    
    // Return to thread-local cache if space available
    ThreadCache& tc = thread_cache_;
    if (tc.cached_bytes < ThreadCache::MAX_CACHE_BYTES) {
        block->is_free = true;
        size_t sc_idx = size_class_index(block->size - sizeof(BlockHeader));
        block->next = tc.free_lists[sc_idx];
        tc.free_lists[sc_idx] = block;
        tc.cached_bytes += block->size;
        stats_.active_allocations.fetch_sub(1, std::memory_order_relaxed);
        return;
    }
    
    // Cache full - return to global free list
    // ... implement
}
```

4. **Write comprehensive tests:**
```cpp
TEST(MemoryAllocator, BasicAllocation) {
    auto& alloc = MemoryAllocator::instance();
    
    void* ptr = alloc.allocate(64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 16, 0); // Check alignment
    
    // Write to memory to ensure it's valid
    std::memset(ptr, 0xAA, 64);
    
    alloc.deallocate(ptr);
}

TEST(MemoryAllocator, StressTest) {
    // Allocate and free 1 million blocks in random order
    std::vector<void*> ptrs;
    for (int i = 0; i < 1000000; i++) {
        size_t size = (rand() % 1024) + 8;
        ptrs.push_back(MemoryAllocator::instance().allocate(size));
    }
    
    // Free in random order
    std::shuffle(ptrs.begin(), ptrs.end(), std::mt19937{});
    for (void* ptr : ptrs) {
        MemoryAllocator::instance().deallocate(ptr);
    }
    
    // Check no memory leaks
    auto stats = MemoryAllocator::instance().get_stats();
    EXPECT_EQ(stats.active_allocations.load(), 0);
}

TEST(MemoryAllocator, ThreadSafety) {
    // 8 threads, each allocating 100k blocks
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; t++) {
        threads.emplace_back([]() {
            std::vector<void*> local_ptrs;
            for (int i = 0; i < 100000; i++) {
                local_ptrs.push_back(MemoryAllocator::instance().allocate(64));
            }
            for (void* ptr : local_ptrs) {
                MemoryAllocator::instance().deallocate(ptr);
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    // No crashes = success
}
```

**Deliverables:**
- Full implementation with thread-local caching
- All tests passing under Valgrind (zero leaks)
- Performance benchmark: >5M alloc/free per second single-threaded
- Documentation of size class design decisions

---

## TURN 2 — Memory Defragmentation Engine

**Instructions:**

Implement a defragmentation engine that compacts memory by relocating objects.

**Challenge:** Moving objects breaks pointers! You must implement a handle system or GC-like relocation.

**Requirements:**
- Detect fragmentation (many small free blocks scattered)
- Compact memory by moving objects together
- Update all pointers to moved objects
- Concurrent with allocations (pause-the-world not acceptable for long)
- Reduce fragmentation from 90% to <10% in single defrag pass

**Implement:**
```cpp
class DefragmentationEngine {
public:
    struct DefragStats {
        size_t bytes_moved;
        size_t objects_relocated;
        size_t free_blocks_merged;
        std::chrono::microseconds duration;
    };
    
    // Async defragmentation callback
    using RelocationCallback = std::function<void(void* old_addr, void* new_addr, size_t size)>;
    
    DefragStats defragment(MemoryAllocator& allocator, RelocationCallback callback);
    
    float calculate_fragmentation(const MemoryAllocator& allocator);
    
private:
    struct RelocationPlan {
        std::vector<std::pair<BlockHeader*, void*>> moves; // (current block, target addr)
    };
    
    RelocationPlan plan_relocations(MemoryAllocator& allocator);
    void execute_plan(const RelocationPlan& plan, RelocationCallback callback);
};
```

**Alternative approach: Handle-based allocation (no raw pointers):**
```cpp
// Instead of returning raw pointers, return handles
using Handle = uint64_t;

class HandleAllocator : public MemoryAllocator {
public:
    Handle allocate_handle(size_t size);
    void deallocate_handle(Handle h);
    
    void* resolve(Handle h); // Get current pointer (may change after defrag)
    
    void defragment() {
        // Move objects, update internal handle->pointer mapping
        // Clients' handles remain valid!
    }
    
private:
    std::unordered_map<Handle, void*> handle_table_;
    std::mutex handle_mutex_;
    Handle next_handle_ = 1;
};
```

**Tests:**
```cpp
TEST(Defragmentation, ReducesFragmentation) {
    MemoryAllocator alloc;
    
    // Create fragmented heap: alloc-free-alloc-free pattern
    std::vector<void*> keep;
    for (int i = 0; i < 1000; i++) {
        void* p1 = alloc.allocate(64);
        void* p2 = alloc.allocate(64);
        keep.push_back(p1);
        alloc.deallocate(p2); // Free every other allocation
    }
    
    float frag_before = DefragmentationEngine{}.calculate_fragmentation(alloc);
    EXPECT_GT(frag_before, 0.4); // At least 40% fragmented
    
    // Defragment
    DefragmentationEngine engine;
    auto stats = engine.defragment(alloc, [](void* old, void* Новый, size_t) {});
    
    float frag_after = engine.calculate_fragmentation(alloc);
    EXPECT_LT(frag_after, 0.1); // Less than 10% fragmentation
    
    // Cleanup
    for (void* ptr : keep) alloc.deallocate(ptr);
}

TEST(HandleAllocator, DefragmentationPreservesHandles) {
    HandleAllocator alloc;
    
    // Allocate 1000 objects via handles
    std::vector<Handle> handles;
    for (int i = 0; i < 1000; i++) {
        handles.push_back(alloc.allocate_handle(128));
    }
    
    // Write unique data to each
    for (Handle h : handles) {
        uint64_t* ptr = static_cast<uint64_t*>(alloc.resolve(h));
        *ptr = h; // Store handle as data
    }
    
    // Defragment
    alloc.defragment();
    
    // Handles still valid, data intact
    for (Handle h : handles) {
        uint64_t* ptr = static_cast<uint64_t*>(alloc.resolve(h));
        EXPECT_EQ(*ptr, h);
    }
    
    for (Handle h : handles) alloc.deallocate_handle(h);
}
```

---

## TURN 3 — Force Failure: Subtle Use-After-Free in Defragmentation

**Instructions:**

Introduce a subtle bug where defragmentation moves an object while another thread is accessing it.

**Ask the AI:**
> "Your defragmentation logic moves objects without coordinating with active allocations. What happens when thread A is writing to memory while thread B's defragmentation moves that object? Show the exact failure with a test that triggers data corruption."

**Expected failure:**
- Thread A writes to address X
- Thread B defragments, moves object from X to Y
- Thread A's next write goes to old address X (now invalid)
- Data corruption or segfault

**Test:**
```cpp
TEST(DefragmentationRace, UseAfterMove) {
    HandleAllocator alloc;
    Handle h = alloc.allocate_handle(1024);
    
    std::atomic<bool> stop{false};
    std::atomic<int> corruption_count{0};
    
    // Thread A: continuously writes to object
    std::thread writer([&]() {
        uint64_t counter = 0;
        while (!stop) {
            uint64_t* ptr = static_cast<uint64_t*>(alloc.resolve(h));
            for (int i = 0; i < 128; i++) {
                ptr[i] = counter++;
            }
            
            // Verify integrity
            ptr = static_cast<uint64_t*>(alloc.resolve(h));
            for (int i = 1; i < 128; i++) {
                if (ptr[i] != ptr[i-1] + 1) {
                    corruption_count++;
                }
            }
        }
    });
    
    // Thread B: continuously defragments
    std::thread defragmenter([&]() {
        for (int i = 0; i < 100; i++) {
            alloc.defragment();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        stop = true;
    });
    
    writer.join();
    defragmenter.join();
    
    // With bug: corruption_count > 0
    // Expected: corruption_count == 0
    EXPECT_EQ(corruption_count.load(), 0);
}
```

**Fix required:**
1. Stop-the-world collection (pause all allocator threads during defrag)
2. Read-write locks on objects (allow concurrent reads, exclusive for defrag)
3. Generational approach (only defrag idle regions)

---

## TURN 4 — Zero-Copy Reallocate with In-Place Growth

**Instructions:**

Implement `realloc`-like functionality that grows allocations in-place when possible.

**Challenge:** Growing in-place requires checking if next block is free. Shrinking needs to split blocks without fragmentation.

**Implement:**
```cpp
class MemoryAllocator {
public:
    // Try to resize in-place, returns nullptr if must allocate new
    void* try_realloc_inplace(void* ptr, size_t new_size);
    
    // Full realloc (may move data)
    void* reallocate(void* ptr, size_t new_size);
    
private:
    bool can_grow_inplace(BlockHeader* block, size_t additional_bytes);
    void grow_block_inplace(BlockHeader* block, size_t new_total_size);
    void shrink_block_inplace(BlockHeader* block, size_t new_total_size);
};
```

**Performance requirement:** Real tests:**
```cpp
TEST(Realloc, InPlaceGrowth) {
    MemoryAllocator alloc;
    
    void* ptr = alloc.allocate(1024);
    void* orig_ptr = ptr;
    
    // Fill with data
    std::memset(ptr, 0xAA, 1024);
    
    // Grow to 2048 (should be in-place if next block free)
    ptr = alloc.reallocate(ptr, 2048);
    
    // Check if in-place
    EXPECT_EQ(ptr, orig_ptr); // Same address
    
    // Check data preserved
    for (int i = 0; i < 1024; i++) {
        EXPECT_EQ(static_cast<char*>(ptr)[i], (char)0xAA);
    }
    
    alloc.deallocate(ptr);
}

TEST(Realloc, VectorGrowth) {
    // Simulate std::vector growth pattern
    // Start with 10 elements, grow to 1000 by doubling
    // Measure how often in-place growth succeeds
    
    struct CustomVector {
        void* data = nullptr;
        size_t size = 0;
        size_t capacity = 0;
        MemoryAllocator& alloc;
        
        void push_back(int val) {
            if (size == capacity) {
                size_t new_cap = capacity == 0 ? 10 : capacity * 2;
                void* new_data = alloc.reallocate(data, new_cap * sizeof(int));
                data = new_data;
                capacity = new_cap;
            }
            static_cast<int*>(data)[size++] = val;
        }
    };
    
    MemoryAllocator alloc;
    CustomVector vec{nullptr, 0, 0, alloc};
    
    for (int i = 0; i < 1000; i++) {
        vec.push_back(i);
    }
    
  // Check stats: at least 50% in-place growth
    // ...
}
```

---

## TURN 5 — NUMA-Aware Allocation for Multi-Socket Systems

**Instructions:**

Add NUMA (Non-Uniform Memory Access) awareness for optimal performance on multi-socket servers.

**Requirements:**
- Detect NUMA topology
- Allocate memory on local NUMA node when possible
- Migrate pages to correct node if thread migrates
- Monitor remote vs. local memory access ratios

**Implement:**
```cpp
#include <numa.h>
#include <numaif.h>

class NUMAAllocator : public MemoryAllocator {
public:
    void* allocate_on_node(size_t size, int node);
    void migrate_to_node(void* ptr, size_t size, int target_node);
    
    struct NUMAStats {
        size_t local_allocations;
        size_t remote_allocations;
        size_t migrations;
        std::map<int, size_t> per_node_usage;
    };
    
    NUMAStats get_numa_stats() const;
    
private:
    int get_current_node() const;
    void bind_memory_to_node(void* ptr, size_t size, int node);
};
```

**Tests:**
```cpp
TEST(NUMA, LocalAllocation) {
    if (numa_available() < 0) {
        GTEST_SKIP() << "NUMA not available";
    }
    
    NUMAAllocator alloc;
    
    // Pin thread to node 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    
    // Allocate should use node 0
    void* ptr = alloc.allocate(1024 * 1024); // 1MB
    
    int allocated_node = -1;
    get_mempolicy(&allocated_node, nullptr, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
    
    EXPECT_EQ(allocated_node, 0);
    
    alloc.deallocate(ptr);
}

TEST(NUMA, MigrationPreservesData) {
    NUMAAllocator alloc;
    
    void* ptr = alloc.allocate_on_node(1024 * 1024, 0);
    
    // Write pattern
    for (int i = 0; i < 1024 * 256; i++) {
        static_cast<int*>(ptr)[i] = i * 37;
    }
    
    // Migrate to node 1
    alloc.migrate_to_node(ptr, 1024 * 1024, 1);
    
    // Verify data intact
    for (int i = 0; i < 1024 * 256; i++) {
        EXPECT_EQ(static_cast<int*>(ptr)[i], i * 37);
    }
    
    alloc.deallocate(ptr);
}
```

---

## TURN 6 — Memory Profiling and Leak Detection

**Instructions:**

Add built-in memory leak detection and allocation profiling without Valgrind.

**Requirements:**
- Track every allocation with stack trace
- Detect leaks at program exit
- Profile hot allocation sites
- Zero performance overhead when disabled
- Generate reports in human-readable format

**Implement:**
```cpp
class AllocationTracker {
public:
    struct AllocationInfo {
        void* address;
        size_t size;
        std::vector<void*> stack_trace;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        std::thread::id thread_id;
    };
    
    void track_allocation(void* ptr, size_t size);
    void track_deallocation(void* ptr);
    
    std::vector<AllocationInfo> get_leaks() const;
    
    struct Profile {
        std::map<std::vector<void*>, size_t> allocation_sites; // stack -> count
        size_t total_allocations;
        size_t total_bytes_allocated;
        std::chrono::microseconds total_allocation_time;
    };
    
    Profile generate_profile() const;
    
    void print_report(std::ostream& out) const;
    
private:
    std::unordered_map<void*, AllocationInfo> active_allocations_;
    std::mutex tracker_mutex_;
    
    std::vector<void*> capture_stack_trace(int max_depth = 16);
    std::string resolve_symbol(void* addr);
};

// Global tracker (optional compilation flag)
#ifdef ENABLE_ALLOCATION_TRACKING
extern AllocationTracker g_allocation_tracker;
#endif
```

**Usage:**
```cpp
TEST(LeakDetection, DetectsMemoryLeak) {
    AllocationTracker tracker;
    MemoryAllocator alloc;
    
    // Intentionally leak
    void* ptr1 = alloc.allocate(128);
    tracker.track_allocation(ptr1, 128);
    
    void* ptr2 = alloc.allocate(256);
    tracker.track_allocation(ptr2, 256);
    alloc.deallocate(ptr2);
    tracker.track_deallocation(ptr2);
    
    // ptr1 not freed - should be detected
    auto leaks = tracker.get_leaks();
    EXPECT_EQ(leaks.size(), 1);
    EXPECT_EQ(leaks[0].address, ptr1);
    EXPECT_EQ(leaks[0].size, 128);
    
    tracker.print_report(std::cout);
}

TEST(Profiling, IdentifiesHotSites) {
    AllocationTracker tracker;
    MemoryAllocator alloc;
    
    auto allocate_from_function_A = [&]() {
        void* ptr = alloc.allocate(64);
        tracker.track_allocation(ptr, 64);
        return ptr;
    };
    
    auto allocate_from_function_B = [&]() {
        void* ptr = alloc.allocate(128);
        tracker.track_allocation(ptr, 128);
        return ptr;
    };
    
    // Function A called 1000 times
    std::vector<void*> ptrs_a;
    for (int i = 0; i < 1000; i++) {
        ptrs_a.push_back(allocate_from_function_A());
    }
    
    // Function B called 10 times
    std::vector<void*> ptrs_b;
    for (int i = 0; i < 10; i++) {
        ptrs_b.push_back(allocate_from_function_B());
    }
    
    auto profile = tracker.generate_profile();
    EXPECT_EQ(profile.total_allocations, 1010);
    
    // Top allocation site should be function A
    // (stack trace comparison)
}
```

---

## TURN 7 — Huge Page Support for Large Allocations

**Instructions:**

Use huge pages (2MB/1GB) for large allocations to reduce TLB misses.

**Implement:**
```cpp
class HugePageAllocator : public MemoryAllocator {
public:
    void* allocate_huge(size_t size, HugePageSize page_size = HugePageSize::Size_2MB);
    void deallocate_huge(void* ptr);
    
    enum class HugePageSize {
        Size_2MB = 1 << 21,
        Size_1GB = 1 << 30,
    };
    
    static bool huge_pages_available();
    static void enable_transparent_huge_pages();
    
private:
    void* mmap_huge(size_t size, HugePageSize page_size);
};
```

**Benchmark:**
```cpp
BENCHMARK(HugePageVsNormalPage) {
    constexpr size_t SIZE = 1024 * 1024 * 1024; // 1GB
    
    // Normal pages
    auto start = std::chrono::high_resolution_clock::now();
    void* ptr_normal = MemoryAllocator::instance().allocate(SIZE);
    // Touch all pages to fault them in
    for (size_t i = 0; i < SIZE; i += 4096) {
        static_cast<char*>(ptr_normal)[i] = 0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto normal_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Huge pages
    start = std::chrono::high_resolution_clock::now();
    HugePageAllocator huge_alloc;
    void* ptr_huge = huge_alloc.allocate_huge(SIZE);
    for (size_t i = 0; i < SIZE; i += 4096) {
        static_cast<char*>(ptr_huge)[i] = 0;
    }
    end = std::chrono::high_resolution_clock::now();
    auto huge_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Huge pages should be significantly faster (50%+ improvement)
    EXPECT_LT(huge_time.count(), normal_time.count() * 0.5);
}
```

---

## TURN 8 — Force Failure: Memory Corruption from Alignment Bugs

**Ask the AI:**
> "Your allocator doesn't properly handle over-aligned types (e.g., `alignas(64)` for SIMD). What happens when someone allocates a struct with 64-byte alignment but your allocator only guarantees 16-byte? Show the crash with a test using AVX-512 intrinsics."

**Expected failure:**
- Allocate SIMD type with 64-byte alignment requirement
- Allocator returns 16-byte aligned address
- SIMD load/store instruction crashes (SIGSEGV) or gets wrong data

**Test:**
```cpp
#include <immintrin.h>

struct alignas(64) SIMDData {
    __m512 vectors[4]; // AVX-512: requires 64-byte alignment
};

TEST(AlignmentBug, SIMDCrash) {
    MemoryAllocator alloc;
    
    // Buggy allocator ignores alignment
    void* ptr = alloc.allocate(sizeof(SIMDData)); // Only 16-byte aligned!
    
    SIMDData* data = new (ptr) SIMDData;
    
    // This will crash or produce wrong results
    __m512 vec = _mm512_set1_ps(1.0f);
    _mm512_store_ps(&data->vectors[0], vec); // CRASH if not 64-byte aligned
    
    // Never reached
    alloc.deallocate(ptr);
}
```

**Fix:** Add alignment parameter:
```cpp
void* allocate(size_t size, size_t alignment = 16);

// Implementation must ensure returned address % alignment == 0
// May need to waste padding bytes
```

---

## TURN 9 — Allocation Patterns: Benchmarking Real-World Workloads

**Instructions:**

Benchmark against real allocation patterns from databases, web servers, game engines.

**Workloads:**
1. **Database pattern:** Many small allocations (50-200 bytes), held for seconds, then freed in batch
2. **Web server pattern:** Burst allocations during request, all freed at end of request
3. **Game engine pattern:** Mostly temporary allocations per frame, plus long-lived assets
4. **ML training pattern:** Large tensor allocations (GB), reallocated often

**Implement:**
```cpp
class AllocationBenchmark {
public:
    struct WorkloadResult {
        std::chrono::microseconds total_time;
        double allocs_per_second;
        size_t peak_memory_usage;
        float fragmentation_final;
    };
    
    static WorkloadResult run_database_pattern();
    static WorkloadResult run_webserver_pattern();
    static WorkloadResult run_gameengine_pattern();
    static WorkloadResult run_ml_training_pattern();
    
    static void compare_with_malloc();
    static void compare_with_tcmalloc();
};
```

**Benchmark database pattern:**
```cpp
WorkloadResult AllocationBenchmark::run_database_pattern() {
    // Simulate database query processing
    // 10k concurrent queries, each allocates row buffers, hash tables, sort buffers
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<void*>> queries(10000);
    
    for (auto& query : queries) {
        // Each query allocates buffers
        for (int i = 0; i < 100; i++) {
            size_t size = 50 + (rand() % 150);
            query.push_back(MemoryAllocator::instance().allocate(size));
        }
    }
    
    // Queries finish in random order
    std::shuffle(queries.begin(), queries.end(), std::mt19937{});
    
    for (auto& query : queries) {
        for (void* ptr : query) {
            MemoryAllocator::instance().deallocate(ptr);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    return WorkloadResult{
        std::chrono::duration_cast<std::chrono::microseconds>(end - start),
        // ... other metrics
    };
}
```

**Expected results:**
- Custom allocator: 10M alloc/s, <5% fragmentation
- malloc: 5M alloc/s, 20% fragmentation
- tcmalloc: 15M alloc/s, 3% fragmentation

---

## TURN 10 — Security: Hardening Against Exploits

**Instructions:**

Add security features to prevent heap exploitation.

**Mitigations to implement:**
1. **Guard pages:** Unmapped pages before/after heap regions
2. **Canaries:** Magic values to detect buffer overflows
3. **ASLR:** Randomize heap layout
4. **Double-free detection:** Guard bits to detect double frees
5. **Metadata encryption:** Encrypt block headers to prevent corruption

**Implement:**
```cpp
class SecureAllocator : public MemoryAllocator {
public:
    SecureAllocator();
    
    void* allocate(size_t size, size_t alignment = 16) override;
    void deallocate(void* ptr) override;
    
private:
    // Place guard pages around every large allocation
    void add_guard_pages(void* ptr, size_t size);
    
    // Add canary values to detect overflows
    void write_canary(BlockHeader* block);
    bool verify_canary(BlockHeader* block);
    
    // Randomize addresses (ASLR)
    size_t get_random_offset();
    
    // Encrypt metadata
    struct EncryptedHeader {
        uint64_t encrypted_data[4];
    };
    
    BlockHeader* decrypt_header(EncryptedHeader* enc);
    EncryptedHeader* encrypt_header(BlockHeader* header);
    
    uint64_t aslr_base_;
    std::array<uint8_t, 32> encryption_key_;
};
```

**Exploit tests:**
```cpp
TEST(Security, DetectsBufferOverflow) {
    SecureAllocator alloc;
    
    char* buf = static_cast<char*>(alloc.allocate(128));
    
    // Overflow buffer (write past end)
    for (int i = 0; i < 200; i++) {
        buf[i] = 'X'; // Overwrite canary
    }
    
    // Deallocation should detect corrupted canary
    EXPECT_DEATH(alloc.deallocate(buf), "Canary corrupted");
}

TEST(Security, DetectsDoubleFree) {
    SecureAllocator alloc;
    
    void* ptr = alloc.allocate(64);
    alloc.deallocate(ptr);
    
    // Double free should be detected
    EXPECT_DEATH(alloc.deallocate(ptr), "Double free");
}

TEST(Security, GuardPagesPreventOverflow) {
    SecureAllocator alloc;
    
    // Large allocation gets guard pages
    char* buf = static_cast<char*>(alloc.allocate(1024 * 1024));
    
    // Try to write past end into guard page
    // Should segfault thanks to unmapped guard page
    EXPECT_DEATH(buf[1024 * 1024] = 'X', "");
}
```

---

## TURN 11 — Final Integration: Custom Smart Pointers

**Instructions:**

Create smart pointers that work with the custom allocator.

**Implement:**
```cpp
template<typename T>
class unique_ptr {
public:
    unique_ptr() : ptr_(nullptr) {}
    
    explicit unique_ptr(T* ptr) : ptr_(ptr) {}
    
    ~unique_ptr() {
        if (ptr_) {
            ptr_->~T();
            MemoryAllocator::instance().deallocate(ptr_);
        }
    }
    
    // Move only
    unique_ptr(unique_ptr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    unique_ptr& operator=(unique_ptr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    
    void reset(T* ptr = nullptr) {
        if (ptr_) {
            ptr_->~T();
            MemoryAllocator::instance().deallocate(ptr_);
        }
        ptr_ = ptr;
    }
    
private:
    T* ptr_;
};

template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
    void* mem = MemoryAllocator::instance().allocate(sizeof(T), alignof(T));
    T* obj = new (mem) T(std::forward<Args>(args)...);
    return unique_ptr<T>(obj);
}

template<typename T>
class shared_ptr {
public:
    shared_ptr() : ptr_(nullptr), control_(nullptr) {}
    
    explicit shared_ptr(T* ptr) : ptr_(ptr) {
        control_ = static_cast<ControlBlock*>(
            MemoryAllocator::instance().allocate(sizeof(ControlBlock))
        );
        new (control_) ControlBlock{1, 0};
    }
    
    ~shared_ptr() { dec_ref(); }
    
    shared_ptr(const shared_ptr& other) : ptr_(other.ptr_), control_(other.control_) {
        if (control_) control_->strong_count++;
    }
    
    T* get() const { return ptr_; }
    long use_count() const { return control_ ? control_->strong_count : 0; }
    
private:
    struct ControlBlock {
        std::atomic<long> strong_count;
        std::atomic<long> weak_count;
    };
    
    void dec_ref() {
        if (control_ && --control_->strong_count == 0) {
            ptr_->~T();
            MemoryAllocator::instance().deallocate(ptr_);
            
            if (control_->weak_count == 0) {
                control_->~ControlBlock();
                MemoryAllocator::instance().deallocate(control_);
            }
        }
    }
    
    T* ptr_;
    ControlBlock* control_;
};
```

**Final integration test:**
```cpp
TEST(SmartPointers, EndToEndIntegration) {
    // Full system test using all components
    
    struct LargeObject {
        std::array<double, 1024> data;
        LargeObject() { data.fill(3.14); }
    };
    
    // Allocate using smart pointers
    auto ptr1 = make_unique<LargeObject>();
    auto ptr2 = make_unique<LargeObject>();
    
    // Verify data
    EXPECT_DOUBLE_EQ(ptr1->data[0], 3.14);
    
    // Shared ownership
    shared_ptr<int> shared1(static_cast<int*>(
        MemoryAllocator::instance().allocate(sizeof(int))
    ));
    *shared1.get() = 42;
    
    {
        shared_ptr<int> shared2 = shared1;
        EXPECT_EQ(shared1.use_count(), 2);
        EXPECT_EQ(*shared2, 42);
    }
    
    EXPECT_EQ(shared1.use_count(), 1);
    
    // Automatic cleanup
}

TEST(FullSystem, StressTestAllComponents) {
    // Ultimate stress test:
    // - 8 threads allocating/freeing
    // - NUMA-aware allocations
    // - Defragmentation running in background
    // - Leak detection enabled
    // - Security features active
    // - Profile generation
    
    // Run for 60 seconds, verify:
    // - No crashes
    // - No leaks
    // - No corruption
    // - Good performance (>5M alloc/s aggregate)
    // - Low fragmentation (<10%)
}
```

**Deliverables:**
- Production-ready allocator library
- Full test suite (100+ tests, all passing)
- Benchmarks showing superiority over malloc
- Security audit documentation
- User guide and API reference
