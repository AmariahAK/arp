# Task: Contribute Vectorized Execution Engine to PostgreSQL

**Repository:** https://github.com/postgres/postgres  
**Version:** PostgreSQL 17.0  
**Created:** November 22, 2025  
**Language:** C  
**Difficulty:** EXTREME

## Overview
Implement a production-ready vectorized execution engine for PostgreSQL that dramatically improves analytical query performance. This contribution must integrate seamlessly with PostgreSQL's existing executor, pass all regression tests, handle all SQL data types correctly, and provide 3-5x speedup on analytical workloads. The implementation requires deep understanding of PostgreSQL internals, SIMD programming, and database execution models.

**Key Challenge:** This must be production-quality code suitable for upstream merge. You cannot break existing functionality, must follow PostgreSQL coding standards exactly, and need to handle every edge case.

---

## TURN 1 — Understanding PostgreSQL Executor Architecture

**Role:** You are a PostgreSQL core contributor who understands the executor inside-out. You've read executor/README and can explain how tuples flow through the execution tree

.

**Background:** PostgreSQL uses a volcano-style iterator model where each executor node implements Init, Next, and End functions. Tuples are processed one-at-a-time (tuple-at-a-time or "row-based" execution). Vectorized execution processes tuples in batches for better CPU cache utilization and SIMD opportunities.

**Reference:** Study:
- `src/backend/executor/README` - Executor overview
- `src/include/executor/executor.h` - Executor node interface
- `src/backend/executor/nodeSeqscan.c` - Sequential scan implementation
- Papers: "MonetDB/X100" (vectorized execution), "Vectorization vs. Compilation"

**VERY IMPORTANT:**
- Must maintain backward compatibility with all existing executor nodes
- Cannot break planner assumptions
- Must handle all SQL data types (numeric, text, arrays, JSON, etc.)
- Memory management must follow PostgreSQL conventions (palloc/pfree)
- Must respect transaction isolation levels
- No crashes or assertion failures in regression tests

**Goal:** Design the vectorized executor architecture and implement basic infrastructure.

**Instructions:**

1. **Study the current executor model:**

Examine how tuples flow through the executor:
```c
// Current tuple-at-a-time model
TupleTableSlot *
ExecProcNode(PlanState *node)
{
    // Returns one tuple at a time
    switch (nodeTag(node))
    {
        case T_SeqScanState:
            return ExecSeqScan((SeqScanState *) node);
        case T_AggState:
            return ExecAgg((AggState *) node);
        // ... many more node types
    }
}
```

2. **Design vectorized execution model:**

Propose how to extend the executor to support batch processing:
```c
// New vectorized execution interface
typedef struct VectorBatch
{
    int         nbatch;         // Number of tuples in batch (e.g., 1024)
    int         ntypes;         // Number of columns
    Datum      *data;           // Column-oriented data (nbatch * ntypes)
    bool       *nulls;          // NULL bitmap
    
    // Memory context for batch
    MemoryContext batch_mcxt;
} VectorBatch;

// Vectorized execution function signature
typedef VectorBatch *(*ExecVectorizedFunc) (PlanState *pstate);

// Add to PlanState
typedef struct PlanState
{
    NodeTag     type;
    Plan       *plan;
    // ... existing fields ...
    
    // New vectorized execution support
    bool        supports_vectorized;
    ExecVectorizedFunc vectorized_func;
} PlanState;
```

Key design decisions to address:
- **Batch size:** 1024 tuples? Tunable? Cache-line aligned?
- **Column vs row storage:** Store batches in columnar format for SIMD?
- **Memory management:** How to allocate/free batches efficiently?
- **Backwards compatibility:** How to mix vectorized and row-based nodes?
- **Type handling:** How to vectorize all PostgreSQL data types?

3. **Implement basic batch infrastructure:**

```c
// src/backend/executor/execVectorized.c

/*
 * Create a new vector batch
 */
VectorBatch *
MakeVectorBatch(int nbatch, int ntypes)
{
    VectorBatch *batch;
    MemoryContext oldcontext;
    
    // Allocate in executor's per-query memory context
    oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);
    
    batch = (VectorBatch *) palloc(sizeof(VectorBatch));
    batch->nbatch = nbatch;
    batch->ntypes = ntypes;
    
    // Allocate column-aligned data
    batch->data = (Datum *) palloc_aligned(sizeof(Datum) * nbatch * ntypes, 32);
    batch->nulls = (bool *) palloc_aligned(sizeof(bool) * nbatch * ntypes, 32);
    
    batch->batch_mcxt = AllocSetContextCreate(CurrentMemoryContext,
                                               "VectorBatch",
                                               ALLOCSET_DEFAULT_SIZES);
    
    MemoryContextSwitchTo(oldcontext);
    
    return batch;
}

/*
 * Free a vector batch
 */
void
FreeVectorBatch(VectorBatch *batch)
{
    if (batch == NULL)
        return;
        
    MemoryContextDelete(batch->batch_mcxt);
    pfree(batch->data);
    pfree(batch->nulls);
    pfree(batch);
}

/*
 * Materialize row-based tuples into a vector batch
 */
VectorBatch *
MaterializeToBatch(PlanState *planstate, int max_batch)
{
    VectorBatch *batch;
    TupleTableSlot *slot;
    int         i = 0;
    int         ntypes = planstate->ps_ResultTupleDesc->natts;
    
    batch = MakeVectorBatch(max_batch, ntypes);
    
    // Pull tuples from child node until batch full or depleted
    while (i < max_batch)
    {
        slot = ExecProcNode(planstate);
        if (TupIsNull(slot))
            break;
            
        // Copy tuple data into batch (column-oriented)
        slot_getallattrs(slot);
        for (int col = 0; col < ntypes; col++)
        {
            batch->data[col * max_batch + i] = slot->tts_values[col];
            batch->nulls[col * max_batch + i] = slot->tts_isnull[col];
        }
        
        i++;
    }
    
    batch->nbatch = i;  // Actual number filled
    
    return batch;
}
```

4. **Implement vectorized sequential scan:**

```c
// src/backend/executor/nodeSeqscan.c

/*
 * ExecSeqScanVectorized - vectorized version of sequential scan
 */
VectorBatch *
ExecSeqScanVectorized(SeqScanState *node)
{
    Relation    relation;
    TableScanDesc scandesc;
    VectorBatch *batch;
    HeapTuple   tuple;
    int         batch_size = 1024;  // Configurable via GUC
    int         i = 0;
    
    relation = node->ss_currentRelation;
    scandesc = node->ss_currentScanDesc;
    
    batch = MakeVectorBatch(batch_size, relation->rd_att->natts);
    
    // Scan tuples and fill batch
    while (i < batch_size)
    {
        tuple = heap_getnext(scandesc, ForwardScanDirection);
        
        if (!HeapTupleIsValid(tuple))
            break;
            
        // Extract tuple attributes into batch columns
        heap_deform_tuple(tuple,
                          relation->rd_att,
                          &batch->data[i * relation->rd_att->natts],
                          &batch->nulls[i * relation->rd_att->natts]);
        i++;
    }
    
    batch->nbatch = i;
    
    if (i == 0)
    {
        FreeVectorBatch(batch);
        return NULL;  // End of scan
    }
    
    return batch;
}

// Register vectorized capability
void
ExecInitSeqScan(SeqScanState *node, EState *estate, int eflags)
{
    // ... existing initialization ...
    
    // Mark this node as supporting vectorized execution
    node->ps.supports_vectorized = true;
    node->ps.vectorized_func = (ExecVectorizedFunc) ExecSeqScanVectorized;
}
```

5. **Write tests:**

```c
// src/test/regress/sql/vectorized_exec.sql

-- Test basic vectorized sequential scan
CREATE TABLE test_vector (
    id INTEGER,
    value DOUBLE PRECISION,
    text_col TEXT
);

INSERT INTO test_vector SELECT i, i::float * 1.5, 'row_' || i
FROM generate_series(1, 100000) i;

-- Enable vectorized execution
SET enable_vectorized_exec = on;

-- Simple sequential scan
EXPLAIN ANALYZE SELECT * FROM test_vector WHERE value > 50000;

-- Expected: Uses vectorized sequential scan

-- Verify results match row-based execution
SET enable_vectorized_exec = off;
SELECT QUERY INTO result_row_based 
SELECT * FROM test_vector WHERE value > 50000 ORDER BY id;

SET enable_vectorized_exec = on;
SELECT QUERY INTO result_vectorized
SELECT * FROM test_vector WHERE value > 50000 ORDER BY id;

-- Results must be identical
SELECT COUNT(*) FROM (
    SELECT * FROM result_row_based
    EXCEPT
    SELECT * FROM result_vectorized
);  -- Should return 0
```

**Deliverables:**
- Architecture document explaining vectorized execution model
- Basic VectorBatch infrastructure
- Vectorized sequential scan implementation
- Regression tests showing correctness
- Performance measurement showing speedup on simple scans

---

## TURN 2 — SIMD-Optimized Filter Execution

**Instructions:**

Implement vectorized filter evaluation using SIMD for common predicates (>, <, =, AND, OR).

**Challenge:** PostgreSQL supports 50+ data types. Start with integers, floats, then extend.

**Implement:**

```c
// src/backend/executor/execVectorizedFilter.c

/*
 * VectorizedFilterInt64 - Apply > predicate on int64 column using AVX2
 */
static void
VectorizedFilterInt64_GreaterThan(VectorBatch *batch, int col_idx, int64 threshold, bool *result)
{
    int64  *data = (int64 *) &batch->data[col_idx * batch->nbatch];
    int     i;
    
#ifdef USE_AVX2
    // Process 4 int64 values at once with AVX2
    __m256i thresh_vec = _mm256_set1_epi64x(threshold);
    
    for (i = 0; i + 4 <= batch->nbatch; i += 4)
    {
        __m256i data_vec = _mm256_loadu_si256((__m256i *) &data[i]);
        __m256i cmp = _mm256_cmpgt_epi64(data_vec, thresh_vec);
        
        // Extract comparison results
        int mask = _mm256_movemask_epi8(cmp);
        
        result[i + 0] = (mask & 0x000000FF) != 0;
        result[i + 1] = (mask & 0x0000FF00) != 0;
        result[i + 2] = (mask & 0x00FF0000) != 0;
        result[i + 3] = (mask & 0xFF000000) != 0;
    }
#else
    i = 0;
#endif
    
    // Scalar fallback for remaining elements
    for (; i < batch->nbatch; i++)
    {
        result[i] = (data[i] > threshold);
    }
}

/*
 * ExecVectorizedFilter - Apply filter expression to batch
 */
VectorBatch *
ExecVectorizedFilter(FilterState *node, VectorBatch *input_batch)
{
    ExprState  *qual = node->qual;
    bool       *filter_result;
    VectorBatch *output_batch;
    int         output_count = 0;
    int         i;
    
    if (input_batch == NULL || input_batch->nbatch == 0)
        return NULL;
    
    // Allocate filter result bitmap
    filter_result = (bool *) palloc(sizeof(bool) * input_batch->nbatch);
    
    // Evaluate filter on entire batch
    ExecVectorizedQual(qual, input_batch, filter_result);
    
    // Count passing tuples
    for (i = 0; i < input_batch->nbatch; i++)
        if (filter_result[i])
            output_count++;
    
    // Create output batch with only passing tuples
    output_batch = MakeVectorBatch(output_count, input_batch->ntypes);
    
    // Compact: copy passing tuples to output batch
    // (This is where SIMD gather/scatter could help)
    int out_idx = 0;
    for (i = 0; i < input_batch->nbatch; i++)
    {
        if (filter_result[i])
        {
            // Copy all columns for this tuple
            for (int col = 0; col < input_batch->ntypes; col++)
            {
                output_batch->data[col * output_count + out_idx] =
                    input_batch->data[col * input_batch->nbatch + i];
                output_batch->nulls[col * output_count + out_idx] =
                    input_batch->nulls[col * input_batch->nbatch + i];
            }
            out_idx++;
        }
    }
    
    pfree(filter_result);
    
    return output_batch;
}
```

**Benchmark:**
```sql
-- TPC-H Query 6 (highly selective filter)
SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= date '1994-01-01'
  AND l_shipdate < date '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;

-- Row-based: 2500ms
-- Vectorized (no SIMD): 800ms (3x)
-- Vectorized + SIMD: 450ms (5.5x)
```

---

## TURN 3 — Force Failure: Type Confusion Bug

**Instructions:**

Introduce a subtle bug where SIMD code assumes alignment that PostgreSQL doesn't guarantee.

**Ask the AI:**
> "Your SIMD filter code uses _mm256_loadu_si256 which requires 32-byte alignment, but PostgreSQL's palloc doesn't guarantee this for Datum arrays. Show a test where this causes a segfault on specific data, and explain how to fix it."

**Expected failure:**
```c
// Buggy code
__m256i data_vec = _mm256_load_si256((__m256i *) &data[i]);  // Requires alignment!

// When data is not 32-byte aligned → SIGSEGV
```

**Test:**
```sql
-- Create table where data isn't aligned
CREATE TABLE unaligned_test AS 
SELECT i FROM generate_series(1, 100001) i;  -- Odd number ensures misalignment

SET enable_vectorized_exec = on;
SELECT COUNT(*) FROM unaligned_test WHERE i > 50000;
-- CRASH: Unaligned access in SIMD code
```

**Fix:** Use unaligned loads or ensure palloc_aligned.

---

## TURN 4 — Vectorized Hash Join

**Instructions:**

Implement vectorized hash join, the most critical operator for analytical queries.

**Challenges:**
- Build hash table from batch
- Probe hash table with batch
- Handle hash collisions
- Multi-column join keys

**Implement:**

```c
// Vectorized hash join probe
VectorBatch *
ExecVectorizedHashJoin(HashJoinState *node, VectorBatch *outer_batch)
{
    HashJoinTable hashtable = node->hj_HashTable;
    VectorBatch *result_batch;
    int         *probe_results;  // Hash bucket indices
    int         i, match_count = 0;
    
    // Hash outer batch (all join keys at once)
    probe_results = VectorizedHashProbe(hashtable, outer_batch, 
                                         node->hj_OuterHashKeys);
    
    // For each outer tuple, check hash bucket for matches
    for (i = 0; i < outer_batch->nbatch; i++)
    {
        int bucket = probe_results[i];
        
        // Check bucket for matching inner tuples
        HashJoinTuple inner_tuple = hashtable->buckets[bucket];
        
        while (inner_tuple != NULL)
        {
            if (TuplesMatch(outer_batch, i, inner_tuple))
            {
                // Emit joined tuple
                match_count++;
            }
            inner_tuple = inner_tuple->next;
        }
    }
    
    // Build output batch with joined tuples
    result_batch = BuildJoinedBatch(outer_batch, matches, match_count);
    
    return result_batch;
}

// SIMD hash computation
static int *
VectorizedHashProbe(HashJoinTable hashtable, VectorBatch *batch, List *hashkeys)
{
    int        *hash_values = palloc(sizeof(int) * batch->nbatch);
    
    // For simplicity, assume single int64 join key
    int64 *key_data = (int64 *) &batch->data[0];
    
#ifdef USE_AVX2
    // Hash 4 keys at once
    for (int i = 0; i + 4 <= batch->nbatch; i += 4)
    {
        __m256i keys = _mm256_loadu_si256((__m256i *) &key_data[i]);
        
        // Simple hash: multiply by prime, modulo table size
        __m256i hash = _mm256_mul_epi32(keys, _mm256_set1_epi64x(31));
        
        // Extract and modulo
        int64 h[4];
        _mm256_storeu_si256((__m256i *) h, hash);
        
        for (int j = 0; j < 4; j++)
            hash_values[i + j] = h[j] % hashtable->nbuckets;
    }
#endif
    
    return hash_values;
}
```

**Performance test:**
```sql
-- TPC-H Query 3 (join + aggregation)
SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem
WHERE c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND c_mktsegment = 'BUILDING'
GROUP BY l_orderkey;

-- Row-based: 8500ms
-- Vectorized: 2100ms (4x faster)
```

---

## TURN 5 — Vectorized Aggregation with SIMD

**Instructions:**

Implement vectorized GROUP BY aggregation with SIMD-optimized aggregate functions (SUM, COUNT, AVG, MIN, MAX).

**Implement:**

```c
// SIMD sum aggregation
static void
VectorizedSum_Float64(VectorBatch *batch, int col_idx, double *result)
{
    double *data = (double *) &batch->data[col_idx * batch->nbatch];
    bool   *nulls = &batch->nulls[col_idx * batch->nbatch];
    
    __m256d sum_vec = _mm256_setzero_pd();
    
    int i;
    for (i = 0; i + 4 <= batch->nbatch; i += 4)
    {
        // Load 4 doubles
        __m256d vals = _mm256_loadu_pd(&data[i]);
        
        // Check nulls (skip if any null in group of 4)
        if (!nulls[i] && !nulls[i+1] && !nulls[i+2] && !nulls[i+3])
        {
            sum_vec = _mm256_add_pd(sum_vec, vals);
        }
        else
        {
            // Scalar handling for nulls
            for (int j = 0; j < 4; j++)
                if (!nulls[i + j])
                    *result += data[i + j];
        }
    }
    
    // Horizontal sum of SIMD vector
    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    *result += temp[0] + temp[1] + temp[2] + temp[3];
    
    // Scalar remainder
    for (; i < batch->nbatch; i++)
        if (!nulls[i])
            *result += data[i];
}

// Vectorized hash aggregation
VectorBatch *
ExecVectorizedAgg(AggState *node, VectorBatch *input_batch)
{
    HashAggTable *hash_table = node->hash_table;
    
    // For each tuple in batch, update hash table
    for (int i = 0; i < input_batch->nbatch; i++)
    {
        uint32 hash = ComputeGroupHash(input_batch, i, node->grouping_cols);
        
        AggEntry *entry = HashTableLookup(hash_table, hash);
        
        if (entry == NULL)
        {
            // New group
            entry = CreateAggEntry(hash_table, hash);
        }
        
        // Update aggregates for this group
        UpdateAggregates(entry, input_batch, i, node->aggs);
    }
    
    // At end of input, emit results
    return HashTableToVectorBatch(hash_table);
}
```

**Benchmark - TPC-H Query 1:**
```sql
SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= date '1998-12-01'
GROUP BY l_returnflag, l_linestatus;

-- Row-based: 15000ms
-- Vectorized: 2500ms (6x faster)
```

---

## TURN 6 — Query Planner Integration

**Instructions:**

Modify PostgreSQL's planner to choose between row-based and vectorized execution.

**Implement:**

```c
// src/backend/optimizer/plan/planner.c

/*
 * Should we use vectorized execution for this query?
 */
static bool
should_use_vectorized_execution(PlannerInfo *root, Plan *plan)
{
    // Heuristics:
    // 1. Large table scans favor vectorized
    // 2. OLTP queries (many index lookups) favor row-based
    // 3. Analytical queries (aggregations, joins) favor vectorized
    
    if (plan->plan_rows > 10000)  // Large result set
        return true;
        
    // Check for analytical operators
    if (IsA(plan, Agg) || IsA(plan, HashJoin))
        return true;
        
    // Check for index scans (bad for vectorization)
    if (IsA(plan, IndexScan) || IsA(plan, IndexOnlyScan))
        return false;
        
    return false;  // Default to row-based
}

/*
 * Annotate plan tree with vectorization decisions
 */
static void
annotate_plan_vectorization(PlannerInfo *root, Plan *plan)
{
    plan->vectorized = should_use_vectorized_execution(root, plan);
    
    // Recursively annotate child nodes
    if (plan->lefttree)
        annotate_plan_vectorization(root, plan->lefttree);
    if (plan->righttree)
        annotate_plan_vectorization(root, plan->righttree);
}
```

**Cost model adjustments:**
```c
// Vectorized operations have lower per-tuple cost but higher startup
static Cost
cost_vectorized_seqscan(Path *path, PlannerInfo *root)
{
    Cost startup_cost = 1000.0;  // Batch allocation overhead
    Cost run_cost;
    
    // Per-tuple cost reduced by vectorization factor
    double vectorization_factor = 4.0;  // 4x speedup from SIMD
    
    run_cost = (cpu_tuple_cost / vectorization_factor) * path->rows;
    
    // But add batch materialization cost
    run_cost += (path->rows / BATCH_SIZE) * cpu_operator_cost;
    
    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}
```

---

## TURN 7 — Handle Complex Data Types

**Instructions:**

Extend vectorization to handle PostgreSQL's complex types: TEXT, NUMERIC, ARRAY, JSONB.

**Challenges:**
- Variable-length types (TEXT) don't fit in fixed-size SIMD vectors
- NUMERIC requires arbitrary precision
- ARRAY and JSONB have complex structure

**Implement:**

```c
// Vectorized text comparison
static void
VectorizedTextEqual(VectorBatch *batch, int col_idx, text *pattern, bool *result)
{
    // TEXT is variable-length, stored as pointers
    text **text_ptrs = (text **) &batch->data[col_idx * batch->nbatch];
    
    // Cannot use SIMD directly on variable-length data
    // But can SIMD-optimize the comparison loop
    
    int pattern_len = VARSIZE_ANY_EXHDR(pattern);
    char *pattern_data = VARDATA_ANY(pattern);
    
    for (int i = 0; i < batch->nbatch; i++)
    {
        text *val = text_ptrs[i];
        
        if (VARSIZE_ANY_EXHDR(val) != pattern_len)
        {
            result[i] = false;
            continue;
        }
        
        // Use memcmp (potentially SIMD-optimized by compiler)
        result[i] = (memcmp(VARDATA_ANY(val), pattern_data, pattern_len) == 0);
    }
}

// Dictionary encoding for low-cardinality TEXT columns
typedef struct DictionaryEncoder
{
    HTAB       *string_to_id;   // Hash table: string -> integer ID
    char      **id_to_string;   // Array: ID -> string
    int         next_id;
    int         num_unique;
} DictionaryEncoder;

/*
 * For low-cardinality columns (like l_returnflag with 3 values),
 * encode as integers for faster SIMD processing
 */
static void
EncodeTextColumn(VectorBatch *batch, int col_idx, DictionaryEncoder *encoder)
{
    text **text_vals = (text **) &batch->data[col_idx * batch->nbatch];
    int  *encoded = (int *) palloc(sizeof(int) * batch->nbatch);
    
    for (int i = 0; i < batch->nbatch; i++)
    {
        // Look up or insert in dictionary
        int id = DictionaryLookup(encoder, text_vals[i]);
        encoded[i] = id;
    }
    
    // Replace text column with encoded integers
    batch->data[col_idx * batch->nbatch] = (Datum) encoded;
}
```

---

## TURN 8 — NULL Handling and SIMD Bitmaps

**Instructions:**

Optimize NULL handling using SIMD bitmaps instead of per-value bool arrays.

**Implement:**

```c
// Compact NULL bitmap (1 bit per value)
typedef struct NullBitmap
{
    uint64 *bits;       // Bitmap
    int     nbytes;
} NullBitmap;

static NullBitmap *
CreateNullBitmap(int nbatch)
{
    NullBitmap *bitmap = palloc(sizeof(NullBitmap));
    bitmap->nbytes = (nbatch + 63) / 64;  // Round up to uint64
    bitmap->bits = palloc0(sizeof(uint64) * bitmap->nbytes);
    return bitmap;
}

// SIMD NULL checking
static int
CountNonNulls_SIMD(NullBitmap *bitmap, int nbatch)
{
    int count = 0;
    
#ifdef USE_AVX2
    __m256i count_vec = _mm256_setzero_si256();
    
    int i;
    for (i = 0; i + 4 <= bitmap->nbytes; i += 4)
    {
        __m256i bits = _mm256_loadu_si256((__m256i *) &bitmap->bits[i]);
        
        // Use POPCNT to count set bits
        for (int j = 0; j < 4; j++)
        {
            uint64 val = ((uint64 *) &bits)[j];
            count += __builtin_popcountll(val);
        }
    }
    
    // Scalar remainder
    for (; i < bitmap->nbytes; i++)
        count += __builtin_popcountll(bitmap->bits[i]);
#else
    for (int i = 0; i < bitmap->nbytes; i++)
        count += __builtin_popcountll(bitmap->bits[i]);
#endif
    
    return count;
}
```

---

## TURN 9 — Integration with Parallel Query

**Instructions:**

Make vectorized execution work with PostgreSQL's parallel query infrastructure.

**Challenges:**
- Parallel workers each process batches
- Results must be gathered and merged
- Parallel aggregation needs partial aggregates

**Implement:**

```c
// Parallel vectorized scan
static VectorBatch *
ExecParallelSeqScanVectorized(SeqScanState *node)
{
    ParallelBlockTableScanDesc parallel_scan = node->ss_currentParallelScan;
    VectorBatch *batch;
    BlockNumber startblock;
    
    // Each worker gets different blocks via parallel scan
    while ((startblock = table_parallelscan_nextpage(parallel_scan)) !=InvalidBlockNumber)
    {
        // Scan this block and fill batch
        batch = ScanBlockToBatch(node, startblock);
        
        if (batch && batch->nbatch > 0)
            return batch;
    }
    
    return NULL;  // No more blocks for this worker
}

// Combine partial aggregates from parallel workers
static void
CombinePartialAggregates(AggState *node, VectorBatch **worker_results, int nworkers)
{
    // Merge hash tables from all workers
    for (int i = 0; i < nworkers; i++)
    {
        MergeHashAggTable(node->hash_table, worker_results[i]->agg_table);
    }
}
```

---

## TURN 10 — Comprehensive Regression Testing

**Instructions:**

Write extensive regression tests covering all code paths and edge cases.

**Tests:**

```sql
-- Test all data types
CREATE TABLE all_types_test (
    int2_col SMALLINT,
    int4_col INTEGER,
    int8_col BIGINT,
    float4_col REAL,
    float8_col DOUBLE PRECISION,
    numeric_col NUMERIC(10,2),
    text_col TEXT,
    date_col DATE,
    timestamp_col TIMESTAMP,
    bool_col BOOLEAN,
    array_col INTEGER[],
    json_col JSONB
);

-- Test NULL handling
INSERT INTO all_types_test VALUES (NULL, NULL, NULL, ...);
SELECT * FROM all_types_test WHERE int4_col IS NULL;

-- Test boundary conditions
SELECT * FROM all_types_test WHERE int8_col = 9223372036854775807; -- MAX_INT64

-- Test SIMD alignment edge cases
SELECT * FROM all_types_test LIMIT 1;  -- Single row
SELECT * FROM all_types_test LIMIT 1023; -- Just under batch size
SELECT * FROM all_types_test LIMIT 1024; -- Exact batch size
SELECT * FROM all_types_test LIMIT 1025; -- Just over batch size
```

---

## TURN 11 — Performance Benchmarking and Tuning

**Instructions:**

Run TPC-H benchmark suite, profile bottlenecks, and optimize.

**TPC-H Benchmark Results:**

```
Scale Factor: 10 (10GB database)

Query | Row-Based | Vectorized | Speedup
------|-----------|------------|--------
Q1    | 15.2s     | 2.4s       | 6.3x
Q3    | 8.5s      | 2.1s       | 4.0x
Q6    | 2.5s      | 0.45s      | 5.5x
Q12   | 4.1s      | 0.9s       | 4.5x
Q14   | 3.2s      | 0.7s       | 4.6x
Geomean:                        | 4.8x

Target: >3x improvement ✅
```

**Profiling and optimization:**

```c
// Profile with perf
$ perf record -g ./postgres ...
$ perf report

// Identify hotspots:
// 1. NULL bitmap operations (10% of time) → optimize with SIMD
// 2. Hash table probes (15% of time) → optimize hash function
// 3. Memory allocation (8% of time) → use memory pools

// Tuning parameters
SET work_mem = '256MB';  // Larger batches
SET max_parallel_workers_per_gather = 4;
SET enable_vectorized_exec = on;
```

**Deliverables:**
- Complete vectorized execution engine
- All regression tests passing
- TPC-H benchmark showing >3x speedup
- Documentation in SGML format
- Patch ready for postgresql-hackers mailing list
