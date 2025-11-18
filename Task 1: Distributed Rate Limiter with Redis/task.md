# Task: Build a Production-Grade Distributed Rate Limiter

## Overview
Implement a distributed rate limiter in Go using Redis as the backing store. This system must handle 100k+ requests/second across multiple instances with accurate rate limiting, minimal Redis load, and proper handling of edge cases like clock skew and Redis failures.

---

## TURN 1 – Architecture Design & Sliding Window Counter Setup

**Role:** You are a senior distributed systems engineer specializing in high-throughput rate limiting. You understand the trade-offs between token bucket, leaky bucket, fixed window, and sliding window algorithms. You know that naive Redis implementations create hotspots and that Lua scripts are essential for atomicity.

**Background:** We need a rate limiter that supports per-user, per-API-key, and per-IP limiting with multiple time windows (1 min, 1 hour, 1 day). It must be accurate (drift <1%), low-latency (<2ms p99), and resilient to Redis failures.

**Reference:** Study Cloudflare's rate limiting architecture and Redis's EVALSHA for Lua script caching.

**VERY IMPORTANT:** 
- No race conditions between instances
- No hot keys in Redis (use sharding if needed)
- Must degrade gracefully when Redis is slow/down
- Zero memory leaks in long-running tests

**Goal:** Design and implement the core sliding window counter algorithm with Lua script optimization.

**Instructions:**

1. **Propose architecture** covering:
   - Data structure in Redis (sorted sets vs. strings with TTL)
   - Lua script atomicity guarantees
   - How to handle multiple time windows efficiently
   - Clock skew mitigation strategy
   - Local cache layer (optional)

2. **Implement core structure:**
```go
package ratelimiter

type Config struct {
    MaxRequests int
    Window      time.Duration
    RedisAddr   string
}

type RateLimiter struct {
    client *redis.Client
    script *redis.Script
}

func New(cfg Config) (*RateLimiter, error) {
    // Initialize Redis client
    // Load Lua script
    return &RateLimiter{}, nil
}

// Allow checks if request is allowed and records it atomically
func (rl *RateLimiter) Allow(ctx context.Context, key string) (bool, error) {
    // Implement sliding window logic
    // Must be atomic via Lua script
    return false, nil
}
```

3. **Provide Lua script** for atomic check-and-increment:
   - Remove expired entries
   - Count current window requests
   - Add new timestamp if allowed
   - Return allow/deny + current count

4. **Write initial tests:**
```go
func TestBasicRateLimiting(t *testing.T) {
    // Test: 10 requests in 1-second window, limit=5
    // Expected: First 5 pass, next 5 fail
}
```

**Deliverables:**
- Full architecture explanation with diagrams (ASCII art OK)
- Working Go code with Lua script
- Tests passing with 100% accuracy
- Docker Compose file for Redis

---

## TURN 2 – Multi-Window Support & Sharding Strategy

**Instructions:**

Extend the rate limiter to support multiple windows simultaneously (e.g., 100 req/min AND 1000 req/hour).

**Requirements:**
- Single Redis call checks all windows atomically
- Use pipelining or multi-key Lua script
- Must not create O(n²) complexity with window count
- Implement Redis key sharding to avoid hotspots

**Implement:**
```go
type MultiWindowConfig struct {
    Windows []WindowLimit // e.g., [{100, 1min}, {1000, 1hour}]
}

type WindowLimit struct {
    MaxRequests int
    Duration    time.Duration
}

func (rl *RateLimiter) AllowMultiWindow(ctx context.Context, key string) (allowed bool, limitedBy *WindowLimit, err error) {
    // Check all windows atomically
    // Return which window caused rejection if denied
}
```

**Tests:**
```go
func TestMultiWindowEnforcement(t *testing.T) {
    // Config: 5/sec, 100/min
    // Send 6 requests in 1 second → 6th fails (second limit)
    // Wait 1 second, send 95 more → all pass
    // Send 1 more → fails (minute limit)
}

func TestShardingNoHotKeys(t *testing.T) {
    // Monitor Redis with MONITOR command
    // Assert no single key receives >10% of traffic
}
```

**Challenge:** How do you ensure all instances agree on time? Implement clock skew detection and correction.

---

## TURN 3 – Force Failure: Race Condition Under High Concurrency

**Instructions:**

Deliberately introduce a race condition by splitting the Lua script into two Redis calls (check + increment separately).

**Ask the AI:**
> "Your current implementation checks and increments in separate calls. What happens when 1000 goroutines hit the rate limiter simultaneously at the exact limit boundary? Show the race condition with a test."

**Expected failure:**
- Over-limit requests get through (e.g., 150 requests when limit is 100)
- Test should demonstrate >10% over-limit rate

**Test:**
```go
func TestConcurrentRaceCondition(t *testing.T) {
    limit := 100
    concurrent := 1000
    
    var wg sync.WaitGroup
    allowed := atomic.Int32{}
    
    for i := 0; i < concurrent; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            ok, _ := rl.Allow(ctx, "test-key")
            if ok {
                allowed.Add(1)
            }
        }()
    }
    wg.Wait()
    
    // With race condition: allowed > 100
    // Expected: allowed <= 100
    assert.LessOrEqual(t, allowed.Load(), int32(limit))
}
```

**Fix required:** Combine check + increment into single atomic Lua script.

---

## TURN 4 – Local Cache Layer with Probabilistic Admission

**Instructions:**

Add an in-memory cache to reduce Redis load by 80%+ while maintaining accuracy.

**Strategy:** 
- Cache recent allowances with short TTL (100-500ms)
- Use probabilistic counting (e.g., reservoir sampling) for high-frequency keys
- Sync with Redis periodically for accuracy

**Implement:**
```go
type CachedRateLimiter struct {
    *RateLimiter
    cache *ristretto.Cache // or sync.Map
}

func (crl *CachedRateLimiter) Allow(ctx context.Context, key string) (bool, error) {
    // Check local cache first
    // If cache miss or near limit, check Redis
    // Update cache with result
}
```

**Requirements:**
- Cache hit rate >80% under steady load
- Accuracy drift <1% vs. pure Redis implementation
- No thundering herd on cache expiry
- Memory bounded (max 100MB cache)

**Benchmarks:**
```go
func BenchmarkWithCache(b *testing.B) {
    // Target: >100k req/s on single instance
    // Redis load: <20k req/s
}
```

---

## TURN 5 – Graceful Degradation on Redis Failure

**Instructions:**

Implement fallback behavior when Redis is unavailable or slow (>10ms p99).

**Strategies to implement:**
1. **Fail-open mode:** Allow all requests (configurable)
2. **Fail-closed mode:** Deny all requests (configurable)
3. **Local-only mode:** Use in-memory rate limiter with best-effort accuracy

**Implement:**
```go
type FallbackConfig struct {
    Mode            string // "open", "closed", "local"
    HealthCheckInterval time.Duration
    CircuitBreakerThreshold int
}

func (rl *RateLimiter) AllowWithFallback(ctx context.Context, key string) (allowed bool, fromCache bool, err error) {
    // Try Redis with timeout
    // If timeout/error, use fallback strategy
    // Track failures for circuit breaker
}
```

**Tests:**
```go
func TestRedisDown(t *testing.T) {
    // Stop Redis container
    // Config: fail-open
    // Assert: All requests allowed, no errors
}

func TestRedisSlowCircuitBreaker(t *testing.T) {
    // Inject latency in Redis (tc netem or toxiproxy)
    // After 10 slow requests, circuit opens
    // Requests use local fallback
}
```

**Monitoring:** Add Prometheus metrics for fallback events.

---

## TURN 6 – Distributed Tracing & Observability

**Instructions:**

Integrate OpenTelemetry for distributed tracing and add structured logging.

**Implement:**
```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

func (rl *RateLimiter) Allow(ctx context.Context, key string) (bool, error) {
    ctx, span := otel.Tracer("ratelimiter").Start(ctx, "RateLimiter.Allow")
    defer span.End()
    
    // Add span attributes: key, window, result
    // Log cache hit/miss
    // Record Redis latency
}
```

**Requirements:**
- Trace every rate limit decision
- <5% performance overhead from tracing
- Correlate failures across instances via trace context
- Dashboard-ready metrics (Grafana + Tempo)

**Test:**
```go
func TestTracingOverhead(t *testing.T) {
    // Benchmark with/without tracing
    // Assert: <5% latency increase
}
```

---

## TURN 7 – Subtle Bug: Time Drift in Distributed System

**Instructions:**

Introduce a time synchronization bug by using `time.Now()` differently across instances.

**Scenario:** 
- Instance A uses UTC
- Instance B uses local time (PST, -8 hours)

**Ask the AI:**
> "If two instances have 8-hour clock skew, what happens to the sliding window? Show the exact failure mode with a test."

**Expected failure:**
- Instance B sees requests as "expired" too early
- Users get rate limited inconsistently based on which instance they hit

**Test:**
```go
func TestClockSkew(t *testing.T) {
    // Mock time.Now() on instance B to return time - 8 hours
    // Send requests to both instances
    // Assert: Inconsistent rate limiting (one allows, one denies)
}
```

**Fix required:** Use Redis `TIME` command for synchronized clock source.

---

## TURN 8 – Token Bucket Algorithm Comparison

**Instructions:**

Implement a token bucket variant and benchmark against sliding window.

**Implement:**
```go
type TokenBucketLimiter struct {
    // Token refill rate
    // Burst capacity
    // Last refill timestamp
}

func (tbl *TokenBucketLimiter) Allow(ctx context.Context, key string) (bool, error) {
    // Refill tokens based on time elapsed
    // Consume 1 token if available
}
```

**Comparison tests:**
```go
func TestBurstHandling(t *testing.T) {
    // Token bucket: Allows burst up to capacity
    // Sliding window: Strict per-window limit
    // Compare accuracy and Redis load
}
```

**Benchmark:**
```bash
BenchmarkSlidingWindow-8    100000    12000 ns/op    2 redis-ops
BenchmarkTokenBucket-8      100000    8000 ns/op     1 redis-ops
```

**Decision:** Which algorithm for which use case? Document trade-offs.

---

## TURN 9 – Production Load Test: 1M req/s Across 10 Instances

**Instructions:**

Deploy 10 instances + Redis cluster, run load test with realistic traffic patterns.

**Setup:**
```yaml
# docker-compose.yml
services:
  redis-cluster:
    # 3-node cluster with replicas
  
  ratelimiter:
    replicas: 10
    # Resource limits: 1 CPU, 512MB RAM
  
  load-generator:
    # k6 or wrk for traffic generation
```

**Traffic patterns:**
- 80% from top 1000 users (hot keys)
- 20% from long tail (100k users)
- Spike to 2M req/s for 10 seconds

**Requirements:**
- p99 latency <5ms
- p99.9 latency <20ms
- Zero request loss
- Redis CPU <70%
- No memory leaks after 24 hours

**Tests:**
```go
func TestProductionLoad(t *testing.T) {
    // Run for 10 minutes
    // Monitor metrics: latency, error rate, Redis load
    // Assert: All SLOs met
}
```

**Failure modes to test:**
- Redis node failure → automatic failover
- Network partition → circuit breaker activation
- Memory pressure → cache eviction without crash

---

## TURN 10 – Security: Prevent Rate Limit Bypasses

**Instructions:**

Harden the system against common bypass techniques.

**Attack vectors to prevent:**

1. **Key enumeration:** Attacker tries millions of user IDs to find unlocked ones
```go
   func (rl *RateLimiter) AllowWithEnumProtection(ctx context.Context, key string) (bool, error) {
       // Global rate limit on failed lookups
       // Honeypot keys that trigger alerts
   }
```

2. **Clock manipulation:** Client sends past timestamps
```go
   // Validate timestamp is within acceptable range
   // Reject requests >5 seconds in past/future
```

3. **Distributed denial of resources:** 1M unique keys to exhaust Redis memory
```go
   // Implement LRU eviction policy
   // Monitor memory usage, reject new keys if >80% full
```

**Tests:**
```go
func TestEnumerationAttack(t *testing.T) {
    // Try 10k different keys rapidly
    // Assert: Attack detected, subsequent requests blocked
}

func TestMemoryExhaustion(t *testing.T) {
    // Create 1M unique rate limit keys
    // Assert: Redis memory <1GB, old keys evicted
}
```

---

## TURN 11 – Final Integration: HTTP Middleware with Contextual Limits

**Instructions:**

Create production-ready HTTP middleware with per-endpoint, per-user, and per-IP limiting.

**Implement:**
```go
func RateLimitMiddleware(rl *RateLimiter, config EndpointConfig) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // Extract user ID from JWT
            // Extract IP from X-Forwarded-For
            // Build composite key: "user:{id}:endpoint:{path}:ip:{ip}"
            
            allowed, limitedBy, err := rl.AllowMultiWindow(r.Context(), key)
            if !allowed {
                w.Header().Set("X-RateLimit-Limit", strconv.Itoa(limitedBy.MaxRequests))
                w.Header().Set("X-RateLimit-Remaining", "0")
                w.Header().Set("X-RateLimit-Reset", resetTime.Format(time.RFC3339))
                w.WriteHeader(http.StatusTooManyRequests)
                json.NewEncoder(w).Encode(map[string]string{
                    "error": "rate limit exceeded",
                })
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}

type EndpointConfig struct {
    Path    string
    Limits  []WindowLimit
    Scope   []string // "user", "ip", "global"
}
```

**Requirements:**
- Standard rate limit headers (RateLimit-*)
- Graceful degradation on middleware errors
- Configurable via YAML/environment variables
- Zero-downtime config reload

**Final test:**
```go
func TestEndToEnd(t *testing.T) {
    // Spin up full HTTP server with middleware
    // Simulate 1000 concurrent users hitting 10 endpoints
    // Assert: Correct limits enforced per endpoint
    // Assert: No crashes, memory leaks, or panics
}
```

**Demo:** Include curl commands showing rate limit headers in action.