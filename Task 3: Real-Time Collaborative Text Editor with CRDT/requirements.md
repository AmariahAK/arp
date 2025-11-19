# Requirements

## Prerequisites
- Node.js 20+ and npm/pnpm installed
- TypeScript 5.0+ configured
- WebSocket server capability (ws library)
- Browser with SharedArrayBuffer support (for testing)
- Basic understanding of distributed systems and conflict resolution

## Initial Setup
The developer should provide:
1. A TypeScript/Node.js development environment
2. A modern browser for testing the collaborative editor
3. Ability to run multiple client instances locally
4. Docker for optional Redis pub/sub setup (Turn 8+)

## Dependencies
- **Core:**
  - `typescript` (5.0+)
  - `ws` (WebSocket library)
  - `yjs` (for reference/comparison only, NOT to be used in implementation)
  
- **Testing:**
  - `vitest` or `jest`
  - `@testing-library/dom`
  - `puppeteer` (for multi-client simulation)
  
- **Optional (later turns):**
  - `redis` (for pub/sub scaling)
  - `ioredis` (Redis client)
  - `compression` (for delta compression)

## Testing Environment
- Minimum 4 CPU cores for concurrency tests
- At least 4GB RAM available
- Network access for WebSocket connections
- Multiple browser instances (Chrome recommended)

## Domain Knowledge Required
- Understanding of CRDTs (Conflict-free Replicated Data Types)
- Operational Transformation (OT) concepts
- Lamport timestamps and vector clocks
- WebSocket protocol
- Basic cryptography (for Turn 10)