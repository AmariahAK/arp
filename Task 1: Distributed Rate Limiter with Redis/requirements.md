# Requirements

## Prerequisites
- Docker and Docker Compose installed
- Go 1.21+ installed
- Redis 7.0+ (via Docker)
- Basic understanding of distributed systems

## Initial Setup
The developer should provide:
1. A running Redis instance (docker-compose.yml will be provided in Turn 1)
2. Go development environment configured
3. Access to run integration tests locally

## Dependencies
- `github.com/redis/go-redis/v9`
- `github.com/stretchr/testify` (for testing)
- Standard Go library packages

## Testing Environment
- Minimum 4 CPU cores for concurrency tests
- At least 2GB RAM available
- Network access for Redis connections