# Requirements

## Prerequisites
- Elixir 1.15+ and Erlang/OTP 26+
- PostgreSQL 15+ (for testing participant databases)
- Docker and Docker Compose
- Understanding of distributed systems and consensus algorithms
- Network partition simulation tools (toxiproxy or tc)

## Initial Setup
The developer should provide:
1. Elixir development environment with Mix
2. Docker cluster with at least 5 nodes
3. PostgreSQL instances for testing
4. Network chaos engineering tools installed

## Dependencies
- No external transaction libraries (Sage, Ecto.Multi allowed only for DB interaction)
- Must implement 2PC from scratch
- Allowed libraries:
  - `gen_statem` for state machines
  - `Phoenix.PubSub` for distributed messaging
  - `libcluster` for node discovery
  - `ecto` for database interaction only
  - `telemetry` for metrics

## Testing Environment
- Minimum 5 Erlang nodes in distributed cluster
- Network partition simulation capability
- PostgreSQL instances for each participant
- Ability to introduce arbitrary latency and packet loss

## Performance Requirements
- Handle 1000+ concurrent transactions
- Transaction commit latency <100ms (p99)
- Survive coordinator crash and recover automatically
- Handle network partitions gracefully
- Support 100k+ transactions per hour sustained load
