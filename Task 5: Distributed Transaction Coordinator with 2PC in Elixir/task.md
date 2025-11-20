# Task: Build a Production-Grade Distributed Transaction Coordinator with Two-Phase Commit

## Overview
Implement a fault-tolerant distributed transaction coordinator using the Two-Phase Commit (2PC) protocol in Elixir/Erlang OTP. This system must handle distributed transactions across heterogeneous databases, survive coordinator crashes, handle network partitions correctly, prevent distributed deadlocks, and maintain ACID guarantees while achieving high throughput (1000+ TPS) with sub-100ms latency.

**Key Challenge:** You CANNOT use existing distributed transaction frameworks (Sage allowed only for local DB ops). Implement 2PC from scratch using GenServers, supervisors, and distributed Erlang.

---

## TURN 1 — Core 2PC Protocol with GenServer State Machines

**Role:** You are a distributed systems engineer who has built transactional systems at scale. You understand the fundamental impossibility results (FLP, CAP), know why 2PC blocks on coordinator failure, and can implement recovery protocols.

**Background:** Two-Phase Commit is a distributed consensus protocol that ensures all participants either commit or abort a transaction atomically. It consists of a PREPARE phase (voting) and a COMMIT phase (decision propagation). The protocol blocks if the coordinator crashes during the commit phase.

**Reference:** Study:
- Jim Gray's "Notes on Data Base Operating Systems" (original 2PC paper)
- "Consensus on Transaction Commit" (Lamport)
- Erlang/OTP design principles (gen_statem, supervision trees)
- Paxos Commit as an alternative

**VERY IMPORTANT:**
- Coordinator and participants must persist state to disk before responding
- Crash recovery must handle all failure scenarios correctly
- No partial commits (atomicity violation)
- No lost commits (durability violation)
- Must handle participant timeouts and coordinator crashes
- Network partitions must not cause split-brain

**Goal:** Implement core 2PC coordinator and participant using gen_statem.

**Instructions:**

1. **Design the protocol flow:**
   - Client requests transaction begin, gets TxID
   - Client performs operations on multiple participants
   - Client requests commit
   - Coordinator sends PREPARE to all participants
   - Participants vote YES/NO (if YES, must be able to commit)
   - If all YES: Coordinator writes COMMIT decision to disk, sends COMMIT
   - If any NO/timeout: Coordinator writes ABORT decision, sends ABORT
   - Participants complete commit/abort and ACK
   - Coordinator releases resources

2. **Implement coordinator state machine:**
```elixir
defmodule DistributedTx.Coordinator do
  @moduledoc """
  Two-Phase Commit Coordinator
  
  States:
  - :idle -> :preparing -> :committing/:aborting -> :completed
  
  Must persist state transitions to survive crashes.
  """
  
  use GenStateMachine, callback_mode: [:state_functions, :state_enter]
  
  require Logger
  
  defmodule State do
    @enforce_keys [:tx_id, :participants, :client_pid]
    defstruct [
      :tx_id,
      :participants,        # %{participant_node => participant_pid}
      :client_pid,
      :prepare_timeout,
      :commit_timeout,
      votes: %{},           # %{participant_node => :yes | :no}
      decision: nil,        # :commit | :abort
      acks: MapSet.new(),   # Set of participant_nodes that ACKed
      start_time: nil
    ]
  end
  
  # Client API
  def start_link(tx_id, participants, client_pid) do
    GenStateMachine.start_link(__MODULE__, {tx_id, participants, client_pid})
  end
  
  def prepare(coordinator_pid) do
    GenStateMachine.call(coordinator_pid, :prepare, 30_000)
  end
  
  # Participant callbacks
  def vote(coordinator_pid, participant_node, vote) when vote in [:yes, :no] do
    GenStateMachine.cast(coordinator_pid, {:vote, participant_node, vote})
  end
  
  def ack(coordinator_pid, participant_node) do
    GenStateMachine.cast(coordinator_pid, {:ack, participant_node})
  end
  
  # GenStateMachine callbacks
  @impl true
  def init({tx_id, participants, client_pid}) do
    state = %State{
      tx_id: tx_id,
      participants: participants,
      client_pid: client_pid,
      prepare_timeout: 5_000,
      commit_timeout: 10_000,
      start_time: System.monotonic_time(:millisecond)
    }
    
    # Write initial state to disk (WAL)
    :ok = WAL.write_tx_start(tx_id, participants)
    
    {:ok, :idle, state}
  end
  
  # State: idle
  def idle(:enter, _old_state, state) do
    Logger.debug("TX #{state.tx_id}: Coordinator idle")
    :keep_state_and_data
  end
  
  def idle({:call, from}, :prepare, state) do
    Logger.info("TX #{state.tx_id}: Starting PREPARE phase")
    
    # Write PREPARING to disk
    :ok = WAL.write_tx_preparing(state.tx_id)
    
    # Send PREPARE to all participants
    for {node, pid} <- state.participants do
      send_prepare(node, pid, state.tx_id)
    end
    
    # Set timeout for prepare phase
    timeout = state.prepare_timeout
    
    {:next_state, :preparing, state,
     [{:reply, from, :ok}, {{:timeout, :prepare}, timeout, nil}]}
  end
  
  # State: preparing (collecting votes)
  def preparing(:enter, _old_state, state) do
    Logger.debug("TX #{state.tx_id}: Entered PREPARING state")
    :keep_state_and_data
  end
  
  def preparing(:cast, {:vote, node, vote}, state) do
    Logger.debug("TX #{state.tx_id}: Received #{vote} vote from #{node}")
    
    new_votes = Map.put(state.votes, node, vote)
    new_state = %{state | votes: new_votes}
    
    # Check if all votes received
    if map_size(new_votes) == map_size(state.participants) do
      decide_and_commit(new_state)
    else
      {:keep_state, new_state}
    end
  end
  
  def preparing({:timeout, :prepare}, _content, state) do
    Logger.warn("TX #{state.tx_id}: PREPARE timeout - aborting")
    
    # Timeout = implicit NO vote for missing participants
    decide_and_commit(%{state | decision: :abort})
  end
  
  # State: committing
  def committing(:enter, _old_state, state) do
    Logger.info("TX #{state.tx_id}: Committing transaction")
    
    # Write COMMIT decision to disk (CRITICAL - must be durable)
    :ok = WAL.write_tx_decision(state.tx_id, :commit)
    
    # Send COMMIT to all participants
    for {node, pid} <- state.participants do
      send_commit(node, pid, state.tx_id)
    end
    
    # Set timeout
    {{:timeout, :commit}, state.commit_timeout, nil}
  end
  
  def committing(:cast, {:ack, node}, state) do
    new_acks = MapSet.put(state.acks, node)
    new_state = %{state | acks: new_acks}
    
    if MapSet.size(new_acks) == map_size(state.participants) do
      # All participants ACKed - transaction complete
      Logger.info("TX #{state.tx_id}: All participants committed")
      finalize_transaction(new_state, :commit)
    else
      {:keep_state, new_state}
    end
  end
  
  def committing({:timeout, :commit}, _content, state) do
    Logger.warn("TX #{state.tx_id}: COMMIT timeout - some participants may not have ACKed")
    
    # Keep retrying or mark for recovery
    # In production, would retry indefinitely or escalate
    
    {:keep_state_and_data}
  end
  
  # State: aborting
  def aborting(:enter, _old_state, state) do
    Logger.info("TX #{state.tx_id}: Aborting transaction")
    
    # Write ABORT decision to disk
    :ok = WAL.write_tx_decision(state.tx_id, :abort)
    
    # Send ABORT to all participants (including those that voted YES)
    for {node, pid} <- state.participants do
      send_abort(node, pid, state.tx_id)
    end
    
    {{:timeout, :abort}, state.commit_timeout, nil}
  end
  
  def aborting(:cast, {:ack, node}, state) do
    new_acks = MapSet.put(state.acks, node)
    new_state = %{state | acks: new_acks}
    
    if MapSet.size(new_acks) == map_size(state.participants) do
      Logger.info("TX #{state.tx_id}: All participants aborted")
      finalize_transaction(new_state, :abort)
    else
      {:keep_state, new_state}
    end
  end
  
  # State: completed
  def completed(:enter, _old_state, state) do
    # Notify client
    send(state.client_pid, {:tx_result, state.tx_id, state.decision})
    
    # Clean up WAL after delay (keep for recovery)
    Process.send_after(self(), :cleanup_wal, 60_000)
    
    :keep_state_and_data
  end
  
  # Helper functions
  defp decide_and_commit(state) do
    decision =
      if Enum.all?(state.votes, fn {_node, vote} -> vote == :yes end) do
        :commit
      else
        :abort
      end
    
    next_state = if decision == :commit, do: :committing, else: :aborting
    
    {:next_state, next_state, %{state | decision: decision}}
  end
  
  defp send_prepare(node, pid, tx_id) do
    # TODO: Handle node down
    send({pid, node}, {:prepare, tx_id, self()})
  end
  
  defp send_commit(node, pid, tx_id) do
    send({pid, node}, {:commit, tx_id})
  end
  
  defp send_abort(node, pid, tx_id) do
    send({pid, node}, {:abort, tx_id})
  end
  
  defp finalize_transaction(state, decision) do
    duration = System.monotonic_time(:millisecond) - state.start_time
    Logger.info("TX #{state.tx_id}: Transaction #{decision} in #{duration}ms")
    
    # Record metrics
    :telemetry.execute([:tx, :complete], %{duration: duration}, %{decision: decision})
    
    {:next_state, :completed, state}
  end
end
```

3. **Implement participant state machine:**
```elixir
defmodule DistributedTx.Participant do
  use GenStateMachine, callback_mode: [:state_functions, :state_enter]
  
  defmodule State do
    defstruct [
      :tx_id,
      :coordinator_pid,
      :resource,           # Database connection or resource
      :prepare_data,       # Data prepared for commit
      :decision: nil
    ]
  end
  
  # States: :idle -> :prepared -> :committed/:aborted
  
  def idle(:info, {:prepare, tx_id, coordinator_pid}, state) do
    Logger.info("Participant: Received PREPARE for TX #{tx_id}")
    
    # Execute prepare logic (e.g., acquire locks, validate constraints)
    case prepare_transaction(state.resource, tx_id) do
      {:ok, prepare_data} ->
        # Write PREPARED to disk
        :ok = WAL.write_participant_prepared(tx_id, prepare_data)
        
        # Vote YES
        DistributedTx.Coordinator.vote(coordinator_pid, Node.self(), :yes)
        
        {:next_state, :prepared, %{state | tx_id: tx_id, coordinator_pid: coordinator_pid, prepare_data: prepare_data}}
      
      {:error, reason} ->
        Logger.warn("Participant: Cannot prepare TX #{tx_id}: #{inspect(reason)}")
        
        # Vote NO
        DistributedTx.Coordinator.vote(coordinator_pid, Node.self(), :no)
        
        {:keep_state_and_data}
    end
  end
  
  def prepared(:info, {:commit, tx_id}, state) do
    Logger.info("Participant: Committing TX #{tx_id}")
    
    # Execute commit
    :ok = commit_transaction(state.resource, state.prepare_data)
    
    # Write COMMITTED to disk
    :ok = WAL.write_participant_committed(tx_id)
    
    # ACK to coordinator
    DistributedTx.Coordinator.ack(state.coordinator_pid, Node.self())
    
    {:next_state, :committed, state}
  end
  
  def prepared(:info, {:abort, tx_id}, state) do
    Logger.info("Participant: Aborting TX #{tx_id}")
    
    # Rollback
    :ok = abort_transaction(state.resource, state.prepare_data)
    
    # Write ABORTED to disk
    :ok = WAL.write_participant_aborted(tx_id)
    
    # ACK to coordinator
    DistributedTx.Coordinator.ack(state.coordinator_pid, Node.self())
    
    {:next_state, :aborted, state}
  end
  
  # Helpers
  defp prepare_transaction(resource, tx_id) do
    # Execute pre-commit logic
    # Return {:ok, prepare_data} or {:error, reason}
    raise "Implement me"
  end
  
  defp commit_transaction(resource, prepare_data) do
    # Finalize commit
    raise "Implement me"
  end
  
  defp abort_transaction(resource, prepare_data) do
    # Rollback changes
    raise "Implement me"
  end
end
```

4. **Write-Ahead Log (WAL) for durability:**
```elixir
defmodule DistributedTx.WAL do
  @moduledoc """
  Write-Ahead Log for transaction state persistence.
  Must fsync after every write for durability.
  """
  
  def write_tx_start(tx_id, participants) do
    entry = {:tx_start, tx_id, participants, timestamp()}
    append_and_sync(entry)
  end
  
  def write_tx_preparing(tx_id) do
    entry = {:tx_preparing, tx_id, timestamp()}
    append_and_sync(entry)
  end
  
  def write_tx_decision(tx_id, decision) when decision in [:commit, :abort] do
    entry = {:tx_decision, tx_id, decision, timestamp()}
    append_and_sync(entry)
  end
  
  def write_participant_prepared(tx_id, prepare_data) do
    entry = {:participant_prepared, tx_id, prepare_data, timestamp()}
    append_and_sync(entry)
  end
  
  defp append_and_sync(entry) do
    binary = :erlang.term_to_binary(entry)
    
    File.write!(wal_path(), binary, [:append, :sync])
    
    :ok
  end
  
  defp wal_path do
    Path.join([System.tmp_dir!(), "distributed_tx_wal.log"])
  end
  
  defp timestamp do
    System.system_time(:microsecond)
  end
  
  # Recovery: replay WAL on startup
  def recover_transactions do
    if File.exists?(wal_path()) do
      {:ok, data} = File.read(wal_path())
      parse_wal(data)
    else
      %{}
    end
  end
  
  defp parse_wal(data) do
    # Parse and replay log entries
    # Return map of in-progress transactions
    raise "Implement me"
  end
end
```

5. **Tests:**
```elixir
defmodule DistributedTx.CoordinatorTest do
  use ExUnit.Case
  
  test "successful commit with all YES votes" do
    participants = setup_participants(3)
    
    {:ok, coordinator} = Coordinator.start_link("tx-001", participants, self())
    
    # Start prepare
    :ok = Coordinator.prepare(coordinator)
    
    # Simulate all YES votes
    for {node, _pid} <- participants do
      Coordinator.vote(coordinator, node, :yes)
    end
    
    # Receive commit result
    assert_receive {:tx_result, "tx-001", :commit}, 1000
  end
  
  test "abort with one NO vote" do
    participants = setup_participants(3)
    
    {:ok, coordinator} = Coordinator.start_link("tx-002", participants, self())
    
    :ok = Coordinator.prepare(coordinator)
    
    # Two YES, one NO
    [{node1, _}, {node2, _}, {node3, _}] = Map.to_list(participants)
    Coordinator.vote(coordinator, node1, :yes)
    Coordinator.vote(coordinator, node2, :no)  # One NO
    Coordinator.vote(coordinator, node3, :yes)
    
    assert_receive {:tx_result, "tx-002", :abort}, 1000
  end
  
  test "abort on prepare timeout" do
    participants = setup_participants(3)
    
    {:ok, coordinator} = Coordinator.start_link("tx-003", participants, self())
    
    :ok = Coordinator.prepare(coordinator)
    
    # Don't send votes - timeout should trigger abort
    assert_receive {:tx_result, "tx-003", :abort}, 6000
  end
end
```

**Deliverables:**
- Full 2PC implementation with coordinator and participant
- WAL for durability with fsync
- Tests covering success, abort, and timeout scenarios
- Documentation of protocol guarantees

---

## TURN 2 — Crash Recovery Protocol

**Instructions:**

Implement crash recovery for coordinator and participant failures.

**Scenarios to handle:**
1. Coordinator crashes during PREPARING → Recovery must abort or retry
2. Coordinator crashes after writing COMMIT → Recovery must complete commit
3. Participant crashes during PREPARED → Must commit when coordinator says so
4. Participant crashes before voting → Coordinator treats as NO vote

**Implement:**
```elixir
defmodule DistributedTx.Recovery do
  @moduledoc """
  Handles coordinator and participant crash recovery.
  """
  
  def recover_coordinator(tx_id) do
    case WAL.get_tx_state(tx_id) do
      {:preparing, state} ->
        # Coordinator crashed before decision - ABORT
        Logger.warn("Recovering TX #{tx_id}: Coordinator crashed before decision, aborting")
        broadcast_abort(state.participants, tx_id)
        
      {:decision, :commit, state} ->
        # Coordinator decided COMMIT - must complete
        Logger.info("Recovering TX #{tx_id}: Completing COMMIT")
        broadcast_commit(state.participants, tx_id)
        
      {:decision, :abort, state} ->
        # Coordinator decided ABORT
        Logger.info("Recovering TX #{tx_id}: Completing ABORT")
        broadcast_abort(state.participants, tx_id)
        
      nil ->
        Logger.warn("No state found for TX #{tx_id}")
        :ok
    end
  end
  
  def recover_participant(tx_id) do
    case WAL.get_participant_state(tx_id) do
      {:prepared, prepare_data} ->
        # Participant prepared but didn't get decision - contact coordinator
        Logger.info("Recovering TX #{tx_id}: Participant prepared,querying coordinator")
        query_coordinator_decision(tx_id, prepare_data)
        
      nil ->
        # No prepare record - transaction never reached this participant
        :ok
    end
  end
  
  defp query_coordinator_decision(tx_id, prepare_data) do
    # Contact coordinator to get decision
    # If coordinator also crashed, enter UNCERTAIN state
    # This is the blocking problem of 2PC!
    raise "Implement me"
  end
end
```

**Tests:**
```elixir
test "coordinator crash after COMMIT decision - recovery completes commit" do
  participants = setup_participants(3)
  
  {:ok, coordinator} = Coordinator.start_link("tx-004", participants, self())
  :ok = Coordinator.prepare(coordinator)
  
  # All vote YES
  for {node, _pid} <- participants do
    Coordinator.vote(coordinator, node, :yes)
  end
  
  # Kill coordinator after it writes COMMIT but before all ACKs
  Process.exit(coordinator, :kill)
  
  # Recover
  :ok = Recovery.recover_coordinator("tx-004")
  
  # All participants should be committed
  for {_node, pid} <- participants do
    assert participant_state(pid) == :committed
  end
end

test "participant crash during PREPARED - recovery completes on restart" do
  # Participant votes YES, then crashes
  # Coordinator decides COMMIT
  # Participant restarts and recovers
  # Must complete commit
end
```

---

## TURN 3 — Force Failure: Split Brain from Network Partition

**Instructions:**

Introduce a network partition that causes split-brain.

**Ask the AI:**
> "Your 2PC implementation doesn't handle network partitions correctly. What happens when the coordinator can reach 2 out of 3 participants, but the third participant can't reach the coordinator? Show the exact failure mode where the minority partition makes incorrect decisions."

**Expected failure:**
- Coordinator and 2 participants in majority partition
- 1 participant in minority partition
- Coordinator decides COMMIT (2 YES votes)
- Minority participant thinks transaction failed (timeout)
- Inconsistency: 2 committed, 1 aborted

**Test:**
```elixir
test "network partition causes split brain" do
  # Use :partisan or :inet_tcp_dist tricks to simulate partition
  
  # Setup 3 participants on different nodes
  participants = [
    {:"node1@localhost", pid1},
    {:"node2@localhost", pid2},
    {:"node3@localhost", pid3}
  ]
  
  {:ok, coordinator} = Coordinator.start_link("tx-005", participants, self())
  :ok = Coordinator.prepare(coordinator)
  
  # Partition network: coordinator + node1 + node2 | node3
  :partisan.partition([:"node3@localhost"], [:"coordinator@localhost", :"node1@localhost", :"node2@localhost"])
  
  # node1 and node2 vote YES
  Coordinator.vote(coordinator, :"node1@localhost", :yes)
  Coordinator.vote(coordinator, :"node2@localhost", :yes)
  
  # node3 can't vote (partition)
  # Coordinator times out on node3, but has 2/3 votes
  # What decision does it make?
  
  # Expected: Should abort (don't have unanimous YES)
  # Actual (with bug): Might commit with 2/3 majority
  
  assert_receive {:tx_result, "tx-005", decision}, 6000
  
  # Check consistency
  # All participants must have same decision
  state1 = get_participant_state(pid1)
  state2 = get_participant_state(pid2)
  state3 = get_participant_state(pid3)
  
  assert state1 == state2
  assert state2 == state3  # This will fail with split brain!
end
```

**Fix required:**
- Implement partition detection
- Require unanimous votes (timeout = NO vote)
- Add consensus layer (Paxos/Raft) for coordinator election

---

## TURN 4 — Distributed Deadlock Detection

**Instructions:**

Implement deadlock detection for distributed transactions waiting on each other.

**Example deadlock:**
- TX1 locks row A on DB1, waits for row B on DB2
- TX2 locks row B on DB2, waits for row A on DB1
- Both transactions stuck in PREPARING phase forever

**Implement:**
```elixir
defmodule DistributedTx.DeadlockDetector do
  use GenServer
  
  defmodule State do
    defstruct [
      wait_for_graph: %{},  # tx_id -> [waiting_for_tx_ids]
      transactions: %{}      # tx_id -> metadata
    ]
  end
  
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %State{}, name: __MODULE__)
  end
  
  def register_wait(tx_id, waiting_for_tx_id) do
    GenServer.cast(__MODULE__, {:wait, tx_id, waiting_for_tx_id})
  end
  
  def unregister_wait(tx_id, waiting_for_tx_id) do
    GenServer.cast(__MODULE__, {:unwait, tx_id, waiting_for_tx_id})
  end
  
  @impl true
  def handle_cast({:wait, tx_id, waiting_for_tx_id}, state) do
    # Add edge to wait-for graph
    new_graph = Map.update(state.wait_for_graph, tx_id, [waiting_for_tx_id], fn waiting ->
      [waiting_for_tx_id | waiting] |> Enum.uniq()
    end)
    
    new_state = %{state | wait_for_graph: new_graph}
    
    # Check for cycles (deadlock)
    case detect_cycle(new_graph, tx_id) do
      {:cycle, cycle_txs} ->
        Logger.warn("Deadlock detected: #{inspect(cycle_txs)}")
        abort_youngest_transaction(cycle_txs)
        {:noreply, new_state}
        
      :no_cycle ->
        {:noreply, new_state}
    end
  end
  
  defp detect_cycle(graph, start_tx) do
    # DFS to detect cycle
    detect_cycle_helper(graph, start_tx, [start_tx], MapSet.new([start_tx]))
  end
  
  defp detect_cycle_helper(graph, current, path, visited) do
    case Map.get(graph, current, []) do
      [] ->
        :no_cycle
        
      waiting_for ->
        Enum.reduce_while(waiting_for, :no_cycle, fn next_tx, _acc ->
          cond do
            next_tx in visited ->
              # Cycle found
              cycle_start_idx = Enum.find_index(path, &(&1 == next_tx))
              cycle = Enum.slice(path, cycle_start_idx..-1)
              {:halt, {:cycle, cycle}}
              
            true ->
              new_visited = MapSet.put(visited, next_tx)
              new_path = [next_tx | path]
              
              case detect_cycle_helper(graph, next_tx, new_path, new_visited) do
                {:cycle, _} = result -> {:halt, result}
                :no_cycle -> {:cont, :no_cycle}
              end
          end
        end)
    end
  end
  
  defp abort_youngest_transaction(cycle_txs) do
    # Abort transaction with highest timestamp (youngest)
    youngest = Enum.max_by(cycle_txs, &tx_timestamp/1)
    Logger.info("Aborting youngest transaction in deadlock: #{youngest}")
    
    # Send abort signal
    send_abort_signal(youngest)
  end
end
```

**Tests:**
```elixir
test "detects simple deadlock circle" do
  # TX1 waits for TX2
  # TX2 waits for TX1
  DeadlockDetector.register_wait("tx1", "tx2")
  DeadlockDetector.register_wait("tx2", "tx1")
  
  # One should be aborted
  assert_receive {:tx_aborted, tx_id, :deadlock}, 1000
  assert tx_id in ["tx1", "tx2"]
end

test "detects complex deadlock with 4 transactions" do
  # TX1 -> TX2 -> TX3 -> TX4 -> TX1
  DeadlockDetector.register_wait("tx1", "tx2")
  DeadlockDetector.register_wait("tx2", "tx3")
  DeadlockDetector.register_wait("tx3", "tx4")
  DeadlockDetector.register_wait("tx4", "tx1")  # Closes cycle
  
  # One transaction aborted
  assert_receive {:tx_aborted, _tx_id, :deadlock}, 1000
end
```

---

## TURN 5 — Optimistic 2PC with Presumed Abort

**Instructions:**

Optimize 2PC using Presumed Abort optimization to reduce messages and logging.

**Presumed Abort optimization:**
- Coordinator doesn't log ABORT decisions (presume abort if no log entry)
- Participants don't ACK ABORT messages (coordinator forgets immediately)
- Reduces log writes and messages for abort path (common case)
- Only commits require full logging and ACKs

**Implement:**
```elixir
defmodule DistributedTx.OptimizedCoordinator do
  @moduledoc """
  2PC with Presumed Abort optimization.
  
  Changes from basic 2PC:
  1. No logging of ABORT decision (assume abort if no decision logged)
  2. No ACKs needed for ABORT
  3. Coordinator forgets aborted transactions immediately
  """
  
  # In abort path:
  def aborting(:enter, _old_state, state) do
    Logger.info("TX #{state.tx_id}: Aborting (no logging)")
    
    # DON'T write ABORT to WAL (presumed abort optimization)
    
    # Send ABORT to all participants
    for {node, pid} <- state.participants do
      send_abort(node, pid, state.tx_id)
    end
    
    # Don't wait for ACKs - immediately complete
    finalize_transaction(state, :abort)
  end
  
  # In recovery:
  def recover_transaction(tx_id) do
    case WAL.get_tx_decision(tx_id) do
      {:commit, state} ->
        # Found COMMIT decision - complete it
        complete_commit(state)
        
      nil ->
        # No decision found - presume ABORT
        Logger.info("TX #{tx_id}: No decision found, presuming ABORT")
        :abort
    end
  end
end
```

**Benchmarks:**
```elixir
benchmark "compare basic vs optimized 2PC" do
  # Run 1000 transactions that abort
  
  Enum.each(1..1000, fn i ->
    tx_id = "tx-#{i}"
    
    {:ok, coordinator} = OptimizedCoordinator.start_link(tx_id, participants, self())
    :ok = Coordinator.prepare(coordinator)
    
    # Vote NO to trigger abort
    Coordinator.vote(coordinator, node1, :no)
    
    assert_receive {:tx_result, ^tx_id, :abort}, 1000
  end)
  
  # Check stats:
  # Optimized: ~50% fewer WAL writes
  # Optimized: ~30% faster abort latency
end
```

---

## TURN 6 — Three-Phase Commit (3PC) for Non-Blocking

**Instructions:**

Implement Three-Phase Commit to eliminate blocking on coordinator failure.

**3PC phases:**
1. **PREPARE** (CanCommit): Same as 2PC
2. **PRE-COMMIT**: Coordinator tells participants "going to commit, get ready"
3. **COMMIT**: Final commit

**Key property:** If coordinator crashes after PRE-COMMIT, participants can complete commit without coordinator (non-blocking).

**Tradeoff:** Longer latency (3 phases instead of 2), still subject to network partitions.

**Implement:**
```elixir
defmodule DistributedTx.ThreePhaseCommit do
  # States: idle -> preparing -> precommitting -> committing -> completed
  
  def preparing(:cast, {:vote, node, :yes}, state) do
    new_votes = Map.put(state.votes, node, :yes)
    new_state = %{state | votes: new_votes}
    
    if map_size(new_votes) == map_size(state.participants) and all_yes?(new_votes) do
      # All YES - move to PRECOMMITTING
      {:next_state, :precommitting, new_state}
    else
      {:keep_state, new_state}
    end
  end
  
  # New state: precommitting
  def precommitting(:enter, _old_state, state) do
    Logger.info("TX #{state.tx_id}: Entering PRECOMMIT phase")
    
    # Write PRECOMMIT decision to disk
    :ok = WAL.write_tx_precommit(state.tx_id)
    
    # Send PRECOMMIT to all participants
    for {node, pid} <- state.participants do
      send_precommit(node, pid, state.tx_id)
    end
    
    # Wait for ACKs
    {{:timeout, :precommit}, state.commit_timeout, nil}
  end
  
  def precommitting(:cast, {:ack_precommit, node}, state) do
    new_acks = MapSet.put(state.acks, node)
    
    if MapSet.size(new_acks) == map_size(state.participants) do
      # All acknowledged PRECOMMIT - safe to COMMIT
      {:next_state, :committing, %{state | acks: new_acks}}
    else
      {:keep_state, %{state | acks: new_acks}}
    end
  end
  
  # Recovery with 3PC
  def recover_coordinator(tx_id) do
    case WAL.get_tx_state(tx_id) do
      {:preparing, _} ->
        # Can safely abort
        :abort
        
      {:precommitting, state} ->
        # Past point of no return - must commit
        # This is the key difference from 2PC!
        complete_commit(state)
        
      {:committing, state} ->
        complete_commit(state)
    end
  end
end
```

**Tests comparing 2PC vs 3PC:**
```elixir
test "3PC allows recovery without coordinator in PRECOMMIT phase" do
  # Start 3PC transaction
  participants = setup_participants(3)
  {:ok, coordinator} = ThreePhaseCommit.start_link("tx-3pc-1", participants, self())
  
  # All vote YES
  :ok = Coordinator.prepare(coordinator)
  for {node, _} <- participants do
    Coordinator.vote(coordinator, node, :yes)
  end
  
  # Wait for PRECOMMIT phase
  :timer.sleep(100)
  
  # Kill coordinator in PRECOMMIT
  Process.exit(coordinator, :kill)
  
  # Participants can complete commit without coordinator!
  # They detect coordinator failure and coordinate among themselves
  
  for {_, pid} <- participants do
    assert participant_completes_commit(pid, "tx-3pc-1"), "Participant should complete commit"
  end
end

test "2PC blocks when coordinator crashes in COMMIT phase" do
  # Same test with 2PC - participants block waiting for coordinator
  
  {:ok, coordinator} = TwoPhaseCommit.start_link("tx-2pc-1", participants, self())
  
  # ... same setup ...
  
  # Kill coordinator after COMMIT decision but before sending COMMIT messages
  Process.exit(coordinator, :kill)
  
  # Participants are blocked in PREPARED state
  # They can't commit or abort without coordinator
  
  for {_, pid} <- participants do
    assert participant_state(pid) == :prepared, "Participant blocked in PREPARED"
  end
  
  # Only recovery can unblock
  :ok = Recovery.recover_coordinator("tx-2pc-1")
end
```

---

## TURN 7 — Paxos Commit for Fault-Tolerant Coordinator

**Instructions:**

Replace single coordinator with Paxos-based coordinator ensemble for fault tolerance.

**Problem with single coordinator:** Even with recovery, coordinator is a single point of failure. If it fails, transactions block.

**Solution:** Use Paxos to elect coordinator and replicate transaction decisions across multiple nodes.

**Implement:**
```elixir
defmodule DistributedTx.PaxosCoordinator do
  @moduledoc """
  Fault-tolerant coordinator using Paxos consensus.
  
  - 3 or 5 coordinator nodes form a Paxos ensemble
  - Transaction decisions (COMMIT/ABORT) agreed via Paxos
  - If leader fails, new leader elected automatically
  - No blocking on single node failure
  """
  
  use GenServer
  
  defmodule State do
    defstruct [
      :node_id,
      :ensemble,          # List of coordinator nodes
      :paxos_state,       # Paxos algorithm state
      :active_txs,        # Map of tx_id -> coordinator state
      :is_leader
    ]
  end
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  # Client API - same as before
  def prepare(tx_id) do
    # Forward to current leader
    leader = get_current_leader()
    GenServer.call({__MODULE__, leader}, {:prepare, tx_id})
  end
  
  @impl true
  def handle_call({:prepare, tx_id}, from, state) do
    if state.is_leader do
      # Leader handles prepare
      handle_prepare_as_leader(tx_id, from, state)
    else
      # Forward to leader
      leader = get_current_leader()
      {:reply, {:redirect, leader}, state}
    end
  end
  
  # When all votes collected, use Paxos to agree on decision
  defp decide_transaction(tx_id, votes, state) do
    decision = if all_yes?(votes), do: :commit, else: :abort
    
    # Propose decision via Paxos
    case Paxos.propose(state.paxos_state, {tx_id, decision}) do
      {:ok, {^tx_id, ^decision}, new_paxos_state} ->
        # Decision accepted by majority
        Logger.info("TX #{tx_id}: Paxos consensus reached for #{decision}")
        
        # Broadcast to participants
        broadcast_decision(tx_id, decision, state)
        
        %{state | paxos_state: new_paxos_state}
        
      {:error, reason} ->
        # Paxos failed (shouldn't happen with proper implementation)
        Logger.error("TX #{tx_id}: Paxos consensus failed: #{reason}")
        state
    end
  end
  
  # Leader election via Paxos
  defp elect_leader(state) do
    case Paxos.elect_leader(state.ensemble) do
      {:ok, leader_node} ->
        is_leader = leader_node == state.node_id
        Logger.info("New leader elected: #{leader_node} (am I leader: #{is_leader})")
        %{state | is_leader: is_leader}
        
      {:error, _reason} ->
        state
    end
  end
end
```

**Tests:**
```elixir
test "transaction completes despite coordinator failure" do
  # Start 3 coordinator nodes
  coordinators = start_coordinator_ensemble(3)
  
  participants = setup_participants(2)
  
  # Start transaction on leader
  :ok = PaxosCoordinator.prepare("tx-paxos-1")
  
  # All vote YES
  for {node, _} <- participants do
    PaxosCoordinator.vote("tx-paxos-1", node, :yes)
  end
  
  # Kill current leader
  leader = PaxosCoordinator.get_leader()
  Process.exit(leader, :kill)
  
  # New leader elected automatically
  :timer.sleep(500)  # Wait for election
  
  # Transaction should complete via new leader
  assert_receive {:tx_result, "tx-paxos-1", :commit}, 5000
end

test "handles network partition with majority" do
  coordinators = start_coordinator_ensemble(5)
  
  # Partition: 3 nodes vs 2 nodes
  :partisan.partition(
    Enum.take(coordinators, 3),
    Enum.drop(coordinators, 3)
  )
  
  # Transaction on majority partition succeeds
  # Transaction on minority partition fails (no quorum)
end
```

---

## TURN 8 — Force Failure: Subtle Race in Concurrent Transactions

**Ask the AI:**
> "Your system doesn't handle concurrent transactions accessing the same resources correctly. What happens when two transactions try to prepare the same database row simultaneously? Show a test where both transactions commit despite conflicting writes (lost update problem)."

**Expected failure:**
- TX1 reads row A (value=100)
- TX2 reads row A (value=100)
- TX1 prepares: SET A = 100 + 10 = 110
- TX2 prepares: SET A = 100 + 20 = 120
- Both prepare successfully (no lock conflict detected)
- Both commit
- Final value: 120 (lost TX1's update!)

**Test:**
```elixir
test "concurrent transactions cause lost update" do
  # Setup database with row A = 100
  {:ok, db} = TestDB.start_link(initial_value: 100)
  
  # Start two concurrent transactions
  task1 = Task.async(fn ->
    {:ok, tx1} = DistributedTx.begin(["increment by 10"])
    DistributedTx.read(tx1, db, :row_a)  # Reads 100
    DistributedTx.write(tx1, db, :row_a, 110)
    DistributedTx.commit(tx1)
  end)
  
  task2 = Task.async(fn ->
    {:ok, tx2} = DistributedTx.begin(["increment by 20"])
    DistributedTx.read(tx2, db, :row_a)  # Reads 100
    DistributedTx.write(tx2, db, :row_a, 120)
    DistributedTx.commit(tx2)
  end)
  
  Task.await(task1)
  Task.await(task2)
  
  final_value = TestDB.read(db, :row_a)
  
  # Expected: 130 (100 + 10 + 20)
  # Actual (with bug): 120 (lost update!)
  
  assert final_value == 130, "Lost update detected: #{final_value}"
end
```

**Fix required:** Implement proper locking or MVCC at prepare phase.

---

## TURN 9 — Performance Optimization: Batching and Pipelining

**Instructions:**

Optimize throughput using batching and pipelining of transactions.

**Optimizations:**
1. **Batch PREPARE messages:** Send single message with multiple tx_ids
2. **Pipeline decisions:** Don't wait for all ACKs before starting next tx
3. **Group commit:** Fsync multiple commit decisions together

**Implement:**
```elixir
defmodule DistributedTx.BatchCoordinator do
  @moduledoc """
  Optimized coordinator that batches operations for higher throughput.
  """
  
  defmodule State do
    defstruct [
      pending_prepares: [],          # Queue of txs waiting to prepare
      batch_size: 10,
      batch_timeout_ms: 5,
      wal_buffer: [],                # Buffer for group commit
      last_fsync: nil
    ]
  end
  
  def handle_call({:prepare, tx_id}, from, state) do
    # Add to pending batch
    new_pending = [{tx_id, from} | state.pending_prepares]
    
    if length(new_pending) >= state.batch_size do
      # Batch full - send immediately
      send_batch_prepare(new_pending, state)
      {:noreply, %{state | pending_prepares: []}}
    else
      # Wait for more or timeout
      if state.pending_prepares == [] do
        Process.send_after(self(), :flush_batch, state.batch_timeout_ms)
      end
      
      {:noreply, %{state | pending_prepares: new_pending}}
    end
  end
  
  defp send_batch_prepare(pending_txs, state) do
    tx_ids = Enum.map(pending_txs, fn {tx_id, _from} -> tx_id end)
    
    Logger.info("Sending batch PREPARE for #{length(tx_ids)} transactions")
    
    # Single message to each participant with all tx_ids
    for {node, pid} <- state.participants do
      send({pid, node}, {:batch_prepare, tx_ids, self()})
    end
  end
  
  # Group commit: buffer multiple commit decisions and fsync once
  defp write_decision_with_group_commit(tx_id, decision, state) do
    entry = {:tx_decision, tx_id, decision, timestamp()}
    
    new_buffer = [entry | state.wal_buffer]
    
    should_flush = 
      length(new_buffer) >= 100 or
      (state.last_fsync && timestamp() - state.last_fsync > 10_000)
    
    if should_flush do
      # Flush all buffered decisions in single fsync
      WAL.batch_write_and_sync(new_buffer)
      %{state | wal_buffer: [], last_fsync: timestamp()}
    else
      %{state | wal_buffer: new_buffer}
    end
  end
end
```

**Benchmarks:**
```elixir
benchmark "throughput with batching" do
  # Without batching: ~1000 tx/sec
  # With batching (size=10): ~8000 tx/sec
  # With batching + group commit: ~12000 tx/sec
  
  results = Benchee.run(%{
    "no batching" => fn ->
      DistributedTx.Coordinator.commit("tx-#{:rand.uniform(1_000_000)}")
    end,
    
    "with batching" => fn ->
      DistributedTx.BatchCoordinator.commit("tx-#{:rand.uniform(1_000_000)}")
    end
  }, time: 10, parallel: 8)
  
  # Assert batching is >5x faster
end
```

---

## TURN 10 — Observability: Distributed Tracing

**Instructions:**

Add distributed tracing using OpenTelemetry to trace transactions across nodes.

**Implement:**
```elixir
defmodule DistributedTx.Telemetry do
  require OpenTelemetry.Tracer
  
  def trace_transaction(tx_id, participant_nodes, fun) do
    OpenTelemetry.Tracer.with_span "transaction.execute", %{
      "tx.id" => tx_id,
      "tx.participants" => length(participant_nodes)
    } do
      result = fun.()
      
      # Add result to span
      OpenTelemetry.Tracer.set_attributes(%{
        "tx.result" => elem(result, 0)
      })
      
      result
    end
  end
  
  def trace_prepare_phase(tx_id, fun) do
    OpenTelemetry.Tracer.with_span "transaction.prepare", %{
      "tx.id" => tx_id
    }, do: fun.()
  end
  
  def trace_commit_phase(tx_id, decision, fun) do
    OpenTelemetry.Tracer.with_span "transaction.commit", %{
      "tx.id" => tx_id,
      "tx.decision" => decision
    }, do: fun.()
  end
  
  # Emit metrics
  def emit_transaction_metrics(tx_id, decision, duration_us) do
    :telemetry.execute(
      [:distributed_tx, :complete],
      %{duration: duration_us},
      %{tx_id: tx_id, decision: decision}
    )
  end
end

# Attach telemetry handlers
:telemetry.attach_many(
  "distributed-tx-metrics",
  [
    [:distributed_tx, :complete],
    [:distributed_tx, :prepare],
    [:distributed_tx, :commit]
  ],
  &handle_telemetry_event/4,
  nil
)
```

**Integration with Prometheus:**
```elixir
defmodule DistributedTx.Metrics do
  use Prometheus.Metric
  
  def setup do
    Counter.declare(
      name: :tx_total,
      help: "Total transactions",
      labels: [:decision, :node]
    )
    
    Histogram.declare(
      name: :tx_duration_microseconds,
      help: "Transaction duration",
      labels: [:decision],
      buckets: [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    )
    
    Gauge.declare(
      name: :tx_in_progress,
      help: "Transactions currently in progress"
    )
  end
  
  def record_transaction(decision, duration_us) do
    Counter.inc(name: :tx_total, labels: [decision, Node.self()])
    Histogram.observe([name: :tx_duration_microseconds, labels: [decision]], duration_us)
  end
end
```

---

## TURN 11 — Final Integration: Banking Application

**Instructions:**

Build a complete distributed banking application using the 2PC coordinator.

**Scenario:** Money transfer between accounts on different database shards.

**Implement:**
```elixir
defmodule BankingApp do
  @moduledoc """
  Distributed banking application with ACID guarantees.
  
  Uses 2PC to transfer money between accounts on different databases.
  """
  
  def transfer(from_account, to_account, amount) do
    # Start distributed transaction
    {:ok, tx_id} = DistributedTx.begin()
    
    # Identify which databases hold the accounts
    from_db = get_database_for_account(from_account)
    to_db = get_database_for_account(to_account)
    
    participants = build_participants([from_db, to_db])
    
    try do
      # Execute operations on each participant
      :ok = DistributedTx.execute(tx_id, from_db, fn ->
        debit_account(from_account, amount)
      end)
      
      :ok = DistributedTx.execute(tx_id, to_db, fn ->
        credit_account(to_account, amount)
      end)
      
      # Commit transaction
      case DistributedTx.commit(tx_id, participants) do
        {:ok, :commit} ->
          Logger.info("Transfer successful: #{from_account} -> #{to_account}, amount: #{amount}")
          {:ok, tx_id}
          
        {:ok, :abort} ->
          Logger.warn("Transfer aborted: #{from_account} -> #{to_account}")
          {:error, :transaction_aborted}
      end
    catch
      :error, reason ->
        DistributedTx.abort(tx_id, participants)
        {:error, reason}
    end
  end
  
  defp debit_account(account_id, amount) do
    # Check sufficient balance
    balance = DB.read_balance(account_id)
    
    if balance >= amount do
      DB.write_balance(account_id, balance - amount)
      :ok
    else
      {:error, :insufficient_funds}
    end
  end
  
  defp credit_account(account_id, amount) do
    balance = DB.read_balance(account_id)
    DB.write_balance(account_id, balance + amount)
    :ok
  end
end
```

**End-to-end tests:**
```elixir
test "money transfer maintains consistency" do
  # Setup: Account A (DB1) = $1000, Account B (DB2) = $500
  
  # Transfer $300 from A to B
  {:ok, _tx_id} = BankingApp.transfer("account_A", "account_B", 300)
  
  # Verify final state
  assert DB.read_balance("account_A") == 700
  assert DB.read_balance("account_B") == 800
  
  # Total money conserved
  assert 700 + 800 == 1500
end

test "transfer fails if insufficient funds" do
  # Account A = $100
  
  # Try to transfer $200
  {:error, :transaction_aborted} = BankingApp.transfer("account_A", "account_B", 200)
  
  # Balances unchanged
  assert DB.read_balance("account_A") == 100
end

test "concurrent transfers don't cause race conditions" do
  # Two concurrent transfers from same account
  # Should serialize properly with locking
  
  task1 = Task.async(fn ->
    BankingApp.transfer("account_A", "account_B", 100)
  end)
  
  task2 = Task.async(fn ->
    BankingApp.transfer("account_A", "account_C", 100)
  end)
  
  results = [Task.await(task1), Task.await(task2)]
  
  # One should succeed, one should fail (insufficient funds)
  assert Enum.count(results, &match?({:ok, _}, &1)) == 1
  assert Enum.count(results, &match?({:error, _}, &1)) == 1
end

test "system survives coordinator crash during transfer" do
  # Start transfer
  spawn fn ->
    BankingApp.transfer("account_A", "account_B", 500)
  end
  
  # Crash coordinator mid-transaction
  :timer.sleep(10)
  Process.exit(DistributedTx.Coordinator, :kill)
  
  # Recovery should complete or abort transaction
  :timer.sleep(1000)
  DistributedTx.Recovery.recover_all()
  
  # Verify consistency: either both updated or both unchanged
  balance_a = DB.read_balance("account_A")
  balance_b = DB.read_balance("account_B")
  
  assert (balance_a == 500 and balance_b == 1000) or
         (balance_a == 1000 and balance_b == 500)
end
```

**Performance test:**
```elixir
test "sustains 1000 concurrent transfers" do
  # Run 1000 transfers concurrently
  tasks = for i <- 1..1000 do
    Task.async(fn ->
      BankingApp.transfer("account_#{rem(i, 100)}", "account_#{rem(i+1, 100)}", 10)
    end)
  end
  
  results = Enum.map(tasks, &Task.await(&1, 30_000))
  
  successful = Enum.count(results, &match?({:ok, _}, &1))
  
  Logger.info("Completed #{successful}/1000 transfers")
  
  # Check final consistency: total money unchanged
  total = Enum.sum(for i <- 0..99, do: DB.read_balance("account_#{i}"))
  assert total == 100 * 1000  # Initial balance
end
```

**Deliverables:**
- Production-ready 2PC implementation with full recovery
- Paxos-based coordinator ensemble for fault tolerance
- Performance optimizations (batching, group commit)
- Complete banking application demonstrating correctness
- Comprehensive test suite
- Distributed tracing and metrics
- Documentation and runbook for operations
