# Task: Build a Production-Grade Collaborative Text Editor with Custom CRDT

## Overview
Implement a real-time collaborative text editor using a custom CRDT (Conflict-free Replicated Data Type) algorithm. The system must handle concurrent edits from 100+ users with strong eventual consistency, minimal bandwidth usage (<1KB per operation), sub-100ms latency, and proper handling of network partitions, out-of-order delivery, and malicious clients.

**Key Challenge:** You CANNOT use existing CRDT libraries like Yjs, Automerge, or ShareDB. You must implement the CRDT algorithm from scratch.

---

## TURN 1 – CRDT Algorithm Design: List CRDT with Tombstones

**Role:** You are a senior distributed systems engineer who has built collaborative editing systems at companies like Google Docs, Notion, or Figma. You deeply understand the trade-offs between Operational Transformation and CRDTs, and know that naive implementations leak memory (tombstone accumulation) or diverge under concurrent edits.

**Background:** Collaborative editors must allow multiple users to edit simultaneously without coordination. CRDTs guarantee eventual consistency without consensus. We'll implement a **List CRDT** (similar to RGA - Replicated Growable Array) where each character has a unique ID, and deletions are tombstones.

**Reference:** Read papers:
- "A comprehensive study of Convergent and Commutative Replicated Data Types" (Shapiro et al.)
- "Logoot: A Scalable Optimistic Replication Algorithm for Collaborative Editing" (Weiss et al.)
- Study Google's Realtime API architecture (deprecated but documented)

**VERY IMPORTANT:**
- Characters CANNOT use simple indices (0, 1, 2...) because concurrent inserts conflict
- Each character needs a globally unique position identifier
- Deletions mark characters as tombstones (visible=false) but keep them in the structure
- Operations must be commutative: `apply(op1, apply(op2, state)) === apply(op2, apply(op1, state))`
- No server-side coordination or locking allowed

**Goal:** Design and implement the core CRDT data structure with insertion and deletion operations.

**Instructions:**

1. **Design the position identifier system:**

   Options to consider:
   - **Fractional indexing:** Use strings like "a", "a0", "a1", "b" to represent positions between existing chars
   - **Lamport timestamps:** (timestamp, clientId, counter) tuples
   - **LSEQ/Logoot:** Dense tree with exponential growth
   
   For each option, analyze:
   - Memory overhead per character
   - Position identifier length growth over time
   - Collision probability under concurrent inserts at same position
   - Sorting/comparison performance

2. **Implement the core CRDT class:**
```typescript
// Core data structure
interface CharId {
  // Your position identifier scheme here
  // Must be globally unique and totally ordered
}

interface Char {
  id: CharId;
  value: string;
  visible: boolean; // false = tombstone
  timestamp: number; // Lamport clock
  clientId: string;
}

class TextCRDT {
  private chars: Char[] = [];
  private lamportClock: number = 0;
  private clientId: string;

  constructor(clientId: string) {
    this.clientId = clientId;
  }

  // Insert character at position (as seen by user)
  insert(position: number, char: string): Operation {
    // 1. Find CharId for position between chars[position-1] and chars[position]
    // 2. Create new Char with unique ID
    // 3. Insert into internal array (maintain sorted order by CharId)
    // 4. Increment Lamport clock
    // 5. Return operation for broadcast
    throw new Error("Implement me");
  }

  // Delete character at position (as seen by user)
  delete(position: number): Operation {
    // 1. Find visible char at position
    // 2. Mark as tombstone (visible=false)
    // 3. Increment Lamport clock
    // 4. Return operation for broadcast
    throw new Error("Implement me");
  }

  // Apply remote operation
  apply(op: Operation): void {
    // Must be idempotent - applying same op twice = no-op
    // Must update Lamport clock: max(local, remote) + 1
    throw new Error("Implement me");
  }

  // Get visible text (for rendering)
  toString(): string {
    return this.chars
      .filter(c => c.visible)
      .map(c => c.value)
      .join('');
  }

  // Get current character count (excluding tombstones)
  length(): number {
    return this.chars.filter(c => c.visible).length;
  }
}

interface Operation {
  type: 'insert' | 'delete';
  charId: CharId;
  value?: string; // for insert
  timestamp: number;
  clientId: string;
}
```

3. **Implement CharId comparison and generation:**
```typescript
class CharIdGenerator {
  // Generate ID between prevId and nextId
  static between(prevId: CharId | null, nextId: CharId | null, clientId: string): CharId {
    // Handle edge cases: beginning, end, between
    // Ensure generated IDs are dense (don't grow exponentially)
    throw new Error("Implement me");
  }

  static compare(a: CharId, b: CharId): number {
    // Return -1 if a < b, 0 if equal, 1 if a > b
    // Must define total order
    throw new Error("Implement me");
  }
}
```

4. **Write convergence tests:**
```typescript
describe('CRDT Convergence', () => {
  test('concurrent inserts at same position converge', () => {
    const doc1 = new TextCRDT('client1');
    const doc2 = new TextCRDT('client2');

    // Both insert 'a' at position 0
    const op1 = doc1.insert(0, 'a');
    const op2 = doc2.insert(0, 'a');

    // Apply in different orders
    doc1.apply(op2);
    doc2.apply(op1);

    // Must converge to same text (either "aa" or "aa", but consistent)
    expect(doc1.toString()).toBe(doc2.toString());
    
    // Order determined by CharId comparison
    // Test that character order is deterministic based on IDs
  });

  test('insert-delete-insert converges', () => {
    // Scenario: User A types "hello", User B deletes "el", User A types "world"
    // Must handle concurrent ops correctly
  });

  test('tombstone accumulation after 1000 deletes', () => {
    const doc = new TextCRDT('client1');
    
    // Insert 1000 chars
    for (let i = 0; i < 1000; i++) {
      doc.insert(i, 'x');
    }
    
    // Delete all
    for (let i = 999; i >= 0; i--) {
      doc.delete(i);
    }

    expect(doc.toString()).toBe('');
    // Internal array still has 1000 tombstones
    // We'll optimize this in Turn 4
  });
});
```

**Deliverables:**
- Complete CRDT implementation with position identifier system
- Explanation of your CharId design with trade-off analysis
- Tests demonstrating convergence under concurrent operations
- Performance test: Insert 10k characters, measure memory usage

---

## TURN 2 – WebSocket Server with Operation Broadcasting

**Instructions:**

Build a WebSocket server that broadcasts operations to all connected clients.

**Requirements:**
- Handle 100+ concurrent WebSocket connections
- Guarantee message ordering per connection
- Detect and handle client disconnections
- No operation loss during broadcast
- Backpressure handling (don't overwhelm slow clients)

**Implement:**
```typescript
import { WebSocket, WebSocketServer } from 'ws';

interface Client {
  id: string;
  ws: WebSocket;
  lastSeenTimestamp: number; // For detecting out-of-order delivery
}

class CollaborationServer {
  private wss: WebSocketServer;
  private clients: Map = new Map();
  private operationLog: Operation[] = []; // Persistent log for late joiners

  constructor(port: number) {
    this.wss = new WebSocketServer({ port });
    this.setupHandlers();
  }

  private setupHandlers() {
    this.wss.on('connection', (ws: WebSocket, req) => {
      const clientId = this.generateClientId();
      
      ws.on('message', (data: Buffer) => {
        // Parse operation
        // Validate operation (prevent malicious ops - covered in Turn 10)
        // Broadcast to all other clients
        // Append to operation log
      });

      ws.on('close', () => {
        // Remove client
        // Broadcast disconnect event (optional)
      });

      ws.on('error', (err) => {
        // Log and close connection
      });

      // Send operation history to new client
      this.sendHistory(ws, clientId);
    });
  }

  private broadcast(operation: Operation, excludeClientId: string) {
    const message = JSON.stringify(operation);
    
    for (const [id, client] of this.clients) {
      if (id === excludeClientId) continue;
      
      if (client.ws.readyState === WebSocket.OPEN) {
        // Check backpressure
        if (client.ws.bufferedAmount < 1024 * 1024) { // 1MB buffer
          client.ws.send(message);
        } else {
          // Handle slow client - maybe disconnect or queue
          console.warn(`Client ${id} is slow, buffered: ${client.ws.bufferedAmount}`);
        }
      }
    }
  }

  private sendHistory(ws: WebSocket, clientId: string) {
    // Send last N operations (or full history for now)
    // Client will apply them to catch up
    const history = {
      type: 'history',
      operations: this.operationLog,
    };
    ws.send(JSON.stringify(history));
  }

  private generateClientId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
}

// Start server
const server = new CollaborationServer(8080);
console.log('Collaboration server running on ws://localhost:8080');
```

**Client-side integration:**
```typescript
class CollaborativeEditor {
  private crdt: TextCRDT;
  private ws: WebSocket;
  private pendingOps: Operation[] = []; // Queue for offline ops

  constructor(serverUrl: string) {
    this.crdt = new TextCRDT(this.generateClientId());
    this.connectWebSocket(serverUrl);
  }

  private connectWebSocket(url: string) {
    this.ws = new WebSocket(url);

    this.ws.on('open', () => {
      // Send pending operations
      this.pendingOps.forEach(op => this.ws.send(JSON.stringify(op)));
      this.pendingOps = [];
    });

    this.ws.on('message', (data: Buffer) => {
      const message = JSON.parse(data.toString());
      
      if (message.type === 'history') {
        // Apply all historical operations
        message.operations.forEach((op: Operation) => this.crdt.apply(op));
        this.render();
      } else {
        // Apply single operation
        this.crdt.apply(message);
        this.render();
      }
    });

    this.ws.on('close', () => {
      // Attempt reconnection with exponential backoff
      setTimeout(() => this.connectWebSocket(url), 1000);
    });
  }

  // User types a character
  onInsert(position: number, char: string) {
    const op = this.crdt.insert(position, char);
    
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(op));
    } else {
      this.pendingOps.push(op); // Queue for later
    }
    
    this.render();
  }

  onDelete(position: number) {
    const op = this.crdt.delete(position);
    
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(op));
    } else {
      this.pendingOps.push(op);
    }
    
    this.render();
  }

  private render() {
    // Update DOM (simplified)
    console.log('Current text:', this.crdt.toString());
  }

  private generateClientId(): string {
    return `client-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
}
```

**Tests:**
```typescript
describe('WebSocket Broadcasting', () => {
  test('operation reaches all connected clients', async () => {
    // Start server
    const server = new CollaborationServer(8081);
    
    // Connect 3 clients
    const client1 = new CollaborativeEditor('ws://localhost:8081');
    const client2 = new CollaborativeEditor('ws://localhost:8081');
    const client3 = new CollaborativeEditor('ws://localhost:8081');
    
    await waitForConnections();
    
    // Client 1 inserts 'A'
    client1.onInsert(0, 'A');
    
    await sleep(100); // Allow broadcast
    
    // All clients should have 'A'
    expect(client1.crdt.toString()).toBe('A');
    expect(client2.crdt.toString()).toBe('A');
    expect(client3.crdt.toString()).toBe('A');
  });

  test('late joiner receives full history', async () => {
    // Client 1 makes 100 edits
    // Client 2 joins late
    // Client 2 should catch up to current state
  });

  test('handles client disconnect gracefully', async () => {
    // Client 1 disconnects
    // Remaining clients continue editing
    // Client 1 reconnects and syncs
  });
});
```

**Performance test:**
```typescript
test('handles 100 concurrent clients', async () => {
  const clients = [];
  for (let i = 0; i < 100; i++) {
    clients.push(new CollaborativeEditor('ws://localhost:8081'));
  }
  
  await waitForConnections();
  
  // Each client inserts a character
  clients.forEach((c, i) => c.onInsert(0, String(i)));
  
  await sleep(1000);
  
  // All clients should have same 100-character string
  const expected = clients[0].crdt.toString();
  clients.forEach(c => {
    expect(c.crdt.toString()).toBe(expected);
  });
});
```

---

## TURN 3 – Force Failure: Out-of-Order Message Delivery

**Instructions:**

Deliberately break message ordering to expose CRDT weaknesses.

**Scenario:** Simulate network reordering where operations arrive out of order:
```typescript
// Client A: types "hello"
op1 = insert(0, 'h')  // timestamp=1
op2 = insert(1, 'e')  // timestamp=2
op3 = insert(2, 'l')  // timestamp=3
op4 = insert(3, 'l')  // timestamp=4
op5 = insert(4, 'o')  // timestamp=5

// Network delivers to Client B in order: op1, op3, op2, op5, op4
```

**Ask the AI:**
> "Your CRDT assumes operations arrive in order. What happens when op3 arrives before op2? Show the exact failure mode with a test. Does your Lamport clock help? How do you ensure convergence?"

**Expected failure modes:**
- Characters appear in wrong order temporarily
- CharId generation fails (if it depends on previous CharId)
- Tombstones applied before inserts (delete arrives before insert)

**Test:**
```typescript
test('out-of-order delivery converges correctly', () => {
  const doc1 = new TextCRDT('client1');
  const doc2 = new TextCRDT('client2');

  // Client 1 types "abc"
  const opA = doc1.insert(0, 'a'); // timestamp=1
  const opB = doc1.insert(1, 'b'); // timestamp=2
  const opC = doc1.insert(2, 'c'); // timestamp=3

  // Client 2 receives in order: C, A, B
  doc2.apply(opC); // Should handle missing dependencies
  doc2.apply(opA);
  doc2.apply(opB);

  expect(doc2.toString()).toBe('abc'); // Must still converge
  expect(doc1.toString()).toBe(doc2.toString());
});

test('delete arrives before insert', () => {
  const doc1 = new TextCRDT('client1');
  const doc2 = new TextCRDT('client2');

  // Client 1 inserts 'x' then deletes it
  const insert = doc1.insert(0, 'x');
  const delete = doc1.delete(0);

  // Client 2 receives delete first
  doc2.apply(delete); // Should be buffered or no-op
  doc2.apply(insert); // Now apply insert

  // Both should converge to empty string
  expect(doc1.toString()).toBe('');
  expect(doc2.toString()).toBe('');
});
```

**Fix required:**
- Implement operation buffering for missing dependencies
- Use vector clocks or causal ordering
- Handle "future" operations gracefully

---

## TURN 4 – Garbage Collection: Tombstone Pruning

**Instructions:**

Implement a safe garbage collection mechanism to remove old tombstones without breaking convergence.

**Problem:** After 100k edits with deletions, the CRDT has massive memory overhead from tombstones.

**Requirements:**
- Only prune tombstones that all clients have acknowledged
- Use a distributed checkpoint protocol
- Must not interfere with active editing
- Reduce memory by >80% for documents with many deletions

**Implement:**
```typescript
interface Checkpoint {
  timestamp: number;
  clientAcks: Map; // clientId -> last seen timestamp
}

class GarbageCollector {
  private checkpoints: Checkpoint[] = [];
  private minAckedTimestamp: number = 0;

  // Each client periodically sends acknowledgment
  ackOperation(clientId: string, timestamp: number) {
    // Update checkpoint
    // Calculate new minAckedTimestamp (min across all clients)
  }

  // Prune tombstones older than minAckedTimestamp
  prune(crdt: TextCRDT): number {
    // Remove chars where visible=false AND timestamp < minAckedTimestamp
    // Return number of tombstones removed
    throw new Error("Implement me");
  }

  // Check if safe to prune up to this timestamp
  canPruneUntil(): number {
    // Return minAckedTimestamp
    return this.minAckedTimestamp;
  }
}

// Integrate into TextCRDT
class TextCRDT {
  private gc: GarbageCollector = new GarbageCollector();

  periodicGarbageCollection() {
    const pruneUntil = this.gc.canPruneUntil();
    
    this.chars = this.chars.filter(char => {
      if (!char.visible && char.timestamp < pruneUntil) {
        return false; // Remove tombstone
      }
      return true;
    });
  }

  // Client sends ack after applying operation
  acknowledgeOperation(timestamp: number) {
    this.gc.ackOperation(this.clientId, timestamp);
  }
}
```

**Server-side coordination:**
```typescript
class CollaborationServer {
  private gc: GarbageCollector = new GarbageCollector();

  private handleAck(clientId: string, timestamp: number) {
    this.gc.ackOperation(clientId, timestamp);
    
    // Broadcast safe prune timestamp to all clients
    const pruneUntil = this.gc.canPruneUntil();
    this.broadcast({
      type: 'gc-prune',
      timestamp: pruneUntil,
    }, '');
  }
}
```

**Tests:**
```typescript
test('garbage collection reduces memory', () => {
  const doc = new TextCRDT('client1');
  
  // Insert 10k characters
  for (let i = 0; i < 10000; i++) {
    doc.insert(i, 'x');
  }
  
  const beforeMemory = doc.getMemoryUsage();
  
  // Delete all
  for (let i = 9999; i >= 0; i--) {
    doc.delete(i);
  }
  
  const afterDelete = doc.getMemoryUsage();
  expect(afterDelete).toBeGreaterThan(beforeMemory * 0.9); // Tombstones still there
  
  // Simulate all clients acking
  doc.gc.ackOperation('client1', Date.now());
  doc.periodicGarbageCollection();
  
  const afterGC = doc.getMemoryUsage();
  expect(afterGC).toBeLessThan(beforeMemory * 0.2); // 80% reduction
});

test('garbage collection preserves convergence', () => {
  // Client A and B both edit
  // Client A prunes tombstones
  // Client C joins late (needs tombstones for correctness)
  // Must still converge
});

test('prevents premature pruning', () => {
  // Client A is offline
  // Other clients prune
  // Client A comes back online
  // Must detect missing operations and re-sync
});
```

---

## TURN 5 – Cursor Preservation and Intention Preservation

**Instructions:**

Implement cursor tracking that maintains user intention despite concurrent edits.

**Problem:** User A types at position 5. User B inserts 10 characters at position 0. User A's cursor should move to position 15, not stay at 5.

**Requirements:**
- Track cursor position relative to CharIds, not integer indices
- Update cursor positions when remote operations arrive
- Handle selections (start/end positions)
- Preserve intention for undo/redo

**Implement:**
```typescript
interface Cursor {
  clientId: string;
  charId: CharId | null; // Position before this CharId (null = end of document)
  type: 'caret' | 'selection';
  selectionEnd?: CharId; // For selections
}

class CursorManager {
  private cursors: Map = new Map();

  // Update local cursor position
  setCursor(clientId: string, position: number, crdt: TextCRDT) {
    // Convert integer position to CharId
    const charId = crdt.getCharIdAtPosition(position);
    
    this.cursors.set(clientId, {
      clientId,
      charId,
      type: 'caret',
    });
  }

  // When remote operation applied, update all cursors
  updateCursorsAfterOperation(op: Operation, crdt: TextCRDT) {
    for (const [clientId, cursor] of this.cursors) {
      if (op.type === 'insert') {
        // If insert happened before cursor, cursor CharId stays same (intention preserved)
        // No action needed - cursor naturally moves
      } else if (op.type === 'delete') {
        // If deleted char is cursor position, move cursor
        if (cursor.charId && this.isSameCharId(cursor.charId, op.charId)) {
          cursor.charId = this.getNextValidCharId(crdt, op.charId);
        }
      }
    }
  }

  // Convert CharId back to integer position (for rendering)
  getCursorPosition(clientId: string, crdt: TextCRDT): number {
    const cursor = this.cursors.get(clientId);
    if (!cursor || !cursor.charId) {
      return crdt.length(); // End of document
    }
    
    return crdt.getPositionOfCharId(cursor.charId);
  }

  private isSameCharId(a: CharId, b: CharId): boolean {
    // Compare CharIds
    return CharIdGenerator.compare(a, b) === 0;
  }

  private getNextValidCharId(crdt: TextCRDT, deletedId: CharId): CharId | null {
    // Find next visible character after deletedId
    throw new Error("Implement me");
  }
}

// Extend TextCRDT with cursor methods
class TextCRDT {
  getCharIdAtPosition(position: number): CharId | null {
    const visibleChars = this.chars.filter(c => c.visible);
    return position < visibleChars.length ? visibleChars[position].id : null;
  }

  getPositionOfCharId(charId: CharId): number {
    let position = 0;
    for (const char of this.chars) {
      if (char.visible) {
        if (CharIdGenerator.compare(char.id, charId) === 0) {
          return position;
        }
        position++;
      }
    }
    return position; // Not found
  }
}
```

**Tests:**
```typescript
test('cursor moves with remote inserts', () => {
  const doc1 = new TextCRDT('client1');
  const doc2 = new TextCRDT('client2');
  const cursors = new CursorManager();

  // Client 1 has cursor at position 5
  doc1.insert(0, 'hello');
  cursors.setCursor('client1', 5, doc1);

  // Client 2 inserts 'XXX' at position 0
  const op = doc2.insert(0, 'XXX');
  doc1.apply(op);

  // Update cursors after operation
  cursors.updateCursorsAfterOperation(op, doc1);

  // Client 1's cursor should now be at position 8 (5 + 3)
  expect(cursors.getCursorPosition('client1', doc1)).toBe(8);
  expect(doc1.toString()).toBe('XXXhello');
});

test('cursor preserved when character deleted before it', () => {
  const doc = new TextCRDT('client1');
  const cursors = new CursorManager();

  doc.insert(0, 'abcdef');
  cursors.setCursor('client1', 4); // Between 'd' and 'e'

  // Delete 'b' (position 1)
  const op = doc.delete(1);
  cursors.updateCursorsAfterOperation(op, doc);

  // Cursor should now be at position 3 (moved left by 1)
  expect(cursors.getCursorPosition('client1', doc)).toBe(3);
});

test('cursor moves when its character is deleted', () => {
  // Cursor on 'c'
  // Delete 'c'
  // Cursor should move to next character 'd'
});
```

---

## TURN 6 – Undo/Redo with CRDT Semantics

**Instructions:**

Implement undo/redo that works correctly with concurrent edits from other users.

**Challenge:** Traditional undo (reverse last operation) doesn't work in collaborative settings. If you type "A", another user types "B", then you undo, should "A" disappear or "B"?

**Approach:** Implement **selective undo** - only undo operations from specific client, not global history.

**Implement:**
```typescript
interface UndoOperation {
  original: Operation;
  inverse: Operation; // Operation to reverse it
  timestamp: number;
}

class UndoManager {
  private undoStack: UndoOperation[] = [];
  private redoStack: UndoOperation[] = [];
  private maxStackSize: number = 100;

  // Record operation for undo
  recordOperation(op: Operation, crdt: TextCRDT) {
    const inverse = this.createInverseOperation(op, crdt);
    
    this.undoStack.push({
      original: op,
      inverse,
      timestamp: op.timestamp,
    });

    // Clear redo stack
    this.redoStack = [];

    // Limit stack size
    if (this.undoStack.length > this.maxStackSize) {
      this.undoStack.shift();
    }
  }

  private createInverseOperation(op: Operation, crdt: TextCRDT): Operation {
    if (op.type === 'insert') {
      // Inverse of insert is delete
      return {
        type: 'delete',
        charId: op.charId,
        timestamp: Date.now(),
        clientId: op.clientId,
      };
    } else {
      // Inverse of delete is... tricky!
      // We can't truly "undo" a delete in CRDT without re-inserting
      // Option 1: Mark as visible=true again (violates CRDT semantics)
      // Option 2: Insert new character at same position (correct but duplicates)
      
      // Use Option 2 for correctness
      const char = crdt.getChar(op.charId);
      return {
        type: 'insert',
        charId: this.generateNewCharId(op.charId), // New ID!
        value: char.value,
        timestamp: Date.now(),
        clientId: op.clientId,
      };
    }
  }

  undo(crdt: TextCRDT): Operation | null {
    const undoOp = this.undoStack.pop();
    if (!undoOp) return null;

    this.redoStack.push(undoOp);

    // Apply inverse operation
    crdt.apply(undoOp.inverse);
    
    return undoOp.inverse; // Broadcast to other clients
  }

  redo(crdt: TextCRDT): Operation | null {
    const redoOp = this.redoStack.pop();
    if (!redoOp) return null;

    this.undoStack.push(redoOp);

    // Re-apply original operation
    crdt.apply(redoOp.original);
    
    return redoOp.original; // Broadcast
  }

  // Undo only operations from specific client (selective undo)
  undoForClient(clientId: string, crdt: TextCRDT): Operation | null {
    // Find most recent operation from this client
    const index = this.undoStack.findLastIndex(op => op.original.clientId === clientId);
    if (index === -1) return null;

    const undoOp = this.undoStack.splice(index, 1)[0];
    this.redoStack.push(undoOp);

    crdt.apply(undoOp.inverse);
    return undoOp.inverse;
  }

  private generateNewCharId(baseId: CharId): CharId {
    // Generate new unique ID near the base position
    throw new Error("Implement me");
  }
}
Tests:
typescripttest('undo own operation preserves other users edits', () => {
  const doc1 = new TextCRDT('client1');
  const doc2 = new TextCRDT('client2');
  const undo1 = new UndoManager();

  // Client 1 types "A"
  const op1 = doc1.insert(0, 'A');
  undo1.recordOperation(op1, doc1);
  doc2.apply(op1);

  // Client 2 types "B"
  const op2 = doc2.insert(1, 'B');
  doc1.apply(op2);

  // Both have "AB"
  expect(doc1.toString()).toBe('AB');
  expect(doc2.toString()).toBe('AB');

  // Client 1 undoes (removes "A")
  const undoOp = undo1.undo(doc1);
  doc2.apply(undoOp!);

  // Both should have "B" only
  expect(doc1.toString()).toBe('B');
  expect(doc2.toString()).toBe('B');
});

test('redo after undo restores state', () => {
  const doc = new TextCRDT('client1');
  const undo = new UndoManager();

  const op = doc.insert(0, 'X');
  undo.recordOperation(op, doc);

  expect(doc.toString()).toBe('X');

  undo.undo(doc);
  expect(doc.toString()).toBe('');

  undo.redo(doc);
  expect(doc.toString()).toBe('X');
});

test('undo stack clears on new edit', () => {
  const doc = new TextCRDT('client1');
  const undo = new UndoManager();

  // Type "ABC"
  ['A', 'B', 'C'].forEach((c, i) => {
    const op = doc.insert(i, c);
    undo.recordOperation(op, doc);
  });

  // Undo twice: "ABC" -> "AB" -> "A"
  undo.undo(doc);
  undo.undo(doc);
  expect(doc.toString()).toBe('A');

  // Type "D" (should clear redo stack)
  const op = doc.insert(1, 'D');
  undo.recordOperation(op, doc);
  expect(doc.toString()).toBe('AD');

  // Redo should do nothing (stack was cleared)
  undo.redo(doc);
  expect(doc.toString()).toBe('AD'); // Still "AD"
});

test('selective undo - undo only own operations', () => {
  const doc1 = new TextCRDT('client1');
  const doc2 = new TextCRDT('client2');
  const undo1 = new UndoManager();
  const undo2 = new UndoManager();

  // Interleaved edits: client1 types "A", client2 types "B", client1 types "C"
  const opA = doc1.insert(0, 'A');
  undo1.recordOperation(opA, doc1);
  doc2.apply(opA);

  const opB = doc2.insert(1, 'B');
  undo2.recordOperation(opB, doc2);
  doc1.apply(opB);

  const opC = doc1.insert(2, 'C');
  undo1.recordOperation(opC, doc1);
  doc2.apply(opC);

  // Both have "ABC"
  expect(doc1.toString()).toBe('ABC');

  // Client1 undoes - should remove "C", not "B"
  const undoOp = undo1.undoForClient('client1', doc1);
  doc2.apply(undoOp!);

  expect(doc1.toString()).toBe('AB');
  expect(doc2.toString()).toBe('AB');
});

TURN 7 – Rich Text Formatting with Inline Attributes
Instructions:
Extend the CRDT to support rich text formatting (bold, italic, links, etc.) without breaking convergence.
Challenge: Formatting spans must merge/split correctly under concurrent edits.
Approach: Store formatting as character-level attributes, not separate spans.
Implement:
typescriptinterface CharAttributes {
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
  link?: string;
  fontSize?: number;
  color?: string;
}

interface Char {
  id: CharId;
  value: string;
  visible: boolean;
  timestamp: number;
  clientId: string;
  attributes: CharAttributes; // Add formatting
}

class TextCRDT {
  // Insert with formatting
  insertFormatted(position: number, char: string, attributes: CharAttributes): Operation {
    const prevId = this.getCharIdAtPosition(position - 1);
    const nextId = this.getCharIdAtPosition(position);
    const charId = CharIdGenerator.between(prevId, nextId, this.clientId);

    const newChar: Char = {
      id: charId,
      value: char,
      visible: true,
      timestamp: ++this.lamportClock,
      clientId: this.clientId,
      attributes, // Include formatting
    };

    this.insertChar(newChar);

    return {
      type: 'insert',
      charId,
      value: char,
      timestamp: newChar.timestamp,
      clientId: this.clientId,
      attributes, // Broadcast formatting
    };
  }

  // Apply formatting to range
  format(startPos: number, endPos: number, attributes: Partial<CharAttributes>): Operation[] {
    const operations: Operation[] = [];

    for (let i = startPos; i < endPos; i++) {
      const char = this.getVisibleCharAt(i);
      if (!char) continue;

      // Merge attributes
      char.attributes = { ...char.attributes, ...attributes };

      // Create format operation
      operations.push({
        type: 'format',
        charId: char.id,
        attributes,
        timestamp: ++this.lamportClock,
        clientId: this.clientId,
      } as FormatOperation);
    }

    return operations;
  }

  // Apply remote format operation
  applyFormat(op: FormatOperation) {
    const char = this.getChar(op.charId);
    if (!char) return;

    // Merge attributes with last-write-wins
    char.attributes = { ...char.attributes, ...op.attributes };

    // Update Lamport clock
    this.lamportClock = Math.max(this.lamportClock, op.timestamp) + 1;
  }

  // Render with formatting (HTML output)
  toHTML(): string {
    let html = '';
    let openTags: string[] = [];

    for (const char of this.chars) {
      if (!char.visible) continue;

      // Close tags that are no longer active
      const newTags = this.getHTMLTags(char.attributes);
      
      // Simple approach: close all, reopen needed
      openTags.reverse().forEach(tag => {
        html += `</${tag}>`;
      });
      openTags = [];

      // Open new tags
      newTags.forEach(tag => {
        html += `<${tag}>`;
        openTags.push(tag.split(' ')[0]); // Store tag name only
      });

      html += this.escapeHTML(char.value);
    }

    // Close remaining tags
    openTags.reverse().forEach(tag => {
      html += `</${tag}>`;
    });

    return html;
  }

  private getHTMLTags(attrs: CharAttributes): string[] {
    const tags: string[] = [];
    if (attrs.bold) tags.push('strong');
    if (attrs.italic) tags.push('em');
    if (attrs.underline) tags.push('u');
    if (attrs.link) tags.push(`a href="${attrs.link}"`);
    if (attrs.color) tags.push(`span style="color:${attrs.color}"`);
    return tags;
  }

  private escapeHTML(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
}

interface FormatOperation extends Operation {
  type: 'format';
  attributes: Partial<CharAttributes>;
}
Tests:
typescripttest('bold formatting persists across edits', () => {
  const doc = new TextCRDT('client1');

  // Type "hello"
  doc.insert(0, 'h');
  doc.insert(1, 'e');
  doc.insert(2, 'l');
  doc.insert(3, 'l');
  doc.insert(4, 'o');

  // Make "ell" bold (positions 1-3)
  doc.format(1, 4, { bold: true });

  expect(doc.toHTML()).toBe('h<strong>ell</strong>o');
});

test('concurrent formatting converges', () => {
  const doc1 = new TextCRDT('client1');
  const doc2 = new TextCRDT('client2');

  // Both have "test"
  ['t', 'e', 's', 't'].forEach((c, i) => {
    const op = doc1.insert(i, c);
    doc2.apply(op);
  });

  // Client 1 makes "te" bold
  const ops1 = doc1.format(0, 2, { bold: true });
  
  // Client 2 makes "st" italic (overlapping at 't')
  const ops2 = doc2.format(2, 4, { italic: true });

  // Apply cross-operations
  ops1.forEach(op => doc2.applyFormat(op as FormatOperation));
  ops2.forEach(op => doc1.applyFormat(op as FormatOperation));

  // Both should converge: <strong>te</strong><em>st</em>
  // Or if overlapping is supported: <strong>te</strong><strong><em>s</em></strong><em>t</em>
  expect(doc1.toHTML()).toBe(doc2.toHTML());
});

test('formatting preserved when inserting in middle', () => {
  const doc = new TextCRDT('client1');

  doc.insert(0, 'h');
  doc.insert(1, 'i');
  
  // Make "hi" bold
  doc.format(0, 2, { bold: true });

  // Insert 'X' in middle
  doc.insertFormatted(1, 'X', { bold: true }); // Should inherit bold

  expect(doc.toHTML()).toBe('<strong>hXi</strong>');
});

test('formatting conflict resolution - last write wins', () => {
  const doc1 = new TextCRDT('client1');
  const doc2 = new TextCRDT('client2');

  // Both have "test"
  ['t', 'e', 's', 't'].forEach((c, i) => {
    const op = doc1.insert(i, c);
    doc2.apply(op);
  });

  // Client 1 makes first char red
  const op1 = doc1.format(0, 1, { color: 'red' })[0];
  
  // Client 2 makes first char blue (concurrent)
  const op2 = doc2.format(0, 1, { color: 'blue' })[0];

  // Apply cross-operations
  doc1.applyFormat(op2 as FormatOperation);
  doc2.applyFormat(op1 as FormatOperation);

  // Should converge based on timestamp (last write wins)
  expect(doc1.toHTML()).toBe(doc2.toHTML());
  
  // Whichever operation has higher timestamp wins
  const winner = op1.timestamp > op2.timestamp ? 'red' : 'blue';
  expect(doc1.toHTML()).toContain(winner);
});
```

---

## TURN 8 – Horizontal Scaling with Redis Pub/Sub

**Instructions:**

Scale to multiple server instances using Redis pub/sub for operation broadcasting.

**Architecture:**
```
[Client 1] --> [Server A] --\
[Client 2] --> [Server A] ---|
[Client 3] --> [Server B] ---|---> [Redis Pub/Sub] --> [All Servers]
[Client 4] --> [Server C] ---|
[Client 5] --> [Server C] --/
Requirements:

Servers are stateless (no shared memory)
Operations broadcast via Redis, not direct server-to-server
Handle Redis failures gracefully (fall back to direct broadcast)
Document persistence via Redis or separate DB

Implement:
typescriptimport Redis from 'ioredis';

class ScalableCollaborationServer {
  private wss: WebSocketServer;
  private redis: Redis;
  private redisSub: Redis; // Separate connection for subscriptions
  private clients: Map<string, Client> = new Map();
  private documentChannel: string = 'document:operations';

  constructor(port: number, redisUrl: string) {
    this.wss = new WebSocketServer({ port });
    this.redis = new Redis(redisUrl);
    this.redisSub = new Redis(redisUrl);
    
    this.setupRedisSubscription();
    this.setupWebSocketHandlers();
  }

  private setupRedisSubscription() {
    this.redisSub.subscribe(this.documentChannel, (err) => {
      if (err) {
        console.error('Redis subscription error:', err);
        // Fall back to in-memory broadcast
      }
    });

    this.redisSub.on('message', (channel, message) => {
      if (channel !== this.documentChannel) return;

      const operation = JSON.parse(message);
      
      // Broadcast to local clients only
      this.broadcastToLocalClients(operation);
    });
  }

  private setupWebSocketHandlers() {
    this.wss.on('connection', (ws: WebSocket) => {
      const clientId = this.generateClientId();
      
      this.clients.set(clientId, {
        id: clientId,
        ws,
        lastSeenTimestamp: 0,
      });

      ws.on('message', async (data: Buffer) => {
        try {
          const operation = JSON.parse(data.toString());
          
          // Validate operation
          if (!this.validateOperation(operation)) {
            ws.send(JSON.stringify({ error: 'Invalid operation' }));
            return;
          }

          // Publish to Redis (reaches all servers)
          await this.redis.publish(
            this.documentChannel,
            JSON.stringify(operation)
          );

          // Also persist to operation log
          await this.persistOperation(operation);

        } catch (err) {
          console.error('Error handling message:', err);
        }
      });

      ws.on('close', () => {
        this.clients.delete(clientId);
      });

      // Send operation history
      this.sendHistoryToClient(ws, clientId);
    });
  }

  private broadcastToLocalClients(operation: Operation) {
    const message = JSON.stringify(operation);

    for (const [id, client] of this.clients) {
      if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(message);
      }
    }
  }

  private async persistOperation(operation: Operation) {
    // Store in Redis sorted set (ordered by timestamp)
    await this.redis.zadd(
      'document:history',
      operation.timestamp,
      JSON.stringify(operation)
    );

    // Keep only last 10k operations
    await this.redis.zremrangebyrank('document:history', 0, -10001);
  }

  private async sendHistoryToClient(ws: WebSocket, clientId: string) {
    // Fetch operation history from Redis
    const history = await this.redis.zrange('document:history', 0, -1);
    
    const operations = history.map(op => JSON.parse(op));

    ws.send(JSON.stringify({
      type: 'history',
      operations,
    }));
  }

  private validateOperation(operation: Operation): boolean {
    // Validate structure and prevent malicious operations (Turn 10)
    return (
      operation.type &&
      operation.charId &&
      operation.timestamp &&
      operation.clientId
    );
  }

  private generateClientId(): string {
    return `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
}

// Start multiple instances
const server1 = new ScalableCollaborationServer(8080, 'redis://localhost:6379');
const server2 = new ScalableCollaborationServer(8081, 'redis://localhost:6379');
const server3 = new ScalableCollaborationServer(8082, 'redis://localhost:6379');

console.log('Scaled servers running on ports 8080, 8081, 8082');
Load Balancer Configuration (nginx):
nginxupstream collaboration_servers {
  least_conn;
  server localhost:8080;
  server localhost:8081;
  server localhost:8082;
}

server {
  listen 80;
  
  location / {
    proxy_pass http://collaboration_servers;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
  }
}
Tests:
typescripttest('operations reach clients on different servers', async () => {
  // Start 3 servers
  const server1 = new ScalableCollaborationServer(8080, 'redis://localhost:6379');
  const server2 = new ScalableCollaborationServer(8081, 'redis://localhost:6379');
  const server3 = new ScalableCollaborationServer(8082, 'redis://localhost:6379');

  // Connect clients to different servers
  const client1 = new CollaborativeEditor('ws://localhost:8080');
  const client2 = new CollaborativeEditor('ws://localhost:8081');
  const client3 = new CollaborativeEditor('ws://localhost:8082');

  await waitForConnections();

  // Client 1 inserts 'A'
  client1.onInsert(0, 'A');

  await sleep(200); // Allow Redis broadcast

  // All clients should have 'A'
  expect(client1.crdt.toString()).toBe('A');
  expect(client2.crdt.toString()).toBe('A');
  expect(client3.crdt.toString()).toBe('A');
});

test('handles Redis failure gracefully', async () => {
  // Stop Redis
  await stopRedis();

  // Servers should fall back to in-memory broadcast
  // Or enter degraded mode

  const client1 = new CollaborativeEditor('ws://localhost:8080');
  const client2 = new CollaborativeEditor('ws://localhost:8080'); // Same server

  // Operations still work locally
  client1.onInsert(0, 'X');
  
  await sleep(100);
  
  expect(client2.crdt.toString()).toBe('X'); // Same server works

  // Different servers won't sync (degraded mode)
});

test('server restart recovers from persistent log', async () => {
  const server = new ScalableCollaborationServer(8080, 'redis://localhost:6379');
  const client = new CollaborativeEditor('ws://localhost:8080');

  // Make 100 edits
  for (let i = 0; i < 100; i++) {
    client.onInsert(i, 'x');
  }

  await sleep(500);

  // Restart server
  await server.close();
  const server2 = new ScalableCollaborationServer(8080, 'redis://localhost:6379');

  // New client joins
  const client2 = new CollaborativeEditor('ws://localhost:8080');
  
  await sleep(500);

  // Should receive full history from Redis
  expect(client2.crdt.toString()).toBe('x'.repeat(100));
});

TURN 9 – Performance Optimization: Delta Compression
Instructions:
Reduce bandwidth by 90% using delta compression and binary encoding.
Current problem: Every operation sends full CharId (possibly 20+ bytes) + metadata. For 1000 operations/sec, this is 20KB/sec per client.
Optimizations:

Binary encoding: Replace JSON with MessagePack or custom binary format
Delta encoding: Send position deltas instead of absolute CharIds
Batch operations: Combine multiple operations into single message
Compression: Use gzip/brotli for large messages

Implement:
typescriptimport msgpack from 'msgpack-lite';
import zlib from 'zlib';

class BinaryOperationEncoder {
  // Encode operation to binary
  encode(op: Operation): Buffer {
    // Use MessagePack for compact binary
    const packed = msgpack.encode({
      t: this.encodeType(op.type),
      c: this.encodeCharId(op.charId),
      v: op.value,
      ts: op.timestamp,
      cl: this.encodeClientId(op.clientId),
      a: op.attributes, // formatting
    });

    // Compress if over 100 bytes
    if (packed.length > 100) {
      return zlib.gzipSync(packed);
    }

    return packed;
  }

  decode(buffer: Buffer): Operation {
    // Detect if compressed
    let data = buffer;
    if (this.isGzipped(buffer)) {
      data = zlib.gunzipSync(buffer);
    }

    const obj = msgpack.decode(data);

    return {
      type: this.decodeType(obj.t),
      charId: this.decodeCharId(obj.c),
      value: obj.v,
      timestamp: obj.ts,
      clientId: this.decodeClientId(obj.cl),
      attributes: obj.a,
    };
  }

  private encodeType(type: string): number {
    return type === 'insert' ? 0 : type === 'delete' ? 1 : 2; // format
  }

  private decodeType(code: number): string {
    return ['insert', 'delete', 'format'][code];
  }

  private encodeCharId(charId: CharId): any {
    // Implement efficient CharId encoding
    // E.g., if using fractional indexing, encode string efficiently
    throw new Error("Implement me");
  }

  private decodeCharId(encoded: any): CharId {
    throw new Error("Implement me");
  }

  private encodeClientId(clientId: string): number {
    // Use client ID pool with integer mapping
    // First time seeing client? Add to pool
    throw new Error("Implement me");
  }

  private decodeClientId(code: number): string {
    throw new Error("Implement me");
  }

  private isGzipped(buffer: Buffer): boolean {
    return buffer[0] === 0x1f && buffer[1] === 0x8b;
  }
}

// Batch multiple operations
class OperationBatcher {
  private batch: Operation[] = [];
  private batchTimeout: NodeJS.Timeout | null = null;
  private maxBatchSize: number = 50;
  private maxBatchDelay: number = 50; // ms

  add(op: Operation, sendCallback: (ops: Operation[]) => void) {
    this.batch.push(op);

    if (this.batch.length >= this.maxBatchSize) {
      this.flush(sendCallback);
    } else if (!this.batchTimeout) {
      this.batchTimeout = setTimeout(() => {
        this.flush(sendCallback);
      }, this.maxBatchDelay);
    }
  }

  private flush(sendCallback: (ops: Operation[]) => void) {
    if (this.batch.length === 0) return;

    sendCallback(this.batch);
    this.batch = [];

    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }
  }
}

// Integrate into server
class OptimizedCollaborationServer extends ScalableCollaborationServer {
  private encoder = new BinaryOperationEncoder();
  private batchers: Map<string, OperationBatcher> = new Map();

  protected setupWebSocketHandlers() {
    this.wss.on('connection', (ws: WebSocket) => {
      const clientId = this.generateClientId();
      const batcher = new OperationBatcher();
      this.batchers.set(clientId, batcher);

      // Use binary mode
      ws.binaryType = 'arraybuffer';

      ws.on('message', (data: ArrayBuffer) => {
        const operation = this.encoder.decode(Buffer.from(data));
        
        // Batch before broadcasting
        batcher.add(operation, (ops) => {
          this.broadcastBatch(ops, clientId);
        });
      });

      ws.on('close', () => {
        this.batchers.delete(clientId);
      });
    });
  }

  private broadcastBatch(operations: Operation[], excludeClientId: string) {
    // Encode batch
    const encoded = msgpack.encode(operations);
    const compressed = zlib.gzipSync(encoded);

    for (const [id, client] of this.clients) {
      if (id === excludeClientId) continue;
      
      if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(compressed);
      }
    }
  }
}
Tests:
typescripttest('binary encoding reduces message size', () => {
  const encoder = new BinaryOperationEncoder();
  
  const operation: Operation = {
    type: 'insert',
    charId: { /* complex ID */ },
    value: 'x',
    timestamp: Date.now(),
    clientId: 'client-123-456-789',
    attributes: { bold: true },
  };

  const jsonSize = JSON.stringify(operation).length;
  const binarySize = encoder.encode(operation).length;

  console.log(`JSON: ${jsonSize} bytes, Binary: ${binarySize} bytes`);
  expect(binarySize).toBeLessThan(jsonSize * 0.5); // At least 50% reduction
});

test('batching reduces message count', async () => {
  const batcher = new OperationBatcher();
  let sendCount = 0;
  
  const sendCallback = (ops: Operation[]) => {
    sendCount++;
  };

  // Add 100 operations rapidly
  for (let i = 0; i < 100; i++) {
    batcher.add({
      type: 'insert',
      charId: { /* */ },
      value: 'x',
      timestamp: Date.now(),
      clientId: 'client1',
    }, sendCallback);
  }

  await sleep(100); // Wait for batch flush

  // Should have sent 2-3 batches, not 100 messages
  expect(sendCount).toBeLessThan(5);
});

test('compression effective on large batches', () => {
  const operations: Operation[] = [];
  
  for (let i = 0; i < 1000; i++) {
    operations.push({
      type: 'insert',
      charId: { /* */ },
      value: 'x',
      timestamp: Date.now() + i,
      clientId: 'client1',
    });
  }

  const uncompressed = msgpack.encode(operations);
  const compressed = zlib.gzipSync(uncompressed);

  console.log(`Uncompressed: ${uncompressed.length} bytes, Compressed: ${compressed.length} bytes`);
  expect(compressed.length).toBeLessThan(uncompressed.length * 0.3); // 70%+ compression
});

test('bandwidth usage under load', async () => {
  // Simulate 100 clients typing at 60 chars/min
  // Measure total bandwidth
  
  const bytesPerOperation = 50; // Binary encoded
  const operationsPerMinute = 100 * 60;
  const bandwidth = bytesPerOperation * operationsPerMinute;

  console.log(`Bandwidth: ${(bandwidth / 1024).toFixed(2)} KB/min`);
  
  // Target: <100 KB/min for 100 clients
  expect(bandwidth).toBeLessThan(100 * 1024);
});

TURN 10 – Security: Prevent Malicious Operations
Instructions:
Harden the system against malicious clients sending crafted operations to corrupt the document or DOS other users.
Attack vectors:

Timestamp manipulation: Client sends operations with future timestamps to win conflicts
CharId collision: Attacker sends same CharId as existing character
Memory exhaustion: Send millions of operations to fill server memory
Invalid operations: Malformed data crashes parser
Position spoofing: Insert at impossible positions

Implement security measures:
typescriptclass OperationValidator {
  private clientRateLimits: Map<string, RateLimiter> = new Map();
  private suspiciousActivityLog: Map<string, number> = new Map();

  validate(op: Operation, clientId: string): ValidationResult {
    const errors: string[] = [];

    // 1. Rate limiting
    if (!this.checkRateLimit(clientId)) {
      return { valid: false, errors: ['Rate limit exceeded'] };
    }

    // 2. Timestamp validation (must be within reasonable range)
    const now = Date.now();
    if (op.timestamp > now + 5000) {
      errors.push('Timestamp in future');
      this.logSuspicious(clientId, 'future_timestamp');
    }
    if (op.timestamp < now - 3600000) {
      errors.push('Timestamp too old');
    }

    // 3. CharId validation
    if (!this.validateCharId(op.charId)) {
      errors.push('Invalid CharId structure');
    }

    // 4. Value validation
    if (op.type === 'insert' && !this.validateValue(op.value)) {
      errors.push('Invalid character value');
    }

    // 5. Client ID validation
    if (op.clientId !== clientId) {
      errors.push('Client ID mismatch');
      this.logSuspicious(clientId, 'id_spoofing');
    }

    // 6. Attribute validation (prevent XSS in rich text)
    if (op.attributes && !this.validateAttributes(op.attributes)) {
      errors.push('Invalid attributes');
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  private checkRateLimit(clientId: string): boolean {
    let limiter = this.clientRateLimits.get(clientId);
    
    if (!limiter) {
      limiter = new RateLimiter(100, 1000); // 100 ops per second
      this.clientRateLimits.set(clientId, limiter);
    }

    return limiter.allow();
  }

  private validateCharId(charId: CharId): boolean {
    // Check structure is valid
    // Prevent excessively long IDs (DOS via memory)
    if (JSON.stringify(charId).length > 1000) {
      return false; // Prevent memory exhaustion
    }

    // Check CharId has required fields based on your implementation
    // E.g., if using fractional indexing, validate string format
    return true;
  }

  private validateValue(value?: string): boolean {
    if (!value) return false;
    
    // Only single character allowed per operation
    if (value.length !== 1) return false;
    
    // Prevent control characters (except newline/tab)
    const code = value.charCodeAt(0);
    if (code < 32 && code !== 10 && code !== 9) {
      return false;
    }

    return true;
  }

  private validateAttributes(attrs: CharAttributes): boolean {
    // Prevent XSS via malicious links or styles
    if (attrs.link) {
      try {
        const url = new URL(attrs.link);
        // Only allow http(s) protocols
        if (!['http:', 'https:'].includes(url.protocol)) {
          return false;
        }
      } catch {
        return false; // Invalid URL
      }
    }

    // Prevent CSS injection
    if (attrs.color && !/^#[0-9A-Fa-f]{6}$/.test(attrs.color)) {
      return false;
    }

    // Validate font size range
    if (attrs.fontSize && (attrs.fontSize < 8 || attrs.fontSize > 72)) {
      return false;
    }

    return true;
  }

  private logSuspicious(clientId: string, activity: string) {
    const count = (this.suspiciousActivityLog.get(clientId) || 0) + 1;
    this.suspiciousActivityLog.set(clientId, count);

    if (count > 10) {
      // Ban client or trigger alert
      console.error(`Client ${clientId} engaged in suspicious activity: ${activity}`);
    }
  }
}

class RateLimiter {
  private tokens: number;
  private lastRefill: number;
  private maxTokens: number;
  private refillRate: number; // tokens per ms

  constructor(maxOpsPerSecond: number, windowMs: number) {
    this.maxTokens = maxOpsPerSecond;
    this.tokens = maxOpsPerSecond;
    this.refillRate = maxOpsPerSecond / windowMs;
    this.lastRefill = Date.now();
  }

  allow(): boolean {
    this.refill();
    
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    
    return false;
  }

  private refill() {
    const now = Date.now();
    const elapsed = now - this.lastRefill;
    const tokensToAdd = elapsed * this.refillRate;
    
    this.tokens = Math.min(this.maxTokens, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
}

// Integrate into server
class SecureCollaborationServer extends OptimizedCollaborationServer {
  private validator = new OperationValidator();
  private bannedClients: Set<string> = new Set();

  protected setupWebSocketHandlers() {
    this.wss.on('connection', (ws: WebSocket, req) => {
      const clientId = this.generateClientId();
      const clientIP = req.socket.remoteAddress;

      // Check if IP is banned
      if (this.isBanned(clientIP!)) {
        ws.close(4001, 'Banned');
        return;
      }

      ws.on('message', async (data: ArrayBuffer) => {
        try {
          const operation = this.encoder.decode(Buffer.from(data));
          
          // Validate operation
          const validation = this.validator.validate(operation, clientId);
          
          if (!validation.valid) {
            ws.send(JSON.stringify({
              type: 'error',
              errors: validation.errors,
            }));
            
            // Three strikes and you're out
            this.recordViolation(clientId, clientIP!);
            return;
          }

          // Additional CRDT-specific validation
          if (!this.validateCRDTSemantics(operation)) {
            ws.close(4002, 'Invalid CRDT operation');
            this.banClient(clientIP!);
            return;
          }

          // Operation is valid, broadcast
          await this.broadcastOperation(operation, clientId);

        } catch (err) {
          console.error('Error processing operation:', err);
          ws.send(JSON.stringify({
            type: 'error',
            errors: ['Malformed operation'],
          }));
        }
      });
    });
  }

  private validateCRDTSemantics(op: Operation): boolean {
    // Check that operation makes sense in CRDT context
    // E.g., CharId should be between valid boundaries
    // Delete operations should reference existing characters
    
    return true; // Implement based on your CRDT design
  }

  private recordViolation(clientId: string, ip: string) {
    // Implement violation tracking
    // After 3 violations, ban IP
  }

  private banClient(ip: string) {
    this.bannedClients.add(ip);
    console.warn(`Banned client: ${ip}`);
  }

  private isBanned(ip: string): boolean {
    return this.bannedClients.has(ip);
  }
}
Tests:
typescriptdescribe('Security - Malicious Operations', () => {
  test('rejects operations with future timestamps', () => {
    const validator = new OperationValidator();
    
    const op: Operation = {
      type: 'insert',
      charId: { /* valid */ },
      value: 'x',
      timestamp: Date.now() + 10000, // 10 seconds in future
      clientId: 'client1',
    };

    const result = validator.validate(op, 'client1');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Timestamp in future');
  });

  test('rejects multi-character inserts', () => {
    const validator = new OperationValidator();
    
    const op: Operation = {
      type: 'insert',
      charId: { /* valid */ },
      value: 'hello', // Should be single char
      timestamp: Date.now(),
      clientId: 'client1',
    };

    const result = validator.validate(op, 'client1');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Invalid character value');
  });

  test('rate limits excessive operations', () => {
    const validator = new OperationValidator();
    
    // Send 200 operations rapidly (limit is 100/sec)
    let rejected = 0;
    
    for (let i = 0; i < 200; i++) {
      const op: Operation = {
        type: 'insert',
        charId: { /* valid */ },
        value: 'x',
        timestamp: Date.now(),
        clientId: 'client1',
      };

      const result = validator.validate(op, 'client1');
      if (!result.valid) rejected++;
    }

    // At least half should be rejected
    expect(rejected).toBeGreaterThan(50);
  });

  test('prevents XSS via malicious links', () => {
    const validator = new OperationValidator();
    
    const op: Operation = {
      type: 'format',
      charId: { /* valid */ },
      timestamp: Date.now(),
      clientId: 'client1',
      attributes: {
        link: 'javascript:alert(1)', // XSS attempt
      },
    };

    const result = validator.validate(op, 'client1');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Invalid attributes');
  });

  test('prevents memory exhaustion via huge CharIds', () => {
    const validator = new OperationValidator();
    
    const hugeCharId = {
      data: 'x'.repeat(10000), // 10KB CharId
    };

    const op: Operation = {
      type: 'insert',
      charId: hugeCharId as any,
      value: 'x',
      timestamp: Date.now(),
      clientId: 'client1',
    };

    const result = validator.validate(op, 'client1');
    expect(result.valid).toBe(false);
  });

  test('detects and bans repeat offenders', async () => {
    const server = new SecureCollaborationServer(8080, 'redis://localhost:6379');
    const ws = new WebSocket('ws://localhost:8080');

    await waitForConnection(ws);

    // Send 20 malicious operations
    for (let i = 0; i < 20; i++) {
      ws.send(JSON.stringify({
        type: 'insert',
        charId: {},
        value: 'hello', // Invalid (multi-char)
        timestamp: Date.now(),
        clientId: 'malicious',
      }));
    }

    await sleep(100);

    // Connection should be closed/banned
    expect(ws.readyState).toBe(WebSocket.CLOSED);
  });

  test('prevents client ID spoofing', () => {
    const validator = new OperationValidator();
    
    const op: Operation = {
      type: 'insert',
      charId: { /* valid */ },
      value: 'x',
      timestamp: Date.now(),
      clientId: 'victim-client', // Pretending to be someone else
    };

    // Actual client is 'attacker'
    const result = validator.validate(op, 'attacker');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Client ID mismatch');
  });
});

TURN 11 – Comprehensive Integration: Full Editor UI
Instructions:
Build a complete web-based editor UI integrating all features from previous turns.
Requirements:

Real-time collaborative editing with visible user cursors
Rich text formatting toolbar
Undo/redo buttons
User presence indicators (who's online)
Connection status indicator
Performance metrics overlay (latency, ops/sec)
Responsive design (mobile-friendly)

Implement:
typescript// Frontend: editor.html + editor.ts

interface User {
  id: string;
  name: string;
  color: string;
  cursor: number;
}

class CollaborativeEditorUI {
  private editor: CollaborativeEditor;
  private cursorManager: CursorManager;
  private undoManager: UndoManager;
  private users: Map<string, User> = new Map();
  private editorElement: HTMLDivElement;
  private toolbarElement: HTMLDivElement;
  private statusElement: HTMLDivElement;

  constructor(serverUrl: string) {
    this.editor = new CollaborativeEditor(serverUrl);
    this.cursorManager = new CursorManager();
    this.undoManager = new UndoManager();
    
    this.setupUI();
    this.setupEventHandlers();
    this.startMetricsMonitoring();
  }

  private setupUI() {
    // Create editor container
    this.editorElement = document.createElement('div');
    this.editorElement.id = 'editor';
    this.editorElement.contentEditable = 'true';
    this.editorElement.style.cssText = `
      width: 100%;
      min-height: 500px;
      padding: 20px;
      border: 1px solid #ccc;
      font-family: 'Monaco', 'Courier New', monospace;
      font-size: 14px;
      outline: none;
    `;

    // Create toolbar
    this.toolbarElement = this.createToolbar();

    // Create status bar
    this.statusElement = this.createStatusBar();

    // Assemble
    document.body.appendChild(this.toolbarElement);
    document.body.appendChild(this.editorElement);
    document.body.appendChild(this.statusElement);

    // Add CSS for user cursors
    this.injectCursorStyles();
  }

  private createToolbar(): HTMLDivElement {
    const toolbar = document.createElement('div');
    toolbar.id = 'toolbar';
    toolbar.style.cssText = `
      display: flex;
      gap: 10px;
      padding: 10px;
      background: #f5f5f5;
      border-bottom: 1px solid #ccc;
    `;

    // Add formatting buttons
    const buttons = [
      { label: 'B', action: 'bold', style: 'font-weight: bold' },
      { label: 'I', action: 'italic', style: 'font-style: italic' },
      { label: 'U', action: 'underline', style: 'text-decoration: underline' },
      { label: '↶', action: 'undo', style: '' },
      { label: '↷', action: 'redo', style: '' },
    ];

    buttons.forEach(btn => {
      const button = document.createElement('button');
      button.textContent = btn.label;
      button.style.cssText = `
        padding: 5px 10px;
        cursor: pointer;
        ${btn.style}
      `;
      button.onclick = () => this.handleToolbarAction(btn.action);
      toolbar.appendChild(button);
    });

    // Add color picker
    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.onchange = (e) => {
      const color = (e.target as HTMLInputElement).value;
      this.applyFormatting({ color });
    };
    toolbar.appendChild(colorPicker);

    return toolbar;
  }

  private createStatusBar(): HTMLDivElement {
    const status = document.createElement('div');
    status.id = 'status';
    status.style.cssText = `
      display: flex;
      justify-content: space-between;
      padding: 10px;
      background: #f5f5f5;
      border-top: 1px solid #ccc;
      font-size: 12px;
    `;

    status.innerHTML = `
      <div id="connection-status">
        <span id="status-indicator" style="width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: grey;"></span>
        <span id="status-text">Connecting...</span>
      </div>
      <div id="users-online">
        <span>👥 Users: <span id="user-count">0</span></span>
      </div>
      <div id="performance">
        <span>⚡ Latency: <span id="latency">--</span>ms</span>
        <span style="margin-left: 10px;">📊 Ops/sec: <span id="ops-per-sec">0</span></span>
      </div>
    `;

    return status;
  }

  private setupEventHandlers() {
    // Handle user input
    this.editorElement.addEventListener('input', (e) => {
      this.handleInput(e);
    });

    this.editorElement.addEventListener('keydown', (e) => {
      this.handleKeyDown(e);
    });

    // Handle selection changes (for cursor broadcasting)
    document.addEventListener('selectionchange', () => {
      this.handleSelectionChange();
    });

    // Handle remote operations
    this.editor.on('operation', (op: Operation) => {
      this.handleRemoteOperation(op);
    });

    this.editor.on('connected', () => {
      this.updateConnectionStatus(true);
    });

    this.editor.on('disconnected', () => {
      this.updateConnectionStatus(false);
    });

    this.editor.on('user-joined', (user: User) => {
      this.users.set(user.id, user);
      this.updateUserCount();
      this.renderRemoteCursor(user);
    });

    this.editor.on('user-left', (userId: string) => {
      this.users.delete(userId);
      this.updateUserCount();
      this.removeRemoteCursor(userId);
    });

    this.editor.on('cursor-update', (userId: string, position: number) => {
      const user = this.users.get(userId);
      if (user) {
        user.cursor = position;
        this.updateRemoteCursor(user);
      }
    });
  }

  private handleInput(e: InputEvent) {
    // Get current selection
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const range = selection.getRangeAt(0);
    const position = this.getCaretPosition();

    if (e.inputType === 'insertText' && e.data) {
      // User typed a character
      const op = this.editor.crdt.insert(position, e.data);
      this.undoManager.recordOperation(op, this.editor.crdt);
      this.editor.broadcastOperation(op);
      
    } else if (e.inputType === 'deleteContentBackward') {
      // User pressed backspace
      if (position > 0) {
        const op = this.editor.crdt.delete(position - 1);
        this.undoManager.recordOperation(op, this.editor.crdt);
        this.editor.broadcastOperation(op);
      }
    }

    // Update cursor position
    this.broadcastCursorPosition();
  }

  private handleKeyDown(e: KeyboardEvent) {
    // Ctrl+Z / Cmd+Z for undo
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
      e.preventDefault();
      this.performUndo();
    }

    // Ctrl+Shift+Z / Cmd+Shift+Z for redo
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
      e.preventDefault();
      this.performRedo();
    }

    // Ctrl+B for bold, etc.
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
      e.preventDefault();
      this.applyFormatting({ bold: true });
    }
  }

  private handleToolbarAction(action: string) {
    switch (action) {
      case 'bold':
        this.applyFormatting({ bold: true });
        break;
      case 'italic':
        this.applyFormatting({ italic: true });
        break;
      case 'underline':
        this.applyFormatting({ underline: true });
        break;
      case 'undo':
        this.performUndo();
        break;
      case 'redo':
        this.performRedo();
        break;
    }
  }

  private applyFormatting(attributes: Partial<CharAttributes>) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const range = selection.getRangeAt(0);
    const startPos = this.getPositionFromNode(range.startContainer, range.startOffset);
    const endPos = this.getPositionFromNode(range.endContainer, range.endOffset);

    const operations = this.editor.crdt.format(startPos, endPos, attributes);
    operations.forEach(op => {
      this.undoManager.recordOperation(op, this.editor.crdt);
      this.editor.broadcastOperation(op);
    });

    this.render();
  }

  private performUndo() {
    const op = this.undoManager.undo(this.editor.crdt);
    if (op) {
      this.editor.broadcastOperation(op);
      this.render();
    }
  }

  private performRedo() {
    const op = this.undoManager.redo(this.editor.crdt);
    if (op) {
      this.editor.broadcastOperation(op);
      this.render();
    }
  }

  private handleRemoteOperation(op: Operation) {
    this.editor.crdt.apply(op);
    this.render();
    
    // Update performance metrics
    this.recordOperation();
  }

  private render() {
    // Convert CRDT to HTML
    const html = this.editor.crdt.toHTML();
    
    // Preserve cursor position
    const cursorPos = this.getCaretPosition();
    
    // Update DOM
    this.editorElement.innerHTML = html;
    
    // Restore cursor
    this.setCaretPosition(cursorPos);

    // Render remote cursors
    this.renderAllRemoteCursors();
  }

  private getCaretPosition(): number {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return 0;

    const range = selection.getRangeAt(0);
    return this.getPositionFromNode(range.startContainer, range.startOffset);
  }

  private getPositionFromNode(node: Node, offset: number): number {
    // Walk the DOM tree to calculate text position
    // This is simplified - production needs proper tree walking
    let position = 0;
    const walker = document.createTreeWalker(
      this.editorElement,
      NodeFilter.SHOW_TEXT,
      null
    );

    let currentNode;
    while ((currentNode = walker.nextNode())) {
      if (currentNode === node) {
        return position + offset;
      }
      position += currentNode.textContent?.length || 0;
    }

    return position;
  }

  private setCaretPosition(position: number) {
    // Set cursor to specific position
    const walker = document.createTreeWalker(
      this.editorElement,
      NodeFilter.SHOW_TEXT,
      null
    );

    let currentPos = 0;
    let currentNode;

    while ((currentNode = walker.nextNode())) {
      const nodeLength = currentNode.textContent?.length || 0;
      
      if (currentPos + nodeLength >= position) {
        const range = document.createRange();
        const selection = window.getSelection();
        
        range.setStart(currentNode, position - currentPos);
        range.collapse(true);
        
        selection?.removeAllRanges();
        selection?.addRange(range);
        return;
      }
      
      currentPos += nodeLength;
    }
  }

  private broadcastCursorPosition() {
    const position = this.getCaretPosition();
    this.editor.broadcastCursor(position);
  }

  private handleSelectionChange() {
    // Debounce cursor broadcasts
    clearTimeout(this.cursorBroadcastTimeout);
    this.cursorBroadcastTimeout = setTimeout(() => {
      this.broadcastCursorPosition();
    }, 100);
  }
  private cursorBroadcastTimeout: any;

  private renderRemoteCursor(user: User) {
    // Create cursor element
    const cursor = document.createElement('div');
    cursor.id = `cursor-${user.id}`;
    cursor.className = 'remote-cursor';
    cursor.style.cssText = `
      position: absolute;
      width: 2px;
      height: 20px;
      background: ${user.color};
      pointer-events: none;
      z-index: 1000;
    `;

    // Add user label
    const label = document.createElement('div');
    label.textContent = user.name;
    label.style.cssText = `
      position: absolute;
      top: -20px;
      left: 0;
      background: ${user.color};
      color: white;
      padding: 2px 5px;
      border-radius: 3px;
      font-size: 10px;
      white-space: nowrap;
    `;
    cursor.appendChild(label);

    document.body.appendChild(cursor);
    this.updateRemoteCursor(user);
  }

  private updateRemoteCursor(user: User) {
    const cursor = document.getElementById(`cursor-${user.id}`);
    if (!cursor) return;

    // Calculate pixel position from character position
    const coords = this.getCoordinatesForPosition(user.cursor);
    cursor.style.left = `${coords.x}px`;
    cursor.style.top = `${coords.y}px`;
  }

  private removeRemoteCursor(userId: string) {
    const cursor = document.getElementById(`cursor-${userId}`);
    cursor?.remove();
  }

  private renderAllRemoteCursors() {
    for (const user of this.users.values()) {
      this.updateRemoteCursor(user);
    }
  }

  private getCoordinatesForPosition(position: number): { x: number; y: number } {
    // Calculate pixel coordinates for a given text position
    // Simplified implementation
    const range = document.createRange();
    this.setCaretPosition(position);
    
    const selection = window.getSelection();
    if (selection && selection.rangeCount > 0) {
      const rect = selection.getRangeAt(0).getBoundingClientRect();
      return {
        x: rect.left + window.scrollX,
        y: rect.top + window.scrollY,
      };
    }

    return { x: 0, y: 0 };
  }

  private injectCursorStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .remote-cursor {
        animation: blink 1s infinite;
      }

      @keyframes blink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
      }
    `;
    document.head.appendChild(style);
  }

  private updateConnectionStatus(connected: boolean) {
    const indicator = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');
    
    if (indicator && text) {
      if (connected) {
        indicator.style.background = 'green';
        text.textContent = 'Connected';
      } else {
        indicator.style.background = 'red';
        text.textContent = 'Disconnected';
      }
    }
  }

  private updateUserCount() {
    const countElement = document.getElementById('user-count');
    if (countElement) {
      countElement.textContent = String(this.users.size + 1); // +1 for self
    }
  }

  private startMetricsMonitoring() {
    setInterval(() => {
      this.updateMetrics();
    }, 1000);
  }

  private opsRecorded = 0;
  private lastOpTime = Date.now();
  
  private recordOperation() {
    this.opsRecorded++;
  }

  private updateMetrics() {
    const opsPerSec = this.opsRecorded;
    this.opsRecorded = 0;

    const opsElement = document.getElementById('ops-per-sec');
    if (opsElement) {
      opsElement.textContent = String(opsPerSec);
    }

    // Measure latency (ping-pong with server)
    this.measureLatency();
  }

  private async measureLatency() {
    const start = Date.now();
    await this.editor.ping();
    const latency = Date.now() - start;

    const latencyElement = document.getElementById('latency');
    if (latencyElement) {
      latencyElement.textContent = String(latency);
    }
  }
}

// Initialize editor
const editor = new CollaborativeEditorUI('ws://localhost:8080');
Final Tests:
typescriptdescribe('Full Integration', () => {
  test('end-to-end collaborative editing', async () => {
    // Start server
    const server = new SecureCollaborationServer(8080, 'redis://localhost:6379');

    // Open 3 editor instances
    const page1 = await browser.newPage();
    const page2 = await browser.newPage();
    const page3 = await browser.newPage();

    await page1.goto('http://localhost:3000/editor.html');
    await page2.goto('http://localhost:3000/editor.html');
    await page3.goto('http://localhost:3000/editor.html');

    // Wait for connections
    await sleep(1000);

    // User 1 types "Hello"
    await page1.type('#editor', 'Hello');

    await sleep(500);

    // All pages should show "Hello"
    const text1 = await page1.$eval('#editor', el => el.textContent);
    const text2 = await page2.$eval('#editor', el => el.textContent);
    const text3 = await page3.$eval('#editor', el => el.textContent);

    expect(text1).toBe('Hello');
    expect(text2).toBe('Hello');
    expect(text3).toBe('Hello');
  });

  test('rich text formatting syncs correctly', async () => {
    // User 1 types "test" and makes it bold
    // User 2 should see bold text
  });

  test('cursors visible and accurate', async () => {
    // User 1 moves cursor to position 5
    // User 2 should see User 1's cursor at correct position
  });

  test('undo/redo preserves state across clients', async () => {
    // User 1 types "ABC", User 2 types "123"
    // User 1 undoes
    // All clients should converge to same state
  });

  test('handles 100 simultaneous users', async () => {
    // Stress test with many clients
    // Verify convergence and performance
  });
});
```

---

## Deliverables Summary

The complete collaborative editor should include:

1. **Core CRDT implementation** with custom position identifiers
2. **WebSocket server** with broadcasting
3. **Garbage collection** for tombstones
4. **Cursor tracking** with intention preservation
5. **Undo/redo** with CRDT semantics
6. **Rich text formatting** with attributes
7. **Horizontal scaling** via Redis
8. **Performance optimization** (compression, batching)
9. **Security hardening** against malicious operations
10. **Full UI** with real-time collaboration features

All features must work together seamlessly with:
- Sub-100ms latency
- Zero operation loss
- Strong eventual consistency
- Support for 100+ concurrent users
- <1KB bandwidth per operation
- Memory efficiency via garbage collection
```

---