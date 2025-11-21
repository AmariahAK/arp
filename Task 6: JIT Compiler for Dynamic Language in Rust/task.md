# Task: Build a Production-Grade JIT Compiler for a Dynamic Language

## Overview
Implement a Just-In-Time (JIT) compiler in Rust that translates bytecode from a dynamic scripting language into native x86-64 machine code at runtime. The system must generate optimized code competitive with handwritten assembly, handle dynamic typing with inline caching, implement register allocation and instruction selection, support garbage collector integration, and maintain correctness under aggressive optimizations.

**Key Challenge:** You CANNOT use existing JIT frameworks (Cranelift, LLVM JIT, LuaJIT internals). Everything must be built from scratch: assembler, register allocator, optimization passes.

---

## TURN 1 — Basic Code Generation: Bytecode to x86-64

**Role:** You are a compiler engineer who has built JIT compilers for JavaScript engines (V8, SpiderMonkey) or language VMs (JVM, CLR). You understand calling conventions, ABI requirements, and can hand-code assembly with perfect correctness.

**Background:** Our dynamic language has a stack-based bytecode VM. A JIT compiler translates hot functions from bytecode to native code for 10-100x speedups. The first step is basic code generation without optimizations.

**Reference:** Study:
- LuaJIT architecture (Mike Pall's design)
- V8's TurboFan compiler pipeline
- x86-64 calling conventions (System V AMD64 ABI)
- Inline assembly in Rust (`asm!` macro)

**VERY IMPORTANT:**
- Generated code must respect calling conventions exactly
- Stack must be 16-byte aligned before CALL instructions
- All register allocation must handle callee-saved registers
- No buffer overflows in code buffer
- Generated code must be verifiable via disassembly
- Must handle JIT code buffer exhaustion gracefully

**Goal:** Implement basic JIT compiler that translates simple bytecode to x86-64.

**Instructions:**

1. **Define the bytecode format:**
```rust
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Opcode {
    // Stack operations
    LoadConst,     // Push constant onto stack
    LoadLocal,     // Load local variable  
    StoreLocal,    // Store to local variable
    
    // Arithmetic
    Add,           // Pop two, push sum
    Sub,           // Pop two, push difference
    Mul,           // Pop two, push product
    Div,           // Pop two, push quotient
    
    // Control flow
    Jump,          // Unconditional jump
    JumpIfFalse,   // Conditional jump
    Call,          // Function call
    Return,        // Return from function
    
    // Comparisons
    Equal,
    LessThan,
    GreaterThan,
}

#[derive(Debug)]
pub struct Instruction {
    pub opcode: Opcode,
    pub operand: i32,  // Immediate value, offset, or argument count
}

pub struct Function {
    pub name: String,
    pub bytecode: Vec<Instruction>,
    pub num_locals: usize,
    pub num_params: usize,
}
```

2. **Implement x86-64 assembler:**
```rust
use std::mem;

pub struct Assembler {
    code: Vec<u8>,
    labels: HashMap<String, usize>,
    relocations: Vec<Relocation>,
}

struct Relocation {
    offset: usize,
    label: String,
    kind: RelocKind,
}

enum RelocKind {
    Rel32,  // 32-bit PC-relative
    Abs64,  // 64-bit absolute
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096),
            labels: HashMap::new(),
            relocations: Vec::new(),
        }
    }
    
    // Basic instructions
    pub fn mov_r64_imm64(&mut self, dst: Reg, imm: i64) {
        // MOV rax, imm64 => REX.W + B8 + rd + imm64
        let rex = 0x48 | if dst.needs_rex() { 1 } else { 0 };
        self.emit_u8(rex);
        self.emit_u8(0xB8 + dst.id());
        self.emit_u64(imm as u64);
    }
    
    pub fn mov_r64_r64(&mut self, dst: Reg, src: Reg) {
        // MOV dst, src => REX.W + 89 /r
        let rex = 0x48 | (src.needs_rex() as u8) << 2 | (dst.needs_rex() as u8);
        self.emit_u8(rex);
        self.emit_u8(0x89);
        self.emit_modrm(0b11, src.id(), dst.id());
    }
    
    pub fn add_r64_r64(&mut self, dst: Reg, src: Reg) {
        // ADD dst, src => REX.W + 01 /r
        let rex = 0x48 | (src.needs_rex() as u8) << 2 | (dst.needs_rex() as u8);
        self.emit_u8(rex);
        self.emit_u8(0x01);
        self.emit_modrm(0b11, src.id(), dst.id());
    }
    
    pub fn sub_r64_r64(&mut self, dst: Reg, src: Reg) {
        let rex = 0x48 | (src.needs_rex() as u8) << 2 | (dst.needs_rex() as u8);
        self.emit_u8(rex);
        self.emit_u8(0x29);
        self.emit_modrm(0b11, src.id(), dst.id());
    }
    
    pub fn imul_r64_r64(&mut self, dst: Reg, src: Reg) {
        // IMUL dst, src => REX.W + 0F AF /r
        let rex = 0x48 | (dst.needs_rex() as u8) << 2 | (src.needs_rex() as u8);
        self.emit_u8(rex);
        self.emit_u8(0x0F);
        self.emit_u8(0xAF);
        self.emit_modrm(0b11, dst.id(), src.id());
    }
    
    pub fn push_r64(&mut self, reg: Reg) {
        // PUSH r64 => 50 + rd
        if reg.needs_rex() {
            self.emit_u8(0x41);
        }
        self.emit_u8(0x50 + reg.id());
    }
    
    pub fn pop_r64(&mut self, reg: Reg) {
        // POP r64 => 58 + rd
        if reg.needs_rex() {
            self.emit_u8(0x41);
        }
        self.emit_u8(0x58 + reg.id());
    }
    
    pub fn ret(&mut self) {
        self.emit_u8(0xC3);
    }
    
    pub fn call_r64(&mut self, reg: Reg) {
        // CALL r64 => FF /2
        let rex = 0x48 | (reg.needs_rex() as u8);
        self.emit_u8(rex);
        self.emit_u8(0xFF);
        self.emit_modrm(0b11, 2, reg.id());
    }
    
    // Label support
    pub fn label(&mut self, name: &str) {
        self.labels.insert(name.to_string(), self.code.len());
    }
    
    pub fn jmp_label(&mut self, label: &str) {
        // JMP rel32 => E9 + rel32
        self.emit_u8(0xE9);
        let offset = self.code.len();
        self.emit_u32(0); // Placeholder
        self.relocations.push(Relocation {
            offset,
            label: label.to_string(),
            kind: RelocKind::Rel32,
        });
    }
    
    // Finalize and make code executable
    pub fn finalize(mut self) -> ExecutableCode {
        // Resolve relocations
        for reloc in &self.relocations {
            let target = self.labels[&reloc.label];
            let source = reloc.offset;
            
            match reloc.kind {
                RelocKind::Rel32 => {
                    let rel = (target as isize - source as isize - 4) as i32;
                    self.write_i32_at(source, rel);
                }
                RelocKind::Abs64 => {
                    self.write_u64_at(source, target as u64);
                }
            }
        }
        
        ExecutableCode::new(self.code)
    }
    
    // Helper methods
    fn emit_u8(&mut self, byte: u8) {
        self.code.push(byte);
    }
    
    fn emit_u16(&mut self, val: u16) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }
    
    fn emit_u32(&mut self, val: u32) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }
    
    fn emit_u64(&mut self, val: u64) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }
    
    fn emit_modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        self.emit_u8((mode << 6) | ((reg & 7) << 3) | (rm & 7));
    }
    
    fn write_i32_at(&mut self, offset: usize, val: i32) {
        let bytes = val.to_le_bytes();
        self.code[offset..offset + 4].copy_from_slice(&bytes);
    }
    
    fn write_u64_at(&mut self, offset: usize, val: u64) {
        let bytes = val.to_le_bytes();
        self.code[offset..offset + 8].copy_from_slice(&bytes);
    }
}

// Register definitions
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reg(u8);

impl Reg {
    pub const RAX: Reg = Reg(0);
    pub const RCX: Reg = Reg(1);
    pub const RDX: Reg = Reg(2);
    pub const RBX: Reg = Reg(3);
    pub const RSP: Reg = Reg(4);
    pub const RBP: Reg = Reg(5);
    pub const RSI: Reg = Reg(6);
    pub const RDI: Reg = Reg(7);
    pub const R8: Reg = Reg(8);
    pub const R9: Reg = Reg(9);
    pub const R10: Reg = Reg(10);
    pub const R11: Reg = Reg(11);
    pub const R12: Reg = Reg(12);
    pub const R13: Reg = Reg(13);
    pub const R14: Reg = Reg(14);
    pub const R15: Reg = Reg(15);
    
    fn id(self) -> u8 {
        self.0 & 7
    }
    
    fn needs_rex(self) -> bool {
        self.0 >= 8
    }
}
```

3. **Make code executable:**
```rust
use std::ptr;

pub struct ExecutableCode {
    ptr: *mut u8,
    size: usize,
}

impl ExecutableCode {
    pub fn new(code: Vec<u8>) -> Self {
        use std::os::unix::io::AsRawFd;
        
        let size = code.len();
        let aligned_size = (size + 4095) & !4095; // Page align
        
        unsafe {
            // Allocate executable memory
            let ptr = libc::mmap(
                ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            ) as *mut u8;
            
            if ptr.is_null() {
                panic!("Failed to allocate executable memory");
            }
            
            // Copy code
            ptr::copy_nonoverlapping(code.as_ptr(), ptr, size);
            
            // Make executable
            if libc::mprotect(
                ptr as *mut libc::c_void,
                aligned_size,
                libc::PROT_READ | libc::PROT_EXEC,
            ) != 0 {
                panic!("Failed to make memory executable");
            }
            
            Self { ptr, size: aligned_size }
        }
    }
    
    pub unsafe fn call<R>(&self) -> R {
        let func: extern "C" fn() -> R = mem::transmute(self.ptr);
        func()
    }
    
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for ExecutableCode {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
        }
    }
}
```

4. **Basic JIT compiler:**
```rust
pub struct JitCompiler {
    // Function cache
    compiled_functions: HashMap<String, ExecutableCode>,
}

impl JitCompiler {
    pub fn new() -> Self {
        Self {
            compiled_functions: HashMap::new(),
        }
    }
    
    pub fn compile(&mut self, func: &Function) -> Result<(), CompileError> {
        let mut asm = Assembler::new();
        
        // Prologue: set up stack frame
        asm.push_r64(Reg::RBP);
        asm.mov_r64_r64(Reg::RBP, Reg::RSP);
        
        // Allocate locals on stack
        if func.num_locals > 0 {
            let stack_space = func.num_locals * 8;
            asm.sub_r64_imm(Reg::RSP, stack_space as i32);
        }
        
        // Compile each instruction
        for (pc, instr) in func.bytecode.iter().enumerate() {
            self.compile_instruction(&mut asm, instr, pc)?;
        }
        
        // Epilogue
        asm.mov_r64_r64(Reg::RSP, Reg::RBP);
        asm.pop_r64(Reg::RBP);
        asm.ret();
        
        // Finalize
        let code = asm.finalize();
        self.compiled_functions.insert(func.name.clone(), code);
        
        Ok(())
    }
    
    fn compile_instruction(
        &mut self,
        asm: &mut Assembler,
        instr: &Instruction,
        pc: usize,
    ) -> Result<(), CompileError> {
        match instr.opcode {
            Opcode::LoadConst => {
                // Load immediate value
                asm.mov_r64_imm64(Reg::RAX, instr.operand as i64);
                asm.push_r64(Reg::RAX);
            }
            
            Opcode::Add => {
                // Pop two values, add, push result
                asm.pop_r64(Reg::RAX);
                asm.pop_r64(Reg::RCX);
                asm.add_r64_r64(Reg::RAX, Reg::RCX);
                asm.push_r64(Reg::RAX);
            }
            
            Opcode::Sub => {
                asm.pop_r64(Reg::RCX);  // Second operand
                asm.pop_r64(Reg::RAX);  // First operand
                asm.sub_r64_r64(Reg::RAX, Reg::RCX);
                asm.push_r64(Reg::RAX);
            }
            
            Opcode::Mul => {
                asm.pop_r64(Reg::RAX);
                asm.pop_r64(Reg::RCX);
                asm.imul_r64_r64(Reg::RAX, Reg::RCX);
                asm.push_r64(Reg::RAX);
            }
            
            Opcode::Return => {
                asm.pop_r64(Reg::RAX);  // Return value
                // Handled by epilogue
            }
            
            _ => return Err(CompileError::UnsupportedOpcode(instr.opcode)),
        }
        
        Ok(())
    }
    
    pub unsafe fn execute<R>(&self, func_name: &str) -> Option<R> {
        self.compiled_functions
            .get(func_name)
            .map(|code| code.call())
    }
}
```

5. **Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_addition() {
        // Bytecode: return 5 + 3
        let func = Function {
            name: "add_test".to_string(),
            bytecode: vec![
                Instruction { opcode: Opcode::LoadConst, operand: 5 },
                Instruction { opcode: Opcode::LoadConst, operand: 3 },
                Instruction { opcode: Opcode::Add, operand: 0 },
                Instruction { opcode: Opcode::Return, operand: 0 },
            ],
            num_locals: 0,
            num_params: 0,
        };
        
        let mut jit = JitCompiler::new();
        jit.compile(&func).unwrap();
        
        let result: i64 = unsafe { jit.execute("add_test").unwrap() };
        assert_eq!(result, 8);
    }
    
    #[test]
    fn test_arithmetic_expression() {
        // Bytecode: return (10 + 5) * 3 - 2
        let func = Function {
            name: "expr_test".to_string(),
            bytecode: vec![
                Instruction { opcode: Opcode::LoadConst, operand: 10 },
                Instruction { opcode: Opcode::LoadConst, operand: 5 },
                Instruction { opcode: Opcode::Add, operand: 0 },
                Instruction { opcode: Opcode::LoadConst, operand: 3 },
                Instruction { opcode: Opcode::Mul, operand: 0 },
                Instruction { opcode: Opcode::LoadConst, operand: 2 },
                Instruction { opcode: Opcode::Sub, operand: 0 },
                Instruction { opcode: Opcode::Return, operand: 0 },
            ],
            num_locals: 0,
            num_params: 0,
        };
        
        let mut jit = JitCompiler::new();
        jit.compile(&func).unwrap();
        
        let result: i64 = unsafe { jit.execute("expr_test").unwrap() };
        assert_eq!(result, 43);  // (10 + 5) * 3 - 2 = 45 - 2 = 43
    }
    
    #[test]
    fn test_disassembly_validation() {
        // Compile simple function and verify generated code
        let func = Function {
            name: "simple".to_string(),
            bytecode: vec![
                Instruction { opcode: Opcode::LoadConst, operand: 42 },
                Instruction { opcode: Opcode::Return, operand: 0 },
            ],
            num_locals: 0,
            num_params: 0,
        };
        
        let mut jit = JitCompiler::new();
        jit.compile(&func).unwrap();
        
        // Use capstone to disassemble and verify correctness
        // ...
    }
}
```

**Deliverables:**
- Working x86-64 assembler in Rust
- Basic JIT compiler for stack-based bytecode
- Executable code generation with mprotect
- Tests showing correct code generation
- Disassembly validation

---

## TURN 2 — Register Allocation with Linear Scan

**Instructions:**

Replace stack-based evaluation with register allocation for performance.

**Background:** Stack-based code is slow (many memory accesses). Modern JITs keep values in registers. Implement linear scan register allocation.

**Implement:**
```rust
struct LiveInterval {
    vreg: VirtualReg,
    start: usize,
    end: usize,
    assigned_reg: Option<Reg>,
}

struct RegisterAllocator {
    intervals: Vec<LiveInterval>,
    free_regs: Vec<Reg>,
    spilled: Vec<VirtualReg>,
}

impl RegisterAllocator {
    fn allocate(&mut self) -> HashMap<VirtualReg, Location> {
        // Linear scan algorithm
        self.intervals.sort_by_key(|iv| iv.start);
        
        let mut active: Vec<LiveInterval> = Vec::new();
        let mut allocation = HashMap::new();
        
        for interval in &self.intervals {
            // Expire old intervals
            active.retain(|iv| {
                if iv.end < interval.start {
                    self.free_regs.push(iv.assigned_reg.unwrap());
                    false
                } else {
                    true
                }
            });
            
            // Allocate register
            if let Some(reg) = self.free_regs.pop() {
                interval.assigned_reg = Some(reg);
                active.push(interval.clone());
                allocation.insert(interval.vreg, Location::Reg(reg));
            } else {
                // Spill
                self.spill(interval);
                allocation.insert(interval.vreg, Location::Stack(self.spilled.len()));
            }
        }
        
        allocation
    }
}
```

**Tests: Compare performance stack-based vs register-allocated:**
```rust
#[bench]
fn bench_fibonacci_stack(b: &mut Bencher) {
    // Stack-based: ~500ns
    b.iter(|| {
        // fib(20) with stack operations
    });
}

#[bench]
fn bench_fibonacci_registers(b: &mut Bencher) {
    // Register-allocated: ~50ns (10x faster)
    b.iter(|| {
        // fib(20) with register allocation
    });
}
```

---

## TURN 3 — Force Failure: Incorrect Calling Convention

**Instructions:**

Introduce a bug where callee-saved registers are not preserved.

**Ask the AI:**
> "Your JIT doesn't save callee-saved registers (RBX, R12-R15) before using them. What happens when your JIT function calls a C function that expects these registers to be preserved? Show the exact crash with a test."

**Expected failure:**
- JIT uses RBX without saving
- JIT calls external C function
- C function uses and corrupts RBX
- JIT returns with wrong value in RBX
- Caller's RBX corrupted → crash or wrong results

**Test:**
```rust
extern "C" fn external_function() -> i64 {
    // This function expects RBX to be preserved
    42
}

#[test]
fn test_calling_convention_bug() {
    // Bytecode that uses RBX and calls external function
    let func = Function {
        bytecode: vec![
            LoadConst(100),  // Store in RBX (buggy)
            Call(external_function),
            Add, // Try to add RBX + RAX
            Return,
        ],
        // ...
    };
    
    let mut jit = JitCompiler::new();
    jit.compile(&func).unwrap();
    
    // This should return 142, but RBX is corrupted
    let result: i64 = unsafe { jit.execute("test").unwrap() };
    assert_eq!(result, 142);  // FAILS - RBX corrupted
}
```

**Fix:** Save/restore callee-saved registers in prologue/epilogue.

---

## TURN 4 — Inline Caching for Dynamic Method Dispatch

**Instructions:**

Add inline caching for dynamic method calls to avoid lookup overhead.

**Background:** Dynamic languages have expensive method lookups. Inline caching optimizes by caching the last lookup result.

**Implement:**
```rust
struct InlineCache {
    receiver_type: Option<TypeId>,
    cached_method: Option<*const u8>,
    miss_count: usize,
}

impl JitCompiler {
    fn compile_method_call(&mut self, asm: &mut Assembler, method_name: &str) {
        let ic = InlineCache::new();
        
        // Generate IC stub:
        // 1. Check if receiver type matches cached type
        // 2. If yes, jump directly to cached method (fast path)
        // 3. If no, call slow lookup and update cache
        
        asm.label("ic_check");
        asm.cmp_r64_imm(Reg::RAX, ic.receiver_type.unwrap_or(0) as i64);
        asm.jne_label("ic_miss");
        
        // Fast path - cached hit
        asm.mov_r64_imm64(Reg::RCX, ic.cached_method.unwrap_or(ptr::null()) as i64);
        asm.call_r64(Reg::RCX);
        asm.jmp_label("ic_done");
        
        // Slow path - cache miss
        asm.label("ic_miss");
        self.compile_slow_lookup(asm, method_name);
        
        asm.label("ic_done");
    }
}
```

**Benchmark:**
```rust
// Without IC: 100ns per method call
// With IC: 5ns per method call (20x faster)
```

---

## TURN 5 — SSA Form and Optimization Passes

**Instructions:**

Convert bytecode to SSA (Static Single Assignment) form and implement optimization passes.

**SSA transformations:**
- Constant propagation
- Dead code elimination
- Common subexpression elimination
- Loop-invariant code motion

**Implement:**
```rust
struct SsaBuilder {
    instructions: Vec<SsaInst>,
    phi_nodes: HashMap<BlockId, Vec<PhiNode>>,
}

enum SsaInst {
    BinOp { dst: SsaReg, op: BinOp, lhs: SsaReg, rhs: SsaReg },
    Load { dst: SsaReg, src: Location },
    Store { dst: Location, src: SsaReg },
    Phi { dst: SsaReg, srcs: Vec<(BlockId, SsaReg)> },
}

struct Optimizer {
    ssa: SsaProgram,
}

impl Optimizer {
    fn constant_propagation(&mut self) {
        // Replace uses of constants with immediate values
        for inst in &mut self.ssa.instructions {
            if let SsaInst::BinOp { lhs, rhs, .. } = inst {
                if self.is_constant(*lhs) && self.is_constant(*rhs) {
                    // Fold at compile time
                    *inst = SsaInst::Load {
                        dst: inst.dst(),
                        src: Location::Immediate(self.eval_constant(inst)),
                    };
                }
            }
        }
    }
    
    fn dead_code_elimination(&mut self) {
        // Remove instructions whose results are never used
        let mut used = HashSet::new();
        
        // Mark all used values (backwards pass)
        for inst in self.ssa.instructions.iter().rev() {
            if inst.has_side_effects() || used.contains(&inst.dst()) {
                inst.mark_operands_used(&mut used);
            }
        }
        
        // Remove unused instructions
        self.ssa.instructions.retain(|inst| used.contains(&inst.dst()));
    }
}
```

**Tests:**
```rust
#[test]
fn test_constant_folding() {
    // Input: x = 2 + 3; y = x * 5
    // After optimization: x = 5; y = 25
    
    let optimized = optimize(bytecode);
    assert_eq!(count_runtime_ops(optimized), 0); // All folded to constants
}
```

---

## TURN 6 — Speculative Optimization with Deoptimization

**Instructions:**

Implement speculative optimizations that assume type stability, with deoptimization guards.

**Example:** Assume all numbers are integers, generate fast integer code, insert guard checks.

**Implement:**
```rust
struct Guard {
    check_type: TypeCheck,
    deopt_target: Label,
}

impl JitCompiler {
    fn compile_with_speculation(&mut self, asm: &mut Assembler) {
        // Speculate: both operands are integers
        asm.label("speculative_add");
        
        // Guard: check if RAX is tagged integer
        asm.test_r64_imm(Reg::RAX, 0x1); // Check LSB tag
        asm.jz_label("deopt_add"); // Not integer, deoptimize
        
        // Fast path: integer add
        asm.add_r64_r64(Reg::RAX, Reg::RCX);
        asm.jmp_label("add_done");
        
        // Deoptimization path
        asm.label("deopt_add");
        self.compile_deoptimize(asm);
        
        asm.label("add_done");
    }
    
    fn compile_deoptimize(&mut self, asm: &mut Assembler) {
        // Save state
        // Jump to interpreter
        // Re-compile without speculation
    }
}
```

**Benchmark: 100% integer workload:**
- With speculation: 2ns per add
- Without speculation: 10ns per add (5x slower)

---

## TURN 7 — Garbage Collector Integration

**Instructions:**

Integrate with a garbage collector: emit GC safepoints and stack maps.

**Requirements:**
- Emit safepoint checks at loop backedges and function calls
- Generate stack maps showing where live pointers are
- Support GC moving objects (update pointers)

**Implement:**
```rust
struct StackMap {
    pc_offset: usize,
    live_ptrs: Vec<Location>, // Where are live GC pointers?
}

impl JitCompiler {
    fn emit_safepoint(&mut self, asm: &mut Assembler, stack_map: StackMap) {
        // Check if GC requested
        asm.mov_r64_imm64(Reg::RAX, &GC_REQUESTED as *const _ as i64);
        asm.cmp_byte_ptr_imm(Reg::RAX, 1);
        asm.jne_label("no_gc");
        
        // Save live pointers
        for (i, loc) in stack_map.live_ptrs.iter().enumerate() {
            self.save_ptr(asm, *loc, i);
        }
        
        // Call GC
        asm.call_extern(gc_safepoint as *const u8);
        
        // Reload pointers (may have moved)
        for (i, loc) in stack_map.live_ptrs.iter().enumerate() {
            self.reload_ptr(asm, *loc, i);
        }
        
        asm.label("no_gc");
        
        // Register stack map for this PC
        self.stack_maps.insert(asm.current_offset(), stack_map);
    }
}
```

---

## TURN 8 — Force Failure: Stack Misalignment Crash

**Ask the AI:**
> "Your JIT doesn't maintain 16-byte stack alignment before CALL instructions as required by System V ABI. Show a test where calling a function with SSE instructions crashes due to misaligned stack."

**Expected failure:**
```rust
extern "C" fn sse_function(x: f64) -> f64 {
    // Uses MOVAPS which requires 16-byte aligned stack
    x * 2.0
}

#[test]
fn test_stack_alignment_crash() {
    let func = Function {
        bytecode: vec![
            LoadConst(3.14),
            Call(sse_function),  // Stack not aligned, crash!
            Return,
        ],
        // ...
    };
    
    jit.compile(&func).unwrap();
    let result: f64 = unsafe { jit.execute("test").unwrap() };
    // CRASH: MOVAPS from unaligned address
}
```

**Fix:** Ensure RSP % 16 == 0 before CALL.

---

## TURN 9 — Tiered Compilation

**Instructions:**

Implement tiered compilation: interpreter → baseline JIT → optimizing JIT.

**Tiers:**
1. **Interpreter:** Bytecode execution, collect profiling data
2. **Baseline JIT:** Fast compilation, no optimizations
3. **Optimizing JIT:** Slow compilation, aggressive optimizations

**Implement:**
```rust
struct TieredCompiler {
    interpreter: Interpreter,
    baseline_jit: BaselineJit,
    optimizing_jit: OptimizingJit,
    
    hotness_counters: HashMap<FunctionId, usize>,
}

impl TieredCompiler {
    fn execute(&mut self, func_id: FunctionId) -> Value {
        let counter = self.hotness_counters.entry(func_id).or_insert(0);
        *counter += 1;
        
        match *counter {
            0..=10 => {
                // Tier 0: Interpreter
                self.interpreter.execute(func_id)
            }
            11..=100 => {
                // Tier 1: Baseline JIT (fast compile)
                if !self.baseline_jit.is_compiled(func_id) {
                    self.baseline_jit.compile(func_id);
                }
                self.baseline_jit.execute(func_id)
            }
            _ => {
                // Tier 2: Optimizing JIT (slow compile, fast execution)
                if !self.optimizing_jit.is_compiled(func_id) {
                    self.optimizing_jit.compile_with_profile(func_id, self.get_profile(func_id));
                }
                self.optimizing_jit.execute(func_id)
            }
        }
    }
}
```

**Benchmark results:**
- Cold start (1 call): Interpreter 100ns, Baseline 50ns (but 1ms compile)
- Warm (100 calls): Baseline 10ns avg, Optimizing 2ns avg (but 10ms compile)

---

## TURN 10 — SIMD Vectorization

**Instructions:**

Auto-vectorize loops using SIMD instructions (SSE/AVX).

**Example: Vectorize array addition:**
```rust
// Scalar: for (i = 0; i < n; i++) c[i] = a[i] + b[i]
// Vectorized: process 4 elements at once with SIMD

impl JitCompiler {
    fn vectorize_loop(&mut self, asm: &mut Assembler, loop_body: &[SsaInst]) {
        if !self.can_vectorize(loop_body) {
            return self.compile_scalar(asm, loop_body);
        }
        
        // Vector loop (4 elements per iteration)
        asm.label("vector_loop");
        asm.movaps_xmm_ptr(XMM0, Reg::RSI); // Load a[i..i+4]
        asm.movaps_xmm_ptr(XMM1, Reg::RDI); // Load b[i..i+4]
        asm.addps_xmm_xmm(XMM0, XMM1);     // Add vectors
        asm.movaps_ptr_xmm(Reg::RDX, XMM0); // Store c[i..i+4]
        
        asm.add_r64_imm(Reg::RSI, 16);      // i += 4
        asm.cmp_r64_r64(Reg::RSI, Reg::RCX);
        asm.jl_label("vector_loop");
        
        // Scalar cleanup for remaining elements
        asm.label("scalar_cleanup");
        // ...
    }
}
```

**Benchmark:**
- Scalar loop: 100ns for 1000 elements
- SSE vectorized: 25ns (4x faster)
- AVX vectorized: 12ns (8x faster)

---

## TURN 11 — Final Integration: Complete Dynamic Language

**Instructions:**

Build a complete dynamic language with JIT compilation.

**Features:**
- Dynamic typing with tagging
- Objects and method dispatch
- Closures and upvalues
- Garbage collection
- Exception handling
- Full JIT pipeline

**Example language:**
```javascript
// Fibonacci in our dynamic language
function fib(n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

print(fib(30));
```

**Implementation:**
```rust
struct DynamicLanguage {
    parser: Parser,
    compiler: BytecodeCompiler,
    jit: TieredCompiler,
    gc: GarbageCollector,
}

impl DynamicLanguage {
    fn execute(&mut self, source: &str) -> Result<Value, Error> {
        // Parse
        let ast = self.parser.parse(source)?;
        
        // Compile to bytecode
        let bytecode = self.compiler.compile(&ast)?;
        
        // Execute (interpreter → JIT as needed)
        let result = self.jit.execute(bytecode)?;
        
        Ok(result)
    }
}
```

**Final benchmarks vs other implementations:**
| Benchmark | Our JIT | LuaJIT | V8 (TurboFan) | CPython |
|-----------|---------|--------|---------------|---------|
| fib(30) | 15ms | 12ms | 10ms | 500ms |
| array_sum(1M) | 2ms | 1.5ms | 1ms | 50ms |
| method_calls(100k) | 3ms | 2ms | 2ms | 150ms |

**Deliverables:**
- Complete JIT compiler with all optimization passes
- Tiered compilation system
- GC integration with stack maps
- SIMD vectorization
- Full dynamic language implementation
- Comprehensive benchmark suite
