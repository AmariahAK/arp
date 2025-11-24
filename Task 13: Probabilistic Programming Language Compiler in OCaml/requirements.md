# Requirements: Probabilistic Programming Language Compiler in OCaml

## Task Overview

Build a production-grade compiler for a probabilistic programming language with automatic inference, supporting both exact and approximate inference algorithms. The compiler must handle probabilistic models, perform type inference, optimize probabilistic computations, and generate efficient code.

## Prerequisites

### Required Knowledge
- **Expert-level OCaml programming** (functors, GADTs, modules, type system)
- **Programming language theory** (type systems, lambda calculus, operational semantics)
- **Compiler construction** (parsing, type checking, IR, code generation)
- **Probabilistic inference** (MCMC, variational inference, importance sampling)
- **Probability theory** (distributions, Bayes' theorem, graphical models)
- **Functional programming** (monads, algebraic effects, continuations)

### Software Dependencies
- **OCaml 5.1+** (with effects support)
- **Menhir** (parser generator)
- **ppx_deriving** (code generation)
- **OUnit2** (testing framework)
- **Benchmark** (performance testing)
- **Owl** (scientific computing, for numerical operations)
- **Zarith** (arbitrary precision arithmetic)

**Note:** No existing probabilistic programming frameworks (Pyro, Stan, Edward) can be used directly. Must implement from scratch.

## Performance Requirements

### Inference Performance
- **MCMC sampling:** >1000 samples/second for simple models
- **Variational inference:** Converge in <100 iterations for standard models
- **Exact inference:** Solve models with <20 discrete variables in <1 second
- **Gradient computation:** Automatic differentiation with <10% overhead

### Compilation Performance
- **Type checking:** <100ms for 1000-line programs
- **Optimization:** <500ms for aggressive optimization passes
- **Code generation:** <200ms for typical programs

### Quality Requirements
- **Type safety:** 100% - no runtime type errors
- **Inference correctness:** Match reference implementations within 1% KL divergence
- **Numerical stability:** Handle log-probabilities without underflow
- **Convergence:** Detect and report non-convergent inference

## Technical Constraints

### Language Features
- **Type system:** Hindley-Milner with probabilistic types
- **Distributions:** Normal, Bernoulli, Categorical, Dirichlet, Beta, Gamma, etc.
- **Inference:** `observe` for conditioning, `sample` for random variables
- **Control flow:** if/else, recursion, higher-order functions
- **Data structures:** tuples, records, lists, arrays

### Type System
```ocaml
(* Probabilistic types *)
type 'a dist  (* Distribution over 'a *)
type 'a prob  (* Probabilistic computation returning 'a *)

(* Type inference must handle: *)
val sample : 'a dist -> 'a prob
val observe : 'a dist -> 'a -> unit prob
val return : 'a -> 'a prob
val bind : 'a prob -> ('a -> 'b prob) -> 'b prob
```

### Inference Algorithms
- **Exact:** Enumeration, variable elimination
- **MCMC:** Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS
- **Variational:** Mean-field, ADVI (Automatic Differentiation Variational Inference)
- **Importance sampling:** Sequential Monte Carlo

## Code Quality Standards

### Production Requirements
- **Type safety:** Leverage OCaml's type system fully
- **Memory safety:** No unsafe operations, proper resource cleanup
- **Error handling:** Informative error messages with source locations
- **Determinism:** Reproducible results with fixed random seeds
- **Modularity:** Clean separation of concerns (parser, type checker, inference, codegen)

### Testing
- **Unit tests:** >150 tests covering all components
- **Property-based tests:** QuickCheck-style for type checker, inference
- **Integration tests:** End-to-end compilation and inference
- **Benchmark suite:** Standard probabilistic models
- **Coverage:** >85% code coverage

### Documentation
- **Language specification:** Formal grammar and semantics
- **API documentation:** OCamldoc for all modules
- **User guide:** Tutorial and examples
- **Inference guide:** When to use which algorithm

## Deliverables

### Core Compiler
1. **Lexer and parser** (Menhir-based)
2. **Type checker** (bidirectional type checking)
3. **Intermediate representation** (ANF or CPS)
4. **Optimization passes** (constant folding, dead code elimination)
5. **Code generator** (to OCaml or bytecode)

### Inference Engine
1. **Exact inference** (enumeration, variable elimination)
2. **MCMC samplers** (MH, HMC, NUTS)
3. **Variational inference** (mean-field, ADVI)
4. **Automatic differentiation** (forward and reverse mode)

### Standard Library
1. **Distributions** (10+ common distributions)
2. **Utility functions** (log-sum-exp, softmax, etc.)
3. **Model combinators** (mixture models, hierarchical models)

## Success Criteria

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Type checking speed | <100ms/1000 LOC | <50ms/1000 LOC |
| MCMC sampling rate | >1000 samples/s | >5000 samples/s |
| Inference accuracy | Within 1% KL | Within 0.1% KL |
| Compilation success | >95% valid programs | 100% valid programs |
| Test coverage | >85% | >90% |
| Benchmark performance | Match Stan | Beat Stan |
| Documentation | Complete | Extensive examples |

## Estimated Difficulty

**Time estimate:** 55-70 hours for expert OCaml/PL engineer

**Difficulty level:** EXTREME

**Why it's hard:**
- Requires expertise in 3 distinct domains (PL theory, compilers, probabilistic inference)
- Type system for probabilistic computations is non-trivial
- Inference algorithms are mathematically complex
- Automatic differentiation requires sophisticated program transformation
- Numerical stability is challenging (log-space computations)
- Performance optimization of probabilistic code is difficult
- Debugging probabilistic programs is inherently hard
