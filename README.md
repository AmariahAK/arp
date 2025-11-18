# Amariah Kamau – Frontier Model Red Teaming & Hard-Coding Evals Portfolio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Independent researcher creating the hardest publicly available, repository-grounded, multi-turn coding evaluations on the internet.

These tasks are designed to expose deep, systematic weaknesses in current frontier coding agents on:

- Zero heap allocation (even under GraalVM native-image / Python tracemalloc / Rust)
- Numerical drift in long chains (10⁶ – 10⁹ operations)
- Correct automatic differentiation (vjp/jvp/custom primitives)
- SIMD / AVX-512 / CUDA / Metal fusion without temporaries
- Subtle mathematical correctness (FMA vs ADD drift, denormals, associativity grade projection)
- Template metaprogramming / expression templates / `consteval`
- Real upstream contribution quality (must pass CI of JOML, Eigen, Apache Commons Math, JAX, PyTorch, etc.)

---

## 📁 Repository Structure (Used Across All Evals)

Each evaluation folder strictly follows this format:

/eval-name/
requirements.md ← Technical constraints: hardware, compilers, flags, profilers,
memory/time caps, numeric tolerances, CI requirements.
task.md ← Full multi-turn evaluation prompt.
expected_result.md ← Ground-truth invariants, acceptance tests, proofs, performance
ceilings, and red-team traps.

yaml
Copy code

This structure makes each eval:

- deterministic  
- pipeline-ready  
- reproducible  
- suitable for automated scoring and internal lab eval harnesses  

---

## 📌 Evaluation Depth

Every evaluation is **8–22 turns** and forces models to:

- iteratively debug  
- derive correct algorithms  
- optimize under strict constraints  
- verify proofs or numerical stability  
- output merge-ready, CI-passing code  

These are **not** “toy tasks.”  
They’re designed to fail any model relying on shallow heuristics or pattern-matching.

---

## 🔍 Seeking Contract / Bounty Work

Actively looking for paid work with AI labs to:

- Build ultra-hard custom evals  
- Red-team internal coding agents  
- Design safety-relevant evaluations (cyber, finance, avionics, robotics, bio-risk)  

---

## 📬 Contact

- **Email:** amariah.abish@gmail.com  
- **LinkedIn:** https://www.linkedin.com/in/amariah-kamau-3156412a6/  
- **Portfolio:** https://portfolio-pied-five-61.vercel.app/  

_Last updated: November 18, 2025_