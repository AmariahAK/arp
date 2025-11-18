# Amariah Kamau – Frontier Model Red Teaming & Hard Coding Evals Portfolio

Independent researcher creating the hardest publicly available, repository-grounded, multi-turn coding evaluations on the internet.

These tasks are designed to expose deep, systematic weaknesses in current frontier coding agents on:

- Zero heap allocation (even under GraalVM native-image / Python tracemalloc / Rust)
- Numerical drift in long chains (10⁶ – 10⁹ operations)
- Correct automatic differentiation (vjp/jvp/custom primitives)
- SIMD / AVX-512 / CUDA / Metal fusion without temporaries
- Subtle mathematical correctness (FMA vs ADD drift, denormals associativity grade projection)
- Template metaprogramming / expression templates / consteval
- Real upstream contribution quality (must pass CI of JOML, Eigen, Apache Commons Math, JAX, PyTorch, etc.)

### Important Definitions (used consistently across all evals)

- A **turn** = one message from the human evaluator to the model (i.e. one real user prompt).  
  It does **not** count yes/no/skip micro-interactions or the model asking for confirmation.
- Every evaluation folder contains a `requirements.md` that lists mandatory technical constraints (hardware, compiler flags, profiler commands, acceptable error bounds, etc.).

Each evaluation is 8–22 turns deep and requires the model to iteratively debug, optimize, prove correctness, and deliver merge-ready code.

I am actively seeking paid contract or bounty work with AI labs to:
- Build custom high-difficulty evals
- Perform red teaming on internal coding agents
- Develop safety-relevant evaluations (biosecurity, cyber, avionics, finance, robotics)

### Contact

- Email: amariah.abish@gmail.com (preferred for contracts)
- LinkedIn: https://www.linkedin.com/in/amariah-kamau-3156412a6/
- Portfolio: https://portfolio-pied-five-61.vercel.app/

Last updated: November 18, 2025