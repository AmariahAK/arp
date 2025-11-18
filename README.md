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

Each evaluation is 8–22 turns deep and requires the model to iteratively debug, optimize, prove correctness, and deliver merge-ready code.

I am actively seeking paid contract or bounty work with AI labs to:
- Build custom high-difficulty evals
- Perform red teaming on internal coding agents
- Develop safety-relevant evaluations (biosecurity, cyber, avionics, finance, robotics)

### Current Public Evaluations

| # | Domain                          | Language / Framework        | Turns | Typical Failure Turn |
|---|----------------------------------|-----------------------------|-------|----------------------|
| 01 | JOML – CGA Motors + AVX-512     | Java 21 + Panama Vector API | 18    | 6–9                 |
| 02 | Apache Commons Math – Elliptic Integrals | Java + Vector API | 16    | 7–10                |
| 03 | JAX – Full CGA Custom Primitives| JAX + custom vjp/pmap       | 17    | 4–8                 |
| 04 | Eigen – Dual Quaternion Skinning| C++20 + AVX-512             | 19    | 8–11                |
| 05 | Rust nalgebra – Projective GA   | Rust 1.82 + const generics  | 15    | 5–9                 |
| 06 | PyTorch – Geometric Fabric      | Python + TorchScript        | 20    | 9–13                |

More coming weekly (Rust GPU, Zig, CUDA, Carbon, Mojo, etc.)

### Contact

- Email: amariah.abish@gmail.com (preferred for contracts)
- LinkedIn: https://www.linkedin.com/in/amariah-kamau-3156412a6/
- Portfolio: https://portfolio-pied-five-61.vercel.app/

Last updated: November 18, 2025