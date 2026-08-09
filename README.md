# core-diff
A lightweight, dependency‑free forward‑mode automatic differentiation library in Rust,
built on dual numbers with full operator overloading.

---

## Overview

`core-diff` implements forward‑mode autodiff using dual numbers:

$$ a + b\epsilon \text{ with } \epsilon^2=0 $$

Each dual number carries:
- **value** — the numeric value
- **gradient** — the partial derivatives w.r.t. each tracked variable, stored as a
  fixed-size `[T; N]` array (stack-allocated, no heap)

Arithmetic and math operations propagate derivatives automatically 
using calculus rules (product rule, chain rule, etc.).

This allows computing derivatives alongside function evaluation with zero 
symbolic manipulation and zero numerical approximation.

## Features
- Generic `Dual<T, const N: usize>` type — `N` partial derivatives per value,
  stack-allocated (`[T; N]`), no heap
- Multivariable gradients: seed each input as a unit vector via `Dual::var(value, index)`
- Value + gradient propagation
- Full operator overloading (`+`, `-`, `*`, `/`, and unary `-`)
- Correct derivative rules (product, quotient, chain)
- Jacobian computation for fixed-size residual blocks (`core_diff::jacobian::jacobian`),
  backed by `nalgebra`'s const-generic `SMatrix`/`SVector` (still stack-allocated, no heap)
- Clean, modular architecture
- Extensible toward Jacobians and optimization (see Roadmap)

---

## Example

```rust
use core_diff::dual::Dual;

fn main() {
    // Single-variable: f(x) = x^2 at x = 3
    let x: Dual<f64, 1> = Dual::var(3.0, 0);
    let y = x * x;

    assert_eq!(y.value(), 9.0); // f(x)
    assert_eq!(y.grad(), [6.0]); // f'(x) = 2x

    // Multivariable: f(x, y) = x*y + sin(x) at (x, y) = (2, 3)
    let x: Dual<f64, 2> = Dual::var(2.0, 0);
    let y: Dual<f64, 2> = Dual::var(3.0, 1);
    let f = x * y + x.sin();

    println!("f(x, y)  = {}", f.value());
    println!("gradient = {:?}", f.grad()); // [df/dx, df/dy]
}
```


## Roadmap

- Stage 1 — Dual number struct ✅ Done
- Stage 2 — Operator overloading ✅ Done
- Stage 3 — Math functions (sin, cos, exp, log, powf) ✅ Done
- Stage 4 — Gradient computation ✅ Done
- Stage 5 — Jacobian computation ✅ Done
- Stage 6 — CostFunction trait (Ceres‑style API)
- Stage 7 — Optional nonlinear solver
