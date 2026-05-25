# core-diff
A lightweight, dependency‑free forward‑mode automatic differentiation library in Rust,
built on dual numbers with full operator overloading.

---

## Overview

`core-diff` implements forward‑mode autodiff using dual numbers:

$$ a + b\epsilon \text{ with } \epsilon^2=0 $$

Each dual number carries:
- **value** — the numeric value
- **gradient** — the derivative w.r.t. the active variable

Arithmetic and math operations propagate derivatives automatically 
using calculus rules (product rule, chain rule, etc.).

This allows computing derivatives alongside function evaluation with zero 
symbolic manipulation and zero numerical approximation.

## Features
- Generic `Dual<T>` type
- Value + derivative propogation
- Full operator overloading (`+`, `-`, `*`, `/`, and unary `-`)
- Correct derivative rules (product, quotient, chain)
- Zero dependencies
- Clean, modular architecture
- Extensible for gradients, Jacobians, and optimization

---

## Example

```rust
use core_diff::dual::Dual;

fn main() {
    // Define x as the active variable
    let x = Dual::new(3.0, 1.0);

    // Compute f(x) = x^2
    let y = x * x;

    assert_eq!(y.value, 9.0); // f(x)
    assert_eq!(y.derivative, 6.0); // f'(x) = 2x
}
```
