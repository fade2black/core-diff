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
