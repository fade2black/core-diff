use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T, const N: usize> {
    value: T,
    grad: [T; N],
}

impl<T, const N: usize> Dual<T, N>
where
    T: Copy,
{
    /// Generic constructor
    pub fn new(value: T, grad: [T; N]) -> Self {
        Self { value, grad }
    }

    /// Returns the value of the dual number.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the derivative of the dual number.
    pub fn grad(&self) -> [T; N] {
        self.grad
    }
}

impl<T, const N: usize> Add for Dual<T, N>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            grad: std::array::from_fn(|i| self.grad[i] + rhs.grad[i]),
        }
    }
}

impl<T, const N: usize> Sub for Dual<T, N>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            grad: std::array::from_fn(|i| self.grad[i] - rhs.grad[i]),
        }
    }
}

impl<T, const N: usize> Mul for Dual<T, N>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            grad: std::array::from_fn(|i| self.grad[i] * rhs.value + self.value * rhs.grad[i]),
        }
    }
}

impl<T, const N: usize> Div for Dual<T, N>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value,
            grad: std::array::from_fn(|i| {
                (self.grad[i] * rhs.value - self.value * rhs.grad[i]) / (rhs.value * rhs.value)
            }),
        }
    }
}

impl<T, const N: usize> Neg for Dual<T, N>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            grad: std::array::from_fn(|i| -self.grad[i]),
        }
    }
}

macro_rules! impl_scalar_ops {
    ($($trait_name:ident, $method:ident, $op:tt);* $(;)?) => {
        $(
            impl<const N: usize> $trait_name<f64> for Dual<f64, N> {
                type Output = Self;

                fn $method(self, rhs: f64) -> Self::Output {
                    self $op Self::constant(rhs)
                }
            }

            impl<const N: usize> $trait_name<Dual<f64, N>> for f64 {
                type Output = Dual<f64, N>;

                fn $method(self, rhs: Dual<f64, N>) -> Self::Output {
                    Dual::constant(self) $op rhs
                }
            }
        )*
    };
}

impl_scalar_ops!(
    Add, add, +;
    Sub, sub, -;
    Mul, mul, *;
    Div, div, /;
);

impl<T: Float, const N: usize> Dual<T, N> {
    /// A constant: grad = 0
    pub fn constant(value: T) -> Self {
        Self {
            value,
            grad: [T::zero(); N],
        }
    }

    /// A variable: partial derivative w.r.t. itself is 1 at `index`, 0 elsewhere.
    pub fn var(value: T, index: usize) -> Self {
        let mut grad = [T::zero(); N];
        grad[index] = T::one();
        Self { value, grad }
    }

    pub fn sin(self) -> Self {
        let cos_v = self.value.cos();
        Self {
            value: self.value.sin(),
            grad: std::array::from_fn(|i| self.grad[i] * cos_v),
        }
    }

    pub fn cos(self) -> Self {
        let sin_v = self.value.sin();
        Self {
            value: self.value.cos(),
            grad: std::array::from_fn(|i| self.grad[i] * -sin_v),
        }
    }

    pub fn exp(self) -> Self {
        let v = self.value.exp();
        Self {
            value: v,
            grad: std::array::from_fn(|i| self.grad[i] * v),
        }
    }

    pub fn ln(self) -> Self {
        let inv_v = T::one() / self.value;
        Self {
            value: self.value.ln(),
            grad: std::array::from_fn(|i| self.grad[i] * inv_v),
        }
    }

    pub fn powf(self, n: T) -> Self {
        let deriv = n * self.value.powf(n - T::one());
        Self {
            value: self.value.powf(n),
            grad: std::array::from_fn(|i| self.grad[i] * deriv),
        }
    }

    pub fn sqrt(self) -> Self {
        let v = self.value.sqrt();
        let deriv = T::one() / (v + v);
        Self {
            value: v,
            grad: std::array::from_fn(|i| self.grad[i] * deriv),
        }
    }

    pub fn tan(self) -> Self {
        let cos_v = self.value.cos();
        let deriv = T::one() / (cos_v * cos_v);
        Self {
            value: self.value.tan(),
            grad: std::array::from_fn(|i| self.grad[i] * deriv),
        }
    }

    pub fn asin(self) -> Self {
        let deriv = T::one() / (T::one() - self.value * self.value).sqrt();
        Self {
            value: self.value.asin(),
            grad: std::array::from_fn(|i| self.grad[i] * deriv),
        }
    }

    pub fn acos(self) -> Self {
        let deriv = -T::one() / (T::one() - self.value * self.value).sqrt();
        Self {
            value: self.value.acos(),
            grad: std::array::from_fn(|i| self.grad[i] * deriv),
        }
    }

    pub fn atan2(self, other: Self) -> Self {
        let denom = self.value * self.value + other.value * other.value;
        Self {
            value: self.value.atan2(other.value),
            grad: std::array::from_fn(|i| {
                (other.value * self.grad[i] - self.value * other.grad[i]) / denom
            }),
        }
    }
}

/// Free-function form of `Dual::sin`, so expressions like `sin(x * y)` resolve
/// without needing a macro to rewrite them into method calls.
#[inline]
pub fn sin<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.sin()
}

#[inline]
pub fn cos<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.cos()
}

#[inline]
pub fn exp<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.exp()
}

#[inline]
pub fn ln<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.ln()
}

#[inline]
pub fn powf<T: Float, const N: usize>(d: Dual<T, N>, n: T) -> Dual<T, N> {
    d.powf(n)
}

#[inline]
pub fn sqrt<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.sqrt()
}

#[inline]
pub fn tan<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.tan()
}

#[inline]
pub fn asin<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.asin()
}

#[inline]
pub fn acos<T: Float, const N: usize>(d: Dual<T, N>) -> Dual<T, N> {
    d.acos()
}

#[inline]
pub fn atan2<T: Float, const N: usize>(y: Dual<T, N>, x: Dual<T, N>) -> Dual<T, N> {
    y.atan2(x)
}

#[cfg(test)]
mod tests {
    use super::Dual;

    #[test]
    fn dual_new_stores_value_and_gradient() {
        let d = Dual::new(3.0, [0.5]);
        assert_eq!(d.value, 3.0);
        assert_eq!(d.grad, [0.5]);
    }

    #[test]
    fn dual_constant_is_zero_gradient() {
        let c: Dual<f64, 1> = Dual::constant(5.0);
        assert_eq!(c.value, 5.0);
        assert_eq!(c.grad, [0.0]);
    }

    #[test]
    fn dual_variable_is_one_derivative() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);
        assert_eq!(x.value, 2.0);
        assert_eq!(x.grad, [1.0]);
    }

    #[test]
    fn add_works() {
        let a = Dual::new(2.0, [1.0]);
        let b = Dual::new(3.0, [0.0]);
        let c = a + b;
        assert_eq!(c.value, 5.0);
        assert_eq!(c.grad, [1.0]);
    }

    #[test]
    fn mul_works() {
        let x = Dual::new(3.0, [1.0]);
        let y = x * x; // derivative = 2*x = 6
        assert_eq!(y.value, 9.0);
        assert_eq!(y.grad, [6.0]);
    }

    #[test]
    fn div_works() {
        let x = Dual::new(4.0, [1.0]);
        let y = Dual::new(2.0, [0.0]);
        let z = x / y; // derivative = (1*2 - 4*0) / 4 = 0.5
        assert_eq!(z.value, 2.0);
        assert_eq!(z.grad, [0.5]);
    }

    #[test]
    fn neg_works() {
        let x = Dual::new(4.0, [1.0]);
        let y = -x;
        assert_eq!(y.value, -4.0);
        assert_eq!(y.grad, [-1.0]);
    }

    #[test]
    fn test_sin() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);
        let y = x.sin();
        assert!((y.value() - 2.0_f64.sin()).abs() < 1e-12);
        assert!((y.grad()[0] - 2.0_f64.cos()).abs() < 1e-12);
    }

    #[test]
    fn test_cos() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);
        let y = x.cos();
        assert!((y.value() - 2.0_f64.cos()).abs() < 1e-12);
        assert!((y.grad()[0] + 2.0_f64.sin()).abs() < 1e-12); // derivative = -sin(x)
    }

    #[test]
    fn test_exp() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);
        let y = x.exp();
        assert!((y.value() - 2.0_f64.exp()).abs() < 1e-12);
        assert!((y.grad()[0] - 2.0_f64.exp()).abs() < 1e-12);
    }

    #[test]
    fn test_ln() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);
        let y = x.ln();
        assert!((y.value() - 2.0_f64.ln()).abs() < 1e-12);
        assert!((y.grad()[0] - 0.5).abs() < 1e-12); // derivative = 1/x
    }

    #[test]
    fn test_powf() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);
        let y = x.powf(3.0);
        assert_eq!(y.value(), 8.0);
        assert_eq!(y.grad()[0], 3.0 * 2.0_f64.powf(2.0)); // 3 * x^2
    }

    #[test]
    fn test_polynomial() {
        // f(x) = 1 + x + 2x^2 at x=1
        let x: Dual<f64, 1> = Dual::var(1.0, 0);
        let y = Dual::constant(1.0) + x + Dual::constant(2.0) * x * x;
        assert_eq!(y.value(), 4.0);
        assert_eq!(y.grad(), [5.0]);
    }

    #[test]
    fn test_three_pow_x_plus_cos() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);
        let ln3 = Dual::constant(3.0_f64.ln());
        let three_pow_x = (x * ln3).exp();
        let y = three_pow_x + x.cos();

        assert!((y.value() - 8.583853163452858).abs() < 1e-12);
        assert!((y.grad()[0] - 8.978213171187306).abs() < 1e-12);
    }

    #[test]
    fn multivariable_gradient_of_x_times_y() {
        // f(x, y) = x * y + sin(x) at (x, y) = (2, 3)
        // df/dx = y + cos(x), df/dy = x
        let x: Dual<f64, 2> = Dual::var(2.0, 0);
        let y: Dual<f64, 2> = Dual::var(3.0, 1);

        let f = x * y + x.sin();

        assert!((f.value() - (2.0 * 3.0 + 2.0_f64.sin())).abs() < 1e-12);
        assert!((f.grad()[0] - (3.0 + 2.0_f64.cos())).abs() < 1e-12);
        assert!((f.grad()[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sqrt() {
        let x: Dual<f64, 1> = Dual::var(4.0, 0);
        let y = x.sqrt();
        assert!((y.value() - 2.0).abs() < 1e-12);
        assert!((y.grad()[0] - 0.25).abs() < 1e-12); // derivative = 1/(2*sqrt(x)) = 0.25
    }

    #[test]
    fn test_tan() {
        let x: Dual<f64, 1> = Dual::var(0.5, 0);
        let y = x.tan();
        assert!((y.value() - 0.5_f64.tan()).abs() < 1e-12);
        let expected_deriv = 1.0 / (0.5_f64.cos() * 0.5_f64.cos());
        assert!((y.grad()[0] - expected_deriv).abs() < 1e-12);
    }

    #[test]
    fn test_asin() {
        let x: Dual<f64, 1> = Dual::var(0.5, 0);
        let y = x.asin();
        assert!((y.value() - 0.5_f64.asin()).abs() < 1e-12);
        let expected_deriv = 1.0 / (1.0_f64 - 0.25).sqrt();
        assert!((y.grad()[0] - expected_deriv).abs() < 1e-12);
    }

    #[test]
    fn test_acos() {
        let x: Dual<f64, 1> = Dual::var(0.5, 0);
        let y = x.acos();
        assert!((y.value() - 0.5_f64.acos()).abs() < 1e-12);
        let expected_deriv = -1.0 / (1.0_f64 - 0.25).sqrt();
        assert!((y.grad()[0] - expected_deriv).abs() < 1e-12);
    }

    #[test]
    fn test_atan2() {
        // f(y, x) = atan2(y, x) at (y, x) = (3, 4)
        // df/dy = x/(x^2+y^2), df/dx = -y/(x^2+y^2)
        let y: Dual<f64, 2> = Dual::var(3.0, 0);
        let x: Dual<f64, 2> = Dual::var(4.0, 1);
        let f = y.atan2(x);

        assert!((f.value() - 3.0_f64.atan2(4.0)).abs() < 1e-12);
        assert!((f.grad()[0] - 4.0 / 25.0).abs() < 1e-12);
        assert!((f.grad()[1] - (-3.0 / 25.0)).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_ops() {
        let x: Dual<f64, 1> = Dual::var(2.0, 0);

        let a = x + 1.0;
        assert_eq!(a.value(), 3.0);
        assert_eq!(a.grad(), [1.0]);

        let b = 1.0 + x;
        assert_eq!(b.value(), 3.0);
        assert_eq!(b.grad(), [1.0]);

        let c = x * 3.0;
        assert_eq!(c.value(), 6.0);
        assert_eq!(c.grad(), [3.0]);

        let d = 3.0 * x;
        assert_eq!(d.value(), 6.0);
        assert_eq!(d.grad(), [3.0]);

        let e = x - 0.5;
        assert_eq!(e.value(), 1.5);
        assert_eq!(e.grad(), [1.0]);

        let f = 10.0 - x;
        assert_eq!(f.value(), 8.0);
        assert_eq!(f.grad(), [-1.0]); // d/dx(c - x) = -1

        let g = x / 2.0;
        assert_eq!(g.value(), 1.0);
        assert_eq!(g.grad(), [0.5]);

        let h = 8.0 / x;
        assert_eq!(h.value(), 4.0);
        assert_eq!(h.grad(), [-2.0]); // d/dx(c/x) = -c/x^2 = -8/4 = -2
    }
}
