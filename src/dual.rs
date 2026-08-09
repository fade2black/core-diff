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

impl<const N: usize> Dual<f64, N> {
    /// A constant: grad = 0
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; N],
        }
    }

    /// A variable: partial derivative w.r.t. itself is 1 at `index`, 0 elsewhere.
    pub fn var(value: f64, index: usize) -> Self {
        let mut grad = [0.0; N];
        grad[index] = 1.0;
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
        let inv_v = 1.0 / self.value;
        Self {
            value: self.value.ln(),
            grad: std::array::from_fn(|i| self.grad[i] * inv_v),
        }
    }

    pub fn powf(self, n: f64) -> Self {
        let deriv = n * self.value.powf(n - 1.0);
        Self {
            value: self.value.powf(n),
            grad: std::array::from_fn(|i| self.grad[i] * deriv),
        }
    }
}

/// Free-function form of `Dual::sin`, so expressions like `sin(x * y)` resolve
/// without needing a macro to rewrite them into method calls.
#[inline]
pub fn sin<const N: usize>(d: Dual<f64, N>) -> Dual<f64, N> {
    d.sin()
}

#[inline]
pub fn cos<const N: usize>(d: Dual<f64, N>) -> Dual<f64, N> {
    d.cos()
}

#[inline]
pub fn exp<const N: usize>(d: Dual<f64, N>) -> Dual<f64, N> {
    d.exp()
}

#[inline]
pub fn ln<const N: usize>(d: Dual<f64, N>) -> Dual<f64, N> {
    d.ln()
}

#[inline]
pub fn powf<const N: usize>(d: Dual<f64, N>, n: f64) -> Dual<f64, N> {
    d.powf(n)
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
}
