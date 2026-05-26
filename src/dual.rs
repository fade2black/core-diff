use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T> {
    value: T,
    grad: T,
}

impl<T> Dual<T>
where
    T: Copy,
{
    /// Generic constructor
    pub fn new(value: T, grad: T) -> Self {
        Self { value, grad }
    }

    /// Returns the value of the dual number.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the derivative of the dual number.
    pub fn grad(&self) -> T {
        self.grad
    }
}

impl<T> Add for Dual<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            grad: self.grad + rhs.grad,
        }
    }
}

impl<T> Sub for Dual<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            grad: self.grad - rhs.grad,
        }
    }
}

impl<T> Mul for Dual<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            grad: self.grad * rhs.value + self.value * rhs.grad,
        }
    }
}

impl<T> Div for Dual<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value,
            grad: (self.grad * rhs.value - self.value * rhs.grad) / (rhs.value * rhs.value),
        }
    }
}

impl<T> Neg for Dual<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            grad: -self.grad,
        }
    }
}

impl Dual<f64> {
    /// A constant: grad = 0
    pub fn constant(value: f64) -> Self {
        Self { value, grad: 0.0 }
    }

    /// A variable: grad = 1
    pub fn var(value: f64) -> Self {
        Self { value, grad: 1.0 }
    }

    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            grad: self.grad * self.value.cos(),
        }
    }

    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            grad: self.grad * -self.value.sin(),
        }
    }

    pub fn exp(self) -> Self {
        let v = self.value.exp();
        Self {
            value: v,
            grad: self.grad * v,
        }
    }

    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            grad: self.grad * (1.0 / self.value),
        }
    }

    pub fn powf(self, n: f64) -> Self {
        Self {
            value: self.value.powf(n),
            grad: self.grad * n * self.value.powf(n - 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Dual;

    #[test]
    fn dual_new_stores_value_and_gradient() {
        let d = Dual::new(3.0, 0.5);
        assert_eq!(d.value, 3.0);
        assert_eq!(d.grad, 0.5);
    }

    #[test]
    fn dual_constant_is_zero_gradient() {
        let c = Dual::new(5.0, 0.0);
        assert_eq!(c.value, 5.0);
        assert_eq!(c.grad, 0.0);
    }

    #[test]
    fn dual_variable_is_one_derivative() {
        let x = Dual::new(2.0, 1.0);
        assert_eq!(x.value, 2.0);
        assert_eq!(x.grad, 1.0);
    }

    #[test]
    fn add_works() {
        let a = Dual::new(2.0, 1.0);
        let b = Dual::new(3.0, 0.0);
        let c = a + b;
        assert_eq!(c.value, 5.0);
        assert_eq!(c.grad, 1.0);
    }

    #[test]
    fn mul_works() {
        let x = Dual::new(3.0, 1.0);
        let y = x * x; // derivative = 2*x = 6
        assert_eq!(y.value, 9.0);
        assert_eq!(y.grad, 6.0);
    }

    #[test]
    fn div_works() {
        let x = Dual::new(4.0, 1.0);
        let y = Dual::new(2.0, 0.0);
        let z = x / y; // derivative = (1*2 - 4*0) / 4 = 0.5
        assert_eq!(z.value, 2.0);
        assert_eq!(z.grad, 0.5);
    }

    #[test]
    fn neg_works() {
        let x = Dual::new(4.0, 1.0);
        let y = -x;
        assert_eq!(y.value, -4.0);
        assert_eq!(y.grad, -1.0);
    }

    #[test]
    fn test_sin() {
        let x = Dual::new(2.0, 1.0);
        let y = x.sin();
        assert!((y.value() - 2.0_f64.sin()).abs() < 1e-12);
        assert!((y.grad() - 2.0_f64.cos()).abs() < 1e-12);
    }

    #[test]
    fn test_cos() {
        let x = Dual::new(2.0, 1.0);
        let y = x.cos();
        assert!((y.value() - 2.0_f64.cos()).abs() < 1e-12);
        assert!((y.grad() + 2.0_f64.sin()).abs() < 1e-12); // derivative = -sin(x)
    }

    #[test]
    fn test_exp() {
        let x = Dual::new(2.0, 1.0);
        let y = x.exp();
        assert!((y.value() - 2.0_f64.exp()).abs() < 1e-12);
        assert!((y.grad() - 2.0_f64.exp()).abs() < 1e-12);
    }

    #[test]
    fn test_ln() {
        let x = Dual::new(2.0, 1.0);
        let y = x.ln();
        assert!((y.value() - 2.0_f64.ln()).abs() < 1e-12);
        assert!((y.grad() - 0.5).abs() < 1e-12); // derivative = 1/x
    }

    #[test]
    fn test_powf() {
        let x = Dual::new(2.0, 1.0);
        let y = x.powf(3.0);
        assert_eq!(y.value(), 8.0);
        assert_eq!(y.grad(), 3.0 * 2.0_f64.powf(2.0)); // 3 * x^2
    }

    #[test]
    fn test_polynomial() {
        // f(x) = 1 + x + 2x^2 at x=1
        let x = Dual::new(1.0, 1.0);
        let y = Dual::constant(1.0) + x + Dual::constant(2.0) * x * x;
        assert_eq!(y.value(), 4.0);
        assert_eq!(y.grad(), 5.0);
    }

    #[test]
    fn test_three_pow_x_plus_cos() {
        let x = Dual::new(2.0, 1.0);
        let ln3 = Dual::constant(3.0_f64.ln());
        let three_pow_x = (x * ln3).exp();
        let y = three_pow_x + x.cos();

        assert!((y.value() - 8.583853163452858).abs() < 1e-12);
        assert!((y.grad() - 8.978213171187306).abs() < 1e-12);
    }
}
