use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T> {
    pub value: T,
    pub derivative: T,
}

impl<T> Dual<T>
where
    T: Copy,
{
    /// Generic constructor
    pub fn new(value: T, derivative: T) -> Self {
        Self { value, derivative }
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
            derivative: self.derivative + rhs.derivative,
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
            derivative: self.derivative - rhs.derivative,
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
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
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
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative)
                / (rhs.value * rhs.value),
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
            derivative: -self.derivative,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Dual;

    #[test]
    fn dual_new_stores_value_and_derivative() {
        let d = Dual::new(3.0, 0.5);
        assert_eq!(d.value, 3.0);
        assert_eq!(d.derivative, 0.5);
    }

    #[test]
    fn dual_constant_is_zero_derivative() {
        let c = Dual::new(5.0, 0.0);
        assert_eq!(c.value, 5.0);
        assert_eq!(c.derivative, 0.0);
    }

    #[test]
    fn dual_variable_is_one_derivative() {
        let x = Dual::new(2.0, 1.0);
        assert_eq!(x.value, 2.0);
        assert_eq!(x.derivative, 1.0);
    }

    #[test]
    fn add_works() {
        let a = Dual::new(2.0, 1.0);
        let b = Dual::new(3.0, 0.0);
        let c = a + b;
        assert_eq!(c.value, 5.0);
        assert_eq!(c.derivative, 1.0);
    }

    #[test]
    fn mul_works() {
        let x = Dual::new(3.0, 1.0);
        let y = x * x; // derivative = 2*x = 6
        assert_eq!(y.value, 9.0);
        assert_eq!(y.derivative, 6.0);
    }

    #[test]
    fn div_works() {
        let x = Dual::new(4.0, 1.0);
        let y = Dual::new(2.0, 0.0);
        let z = x / y; // derivative = (1*2 - 4*0) / 4 = 0.5
        assert_eq!(z.value, 2.0);
        assert_eq!(z.derivative, 0.5);
    }

    #[test]
    fn neg_works() {
        let x = Dual::new(4.0, 1.0);
        let y = -x;
        assert_eq!(y.value, -4.0);
        assert_eq!(y.derivative, -1.0);
    }
}
