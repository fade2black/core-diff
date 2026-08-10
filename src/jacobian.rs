use crate::dual::Dual;
use nalgebra::{SMatrix, SVector};

/// Converts the outputs of a residual function — evaluated once with
/// `Dual<f64, N>`-seeded inputs - into a value vector and Jacobian matrix.
///
/// `N` = number of parameters, `M` = number of residuals, both fixed at compile time
pub fn jacobian<const N: usize, const M: usize>(
    outputs: [Dual<f64, N>; M],
) -> (SVector<f64, M>, SMatrix<f64, M, N>) {
    let value = SVector::from_fn(|i, _| outputs[i].value());
    let jac = SMatrix::from_fn(|i, j| outputs[i].grad()[j]);
    (value, jac)
}

macro_rules! seed {
    ($($name:ident = $val:expr),*) => {
        const __N: usize = [ $(stringify!($name)),* ].len();
        let __vals = [ $($val),* ];
        let [ $($name),* ]: [Dual<f64, __N>; __N] = std::array::from_fn(|i| Dual::var(__vals[i], i));
    };
}

#[cfg(test)]
mod tests {
    use super::jacobian;
    use crate::dual::Dual;

    #[test]
    fn jacobian_of_two_residuals_two_params() {
        // f1(x, y) = x^2 + y,     f2(x, y) = sin(x*y)   at (x, y) = (2, 3)
        // df1/dx = 2x,   df1/dy = 1
        // df2/dx = y*cos(x*y),  df2/dy = x*cos(x*y)
        let x: Dual<f64, 2> = Dual::var(2.0, 0);
        let y: Dual<f64, 2> = Dual::var(3.0, 1);

        let outputs = [x * x + y, (x * y).sin()];
        let (value, jac) = jacobian(outputs);

        assert!((value[0] - 7.0).abs() < 1e-12);
        assert!((value[1] - 6.0_f64.sin()).abs() < 1e-12);

        assert!((jac[(0, 0)] - 4.0).abs() < 1e-12); // df1/dx = 2*2
        assert!((jac[(0, 1)] - 1.0).abs() < 1e-12); // df1/dy = 1
        assert!((jac[(1, 0)] - 3.0 * 6.0_f64.cos()).abs() < 1e-12); // df2/dx
        assert!((jac[(1, 1)] - 2.0 * 6.0_f64.cos()).abs() < 1e-12); // df2/dy
    }

    #[test]
    fn seed_creates_correctly_seeded_duals() {
        // f(x, y) = x*y + sin(x) at (x, y) = (2, 3)
        // df/dx = y + cos(x), df/dy = x
        seed!(x = 2.0, y = 3.0);
        let f = x * y + x.sin();

        assert!((f.value() - (2.0 * 3.0 + 2.0_f64.sin())).abs() < 1e-12);
        assert!((f.grad()[0] - (3.0 + 2.0_f64.cos())).abs() < 1e-12);
        assert!((f.grad()[1] - 2.0).abs() < 1e-12);
    }
}
