use core_diff::dual::Dual;
use core_diff::jacobian::jacobian;

fn main() {
    // f1(x, y) = x^2 + y
    // f2(x, y) = sin(x*y)
    // at (x, y) = (2, 3)
    let x: Dual<f64, 2> = Dual::var(2.0, 0);
    let y: Dual<f64, 2> = Dual::var(3.0, 1);

    let outputs = [x * x + y, (x * y).sin()];
    let (value, jac) = jacobian(outputs);

    println!("value = {}", value);
    println!("jacobian =\n{}", jac);
}
