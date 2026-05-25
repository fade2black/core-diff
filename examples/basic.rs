use core_diff::dual::Dual;

fn main() {
    // Define x as the active variable
    let x = Dual::new(3.0, 1.0);

    // Compute f(x) = x^2
    let y = x * x;

    assert_eq!(y.value, 9.0); // f(x)
    assert_eq!(y.derivative, 6.0); // f'(x) = 2x
}
