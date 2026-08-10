use core_diff::dual::sin;
use core_diff::jacobian_fn;

jacobian_fn!(f(x, y) -> {
    x * x + y,
    sin(x * y)
});

fn main() {
    let (value, jac) = f(2.0, 3.0);
    println!("value = {}", value);
    println!("jacobian =\n{}", jac);
}
