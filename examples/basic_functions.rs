use core_diff::dual::Dual;

fn main() {
    // Compute 1 + x + 2x^2 at x = 1
    let x: Dual<f64, 1> = Dual::var(1.0, 0); // value=1, grad=1
    let one = Dual::constant(1.0); // value=1, grad=0
    let two = Dual::constant(2.0); // value=2, grad=0

    let y = one + x + two * x.powf(2.0);

    println!("value = {}", y.value());
    println!("grad  = {:?}", y.grad());

    // Compute sin(x) + exp(x) * x^3 at x=2
    let x: Dual<f64, 1> = Dual::var(2.0, 0); // value=2, grad=1

    let y = x.sin() + x.exp() * x.powf(3.0);

    println!("value = {}", y.value());
    println!("grad  = {:?}", y.grad());

    // Compute 3^x using: 3^x = exp(x * ln(3))
    let three_pow_x = (x * Dual::constant(3.0_f64.ln())).exp(); // grad = 0
    // Compute cos(x)
    let cos_x = x.cos();

    // f(x) = 3^x + cos(x)
    let y = three_pow_x + cos_x;

    println!("value = {}", y.value());
    println!("grad  = {:?}", y.grad());

    // Compute f(x, y) = x*y + sin(x) at (x, y) = (2, 3)
    let x: Dual<f64, 2> = Dual::var(2.0, 0);
    let y: Dual<f64, 2> = Dual::var(3.0, 1);

    let f = x * y + x.sin();

    println!("value = {}", f.value());
    println!("grad  = {:?}", f.grad()); // [df/dx, df/dy]
}
