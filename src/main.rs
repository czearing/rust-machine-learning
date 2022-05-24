mod activations;
mod attention;
use activations::softmax::softmax;
use attention::scaled_dot_product_attention::scaled_dot_product_attention;
use ndarray::{array, Array, ArrayD, IxDyn};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    // let a = Array::random(IxDyn(&[2, 2]), Uniform::new(0., 10.));
    // let array = ArrayD::<f32>::zeros(IxDyn(&[4, 2]));
    // let value = softmax(&a);
    // scaled_dot_product_attention();
    // println!("{}", a);
    // println!("{}", value);

    let q = array![[0.7943, -0.5397], [0.1767, 0.6953], [0.5263, 0.7972]];
    let k = array![[1.9828, -0.5915], [0.3966, -0.1153], [-0.6160, 1.2105]];
    let v = array![[-0.9583, 1.3639], [0.2016, -0.2045], [-0.6284, 0.7721]];

    let result = scaled_dot_product_attention(&q, &k, &v, false);

    println!("{}", result);
}
