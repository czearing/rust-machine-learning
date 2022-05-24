/**
## Sigmoid
Sigmoid activation function. The activation curve is more smooth but can lead to a vanishing gradient.
*/
pub fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + fast_math::exp(-x));
}
