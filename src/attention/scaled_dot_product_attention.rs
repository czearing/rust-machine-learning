use crate::activations::softmax::softmax;
use ndarray::Array2;

pub fn scaled_dot_product_attention(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    mask: bool,
) -> Array2<f32> {
    let transposed_key = key.view().reversed_axes().to_owned();
    let mathmul = query.dot(&transposed_key);
    let column_size = (query.column(0).len() - 1) as f32;
    let attention_logits = mathmul / column_size.sqrt();
    if mask {}
    let attention = softmax(&attention_logits);

    return attention.dot(value);
}
