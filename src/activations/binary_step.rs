/**
## Binary Step
Returns 0 if below the threshold else 1.
*/
pub fn binary_step(x: i32, threshold: i32) -> i32 {
    return if x >= threshold { 1 } else { 0 };
}
