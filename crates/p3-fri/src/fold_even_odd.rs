use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::TwoAdicField;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::instrument;

/// Fold a polynomial
/// ```ignore
/// p(x) = p_even(x^2) + x p_odd(x^2)
/// ```
/// into
/// ```ignore
/// p_even(x) + beta p_odd(x)
/// ```
/// Expects input to be bit-reversed evaluations.
#[instrument(skip_all, level = "debug")]
pub fn fold_even_odd<F: TwoAdicField>(poly: Vec<F>, beta: F) -> Vec<F> {
    // We use the fact that
    //     p_e(x^2) = (p(x) + p(-x)) / 2
    //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
    // that is,
    //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
    //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
    // so
    //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
    //                    = (1/2 + beta/2 g_inv^i) p(g^i)
    //                    + (1/2 - beta/2 g_inv^i) p(g^(n/2 + i))
    let half_len = poly.len() / 2;
    let log_half_len = log2_strict_usize(half_len);
    let g_inv = F::two_adic_generator(log_half_len + 1).inverse();
    let one_half = F::two().inverse();
    let half_beta = beta * one_half;

    // Precompute twiddle factors in bit-reversed order
    let mut powers = g_inv
        .shifted_powers(half_beta)
        .take(half_len)
        .collect_vec();
    reverse_slice_index_bits(&mut powers);

    // Operate directly on the slice — no RowMajorMatrix allocation needed.
    // poly is laid out as [r0_0, r1_0, r0_1, r1_1, ...] (pairs of even/odd).
    //
    // Rewrite: (1/2 + p)*r0 + (1/2 - p)*r1 = 1/2*(r0+r1) + p*(r0-r1)
    // This replaces one EF×EF multiply with a cheaper EF×scalar multiply
    // (one_half is embedded from the base field, so multiplication is component-wise).
    powers
        .into_par_iter()
        .enumerate()
        .map(|(i, power)| {
            let r0 = poly[2 * i];
            let r1 = poly[2 * i + 1];
            let sum = r0 + r1;
            let diff = r0 - r1;
            one_half * sum + power * diff
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use itertools::izip;
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_fold_even_odd() {
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_n = 10;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let even_coeffs = coeffs.iter().cloned().step_by(2).collect_vec();
        let even_evals = dft.dft(even_coeffs);

        let odd_coeffs = coeffs.iter().cloned().skip(1).step_by(2).collect_vec();
        let odd_evals = dft.dft(odd_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(even_evals, odd_evals)
            .map(|(even, odd)| even + beta * odd)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        reverse_slice_index_bits(&mut folded);
        folded = fold_even_odd(folded, beta);
        reverse_slice_index_bits(&mut folded);

        assert_eq!(expected, folded);
    }
}
