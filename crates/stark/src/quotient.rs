use p3_air::Air;
use p3_commit::PolynomialSpace;
use p3_field::{AbstractExtensionField, AbstractField, PackedValue};
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{air::MachineAir, septic_digest::SepticDigest};

use super::{
    folder::ProverConstraintFolder, Chip, Domain, PackedChallenge, PackedVal, StarkGenericConfig,
    Val,
};

/// Computes the quotient values.
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn quotient_values<SC, A, Mat>(
    chip: &Chip<Val<SC>, A>,
    local_cumulative_sum: &SC::Challenge,
    global_cumulative_sum: &SepticDigest<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_trace_on_quotient_domain: Option<Mat>,
    main_trace_on_quotient_domain: Mat,
    permutation_trace_on_quotient_domain: Mat,
    perm_challenges: &[PackedChallenge<SC>],
    alpha: SC::Challenge,
    public_values: &[Val<SC>],
) -> Vec<SC::Challenge>
where
    A: for<'a> Air<ProverConstraintFolder<'a, SC>> + MachineAir<Val<SC>>,
    SC: StarkGenericConfig,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let prep_width =
        preprocessed_trace_on_quotient_domain.as_ref().map_or(1, p3_matrix::Matrix::width);
    let main_width = main_trace_on_quotient_domain.width();
    let perm_width = permutation_trace_on_quotient_domain.width();
    let sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    let ext_degree = SC::Challenge::D;
    let perm_ext_width = perm_width / ext_degree;

    // Bitwise AND mask — quotient_size is always a power of 2.
    let qs_mask = quotient_size - 1;

    assert!(
        quotient_size >= PackedVal::<SC>::WIDTH,
        "quotient size is too small: got {}, expected at least {} for chip {}",
        quotient_size,
        PackedVal::<SC>::WIDTH,
        chip.name()
    );

    // Pre-allocate output to avoid per-iteration collection overhead.
    let mut results = vec![SC::Challenge::zero(); quotient_size];

    results
        .par_chunks_mut(PackedVal::<SC>::WIDTH)
        .enumerate()
        .for_each_init(
            || {
                // Per-thread reusable buffers — allocated once, reused across iterations.
                (
                    vec![PackedVal::<SC>::zero(); prep_width],
                    vec![PackedVal::<SC>::zero(); prep_width],
                    vec![PackedVal::<SC>::zero(); main_width],
                    vec![PackedVal::<SC>::zero(); main_width],
                    vec![PackedChallenge::<SC>::zero(); perm_ext_width],
                    vec![PackedChallenge::<SC>::zero(); perm_ext_width],
                )
            },
            |(prep_local, prep_next, local, next, perm_local, perm_next),
             (chunk_idx, result_chunk)| {
                let i_start = chunk_idx * PackedVal::<SC>::WIDTH;
                let wrap = |i: usize| i & qs_mask;
                let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

                let is_first_row =
                    *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
                let is_last_row =
                    *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
                let is_transition =
                    *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
                let inv_zeroifier =
                    *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range]);

                for col in 0..prep_width {
                    prep_local[col] = PackedVal::<SC>::from_fn(|offset| {
                        preprocessed_trace_on_quotient_domain
                            .as_ref()
                            .map_or(Val::<SC>::zero(), |x| x.get(wrap(i_start + offset), col))
                    });
                    prep_next[col] = PackedVal::<SC>::from_fn(|offset| {
                        preprocessed_trace_on_quotient_domain.as_ref().map_or(
                            Val::<SC>::zero(),
                            |x| x.get(wrap(i_start + next_step + offset), col),
                        )
                    });
                }

                for col in 0..main_width {
                    local[col] = PackedVal::<SC>::from_fn(|offset| {
                        main_trace_on_quotient_domain.get(wrap(i_start + offset), col)
                    });
                    next[col] = PackedVal::<SC>::from_fn(|offset| {
                        main_trace_on_quotient_domain
                            .get(wrap(i_start + next_step + offset), col)
                    });
                }

                for (idx, col) in (0..perm_width).step_by(ext_degree).enumerate() {
                    perm_local[idx] = PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            permutation_trace_on_quotient_domain
                                .get(wrap(i_start + offset), col + i)
                        })
                    });
                    perm_next[idx] = PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            permutation_trace_on_quotient_domain
                                .get(wrap(i_start + next_step + offset), col + i)
                        })
                    });
                }

                let accumulator = PackedChallenge::<SC>::zero();
                let packed_local_cumulative_sum =
                    PackedChallenge::<SC>::from_f(*local_cumulative_sum);

                let mut folder = ProverConstraintFolder {
                    preprocessed: VerticalPair::new(
                        RowMajorMatrixView::new_row(prep_local),
                        RowMajorMatrixView::new_row(prep_next),
                    ),
                    main: VerticalPair::new(
                        RowMajorMatrixView::new_row(local),
                        RowMajorMatrixView::new_row(next),
                    ),
                    perm: VerticalPair::new(
                        RowMajorMatrixView::new_row(perm_local),
                        RowMajorMatrixView::new_row(perm_next),
                    ),
                    perm_challenges,
                    local_cumulative_sum: &packed_local_cumulative_sum,
                    global_cumulative_sum,
                    is_first_row,
                    is_last_row,
                    is_transition,
                    alpha,
                    accumulator,
                    public_values,
                };
                chip.eval(&mut folder);

                let quotient = folder.accumulator * inv_zeroifier;

                for idx_in_packing in 0..PackedVal::<SC>::WIDTH {
                    result_chunk[idx_in_packing] =
                        SC::Challenge::from_base_fn(|coeff_idx| {
                            quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing]
                        });
                }
            },
        );

    results
}
