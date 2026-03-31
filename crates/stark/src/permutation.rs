use crate::{
    air::{InteractionScope, MultiTableAirBuilder},
    lookup::Interaction,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{AirBuilder, ExtensionBuilder, PairBuilder};
use p3_field::AbstractExtensionField;
use p3_field::AbstractField;
use p3_field::{batch_multiplicative_inverse, ExtensionField, Field, PrimeField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use rayon_scan::ScanParallelIterator;
use std::borrow::Borrow;

/// Computes the width of the local permutation trace in terms of extension field elements.
#[must_use]
pub const fn local_permutation_trace_width(nb_interactions: usize, batch_size: usize) -> usize {
    if nb_interactions == 0 {
        return 0;
    }
    nb_interactions.div_ceil(batch_size) + 1
}

/// Populates a local permutation row.
#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub fn populate_local_permutation_row<F: PrimeField, EF: ExtensionField<F>>(
    row: &mut [EF],
    preprocessed_row: &[F],
    main_row: &[F],
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    alpha: EF,
    beta_powers: &[EF],
    batch_size: usize,
) {
    let interaction_chunks = &sends
        .iter()
        .map(|int| (int, true))
        .chain(receives.iter().map(|int| (int, false)))
        .chunks(batch_size);

    // Scratch buffer for batch entries — avoids per-batch Vec allocation.
    // batch_size is typically 2, but we support up to 8 on the stack.
    let mut entries_buf: [(EF, F); 8] = [(EF::zero(), F::zero()); 8];

    // Compute the fingerprints for each batch, using batch inversion to replace
    // per-interaction divisions with a single inversion + multiplications.
    for (value, chunk) in row.iter_mut().zip(interaction_chunks) {
        let mut n_entries = 0;
        for (interaction, is_send) in chunk {
            let mut denominator = alpha;
            // beta_powers[0] = 1, beta_powers[1] = beta, beta_powers[2] = beta^2, etc.
            denominator +=
                beta_powers[0] * EF::from_canonical_usize(interaction.argument_index());
            for (j, columns) in interaction.values.iter().enumerate() {
                denominator += beta_powers[j + 1] * columns.apply::<F, F>(preprocessed_row, main_row);
            }
            let mut mult = interaction.multiplicity.apply::<F, F>(preprocessed_row, main_row);
            if !is_send {
                mult = -mult;
            }
            entries_buf[n_entries] = (denominator, mult);
            n_entries += 1;
        }
        let entries = &entries_buf[..n_entries];

        // Batch inversion: compute product of all denominators, invert once,
        // then recover individual inverses via Montgomery's trick.
        if n_entries == 1 {
            *value = EF::from_base(entries[0].1) / entries[0].0;
        } else {
            // Compute prefix products (on stack for small batch sizes).
            let mut prefix_buf: [EF; 8] = [EF::zero(); 8];
            prefix_buf[0] = entries[0].0;
            for i in 1..n_entries {
                prefix_buf[i] = prefix_buf[i - 1] * entries[i].0;
            }
            // Invert the total product once.
            let mut inv = prefix_buf[n_entries - 1].inverse();
            // Backtrack to get individual inverses and accumulate.
            let mut sum = EF::zero();
            for i in (1..n_entries).rev() {
                let inv_i = inv * prefix_buf[i - 1];
                sum += EF::from_base(entries[i].1) * inv_i;
                inv *= entries[i].0;
            }
            sum += EF::from_base(entries[0].1) * inv;
            *value = sum;
        }
    }
}

/// Returns the sends, receives, and permutation trace width grouped by scope.
#[allow(clippy::type_complexity)]
pub fn scoped_interactions<F: Field>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
) -> (HashMap<InteractionScope, Vec<Interaction<F>>>, HashMap<InteractionScope, Vec<Interaction<F>>>)
{
    // Create a hashmap of scope -> vec<send interactions>.
    let mut sends = sends.to_vec();
    sends.sort_by_key(|k| k.scope);
    let grouped_sends: HashMap<_, _> = sends
        .iter()
        .chunk_by(|int| int.scope)
        .into_iter()
        .map(|(k, values)| (k, values.cloned().collect_vec()))
        .collect();

    // Create a hashmap of scope -> vec<receive interactions>.
    let mut receives = receives.to_vec();
    receives.sort_by_key(|k| k.scope);
    let grouped_receives: HashMap<_, _> = receives
        .iter()
        .chunk_by(|int| int.scope)
        .into_iter()
        .map(|(k, values)| (k, values.cloned().collect_vec()))
        .collect();

    (grouped_sends, grouped_receives)
}

/// Generates the permutation trace for the given chip and main trace based on a variant of `LogUp`.
///
/// Delegates to [`generate_permutation_trace_prescoped`] after scoping interactions.
#[allow(clippy::too_many_lines)]
pub fn generate_permutation_trace<F: PrimeField, EF: ExtensionField<F>>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    preprocessed: Option<&RowMajorMatrix<F>>,
    main: &RowMajorMatrix<F>,
    random_elements: &[EF],
    batch_size: usize,
) -> (RowMajorMatrix<EF>, EF) {
    let empty = vec![];
    let (scoped_sends, scoped_receives) = scoped_interactions(sends, receives);
    let local_sends = scoped_sends.get(&InteractionScope::Local).unwrap_or(&empty);
    let local_receives = scoped_receives.get(&InteractionScope::Local).unwrap_or(&empty);
    generate_permutation_trace_prescoped(local_sends, local_receives, preprocessed, main, random_elements, batch_size)
}

/// Like [`generate_permutation_trace`] but takes pre-scoped local interaction slices,
/// avoiding the per-call clone+sort+hash of `scoped_interactions`.
///
/// Uses blocked batch inversion: instead of inverting per chunk per row (~300 base muls each),
/// processes rows in blocks, computing numerators/products in parallel then batch-inverting per
/// block. Amortized cost: ~5 EF muls per chunk vs ~35 with per-row inversion (7x improvement).
#[allow(clippy::too_many_lines)]
pub fn generate_permutation_trace_prescoped<F: PrimeField, EF: ExtensionField<F>>(
    local_sends: &[Interaction<F>],
    local_receives: &[Interaction<F>],
    preprocessed: Option<&RowMajorMatrix<F>>,
    main: &RowMajorMatrix<F>,
    random_elements: &[EF],
    batch_size: usize,
) -> (RowMajorMatrix<EF>, EF) {
    let local_permutation_width =
        local_permutation_trace_width(local_sends.len() + local_receives.len(), batch_size);

    let height = main.height();
    let permutation_trace_width = local_permutation_width;

    let mut local_cumulative_sum = EF::zero();

    if local_sends.is_empty() && local_receives.is_empty() {
        let permutation_trace = RowMajorMatrix::new(
            vec![EF::zero(); permutation_trace_width * height],
            permutation_trace_width,
        );
        return (permutation_trace, local_cumulative_sum);
    }

    let alpha = random_elements[0];
    let beta = random_elements[1];

    // Precompute beta powers.
    let max_values_len = local_sends
        .iter()
        .chain(local_receives.iter())
        .map(|int| int.values.len())
        .max()
        .unwrap_or(0);
    let num_beta_powers = max_values_len + 1;
    let beta_powers: Vec<EF> = {
        let mut powers = Vec::with_capacity(num_beta_powers);
        let mut current = EF::one();
        for _ in 0..num_beta_powers {
            powers.push(current);
            current *= beta;
        }
        powers
    };

    // Number of batch chunks (columns in perm trace, excluding the cumulative sum column).
    let num_chunks = local_permutation_width - 1;

    // Precompute the interaction chunks structure.
    let all_interactions: Vec<(&Interaction<F>, bool)> = local_sends
        .iter()
        .map(|int| (int, true))
        .chain(local_receives.iter().map(|int| (int, false)))
        .collect();
    let chunks: Vec<&[(&Interaction<F>, bool)]> = all_interactions.chunks(batch_size).collect();
    debug_assert_eq!(chunks.len(), num_chunks);

    // Allocate permutation trace. We fill chunk columns [0..num_chunks] here;
    // the last column (cumulative sum) is filled afterward.
    let mut trace_values = vec![EF::zero(); permutation_trace_width * height];

    // Process rows in parallel blocks. Each block computes products + numerators,
    // batch-inverts the products (amortizing the expensive EF inversion), then writes results.
    const BLOCK_SIZE: usize = 4096;
    let main_width = main.width();
    let main_vals = main.values.as_slice();
    let prep_width = preprocessed.map_or(0, |p| p.width());
    let prep_vals = preprocessed.map(|p| p.values.as_slice());

    trace_values
        .par_chunks_mut(permutation_trace_width * BLOCK_SIZE)
        .enumerate()
        .for_each(|(block_idx, block_trace)| {
            let block_start = block_idx * BLOCK_SIZE;
            let block_rows = block_trace.len() / permutation_trace_width;

            // Temporary buffers for this block.
            let mut products = vec![EF::zero(); block_rows * num_chunks];
            let mut numerators = vec![EF::zero(); block_rows * num_chunks];

            for local_row in 0..block_rows {
                let row_idx = block_start + local_row;
                let main_row = &main_vals[row_idx * main_width..(row_idx + 1) * main_width];
                let prep_row = prep_vals.map_or(&[] as &[F], |pv| {
                    &pv[row_idx * prep_width..(row_idx + 1) * prep_width]
                });
                let base = local_row * num_chunks;

                for (c, chunk) in chunks.iter().enumerate() {
                    match chunk.len() {
                        1 => {
                            let (interaction, is_send) = chunk[0];
                            let mut denom = alpha;
                            denom += beta_powers[0]
                                * EF::from_canonical_usize(interaction.argument_index());
                            for (j, col) in interaction.values.iter().enumerate() {
                                denom += beta_powers[j + 1]
                                    * col.apply::<F, F>(prep_row, main_row);
                            }
                            let mut mult =
                                interaction.multiplicity.apply::<F, F>(prep_row, main_row);
                            if !is_send {
                                mult = -mult;
                            }
                            products[base + c] = denom;
                            numerators[base + c] = EF::from_base(mult);
                        }
                        2 => {
                            let (int0, is_send0) = chunk[0];
                            let (int1, is_send1) = chunk[1];

                            let mut d0 = alpha;
                            d0 += beta_powers[0]
                                * EF::from_canonical_usize(int0.argument_index());
                            for (j, col) in int0.values.iter().enumerate() {
                                d0 += beta_powers[j + 1] * col.apply::<F, F>(prep_row, main_row);
                            }
                            let mut m0 = int0.multiplicity.apply::<F, F>(prep_row, main_row);
                            if !is_send0 {
                                m0 = -m0;
                            }

                            let mut d1 = alpha;
                            d1 += beta_powers[0]
                                * EF::from_canonical_usize(int1.argument_index());
                            for (j, col) in int1.values.iter().enumerate() {
                                d1 += beta_powers[j + 1] * col.apply::<F, F>(prep_row, main_row);
                            }
                            let mut m1 = int1.multiplicity.apply::<F, F>(prep_row, main_row);
                            if !is_send1 {
                                m1 = -m1;
                            }

                            products[base + c] = d0 * d1;
                            numerators[base + c] =
                                EF::from_base(m0) * d1 + EF::from_base(m1) * d0;
                        }
                        n => {
                            let mut denoms: [EF; 8] = [EF::zero(); 8];
                            let mut mults: [F; 8] = [F::zero(); 8];
                            for (i, (interaction, is_send)) in chunk.iter().enumerate() {
                                let mut d = alpha;
                                d += beta_powers[0]
                                    * EF::from_canonical_usize(interaction.argument_index());
                                for (j, col) in interaction.values.iter().enumerate() {
                                    d += beta_powers[j + 1]
                                        * col.apply::<F, F>(prep_row, main_row);
                                }
                                let mut m =
                                    interaction.multiplicity.apply::<F, F>(prep_row, main_row);
                                if !is_send {
                                    m = -m;
                                }
                                denoms[i] = d;
                                mults[i] = m;
                            }
                            let mut prod = EF::one();
                            for i in 0..n {
                                prod *= denoms[i];
                            }
                            let mut num = EF::zero();
                            for i in 0..n {
                                let mut all_but_i = EF::one();
                                for j in 0..n {
                                    if j != i {
                                        all_but_i *= denoms[j];
                                    }
                                }
                                num += EF::from_base(mults[i]) * all_but_i;
                            }
                            products[base + c] = prod;
                            numerators[base + c] = num;
                        }
                    }
                }
            }

            // Batch-invert all products for this block (1 inversion + 2N muls).
            let inv_products = batch_multiplicative_inverse(&products);

            // Write results into trace.
            for local_row in 0..block_rows {
                let base = local_row * num_chunks;
                let trace_base = local_row * permutation_trace_width;
                for c in 0..num_chunks {
                    block_trace[trace_base + c] = numerators[base + c] * inv_products[base + c];
                }
            }
        });

    let mut permutation_trace = RowMajorMatrix::new(trace_values, permutation_trace_width);

    // Compute cumulative sums via parallel scan.
    let zero = EF::zero();
    let local_cumulative_sums = permutation_trace
        .par_rows_mut()
        .map(|row| row[..num_chunks].iter().copied().sum::<EF>())
        .collect::<Vec<_>>();

    let local_cumulative_sums =
        local_cumulative_sums.into_par_iter().scan(|a, b| *a + *b, zero).collect::<Vec<_>>();

    local_cumulative_sum = *local_cumulative_sums.last().unwrap();

    permutation_trace.par_rows_mut().zip_eq(local_cumulative_sums.into_par_iter()).for_each(
        |(row, cum_sum)| {
            row[num_chunks] = cum_sum;
        },
    );

    (permutation_trace, local_cumulative_sum)
}

/// Evaluates the permutation constraints for the given chip.
///
/// In particular, the constraints checked here are:
///     - The running sum column starts at zero.
///     - That the RLC per interaction is computed correctly.
///     - The running sum column ends at the (currently) given cumulative sum.
#[allow(clippy::too_many_lines)]
pub fn eval_permutation_constraints<'a, F, AB>(
    sends: &[Interaction<F>],
    receives: &[Interaction<F>],
    batch_size: usize,
    commit_scope: InteractionScope,
    builder: &mut AB,
) where
    F: Field,
    AB::EF: ExtensionField<F>,
    AB: MultiTableAirBuilder<'a, F = F> + PairBuilder,
    AB: 'a,
{
    let (scoped_sends, scoped_receives) = scoped_interactions(sends, receives);
    let empty = vec![];
    let local_sends = scoped_sends.get(&InteractionScope::Local).unwrap_or(&empty);
    let local_receives = scoped_receives.get(&InteractionScope::Local).unwrap_or(&empty);
    eval_permutation_constraints_prescoped(local_sends, local_receives, batch_size, commit_scope, builder);
}

/// Like [`eval_permutation_constraints`] but takes pre-scoped local interaction slices,
/// avoiding the per-call clone+sort+hash of `scoped_interactions`.
#[allow(clippy::too_many_lines)]
pub fn eval_permutation_constraints_prescoped<'a, F, AB>(
    local_sends: &[Interaction<F>],
    local_receives: &[Interaction<F>],
    batch_size: usize,
    commit_scope: InteractionScope,
    builder: &mut AB,
) where
    F: Field,
    AB::EF: ExtensionField<F>,
    AB: MultiTableAirBuilder<'a, F = F> + PairBuilder,
    AB: 'a,
{
    let local_permutation_width =
        local_permutation_trace_width(local_sends.len() + local_receives.len(), batch_size);

    let permutation_trace_width = local_permutation_width;

    let preprocessed = builder.preprocessed();
    let main = builder.main();
    let perm = builder.permutation().to_row_major_matrix();

    let preprocessed_local = preprocessed.row_slice(0);
    let main_local = main.to_row_major_matrix();
    let main_local = main_local.row_slice(0);
    let main_local: &[AB::Var] = (*main_local).borrow();
    let perm_local = perm.row_slice(0);
    let perm_local: &[AB::VarEF] = (*perm_local).borrow();
    let perm_next = perm.row_slice(1);
    let perm_next: &[AB::VarEF] = (*perm_next).borrow();
    let perm_width = perm.width();

    // Assert that the permutation trace width is correct.
    if perm_width != permutation_trace_width {
        panic!(
            "permutation trace width is incorrect: expected {permutation_trace_width}, got {perm_width}",
        );
    }

    // Get the permutation challenges.
    let permutation_challenges = builder.permutation_randomness();
    let random_elements: Vec<AB::ExprEF> =
        permutation_challenges.iter().map(|x| (*x).into()).collect();
    let local_cumulative_sum = builder.local_cumulative_sum();

    let random_elements = &random_elements[0..2];
    let (alpha, beta) = (&random_elements[0], &random_elements[1]);
    if !local_sends.is_empty() || !local_receives.is_empty() {
        // Precompute beta powers to avoid repeated iterator creation per interaction.
        let max_values_len = local_sends
            .iter()
            .chain(local_receives.iter())
            .map(|int| int.values.len())
            .max()
            .unwrap_or(0);
        let num_beta_powers = max_values_len + 1; // +1 for the argument_index term
        let beta_powers: Vec<AB::ExprEF> = {
            let mut powers = Vec::with_capacity(num_beta_powers);
            let mut current = AB::ExprEF::one();
            for _ in 0..num_beta_powers {
                powers.push(current.clone());
                current = current * beta.clone();
            }
            powers
        };

        // Ensure that each batch sum m_i/f_i is computed correctly.
        let interaction_chunks = &local_sends
            .iter()
            .map(|int| (int, true))
            .chain(local_receives.iter().map(|int| (int, false)))
            .chunks(batch_size);

        // Assert that the i-eth entry is equal to the sum_i m_i/rlc_i by constraints:
        // entry * \prod_i rlc_i = \sum_i m_i * \prod_{j!=i} rlc_j over all columns of the
        // permutation trace except the last column.
        for (entry, chunk) in perm_local[0..perm_local.len() - 1].iter().zip(interaction_chunks) {
            // First, we calculate the random linear combinations and multiplicities with the
            // correct sign depending on wetther the interaction is a send or a receive.
            let mut rlcs: Vec<AB::ExprEF> = Vec::with_capacity(batch_size);
            let mut multiplicities: Vec<AB::Expr> = Vec::with_capacity(batch_size);
            for (interaction, is_send) in chunk {
                let mut rlc = alpha.clone();

                rlc = rlc.clone()
                    + beta_powers[0].clone()
                        * AB::ExprEF::from_canonical_usize(interaction.argument_index());
                for (field, bp) in interaction.values.iter().zip(beta_powers[1..].iter()) {
                    let elem = field.apply::<AB::Expr, AB::Var>(&preprocessed_local, main_local);
                    rlc = rlc.clone() + bp.clone() * elem;
                }
                rlcs.push(rlc);

                let send_factor = if is_send { AB::F::one() } else { -AB::F::one() };
                multiplicities.push(
                    interaction
                        .multiplicity
                        .apply::<AB::Expr, AB::Var>(&preprocessed_local, main_local)
                        * send_factor,
                );
            }

            // Now we can calculate the numerator and denominator of the combined batch.
            let n = rlcs.len();
            let mut mults = multiplicities.into_iter();
            let (product, numerator) = if n == 1 {
                // Fast path: single interaction, no all_but_current needed.
                let product = rlcs[0].clone();
                let numerator = AB::ExprEF::from_base(mults.next().unwrap());
                (product, numerator)
            } else if n == 2 {
                // Fast path: two interactions, direct cross-multiply.
                let product = rlcs[0].clone() * rlcs[1].clone();
                let m0 = mults.next().unwrap();
                let m1 = mults.next().unwrap();
                let numerator = AB::ExprEF::from_base(m0) * rlcs[1].clone()
                    + AB::ExprEF::from_base(m1) * rlcs[0].clone();
                (product, numerator)
            } else {
                // General O(N) path using prefix/suffix products.
                // Build prefix products: prefix[i] = rlcs[0] * ... * rlcs[i].
                let mut prefix = Vec::with_capacity(n);
                let mut acc = AB::ExprEF::one();
                for rlc in rlcs.iter() {
                    acc = acc.clone() * rlc.clone();
                    prefix.push(acc.clone());
                }
                let product = prefix[n - 1].clone();

                // Build suffix products: suffix[i] = rlcs[i] * ... * rlcs[n-1].
                let mut suffix = vec![AB::ExprEF::one(); n];
                acc = AB::ExprEF::one();
                for i in (0..n).rev() {
                    acc = acc.clone() * rlcs[i].clone();
                    suffix[i] = acc.clone();
                }

                // all_but_current[i] = prefix[i-1] * suffix[i+1]
                let mut numerator = AB::ExprEF::zero();
                for (i, m) in mults.enumerate() {
                    let all_but_current = if i == 0 {
                        suffix[1].clone()
                    } else if i + 1 == n {
                        prefix[i - 1].clone()
                    } else {
                        prefix[i - 1].clone() * suffix[i + 1].clone()
                    };
                    numerator = numerator.clone() + AB::ExprEF::from_base(m) * all_but_current;
                }
                (product, numerator)
            };

            // Finally, assert that the entry is equal to the numerator divided by the product.
            let entry: AB::ExprEF = (*entry).into();
            builder.assert_eq_ext(product.clone() * entry.clone(), numerator);
        }

        // Compute the running local and next permutation sums.
        let sum_local = perm_local[..local_permutation_width - 1]
            .iter()
            .map(|x| (*x).into())
            .sum::<AB::ExprEF>();
        let sum_next = perm_next[..local_permutation_width - 1]
            .iter()
            .map(|x| (*x).into())
            .sum::<AB::ExprEF>();
        let phi_local: AB::ExprEF = (*perm_local.last().unwrap()).into();
        let phi_next: AB::ExprEF = (*perm_next.last().unwrap()).into();

        // Assert that cumulative sum is initialized to `phi_local` on the first row.
        builder.when_first_row().assert_eq_ext(phi_local.clone(), sum_local);

        // Assert that the cumulative sum is constrained to `phi_next - phi_local` on the transition
        // rows.
        builder.when_transition().assert_eq_ext(phi_next - phi_local.clone(), sum_next);
        builder.when_last_row().assert_eq_ext(*perm_local.last().unwrap(), *local_cumulative_sum);
    }

    // Handle global cumulative sums.
    // If the chip's scope is `InteractionScope::Global`, the last row's final 14 columns is equal to the global cumulative sum.
    let global_cumulative_sum = builder.global_cumulative_sum();
    if commit_scope == InteractionScope::Global {
        for i in 0..7 {
            builder
                .when_last_row()
                .assert_eq(main_local[main_local.len() - 14 + i], global_cumulative_sum.0.x.0[i]);
            builder
                .when_last_row()
                .assert_eq(main_local[main_local.len() - 7 + i], global_cumulative_sum.0.y.0[i]);
        }
    }
}
