use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, AbstractField, ExtensionField,
    Field, PackedValue, TwoAdicField,
};
use p3_field::scale_vec;
use p3_matrix::bitrev::{BitReversableMatrix, BitReversalPerm};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use p3_merkle_tree::LeafManageable;

// GPU interpolation for PCS open. Re-enabled for <=8 core machines where GPU wins.
#[cfg(target_os = "macos")]
mod gpu_interp {
    use alloc::vec::Vec;
    use p3_field::{
        batch_multiplicative_inverse, cyclic_subgroup_coset_known_order,
        two_adic_coset_zerofier, AbstractField, ExtensionField, Field, TwoAdicField,
    };
    use p3_util::{log2_strict_usize, reverse_slice_index_bits};

    fn gpu_enabled() -> bool {
        use core::sync::atomic::{AtomicU8, Ordering};
        static ENABLED: AtomicU8 = AtomicU8::new(2); // 2 = uninitialized
        let v = ENABLED.load(Ordering::Relaxed);
        if v != 2 { return v == 1; }
        let enabled = std::env::var("METAL_DFT").ok().map_or(false, |v| v == "1");
        ENABLED.store(if enabled { 1 } else { 0 }, Ordering::Relaxed);
        enabled
    }

    /// GPU interpolation for data stored in bit-reversed row order (LDE layout).
    /// Bit-reverses the weights to match the bit-reversed data storage.
    pub fn try_gpu_interpolate_coset_bitrev<F2, EF2>(
        mat_data: &[F2],
        height: usize,
        width: usize,
        shift: F2,
        point: EF2,
    ) -> Option<Vec<EF2>>
    where
        F2: TwoAdicField,
        EF2: ExtensionField<F2> + TwoAdicField,
    {
        if core::mem::size_of::<F2>() != 4 || core::mem::size_of::<EF2>() != 16 {
            return None;
        }
        if !gpu_enabled() {
            return None;
        }
        // Disabled: most PCS interpolation matrices are narrow (1-14 cols).
        // GPU dispatch overhead per matrix exceeds benefit for narrow matrices.
        // Would need batched dispatch across all matrices to amortize overhead.
        return None;
        #[allow(unreachable_code)]
        if height * width < 262144 {
            return None;
        }

        use std::sync::OnceLock;
        static STATE: OnceLock<metal_ntt::device::MetalState> = OnceLock::new();
        let state = STATE.get_or_init(metal_ntt::device::MetalState::new);
        let log_h = log2_strict_usize(height);

        let g = F2::two_adic_generator(log_h);
        let diffs: Vec<EF2> = cyclic_subgroup_coset_known_order(g, shift, height)
            .map(|subgroup_i| point - subgroup_i)
            .collect();
        let diff_invs = batch_multiplicative_inverse(&diffs);
        let mut col_scale: Vec<EF2> = g
            .powers()
            .zip(diff_invs)
            .map(|(sg, diff_inv)| diff_inv * sg)
            .collect();

        // NOTE: The Metal shader already does bit-reversal on the row index
        // when reading col_scale (line: brp_row = reverse_bits(row) >> ...),
        // so col_scale must be passed in NATURAL order — no reversal here.

        let mat_u32: &[u32] = unsafe {
            core::slice::from_raw_parts(mat_data.as_ptr() as *const u32, mat_data.len())
        };
        let col_scale_u32: &[u32] = unsafe {
            core::slice::from_raw_parts(col_scale.as_ptr() as *const u32, col_scale.len() * 4)
        };

        let result_u32 = metal_ntt::dot_product::gpu_columnwise_dot_product(
            state,
            mat_u32,
            col_scale_u32,
            height,
            width,
        );

        let zerofier = two_adic_coset_zerofier::<EF2>(log_h, EF2::from_base(shift), point);
        let denominator = F2::from_canonical_usize(height) * shift.exp_u64(height as u64 - 1);
        let zerofier_scale = zerofier * denominator.inverse();

        let mut result: Vec<EF2> = unsafe {
            let mut v = core::mem::ManuallyDrop::new(result_u32);
            Vec::from_raw_parts(
                v.as_mut_ptr() as *mut EF2,
                width,
                v.capacity() / 4,
            )
        };

        for r in result.iter_mut() {
            *r = *r * zerofier_scale;
        }

        Some(result)
    }
}

use crate::verifier::{self, FriError};
use crate::{prover, FriConfig, FriProof};

#[derive(Debug)]
pub struct TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    // degree bound
    log_n: usize,
    dft: Dft,
    mmcs: InputMmcs,
    fri: FriConfig<FriMmcs>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(log_n: usize, dft: Dft, mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self {
            log_n,
            dft,
            mmcs,
            fri,
            _phantom: PhantomData,
        }
    }

    pub fn fri_config(&self) -> &FriConfig<FriMmcs> {
        &self.fri
    }
}

#[derive(Debug)]
pub enum VerificationError<InputMmcsError, FriMmcsError> {
    InputMmcsError(InputMmcsError),
    FriError(FriError<FriMmcsError>),
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct TwoAdicFriPcsProof<
    Val: Field,
    Challenge: Field,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
> {
    pub fri_proof: FriProof<Challenge, FriMmcs, Val>,
    /// For each query, for each committed batch, query openings for that batch
    pub query_openings: Vec<Vec<BatchOpening<Val, InputMmcs>>>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct BatchOpening<Val: Field, InputMmcs: Mmcs<Val>> {
    pub opened_values: Vec<Vec<Val>>,
    pub opening_proof: <InputMmcs as Mmcs<Val>>::Proof,
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val> + Sync,
    FriMmcs: Mmcs<Challenge> + Sync,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger: CanObserve<FriMmcs::Commitment>
        + CanSample<Challenge>
        + GrindingChallenger<Witness = Val>
        + FieldChallenger<Val>,
    <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>: Clone,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type Proof = TwoAdicFriPcsProof<Val, Challenge, InputMmcs, FriMmcs>;
    type Error = VerificationError<InputMmcs::Error, FriMmcs::Error>;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree);
        assert!(log_n <= self.log_n);
        TwoAdicMultiplicativeCoset {
            log_n,
            shift: Val::one(),
        }
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        // Prepare inputs for batch DFT — validates domains and computes shifts
        let inputs: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let log_n = log2_strict_usize(domain.size());
                assert!(log_n <= self.log_n);
                let shift = Val::generator() / domain.shift;
                (evals, self.fri.log_blowup, shift)
            })
            .collect();

        // Batch DFT: implementations may process all matrices on GPU in one lock acquisition
        let t_dft = std::time::Instant::now();
        let ldes: Vec<_> = self.dft
            .coset_lde_batch_multi(inputs)
            .into_iter()
            .map(|m| m.bit_reverse_rows().to_row_major_matrix())
            .collect();
        let dft_ms = t_dft.elapsed().as_secs_f64() * 1000.0;

        let t_merkle = std::time::Instant::now();
        let result = self.mmcs.commit(ldes);
        let merkle_ms = t_merkle.elapsed().as_secs_f64() * 1000.0;
        if dft_ms + merkle_ms > 100.0 {
            tracing::info!("commit breakdown: dft={dft_ms:.0}ms merkle={merkle_ms:.0}ms total={:.0}ms", dft_ms + merkle_ms);
        }
        result
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        // todo: handle extrapolation for LDEs we don't have
        assert_eq!(domain.shift, Val::generator());
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        assert!(lde.height() >= domain.size());
        lde.split_rows(domain.size()).0.bit_reverse_rows()
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        /*

        A quick rundown of the optimizations in this function:
        We are trying to compute sum_i alpha^i * (p(X) - y)/(X - z),
        for each z an opening point, y = p(z). Each p(X) is given as evaluations in bit-reversed order
        in the columns of the matrices. y is computed by barycentric interpolation.
        X and p(X) are in the base field; alpha, y and z are in the extension.
        The primary goal is to minimize extension multiplications.

        - Instead of computing all alpha^i, we just compute alpha^i for i up to the largest width
        of a matrix, then multiply by an "alpha offset" when accumulating.
              a^0 x0 + a^1 x1 + a^2 x2 + a^3 x3 + ...
            = a^0 ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x0 + a^1 x1 ) + ...
            (see `alpha_pows`, `alpha_pow_offset`, `num_reduced`)

        - For each unique point z, we precompute 1/(X-z) for the largest subgroup opened at this point.
        Since we compute it in bit-reversed order, smaller subgroups can simply truncate the vector.
            (see `inv_denoms`)

        - Then, for each matrix (with columns p_i) and opening point z, we want:
            for each row (corresponding to subgroup element X):
                reduced[X] += alpha_offset * sum_i [ alpha^i * inv_denom[X] * (p_i[X] - y[i]) ]

            We can factor out inv_denom, and expand what's left:
                reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] - alpha^i * y[i] ]

            And separate the sum:
                reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] ] - sum_i [ alpha^i * y[i] ]

            And now the last sum doesn't depend on X, so we can precompute that for the matrix, too.
            So the hot loop (that depends on both X and i) is just:
                sum_i [ alpha^i * p_i[X] ]

            with alpha^i an extension, p_i[X] a base

        */

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                (
                    self.mmcs
                        .get_matrices(data)
                        .into_iter()
                        .map(|m| m.as_view())
                        .collect_vec(),
                    points,
                )
            })
            .collect_vec();
        let mats = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .collect_vec();

        let global_max_height = mats.iter().map(|m| m.height()).max().unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(z - X) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, Val::generator());

        // Evaluate coset representations: collect all (mat, point) jobs, run in parallel,
        // then observe to challenger in deterministic order.
        let all_opened_values = {
            let log_blowup = self.fri.log_blowup;

            // Collect all (mat_idx, point) jobs and structure info.
            let mut jobs: Vec<(usize, Challenge)> = Vec::new();
            let mut structure: Vec<Vec<usize>> = Vec::new();

            let all_mats: Vec<_> = mats_and_points
                .iter()
                .flat_map(|(mats, _)| mats.iter())
                .collect();

            let mut flat_idx = 0;
            for (mats, points) in mats_and_points.iter() {
                let mut round_structure = Vec::new();
                for (_, points_for_mat) in izip!(mats.iter(), points.iter()) {
                    round_structure.push(points_for_mat.len());
                    for &point in points_for_mat.iter() {
                        jobs.push((flat_idx, point));
                    }
                    flat_idx += 1;
                }
                structure.push(round_structure);
            }

            // Run all interpolations in parallel with 2-point fusion when possible.
            // inv_denoms[point][j] = 1/(x_{bitrev(j)} - z) for full coset in bitrev order.
            // For interpolation we need: diff_inv[i] = 1/(z - x_i) = -inv_denoms[bitrev(i, log_h)]
            //
            // Group by matrix: matrices with exactly 2 points (the common case in SP1)
            // use a fused kernel that reads the matrix once for both points.

            // Collect per-matrix job groups: (mat_idx, [points])
            let mut mat_jobs: Vec<(usize, Vec<Challenge>)> = Vec::new();
            let mut job_to_mat: Vec<usize> = Vec::new(); // maps flat job index → mat_jobs index
            {
                let mut flat_idx = 0;
                for (mats, points) in mats_and_points.iter() {
                    for (_, points_for_mat) in izip!(mats.iter(), points.iter()) {
                        let mj_idx = mat_jobs.len();
                        mat_jobs.push((flat_idx, points_for_mat.clone()));
                        for _ in points_for_mat.iter() {
                            job_to_mat.push(mj_idx);
                        }
                        flat_idx += 1;
                    }
                }
            }

            // Run per-matrix interpolations in parallel.
            let mat_results: Vec<Vec<Vec<Challenge>>> = mat_jobs
                .par_iter()
                .map(|(mat_idx, points)| {
                    let mat = all_mats[*mat_idx];
                    let h = mat.height() >> log_blowup;
                    let log_h = log2_strict_usize(h);
                    let (low_coset, _) = mat.split_rows(h);
                    let brp_view = BitReversalPerm::new_view(low_coset);

                    let make_diff_invs = |point: Challenge| -> Vec<Challenge> {
                        let full_inv = inv_denoms.get(&point).unwrap();
                        (0..h).map(|i| -full_inv[reverse_bits_len(i, log_h)]).collect()
                    };

                    if points.len() == 2 {
                        // Fused 2-point: read matrix once for both opening points.
                        let diff_invs0 = make_diff_invs(points[0]);
                        let diff_invs1 = make_diff_invs(points[1]);
                        let (ys0, ys1) = interpolate_coset_precomputed_2point(
                            &brp_view,
                            Val::generator(),
                            points[0], &diff_invs0,
                            points[1], &diff_invs1,
                        );
                        vec![ys0, ys1]
                    } else {
                        // General path for 1 or 3+ points.
                        points.iter().map(|&point| {
                            let diff_invs = make_diff_invs(point);
                            interpolate_coset_precomputed(
                                &brp_view,
                                Val::generator(),
                                point,
                                &diff_invs,
                            )
                        }).collect()
                    }
                })
                .collect();

            // Flatten results in original job order.
            let results: Vec<Vec<Challenge>> = {
                let mut flat = Vec::with_capacity(jobs.len());
                let mut point_offsets: Vec<usize> = vec![0; mat_jobs.len()];
                for &mj_idx in &job_to_mat {
                    let off = point_offsets[mj_idx];
                    flat.push(mat_results[mj_idx][off].clone());
                    point_offsets[mj_idx] += 1;
                }
                flat
            };

            // Reconstruct nested structure and observe in order.
            let mut result_iter = results.into_iter();
            let mut all_opened = Vec::new();
            for round_structure in &structure {
                let mut round_opened = Vec::new();
                for &num_points in round_structure {
                    let mut mat_opened = Vec::new();
                    for _ in 0..num_points {
                        let ys = result_iter.next().unwrap();
                        ys.iter().for_each(|&y| challenger.observe_ext_element(y));
                        mat_opened.push(ys);
                    }
                    round_opened.push(mat_opened);
                }
                all_opened.push(round_opened);
            }
            all_opened
        };
        // Batch combination challenge.
        let alpha: Challenge = challenger.sample_ext_element();

        let global_max_width = mats.iter().map(|m| m.width()).max().unwrap();
        let alpha_reducer = PowersReducer::<Val, Challenge>::new(alpha, global_max_width);

        let mut num_reduced = [0; 32];
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);

        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                let _guard =
                    info_span!("reduce matrix quotient", dims = %mat.dimensions()).entered();

                let log_height = log2_strict_usize(mat.height());
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![Challenge::zero(); mat.height()]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height());

                // Precompute reduce_base(row) once per row — reused across all
                // opening points for this matrix, saving one full O(height×width)
                // pass per extra point (SP1 always opens at 2 points).
                let row_sums: Vec<Challenge> = mat
                    .par_row_slices()
                    .map(|row| alpha_reducer.reduce_base(row))
                    .collect();

                if points_for_mat.len() == 2 {
                    // Fused 2-point accumulation: iterate over reduced_opening once
                    // instead of twice, halving cache misses on the large output vector.
                    let alpha_offset0 = alpha.exp_u64(num_reduced[log_height] as u64);
                    let y0 = alpha_reducer.reduce_ext(&openings_for_mat[0]);
                    let inv0 = inv_denoms.get(&points_for_mat[0]).unwrap();

                    num_reduced[log_height] += mat.width();

                    let alpha_offset1 = alpha.exp_u64(num_reduced[log_height] as u64);
                    let y1 = alpha_reducer.reduce_ext(&openings_for_mat[1]);
                    let inv1 = inv_denoms.get(&points_for_mat[1]).unwrap();

                    num_reduced[log_height] += mat.width();

                    reduced_opening_for_log_height
                        .par_iter_mut()
                        .zip(row_sums.par_iter())
                        .zip(inv0[..mat.height()].par_iter())
                        .zip(inv1[..mat.height()].par_iter())
                        .for_each(|(((reduced, &row_sum), &d0), &d1)| {
                            *reduced += (d0 * alpha_offset0) * (row_sum - y0)
                                      + (d1 * alpha_offset1) * (row_sum - y1);
                        });
                } else {
                    for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                        let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                        let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(openings);

                        let raw_inv_denoms = inv_denoms.get(&point).unwrap();

                        reduced_opening_for_log_height
                            .par_iter_mut()
                            .zip(row_sums.par_iter())
                            .zip(raw_inv_denoms[..mat.height()].par_iter())
                            .for_each(|((reduced_opening, &row_sum), &inv_denom)| {
                                *reduced_opening +=
                                    (inv_denom * alpha_pow_offset) * (row_sum - sum_alpha_pows_times_y);
                            });

                        num_reduced[log_height] += mat.width();
                    }
                }
            }
        }

        let (fri_proof, query_indices) = prover::prove(&self.fri, &reduced_openings, challenger);

        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                rounds
                    .iter()
                    .map(|(data, _)| {
                        let log_max_height = log2_strict_usize(self.mmcs.get_max_height(data));
                        let bits_reduced = log_global_max_height - log_max_height;
                        let reduced_index = index >> bits_reduced;
                        let (opened_values, opening_proof) =
                            self.mmcs.open_batch(reduced_index, data);
                        BatchOpening {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect()
            })
            .collect();

        (
            all_opened_values,
            TwoAdicFriPcsProof {
                fri_proof,
                query_openings,
            },
        )
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Write evaluations to challenger.
        for (_, round) in rounds.iter() {
            for (_, mat) in round.iter() {
                for (_, point) in mat.iter() {
                    point
                        .iter()
                        .for_each(|&opening| challenger.observe_ext_element(opening));
                }
            }
        }
        // Batch combination challenge
        let alpha: Challenge = challenger.sample();

        let fri_challenges =
            verifier::verify_shape_and_sample_challenges(&self.fri, &proof.fri_proof, challenger)
                .map_err(VerificationError::FriError)?;

        let log_global_max_height =
            proof.fri_proof.commit_phase_commits.len() + self.fri.log_blowup;

        let reduced_openings: Vec<[Challenge; 32]> = proof
            .query_openings
            .iter()
            .zip(&fri_challenges.query_indices)
            .map(|(query_opening, &index)| {
                let mut ro = [Challenge::zero(); 32];
                let mut alpha_pow = [Challenge::one(); 32];

                for (batch_opening, (batch_commit, mats)) in izip!(query_opening, &rounds) {
                    let batch_heights = mats
                        .iter()
                        .map(|(domain, _)| domain.size() << self.fri.log_blowup)
                        .collect_vec();
                    let batch_dims = batch_heights
                        .iter()
                        // TODO: MMCS doesn't really need width; we put 0 for now.
                        .map(|&height| Dimensions { width: 0, height })
                        .collect_vec();

                    let batch_max_height = batch_heights.iter().max().expect("Empty batch?");
                    let log_batch_max_height = log2_strict_usize(*batch_max_height);
                    let bits_reduced = log_global_max_height - log_batch_max_height;
                    let reduced_index = index >> bits_reduced;

                    self.mmcs.verify_batch(
                        batch_commit,
                        &batch_dims,
                        reduced_index,
                        &batch_opening.opened_values,
                        &batch_opening.opening_proof,
                    )?;
                    for (mat_opening, (mat_domain, mat_points_and_values)) in
                        izip!(&batch_opening.opened_values, mats)
                    {
                        let log_height = log2_strict_usize(mat_domain.size()) + self.fri.log_blowup;

                        let bits_reduced = log_global_max_height - log_height;
                        let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

                        let x = Val::generator()
                            * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

                        for (z, ps_at_z) in mat_points_and_values {
                            for (&p_at_x, &p_at_z) in izip!(mat_opening, ps_at_z) {
                                let quotient = (-p_at_z + p_at_x) / (-*z + x);
                                ro[log_height] += alpha_pow[log_height] * quotient;
                                alpha_pow[log_height] *= alpha;
                            }
                        }
                    }
                }
                Ok(ro)
            })
            .collect::<Result<Vec<_>, InputMmcs::Error>>()
            .map_err(VerificationError::InputMmcsError)?;

        verifier::verify_challenges(
            &self.fri,
            &proof.fri_proof,
            &fri_challenges,
            &reduced_openings,
        )
        .map_err(VerificationError::FriError)?;

        Ok(())
    }
}

/// Per-round metadata for `open_sequential`: original evaluations for LDE recomputation.
/// If `None`, the ProverData's existing leaves are used. If `Some`, LDE is recomputed
/// on demand and leaves are temporarily restored in the ProverData.
pub type RoundOriginals<Val> = Option<Vec<(TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>)>>;

/// Trait for PCS implementations that support memory-efficient sequential opening.
pub trait PcsOpenSequential<Val: TwoAdicField, Challenge: ExtensionField<Val>, Challenger> {
    /// The proof type produced.
    type Proof;
    /// The prover data type.
    type ProverData;

    /// Memory-efficient open that processes rounds sequentially.
    /// For rounds where `round_originals[i]` is `Some(evals)`, the LDE is recomputed
    /// from the raw evaluations, used, then freed. At most ONE round's LDE is live at a time.
    fn open_sequential(
        &self,
        rounds: &mut [(&mut Self::ProverData, Vec<Vec<Challenge>>)],
        round_originals: &mut [RoundOriginals<Val>],
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof);

    /// Memory-efficient open using pre-cached LDE leaves.
    /// `saved_ldes[i]` is `Some(ldes)` for rounds whose leaves were taken from commit data,
    /// `None` for rounds that still have their leaves.
    fn open_sequential_cached(
        &self,
        rounds: &mut [(&mut Self::ProverData, Vec<Vec<Challenge>>)],
        saved_ldes: Vec<Option<Vec<RowMajorMatrix<Val>>>>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof);

    /// Like `open_sequential_cached` but accepts a separate read-only preprocessed round,
    /// avoiding the need to clone preprocessed prover data.
    fn open_sequential_cached_split(
        &self,
        preprocessed_data: &Self::ProverData,
        preprocessed_points: Vec<Vec<Challenge>>,
        rounds: &mut [(&mut Self::ProverData, Vec<Vec<Challenge>>)],
        saved_ldes: Vec<Option<Vec<RowMajorMatrix<Val>>>>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof);
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
{
    /// Recompute LDE matrices from original evaluations, taking ownership to avoid cloning.
    pub fn recompute_lde_owned(
        &self,
        original_evals: Vec<(TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>)>,
    ) -> Vec<RowMajorMatrix<Val>> {
        original_evals
            .into_iter()
            .map(|(domain, evals)| {
                let shift = Val::generator() / domain.shift;
                self.dft
                    .coset_lde_batch(evals, self.fri.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect()
    }

    /// Access the underlying InputMmcs.
    pub fn mmcs(&self) -> &InputMmcs {
        &self.mmcs
    }

    /// Access the FRI config.
    pub fn fri(&self) -> &FriConfig<FriMmcs> {
        &self.fri
    }

    /// Access the DFT.
    pub fn dft(&self) -> &Dft {
        &self.dft
    }

    /// The log blowup factor.
    pub fn log_blowup(&self) -> usize {
        self.fri.log_blowup
    }

    /// Memory-efficient version of `open` using pre-cached LDE leaves.
    ///
    /// For rounds where `saved_ldes[i]` is `Some(ldes)`, the LDE leaves were taken from the
    /// ProverData after commit. They are restored one round at a time: restore → use → take back,
    /// so at most ONE extra round's LDE is in memory at a time.
    ///
    /// For rounds where `saved_ldes[i]` is `None`, the ProverData's existing leaves are used.
    ///
    /// Produces the same proof as `open()`.
    #[allow(clippy::too_many_lines)]
    /// Memory-efficient version of `open` using pre-cached LDE leaves.
    ///
    /// `saved_ldes[i]` is `Some(ldes)` for rounds whose leaves were taken from commit data,
    /// `None` for rounds that still have their leaves. LDEs are restored and freed one round
    /// at a time, keeping peak memory low.
    ///
    /// Produces the same proof as `open()`.
    #[allow(clippy::too_many_lines)]
    pub fn open_sequential_cached<Challenge, Challenger>(
        &self,
        rounds: &mut [(
            &mut <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>,
            Vec<Vec<Challenge>>,
        )],
        mut saved_ldes: Vec<Option<Vec<RowMajorMatrix<Val>>>>,
        challenger: &mut Challenger,
    ) -> (
        OpenedValues<Challenge>,
        TwoAdicFriPcsProof<Val, Challenge, InputMmcs, FriMmcs>,
    )
    where
        Challenge: TwoAdicField + ExtensionField<Val>,
        FriMmcs: Mmcs<Challenge>,
        Challenger: CanObserve<FriMmcs::Commitment>
            + CanSample<Challenge>
            + GrindingChallenger<Witness = Val>
            + FieldChallenger<Val>,
        <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>: Clone + LeafManageable<Val>,
    {
        assert_eq!(rounds.len(), saved_ldes.len());

        // Phase 0: Compute dimensions. Temporarily restore LDEs to read dimensions, then take back.
        let mut global_max_height: usize = 0;
        let mut global_max_width: usize = 0;
        let round_dims: Vec<Vec<Dimensions>> = rounds
            .iter_mut()
            .zip(saved_ldes.iter_mut())
            .map(|((data, _), saved)| {
                let needs_restore = saved.is_some();
                if let Some(ldes) = saved.take() {
                    data.restore_leaves_rm(ldes);
                }
                let dims: Vec<Dimensions> = self.mmcs
                    .get_matrices(*data)
                    .iter()
                    .map(|m| {
                        let d = m.dimensions();
                        global_max_height = global_max_height.max(d.height);
                        global_max_width = global_max_width.max(d.width);
                        d
                    })
                    .collect();
                if needs_restore {
                    *saved = Some(data.take_leaves_rm());
                }
                dims
            })
            .collect();

        let log_global_max_height = log2_strict_usize(global_max_height);

        // Pre-compute inv_denoms from dimensions
        let all_dims_and_points: Vec<(Vec<Dimensions>, &Vec<Vec<Challenge>>)> = round_dims
            .iter()
            .zip(rounds.iter())
            .map(|(dims, (_, points))| (dims.clone(), points))
            .collect();
        let inv_denoms = compute_inverse_denominators_from_dims(&all_dims_and_points, self.fri.log_blowup, Val::generator());

        let mut all_opened_values: OpenedValues<Challenge> = Vec::new();

        // Phase 1: Interpolation — restore LDE one round at a time, compute opened values, take back.
        let log_blowup = self.fri.log_blowup;
        for (ri, (data, points)) in rounds.iter_mut().enumerate() {
            let has_saved = saved_ldes[ri].is_some();
            if let Some(ldes) = saved_ldes[ri].take() {
                data.restore_leaves_rm(ldes);
            }

            let mats: Vec<_> = self.mmcs.get_matrices(*data)
                .into_iter().map(|m| m.as_view()).collect_vec();

            let round_opened_values: Vec<Vec<Vec<Challenge>>> = mats.par_iter()
                .zip(points.par_iter())
                .map(|(mat, points_for_mat)| {
                    points_for_mat.iter().map(|&point| {
                        let h = mat.height() >> log_blowup;
                        let log_h = log2_strict_usize(h);
                        let (low_coset, _) = mat.split_rows(h);
                        let full_inv = inv_denoms.get(&point).unwrap();
                        let diff_invs: Vec<Challenge> = (0..h)
                            .map(|i| -full_inv[reverse_bits_len(i, log_h)])
                            .collect();
                        interpolate_coset_precomputed(
                            &BitReversalPerm::new_view(low_coset),
                            Val::generator(), point, &diff_invs,
                        )
                    }).collect_vec()
                }).collect();

            for mat_values in &round_opened_values {
                for ys in mat_values {
                    ys.iter().for_each(|&y| challenger.observe_ext_element(y));
                }
            }
            all_opened_values.push(round_opened_values);

            if has_saved {
                saved_ldes[ri] = Some(data.take_leaves_rm());
            }
        }

        // Sample alpha AFTER all opened values are observed
        let alpha: Challenge = challenger.sample_ext_element();
        let alpha_reducer = PowersReducer::<Val, Challenge>::new(alpha, global_max_width);

        let mut num_reduced = [0usize; 32];
        let mut reduced_openings: [Option<Vec<Challenge>>; 32] = core::array::from_fn(|_| None);

        // Phase 2: Row reduction — restore cached LDEs, reduce, keep loaded for query phase.
        for (ri, (data, points)) in rounds.iter_mut().enumerate() {
            if let Some(ldes) = saved_ldes[ri].take() {
                data.restore_leaves_rm(ldes);
            }

            let mats: Vec<_> = self.mmcs.get_matrices(*data)
                .into_iter().map(|m| m.as_view()).collect_vec();

            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), all_opened_values[ri].iter())
            {
                let log_height = log2_strict_usize(mat.height());
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![Challenge::zero(); mat.height()]);

                for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                    let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(openings);

                    let raw_inv_denoms = inv_denoms.get(&point).unwrap();
                    let scaled_inv_denoms: Vec<Challenge> = raw_inv_denoms[..mat.height()]
                        .par_iter()
                        .map(|&inv_denom| inv_denom * alpha_pow_offset)
                        .collect();

                    reduced_opening_for_log_height
                        .par_iter_mut()
                        .zip_eq(mat.par_row_slices())
                        .zip(scaled_inv_denoms.par_iter())
                        .for_each(|((reduced_opening, row), &scaled_inv)| {
                            let row_sum = alpha_reducer.reduce_base(row);
                            *reduced_opening += scaled_inv * (row_sum - sum_alpha_pows_times_y);
                        });

                    num_reduced[log_height] += mat.width();
                }
            }
            // Keep LDEs loaded for query phase
        }

        // FRI prove
        let (fri_proof, query_indices) =
            prover::prove(&self.fri, &reduced_openings, challenger);

        // Phase 3: Query openings
        let num_queries = query_indices.len();
        let num_rounds = rounds.len();
        let mut all_query_openings: Vec<Vec<BatchOpening<Val, InputMmcs>>> =
            (0..num_queries).map(|_| Vec::with_capacity(num_rounds)).collect();

        for (data, _) in rounds.iter_mut() {
            let log_max_height = log2_strict_usize(self.mmcs.get_max_height(*data));
            let bits_reduced = log_global_max_height - log_max_height;

            for (qi, &index) in query_indices.iter().enumerate() {
                let reduced_index = index >> bits_reduced;
                let (opened_values, opening_proof) = self.mmcs.open_batch(reduced_index, *data);
                all_query_openings[qi].push(BatchOpening { opened_values, opening_proof });
            }
        }

        (
            all_opened_values,
            TwoAdicFriPcsProof { fri_proof, query_openings: all_query_openings },
        )
    }

    /// Like `open_sequential_cached` but the preprocessed round is passed as a read-only
    /// reference, avoiding a clone of the (potentially large) preprocessed prover data.
    #[allow(clippy::too_many_lines)]
    /// Thin wrapper around the proven-correct `open` function.
    /// Restores all saved LDEs, then delegates to `open`.
    pub fn open_sequential_cached_split<Challenge, Challenger>(
        &self,
        preprocessed_data: &<InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>,
        preprocessed_points: Vec<Vec<Challenge>>,
        rounds: &mut [(
            &mut <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>,
            Vec<Vec<Challenge>>,
        )],
        mut saved_ldes: Vec<Option<Vec<RowMajorMatrix<Val>>>>,
        challenger: &mut Challenger,
    ) -> (
        OpenedValues<Challenge>,
        TwoAdicFriPcsProof<Val, Challenge, InputMmcs, FriMmcs>,
    )
    where
        Challenge: TwoAdicField + ExtensionField<Val>,
        Dft: Sync,
        InputMmcs: Sync,
        FriMmcs: Mmcs<Challenge> + Sync,
        Challenger: CanObserve<FriMmcs::Commitment>
            + CanSample<Challenge>
            + GrindingChallenger<Witness = Val>
            + FieldChallenger<Val>,
        <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>: Clone + LeafManageable<Val>,
    {
        // Restore all LDEs before delegating to the proven-correct `open` function.
        for (ri, (data, _)) in rounds.iter_mut().enumerate() {
            if let Some(ldes) = saved_ldes[ri].take() {
                data.restore_leaves_rm(ldes);
            }
        }

        // Build the rounds vec in the format `open` expects:
        // [preprocessed, main, perm, quotient]
        let mut open_rounds: Vec<(
            &<InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>,
            Vec<Vec<Challenge>>,
        )> = Vec::with_capacity(1 + rounds.len());
        open_rounds.push((preprocessed_data, preprocessed_points));
        for (data, points) in rounds.iter() {
            open_rounds.push((*data, points.clone()));
        }

        self.open(open_rounds, challenger)
    }
}

// Implement the PcsOpenSequential trait for TwoAdicFriPcs, delegating to the inherent method.
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger>
    PcsOpenSequential<Val, Challenge, Challenger> for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val> + Sync,
    FriMmcs: Mmcs<Challenge> + Sync,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger: CanObserve<FriMmcs::Commitment>
        + CanSample<Challenge>
        + GrindingChallenger<Witness = Val>
        + FieldChallenger<Val>,
    <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>: Clone + LeafManageable<Val>,
{
    type Proof = TwoAdicFriPcsProof<Val, Challenge, InputMmcs, FriMmcs>;
    type ProverData = <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>;

    fn open_sequential(
        &self,
        rounds: &mut [(&mut Self::ProverData, Vec<Vec<Challenge>>)],
        round_originals: &mut [RoundOriginals<Val>],
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        TwoAdicFriPcs::open_sequential(self, rounds, round_originals, challenger)
    }

    fn open_sequential_cached(
        &self,
        rounds: &mut [(&mut Self::ProverData, Vec<Vec<Challenge>>)],
        saved_ldes: Vec<Option<Vec<RowMajorMatrix<Val>>>>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        TwoAdicFriPcs::open_sequential_cached(self, rounds, saved_ldes, challenger)
    }

    fn open_sequential_cached_split(
        &self,
        preprocessed_data: &Self::ProverData,
        preprocessed_points: Vec<Vec<Challenge>>,
        rounds: &mut [(&mut Self::ProverData, Vec<Vec<Challenge>>)],
        saved_ldes: Vec<Option<Vec<RowMajorMatrix<Val>>>>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        TwoAdicFriPcs::open_sequential_cached_split(self, preprocessed_data, preprocessed_points, rounds, saved_ldes, challenger)
    }
}

/// Like `compute_inverse_denominators` but works from dimensions + blowup, not full matrices.
/// Used by `open_sequential` when leaf matrices may not be present.
#[instrument(skip_all)]
pub fn compute_inverse_denominators_from_dims<F: TwoAdicField, EF: ExtensionField<F>>(
    dims_and_points: &[(Vec<Dimensions>, &Vec<Vec<EF>>)],
    log_blowup: usize,
    coset_shift: F,
) -> LinearMap<EF, Vec<EF>> {
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (dims, points) in dims_and_points {
        for (dim, points_for_mat) in izip!(dims, *points) {
            let log_height = log2_strict_usize(dim.height);
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    let max_log_height = *max_log_height_for_point.values().max().unwrap();
    let mut subgroup = cyclic_subgroup_coset_known_order(
        F::two_adic_generator(max_log_height),
        coset_shift,
        1 << max_log_height,
    )
    .collect_vec();
    reverse_slice_index_bits(&mut subgroup);

    let entries: Vec<_> = max_log_height_for_point.into_iter().collect();
    let results: Vec<_> = entries
        .into_par_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    &subgroup[..(1 << log_height)]
                        .iter()
                        .map(|&x| EF::from_base(x) - z)
                        .collect_vec(),
                ),
            )
        })
        .collect();
    let mut map = LinearMap::new();
    for (z, inv) in results {
        map.insert(z, inv);
    }
    map
}

#[instrument(skip_all)]
pub fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    coset_shift: F,
) -> LinearMap<EF, Vec<EF>> {
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height());
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    // Compute the largest subgroup we will use, in bitrev order.
    let max_log_height = *max_log_height_for_point.values().max().unwrap();
    let mut subgroup = cyclic_subgroup_coset_known_order(
        F::two_adic_generator(max_log_height),
        coset_shift,
        1 << max_log_height,
    )
    .collect_vec();
    reverse_slice_index_bits(&mut subgroup);

    let entries: Vec<_> = max_log_height_for_point.into_iter().collect();
    let results: Vec<_> = entries
        .into_par_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    &subgroup[..(1 << log_height)]
                        .iter()
                        .map(|&x| EF::from_base(x) - z)
                        .collect_vec(),
                ),
            )
        })
        .collect();
    let mut map = LinearMap::new();
    for (z, inv) in results {
        map.insert(z, inv);
    }
    map
}

pub struct PowersReducer<F: Field, EF> {
    pub powers: Vec<EF>,
    // If EF::D = 2 and powers is [01 23 45 67],
    // this holds [[02 46] [13 57]]
    pub transposed_packed: Vec<Vec<F::Packing>>,
}

impl<F: Field, EF: ExtensionField<F>> PowersReducer<F, EF> {
    pub fn new(base: EF, max_width: usize) -> Self {
        let powers: Vec<EF> = base
            .powers()
            .take(max_width.next_multiple_of(F::Packing::WIDTH))
            .collect();

        let transposed_packed: Vec<Vec<F::Packing>> = transpose_vec(
            (0..EF::D)
                .map(|d| {
                    F::Packing::pack_slice(
                        &powers.iter().map(|a| a.as_base_slice()[d]).collect_vec(),
                    )
                    .to_vec()
                })
                .collect(),
        );

        Self {
            powers,
            transposed_packed,
        }
    }

    // Compute sum_i base^i * x_i
    pub fn reduce_ext(&self, xs: &[EF]) -> EF {
        self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()
    }

    // Same as `self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()`
    pub fn reduce_base(&self, xs: &[F]) -> EF {
        let (xs_packed, xs_sfx) = F::Packing::pack_slice_with_suffix(xs);
        let mut sums = (0..EF::D).map(|_| F::Packing::zero()).collect::<Vec<_>>();
        for (&x, pows) in izip!(xs_packed, &self.transposed_packed) {
            for d in 0..EF::D {
                sums[d] += x * pows[d];
            }
        }
        let packed_sum = EF::from_base_fn(|d| sums[d].as_slice().iter().copied().sum());
        let sfx_sum = xs_sfx
            .iter()
            .zip(&self.powers[(xs_packed.len() * F::Packing::WIDTH)..])
            .map(|(&x, &pow)| pow * x)
            .sum::<EF>();
        packed_sum + sfx_sum
    }
}

fn transpose_vec<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

/// Like `interpolate_coset` but takes precomputed `diff_invs` to skip `batch_multiplicative_inverse`.
/// `diff_invs[i] = 1 / (point - shift * g^i)` in natural order over the coset.
fn interpolate_coset_precomputed<F, EF, Mat>(
    coset_evals: &Mat,
    shift: F,
    point: EF,
    diff_invs: &[EF],
) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Mat: Matrix<F>,
{
    let height = coset_evals.height();
    let log_height = log2_strict_usize(height);
    let g = F::two_adic_generator(log_height);

    let col_scale: Vec<_> = g
        .powers()
        .zip(diff_invs)
        .map(|(sg, &diff_inv)| diff_inv * sg)
        .collect();

    let sum = coset_evals.columnwise_dot_product(&col_scale);

    let zerofier = p3_field::two_adic_coset_zerofier::<EF>(log_height, EF::from_base(shift), point);
    let denominator = F::from_canonical_usize(height) * shift.exp_u64(height as u64 - 1);
    scale_vec(zerofier * denominator.inverse(), sum)
}

/// Two-point fused interpolation: reads the matrix once for both opening points.
/// Returns (ys_for_point0, ys_for_point1).
fn interpolate_coset_precomputed_2point<F, EF, Mat>(
    coset_evals: &Mat,
    shift: F,
    point0: EF,
    diff_invs0: &[EF],
    point1: EF,
    diff_invs1: &[EF],
) -> (Vec<EF>, Vec<EF>)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Mat: Matrix<F> + Sync,
{
    let height = coset_evals.height();
    let width = coset_evals.width();
    let log_height = log2_strict_usize(height);
    let g = F::two_adic_generator(log_height);

    // Precompute col_scale for both points.
    let col_scale0: Vec<_> = g
        .powers()
        .zip(diff_invs0)
        .map(|(sg, &diff_inv)| diff_inv * sg)
        .collect();
    let col_scale1: Vec<_> = g
        .powers()
        .zip(diff_invs1)
        .map(|(sg, &diff_inv)| diff_inv * sg)
        .collect();

    // Fused columnwise dot product: read each row once, accumulate for both points.
    use p3_maybe_rayon::prelude::*;
    let (sum0, sum1) = coset_evals
        .par_rows()
        .zip(col_scale0.par_iter())
        .zip(col_scale1.par_iter())
        .par_fold_reduce(
            || (vec![EF::zero(); width], vec![EF::zero(); width]),
            |(mut acc0, mut acc1), ((row, &s0), &s1)| {
                for (j, x) in row.into_iter().enumerate() {
                    acc0[j] += s0 * x;
                    acc1[j] += s1 * x;
                }
                (acc0, acc1)
            },
            |(mut a0, mut a1), (b0, b1)| {
                for (l, r) in a0.iter_mut().zip(b0) {
                    *l += r;
                }
                for (l, r) in a1.iter_mut().zip(b1) {
                    *l += r;
                }
                (a0, a1)
            },
        );

    let z0 = p3_field::two_adic_coset_zerofier::<EF>(log_height, EF::from_base(shift), point0);
    let z1 = p3_field::two_adic_coset_zerofier::<EF>(log_height, EF::from_base(shift), point1);
    let denominator = F::from_canonical_usize(height) * shift.exp_u64(height as u64 - 1);
    let denom_inv = denominator.inverse();
    let scale0 = z0 * denom_inv;
    let scale1 = z1 * denom_inv;

    (scale_vec(scale0, sum0), scale_vec(scale1, sum1))
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractExtensionField;
    use rand::{thread_rng, Rng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_powers_reducer() {
        let mut rng = thread_rng();
        let alpha: EF = rng.gen();
        let n = 1000;
        let sizes = [5, 110, 512, 999, 1000];
        let r = PowersReducer::<F, EF>::new(alpha, n);

        // check reduce_ext
        for size in sizes {
            let xs: Vec<EF> = (0..size).map(|_| rng.gen()).collect();
            assert_eq!(
                r.reduce_ext(&xs),
                xs.iter()
                    .enumerate()
                    .map(|(i, &x)| alpha.exp_u64(i as u64) * x)
                    .sum()
            );
        }

        // check reduce_base
        for size in sizes {
            let xs: Vec<F> = (0..size).map(|_| rng.gen()).collect();
            assert_eq!(
                r.reduce_base(&xs),
                xs.iter()
                    .enumerate()
                    .map(|(i, &x)| alpha.exp_u64(i as u64) * EF::from_base(x))
                    .sum()
            );
        }

        // bench reduce_base
        /*
        use core::hint::black_box;
        use std::time::Instant;
        let samples = 1_000;
        for i in 0..5 {
            let xs: Vec<F> = (0..999).map(|_| rng.gen()).collect();
            let t0 = Instant::now();
            for _ in 0..samples {
                black_box(r.reduce_base_slow(black_box(&xs)));
            }
            let dt_slow = t0.elapsed();
            let t0 = Instant::now();
            for _ in 0..samples {
                black_box(r.reduce_base(black_box(&xs)));
            }
            let dt_fast = t0.elapsed();
            println!("sample {i}: slow: {dt_slow:?} fast: {dt_fast:?}");
        }
        */
    }
}
