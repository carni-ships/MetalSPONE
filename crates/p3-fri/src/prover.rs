use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::{info_span, instrument};

use crate::fold_even_odd::{fold_even_odd, fold_from_committed};
use crate::{CommitPhaseProofStep, FriConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    challenger: &mut Challenger,
) -> (FriProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: GrindingChallenger + CanObserve<M::Commitment> + FieldChallenger<F>,
{
    let log_max_height = input.iter().rposition(Option::is_some).unwrap();

    let commit_phase_result = commit_phase(config, input, log_max_height, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let parallel_queries = std::env::var("PARALLEL_FOLD_QUERIES")
        .ok()
        .map_or(false, |v| v == "1");

    let query_proofs = info_span!("query phase").in_scope(|| {
        if parallel_queries {
            prove_query_phase_parallel(config, commit_phase_result, &query_indices)
        } else {
            prove_query_phase_sequential(config, commit_phase_result, &query_indices)
        }
    });

    (
        FriProof {
            commit_phase_commits: query_proofs.0,
            query_proofs: query_proofs.1,
            final_poly: query_proofs.2,
            pow_witness,
        },
        query_indices,
    )
}

/// Sequential query phase - works with any Mmcs implementation
/// Returns (commits, query_proofs, final_poly)
fn prove_query_phase_sequential<F, EF, M>(
    config: &FriConfig<M>,
    commit_phase_result: CommitPhaseResult<EF, M>,
    query_indices: &[usize],
) -> (Vec<M::Commitment>, Vec<QueryProof<EF, M>>, EF)
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
{
    let query_proofs: Vec<QueryProof<EF, M>> = query_indices
        .iter()
        .map(|&index| answer_query(config, &commit_phase_result.data, index))
        .collect();

    (
        commit_phase_result.commits,
        query_proofs,
        commit_phase_result.final_poly,
    )
}

/// Parallel query phase - requires M: Sync and M::ProverData: Sync
/// Only called when PARALLEL_FOLD_QUERIES=1 is set
/// Returns (commits, query_proofs, final_poly)
#[cfg(feature = "parallel_fold")]
fn prove_query_phase_parallel<F, EF, M>(
    config: &FriConfig<M>,
    commit_phase_result: CommitPhaseResult<EF, M>,
    query_indices: &[usize],
) -> (Vec<M::Commitment>, Vec<QueryProof<EF, M>>, EF)
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF> + Sync,
    M::ProverData<DenseMatrix<EF>>: Sync,
{
    let commits = commit_phase_result.commits.clone();
    let final_poly = commit_phase_result.final_poly;

    let query_proofs: Vec<QueryProof<EF, M>> = query_indices
        .par_iter()
        .map(|&index| answer_query(config, &commit_phase_result.data, index))
        .collect();

    (commits, query_proofs, final_poly)
}

/// Fallback for sequential when parallel_fold feature is not enabled
#[cfg(not(feature = "parallel_fold"))]
fn prove_query_phase_parallel<F, EF, M>(
    config: &FriConfig<M>,
    commit_phase_result: CommitPhaseResult<EF, M>,
    query_indices: &[usize],
) -> (Vec<M::Commitment>, Vec<QueryProof<EF, M>>, EF)
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
{
    // Just delegate to sequential
    prove_query_phase_sequential(config, commit_phase_result, query_indices)
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> QueryProof<F, M>
where
    F: Field,
    M: Mmcs<F>,
{
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index_pair, commit);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row[index_i_sibling % 2];

            CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            }
        })
        .collect();

    QueryProof {
        commit_phase_openings,
    }
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    log_max_height: usize,
    challenger: &mut Challenger,
) -> CommitPhaseResult<EF, M>
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: CanObserve<M::Commitment> + FieldChallenger<F>,
{
    let mut current = input[log_max_height].as_ref().unwrap().clone();

    let mut commits = vec![];
    let mut data = vec![];
    for log_folded_height in (config.log_blowup..log_max_height).rev() {
        // Give current to commit_matrix (no clone). Read back from ProverData for folding.
        let leaves = RowMajorMatrix::new(current, 2);
        let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);

        let beta: EF = challenger.sample_ext_element();

        // Fold from committed data — avoids cloning the full Vec before commit.
        let mats = config.mmcs.get_matrices(&prover_data);
        debug_assert_eq!(mats.len(), 1);
        current = fold_from_committed(mats[0], beta);

        data.push(prover_data);

        if let Some(v) = &input[log_folded_height] {
            current.iter_mut().zip_eq(v).for_each(|(c, v)| *c += *v);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(current.len(), config.blowup());
    let final_poly = current[0];
    for x in current {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: F,
}
