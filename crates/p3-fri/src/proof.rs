use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

/// SAFETY: CommitPhaseProofStep is Send if F: Send.
/// All actual Mmcs implementations used in SP1 (MetalMmcs, FieldMerkleTreeMmcs, ExtensionMmcs)
/// have Proof types that are Send (Vec<[F; N]> where F: Send). This impl assumes Proof: Send
/// which holds for all SP1 Mmcs implementations.
unsafe impl<F, M> Send for CommitPhaseProofStep<F, M>
where
    F: Field + Send,
    M: Mmcs<F>,
{
}

/// SAFETY: QueryProof is Send if F: Send.
/// All actual Mmcs implementations used in SP1 have Proof types that are Send.
unsafe impl<F, M> Send for QueryProof<F, M>
where
    F: Field + Send,
    M: Mmcs<F>,
{
}

/// SAFETY: FriProof is Send if F: Send and Witness: Send.
unsafe impl<F, M, Witness> Send for FriProof<F, M, Witness>
where
    F: Field + Send,
    M: Mmcs<F>,
    Witness: Send,
{
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize",
    deserialize = "Witness: Deserialize<'de>"
))]
pub struct FriProof<F: Field, M: Mmcs<F>, Witness> {
    pub commit_phase_commits: Vec<M::Commitment>,
    pub query_proofs: Vec<QueryProof<F, M>>,
    // This could become Vec<FC::Challenge> if this library was generalized to support non-constant
    // final polynomials.
    pub final_poly: F,
    pub pow_witness: Witness,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct QueryProof<F: Field, M: Mmcs<F>> {
    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    pub commit_phase_openings: Vec<CommitPhaseProofStep<F, M>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
// #[serde(bound(serialize = "F: Serialize"))]
#[serde(bound = "")]
pub struct CommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    /// The opening of the commit phase codeword at the sibling location.
    // This may change to Vec<FC::Challenge> if the library is generalized to support other FRI
    // folding arities besides 2, meaning that there can be multiple siblings.
    pub sibling_value: F,

    pub opening_proof: M::Proof,
}
