#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use p3_matrix::dense::RowMajorMatrix;

mod merkle_tree;
mod mmcs;

pub use merkle_tree::*;
pub use mmcs::*;

/// Trait for MMCS ProverData that supports taking/restoring leaf matrices.
/// This enables memory-efficient proving by dropping LDE leaves between phases.
pub trait LeafManageable<F> {
    fn take_leaves_rm(&mut self) -> Vec<RowMajorMatrix<F>>;
    fn restore_leaves_rm(&mut self, leaves: Vec<RowMajorMatrix<F>>);
    fn has_leaves_rm(&self) -> bool;
    fn max_height_rm(&self) -> usize;
}

impl<F: Clone + Send + Sync, W: Clone, const DIGEST_ELEMS: usize>
    LeafManageable<F> for FieldMerkleTree<F, W, RowMajorMatrix<F>, DIGEST_ELEMS>
{
    fn take_leaves_rm(&mut self) -> Vec<RowMajorMatrix<F>> {
        self.take_leaves()
    }
    fn restore_leaves_rm(&mut self, leaves: Vec<RowMajorMatrix<F>>) {
        self.restore_leaves(leaves);
    }
    fn has_leaves_rm(&self) -> bool {
        self.has_leaves()
    }
    fn max_height_rm(&self) -> usize {
        self.max_height()
    }
}
