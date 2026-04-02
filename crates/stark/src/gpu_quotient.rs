//! GPU-accelerated quotient computation via Metal constraint evaluation.
//!
//! This module provides [`quotient_values_gpu`] as a drop-in replacement for
//! [`crate::quotient_values`] that compiles constraints to IR and dispatches
//! them on the GPU, one thread per quotient-domain row.
//!
//! Only available on macOS (where Metal is supported).

#![cfg(target_os = "macos")]

use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::gpu_ir::{compile_constraints, ConstraintCompiler};
use crate::septic_digest::SepticDigest;
use crate::Chip;

use metal_ntt::constraints::ChipDispatch;

type Challenge = BinomialExtensionField<BabyBear, 4>;

/// Convert a `&[BabyBear]` to `&[u32]` (zero-cost).
/// BabyBear is `#[repr(transparent)]` wrapping u32 in Montgomery form.
fn bb_as_u32(s: &[BabyBear]) -> &[u32] {
    // SAFETY: BabyBear is repr(transparent) wrapping u32.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u32, s.len()) }
}

/// Get the raw Montgomery-form u32 from a BabyBear value.
fn bb_to_raw(v: BabyBear) -> u32 {
    // SAFETY: BabyBear is repr(transparent) wrapping u32.
    unsafe { std::mem::transmute(v) }
}

/// Create a BabyBear from a raw Montgomery-form u32.
fn bb_from_raw(v: u32) -> BabyBear {
    // SAFETY: BabyBear is repr(transparent) wrapping u32.
    unsafe { std::mem::transmute(v) }
}

/// Prepare a GPU dispatch for a single chip's constraint evaluation.
/// Returns None if the chip isn't GPU-eligible (too few rows or too many registers).
/// The dispatch can then be encoded into a shared command buffer for batched execution.
#[allow(clippy::too_many_arguments)]
pub fn prepare_quotient_dispatch<A, D: PolynomialSpace<Val = BabyBear>>(
    chip: &Chip<BabyBear, A>,
    prep_width: usize,
    main_width_from_air: usize,
    commit_scope: crate::air::InteractionScope,
    local_cumulative_sum: &Challenge,
    global_cumulative_sum: &SepticDigest<BabyBear>,
    trace_domain: D,
    quotient_domain: D,
    preprocessed_trace_on_quotient_domain: &Option<RowMajorMatrix<BabyBear>>,
    main_trace_on_quotient_domain: &RowMajorMatrix<BabyBear>,
    permutation_trace_on_quotient_domain: &RowMajorMatrix<BabyBear>,
    perm_challenges: &[Challenge],
    alpha: Challenge,
    public_values: &[BabyBear],
    metal_state: &metal_ntt::device::MetalState,
) -> Option<ChipDispatch>
where
    A: for<'a> Air<ConstraintCompiler<'a>>,
{
    let quotient_size = quotient_domain.size();

    if quotient_size < 8192 {
        return None;
    }

    let main_width = main_trace_on_quotient_domain.width();
    let prep_width_trace = preprocessed_trace_on_quotient_domain
        .as_ref()
        .map_or(1, |m| m.width());
    let perm_width = permutation_trace_on_quotient_domain.width();

    let qdb = log2_strict_usize(quotient_size) - log2_strict_usize(trace_domain.size());
    let next_step = 1usize << qdb;

    // Compile constraint program
    let program = compile_constraints(
        chip,
        prep_width,
        main_width_from_air,
        commit_scope,
        local_cumulative_sum,
        global_cumulative_sum,
        perm_challenges,
        alpha,
        public_values,
    );

    tracing::debug!(
        "GPU constraint compile: quotient_size={} num_regs={} num_instr={}",
        quotient_size, program.num_regs, program.num_instructions()
    );

    if program.num_regs > 4096 {
        return None;
    }

    // Extract flat u32 trace data (zero-copy)
    let main_u32 = bb_as_u32(main_trace_on_quotient_domain.values.as_slice());

    let empty: Vec<u32> = vec![];
    let prep_u32 = preprocessed_trace_on_quotient_domain
        .as_ref()
        .map_or(empty.as_slice(), |prep| bb_as_u32(prep.values.as_slice()));

    let perm_u32 = bb_as_u32(permutation_trace_on_quotient_domain.values.as_slice());

    // Pack selectors in Montgomery form
    let sels = trace_domain.selectors_on_coset(quotient_domain);
    let mut selector_data = Vec::with_capacity(quotient_size * 4);
    for i in 0..quotient_size {
        selector_data.push(bb_to_raw(sels.is_first_row[i]));
        selector_data.push(bb_to_raw(sels.is_last_row[i]));
        selector_data.push(bb_to_raw(sels.is_transition[i]));
        selector_data.push(bb_to_raw(sels.inv_zeroifier[i]));
    }

    // Create GPU dispatch (copies trace data to Metal buffers)
    Some(ChipDispatch::new(
        metal_state,
        &program.words,
        program.num_regs,
        program.accumulator_regs,
        main_u32,
        main_width as u32,
        prep_u32,
        prep_width_trace as u32,
        perm_u32,
        perm_width as u32,
        &selector_data,
        quotient_size as u32,
        next_step as u32,
    ))
}

/// Convert a completed dispatch's results to Challenge (EF4) values.
pub fn dispatch_to_quotient_values(dispatch: &ChipDispatch) -> Vec<Challenge> {
    dispatch
        .read_results()
        .chunks_exact(4)
        .map(|chunk| Challenge::from_base_fn(|i| bb_from_raw(chunk[i])))
        .collect()
}

/// GPU-accelerated quotient values computation (single-chip convenience wrapper).
///
/// Drop-in replacement for [`crate::quotient_values`] specialized to
/// BabyBear + EF4. Compiles the chip's constraints to IR, dispatches
/// a Metal kernel over all quotient-domain rows, and returns EF4 quotient values.
#[allow(clippy::too_many_arguments)]
pub fn quotient_values_gpu<A, D: PolynomialSpace<Val = BabyBear>>(
    chip: &Chip<BabyBear, A>,
    chip_name: &str,
    prep_width: usize,
    main_width_from_air: usize,
    commit_scope: crate::air::InteractionScope,
    local_cumulative_sum: &Challenge,
    global_cumulative_sum: &SepticDigest<BabyBear>,
    trace_domain: D,
    quotient_domain: D,
    preprocessed_trace_on_quotient_domain: Option<RowMajorMatrix<BabyBear>>,
    main_trace_on_quotient_domain: RowMajorMatrix<BabyBear>,
    permutation_trace_on_quotient_domain: RowMajorMatrix<BabyBear>,
    perm_challenges: &[Challenge],
    alpha: Challenge,
    public_values: &[BabyBear],
    metal_state: &metal_ntt::device::MetalState,
) -> Option<Vec<Challenge>>
where
    A: for<'a> Air<ConstraintCompiler<'a>>,
{
    let _ = chip_name; // used for logging only
    let dispatch = prepare_quotient_dispatch(
        chip,
        prep_width,
        main_width_from_air,
        commit_scope,
        local_cumulative_sum,
        global_cumulative_sum,
        trace_domain,
        quotient_domain,
        &preprocessed_trace_on_quotient_domain,
        &main_trace_on_quotient_domain,
        &permutation_trace_on_quotient_domain,
        perm_challenges,
        alpha,
        public_values,
        metal_state,
    )?;

    // Single-chip dispatch
    let cmd = metal_state.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    dispatch.encode(metal_state, encoder);
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    Some(dispatch_to_quotient_values(&dispatch))
}
