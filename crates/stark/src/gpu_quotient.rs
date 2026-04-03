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
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::gpu_ir::{compile_constraints, ConstraintCompiler};
use crate::septic_digest::SepticDigest;
use crate::Chip;

use metal_ntt::constraints::ChipDispatch;

type Challenge = BinomialExtensionField<BabyBear, 4>;
type InnerPcs = crate::bb31_poseidon2::InnerPcs;

/// Get raw LDE slices from PCS prover data (bit-reversed row order).
///
/// SAFETY: Caller must verify that `SC::Pcs` has the same layout as `InnerPcs`.
/// This is guaranteed when `Val<SC>` is BabyBear (verified by size check in caller).
pub unsafe fn get_lde_slices_from_pcs<'a, SC: crate::StarkGenericConfig>(
    pcs: &SC::Pcs,
    prover_data: &'a crate::PcsProverData<SC>,
    quotient_sizes: &[usize],
) -> Vec<(&'a [BabyBear], usize)> {
    let concrete_pcs: &InnerPcs = &*(pcs as *const SC::Pcs as *const InnerPcs);
    let concrete_data = &*(prover_data as *const crate::PcsProverData<SC>
        as *const <InnerPcs as p3_commit::Pcs<Challenge, crate::bb31_poseidon2::InnerChallenger>>::ProverData);
    concrete_pcs.get_lde_slices(concrete_data, quotient_sizes)
}

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

    let tier = if program.num_regs <= 256 { 256 }
        else if program.num_regs <= 512 { 512 }
        else if program.num_regs <= 1024 { 1024 }
        else if program.num_regs <= 2048 { 2048 }
        else { 4096 };
    tracing::debug!(
        "GPU constraint compile: quotient_size={} num_regs={} tier={} num_instr={}",
        quotient_size, program.num_regs, tier, program.num_instructions()
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

    // Create GPU dispatch (zero-copy for page-aligned trace data)
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

/// Compute selector data for a (trace_domain, quotient_domain) pair.
///
/// Returns packed u32 selector data in Montgomery form: [is_first, is_last, is_transition, inv_zero]
/// interleaved per quotient-domain row.
pub fn compute_selector_data<D: PolynomialSpace<Val = BabyBear>>(
    trace_domain: D,
    quotient_domain: D,
) -> Vec<u32> {
    let quotient_size = quotient_domain.size();
    let sels = trace_domain.selectors_on_coset(quotient_domain);
    let mut selector_data = Vec::with_capacity(quotient_size * 4);
    for i in 0..quotient_size {
        selector_data.push(bb_to_raw(sels.is_first_row[i]));
        selector_data.push(bb_to_raw(sels.is_last_row[i]));
        selector_data.push(bb_to_raw(sels.is_transition[i]));
        selector_data.push(bb_to_raw(sels.inv_zeroifier[i]));
    }
    selector_data
}

/// Prepare a GPU dispatch using raw LDE data in bit-reversed row order.
///
/// Avoids materializing the quotient-domain evaluation entirely: the GPU kernel
/// does bit-reversal indexing to read trace rows directly from the stored LDE.
/// Saves hundreds of MB of allocation + copy per large chip.
///
/// `cached_selectors`: if provided, uses pre-computed selector data instead of
/// computing selectors from scratch. Use `compute_selector_data` to precompute.
///
/// Returns None if the chip isn't GPU-eligible or LDE sizes don't match quotient domain.
#[allow(clippy::too_many_arguments)]
pub fn prepare_quotient_dispatch_bitrev<A, D: PolynomialSpace<Val = BabyBear>>(
    chip: &Chip<BabyBear, A>,
    prep_width: usize,
    main_width_from_air: usize,
    commit_scope: crate::air::InteractionScope,
    local_cumulative_sum: &Challenge,
    global_cumulative_sum: &SepticDigest<BabyBear>,
    trace_domain: D,
    quotient_domain: D,
    // Raw LDE data in bit-reversed row order (from MMCS prover_data)
    main_lde: &[BabyBear],   // height × width u32s
    main_lde_width: usize,
    prep_lde: Option<&[BabyBear]>,  // height × width u32s, or None
    prep_lde_width: usize,
    perm_lde: &[BabyBear],   // height × width u32s
    perm_lde_width: usize,
    perm_challenges: &[Challenge],
    alpha: Challenge,
    public_values: &[BabyBear],
    cached_selectors: Option<&[u32]>,
    metal_state: &metal_ntt::device::MetalState,
) -> Option<ChipDispatch>
where
    A: for<'a> Air<ConstraintCompiler<'a>>,
{
    let quotient_size = quotient_domain.size();

    if quotient_size < 8192 {
        return None;
    }

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

    if program.num_regs > 4096 {
        return None;
    }

    // Verify LDE heights match quotient domain (must be equal for bitrev to work).
    let main_lde_height = main_lde.len() / main_lde_width;
    let perm_lde_height = perm_lde.len() / perm_lde_width;
    if main_lde_height != quotient_size || perm_lde_height != quotient_size {
        tracing::debug!(
            "bitrev dispatch: LDE height mismatch (main={}, perm={}, quotient={}), falling back",
            main_lde_height, perm_lde_height, quotient_size
        );
        return None;
    }
    if let Some(prep) = prep_lde {
        let prep_lde_height = prep.len() / prep_lde_width;
        if prep_lde_height != quotient_size {
            return None;
        }
    }

    // Get raw u32 pointers to LDE data (zero-cost)
    let main_u32 = bb_as_u32(main_lde);
    let prep_u32_owned: Vec<u32>;
    let prep_u32 = match prep_lde {
        Some(prep) => bb_as_u32(prep),
        None => { prep_u32_owned = vec![]; &prep_u32_owned }
    };
    let perm_u32 = bb_as_u32(perm_lde);

    // Use cached selectors if provided, otherwise compute fresh.
    let selector_data_owned;
    let selector_data: &[u32] = match cached_selectors {
        Some(cached) => {
            debug_assert_eq!(cached.len(), quotient_size * 4);
            cached
        }
        None => {
            selector_data_owned = compute_selector_data(trace_domain, quotient_domain);
            &selector_data_owned
        }
    };

    Some(ChipDispatch::new_bitrev(
        metal_state,
        &program.words,
        program.num_regs,
        program.accumulator_regs,
        main_u32,
        main_lde_width as u32,
        prep_u32,
        prep_lde_width.max(1) as u32,
        perm_u32,
        perm_lde_width as u32,
        selector_data,
        quotient_size as u32,
        next_step as u32,
    ))
}

/// Convert a completed dispatch's results to Challenge (EF4) values.
///
/// Uses zero-intermediate-allocation path: copies directly from Metal buffer
/// as Challenge values, since Challenge = BinomialExtensionField<BabyBear, 4>
/// has the same layout as [u32; 4] (4 contiguous Montgomery-form u32s).
pub fn dispatch_to_quotient_values(dispatch: &ChipDispatch) -> Vec<Challenge> {
    // SAFETY: Challenge = BinomialExtensionField<BabyBear, 4> is 4 contiguous
    // BabyBear values, each repr(transparent) u32 in Montgomery form.
    // Same layout as [u32; 4], matching the GPU output format.
    unsafe { dispatch.read_results_as::<Challenge>() }
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
