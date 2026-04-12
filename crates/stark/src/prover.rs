use crate::septic_curve::SepticCurve;
use crate::septic_digest::SepticDigest;
use crate::septic_extension::SepticExtension;
use crate::{air::InteractionScope, AirOpenedValues, ChipOpenedValues, ShardOpenedValues};
use core::fmt::Display;
use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_fri::PcsOpenSequential;
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::LeafManageable;
use p3_util::log2_strict_usize;
use serde::{de::DeserializeOwned, Serialize};
use std::{cmp::Reverse, error::Error, time::Instant};

use super::{
    quotient_values, Com, OpeningProof, StarkGenericConfig, StarkMachine, StarkProvingKey, Val,
    VerifierConstraintFolder,
};
use crate::{
    air::MachineAir, lookup::InteractionBuilder, opts::SP1CoreOpts, record::MachineRecord,
    Challenger, DebugConstraintBuilder, MachineChip, MachineProof, PackedChallenge, PcsProverData,
    ProverConstraintFolder, ShardCommitment, ShardMainData, ShardProof, StarkVerifyingKey,
};

/// Check if GPU constraint evaluation is enabled.
/// Enabled when METAL_CONSTRAINTS=1 or when METAL_DFT=1 (auto-enable with GPU DFT).
/// Disable explicitly with METAL_CONSTRAINTS=0.
#[cfg(target_os = "macos")]
fn gpu_constraints_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Explicit METAL_CONSTRAINTS takes priority
        if let Ok(v) = std::env::var("METAL_CONSTRAINTS") {
            return v == "1";
        }
        // Auto-enable when METAL_DFT=1 (GPU acceleration already active)
        std::env::var("METAL_DFT").ok().map_or(false, |v| v == "1")
    })
}

/// Get the global Metal state singleton.
#[cfg(target_os = "macos")]
fn global_metal() -> &'static metal_ntt::device::MetalState {
    use std::sync::OnceLock;
    static STATE: OnceLock<metal_ntt::device::MetalState> = OnceLock::new();
    STATE.get_or_init(metal_ntt::device::MetalState::new)
}

/// Commit traces without cloning them.
///
/// On macOS with METAL_DFT=1, uses MetalDft::coset_lde_batch_multi_ref to do the DFT
/// from borrowed trace matrices, then commits the resulting LDEs via the MMCS.
/// This avoids cloning ~1.2GB of trace data that the trait-based `pcs.commit()` requires.
///
/// On non-macOS or when GPU DFT is not available, falls back to the standard clone path.
fn commit_traces_borrowed<SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    named_traces: &[(String, RowMajorMatrix<Val<SC>>)],
) -> (Com<SC>, PcsProverData<SC>) {
    #[cfg(target_os = "macos")]
    {
        use p3_baby_bear::BabyBear;
        use p3_field::TwoAdicField;
        use p3_matrix::bitrev::BitReversableMatrix;

        // Only use the borrowed path when Val = BabyBear (always true for SP1)
        if std::mem::size_of::<Val<SC>>() == std::mem::size_of::<BabyBear>() {
            // Get the concrete PCS to access log_blowup and commit_ldes
            type InnerPcs = crate::bb31_poseidon2::InnerPcs;
            let concrete_pcs: &InnerPcs = unsafe {
                &*(pcs as *const SC::Pcs as *const InnerPcs)
            };

            let metal_dft = concrete_pcs.dft();
            let log_blowup = concrete_pcs.log_blowup();

            // Build ref inputs for the DFT (BabyBear types via pointer cast)
            // Domain shift is always Val::one() for natural domains, so
            // coset shift = generator() / one() = generator().
            let coset_shift = BabyBear::generator();
            let ref_inputs: Vec<(&RowMajorMatrix<BabyBear>, usize, BabyBear)> = named_traces
                .iter()
                .map(|(_, trace)| {
                    let bb_trace: &RowMajorMatrix<BabyBear> = unsafe {
                        &*(trace as *const RowMajorMatrix<Val<SC>>
                            as *const RowMajorMatrix<BabyBear>)
                    };
                    (bb_trace, log_blowup, coset_shift)
                })
                .collect();

            let t_dft = Instant::now();
            let ldes: Vec<RowMajorMatrix<BabyBear>> = metal_dft
                .coset_lde_batch_multi_ref(&ref_inputs)
                .into_iter()
                .map(|m| m.bit_reverse_rows().to_row_major_matrix())
                .collect();
            let dft_ms = t_dft.elapsed().as_secs_f64() * 1000.0;

            let t_merkle = Instant::now();
            let (commit, data) = concrete_pcs.commit_ldes(ldes);
            let merkle_ms = t_merkle.elapsed().as_secs_f64() * 1000.0;
            if dft_ms + merkle_ms > 100.0 {
                tracing::info!(
                    "commit_borrowed: dft={dft_ms:.0}ms merkle={merkle_ms:.0}ms total={:.0}ms",
                    dft_ms + merkle_ms
                );
            }

            // Transmute concrete types back to generic SC types
            return unsafe {
                let commit_sc: Com<SC> = std::mem::transmute_copy(&commit);
                std::mem::forget(commit);
                let data_sc: PcsProverData<SC> = std::ptr::read(&data as *const _ as *const _);
                std::mem::forget(data);
                (commit_sc, data_sc)
            };
        }
    }

    // Fallback: standard clone path
    let domains_and_traces = named_traces
        .iter()
        .map(|(_, trace)| {
            let domain = pcs.natural_domain_for_degree(trace.height());
            (domain, trace.to_owned())
        })
        .collect::<Vec<_>>();
    pcs.commit(domains_and_traces)
}

/// An algorithmic & hardware independent prover implementation for any [`MachineAir`].
pub trait MachineProver<SC: StarkGenericConfig, A: MachineAir<SC::Val>>:
    'static + Send + Sync
{
    /// The type used to store the traces.
    type DeviceMatrix: Matrix<SC::Val>;

    /// The type used to store the polynomial commitment schemes data.
    type DeviceProverData;

    /// The type used to store the proving key.
    type DeviceProvingKey: MachineProvingKey<SC>;

    /// The type used for error handling.
    type Error: Error + Send + Sync;

    /// Create a new prover from a given machine.
    fn new(machine: StarkMachine<SC, A>) -> Self;

    /// A reference to the machine that this prover is using.
    fn machine(&self) -> &StarkMachine<SC, A>;

    /// Setup the preprocessed data into a proving and verifying key.
    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, StarkVerifyingKey<SC>);

    /// Setup the proving key given a verifying key. This is similar to `setup` but faster since
    /// some computed information is already in the verifying key.
    fn pk_from_vk(
        &self,
        program: &A::Program,
        vk: &StarkVerifyingKey<SC>,
    ) -> Self::DeviceProvingKey;

    /// Copy the proving key from the host to the device.
    fn pk_to_device(&self, pk: &StarkProvingKey<SC>) -> Self::DeviceProvingKey;

    /// Copy the proving key from the device to the host.
    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> StarkProvingKey<SC>;

    /// Generate the main traces.
    fn generate_traces(&self, record: &A::Record) -> Vec<(String, RowMajorMatrix<Val<SC>>)> {
        let shard_chips = self.shard_chips(record).collect::<Vec<_>>();

        // For each chip, generate the trace.
        let parent_span = tracing::debug_span!("generate traces for shard");
        parent_span.in_scope(|| {
            shard_chips
                .par_iter()
                .map(|chip| {
                    let chip_name = chip.name();
                    let begin = Instant::now();
                    let trace = chip.generate_trace(record, &mut A::Record::default());
                    tracing::debug!(
                        parent: &parent_span,
                        "generated trace for chip {} in {:?}",
                        chip_name,
                        begin.elapsed()
                    );
                    (chip_name, trace)
                })
                .collect::<Vec<_>>()
        })
    }

    /// Commit to the main traces.
    fn commit(
        &self,
        record: &A::Record,
        traces: Vec<(String, RowMajorMatrix<Val<SC>>)>,
    ) -> ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>;

    /// Observe the main commitment and public values and update the challenger.
    fn observe(
        &self,
        challenger: &mut SC::Challenger,
        commitment: Com<SC>,
        public_values: &[SC::Val],
    ) {
        // Observe the commitment.
        challenger.observe(commitment);

        // Observe the public values.
        challenger.observe_slice(public_values);
    }

    /// Compute the openings of the traces.
    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        data: ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>,
        challenger: &mut SC::Challenger,
    ) -> Result<ShardProof<SC>, Self::Error>;

    /// Generate a proof for the given records.
    fn prove(
        &self,
        pk: &Self::DeviceProvingKey,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>;

    /// The stark config for the machine.
    fn config(&self) -> &SC {
        self.machine().config()
    }

    /// The number of public values elements.
    fn num_pv_elts(&self) -> usize {
        self.machine().num_pv_elts()
    }

    /// The chips that will be necessary to prove this record.
    fn shard_chips<'a, 'b>(
        &'a self,
        record: &'b A::Record,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
        SC: 'b,
    {
        self.machine().shard_chips(record)
    }

    /// Debug the constraints for the given inputs.
    fn debug_constraints(
        &self,
        pk: &StarkProvingKey<SC>,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
    ) where
        SC::Val: PrimeField32,
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        self.machine().debug_constraints(pk, records, challenger);
    }
}

/// A proving key for any [`MachineAir`] that is agnostic to hardware.
pub trait MachineProvingKey<SC: StarkGenericConfig>: Send + Sync {
    /// The main commitment.
    fn preprocessed_commit(&self) -> Com<SC>;

    /// The start pc.
    fn pc_start(&self) -> Val<SC>;

    /// The initial global cumulative sum.
    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>>;

    /// Observe itself in the challenger.
    fn observe_into(&self, challenger: &mut Challenger<SC>);
}

/// A prover implementation based on x86 and ARM CPUs.
pub struct CpuProver<SC: StarkGenericConfig, A> {
    machine: StarkMachine<SC, A>,
}

/// An error that occurs during the execution of the [`CpuProver`].
#[derive(Debug, Clone, Copy)]
pub struct CpuProverError;

impl<SC, A> MachineProver<SC, A> for CpuProver<SC, A>
where
    SC: 'static + StarkGenericConfig + Send + Sync,
    A: MachineAir<SC::Val>
        + for<'a> Air<ProverConstraintFolder<'a, SC>>
        + Air<InteractionBuilder<Val<SC>>>
        + for<'a> Air<VerifierConstraintFolder<'a, SC>>
        + for<'a> Air<crate::gpu_ir::ConstraintCompiler<'a>>,
    A::Record: MachineRecord<Config = SP1CoreOpts>,
    SC::Val: PrimeField32 + TwoAdicField,
    SC::Domain: Into<TwoAdicMultiplicativeCoset<Val<SC>>>,
    Com<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync + Serialize + DeserializeOwned + Clone + LeafManageable<Val<SC>>,
    OpeningProof<SC>: Send + Sync,
    SC::Challenger: Clone,
    SC::Pcs: PcsOpenSequential<
        Val<SC>,
        SC::Challenge,
        SC::Challenger,
        Proof = OpeningProof<SC>,
        ProverData = PcsProverData<SC>,
    >,
{
    type DeviceMatrix = RowMajorMatrix<Val<SC>>;
    type DeviceProverData = PcsProverData<SC>;
    type DeviceProvingKey = StarkProvingKey<SC>;
    type Error = CpuProverError;

    fn new(machine: StarkMachine<SC, A>) -> Self {
        Self { machine }
    }

    fn machine(&self) -> &StarkMachine<SC, A> {
        &self.machine
    }

    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, StarkVerifyingKey<SC>) {
        self.machine().setup(program)
    }

    fn pk_from_vk(
        &self,
        program: &A::Program,
        vk: &StarkVerifyingKey<SC>,
    ) -> Self::DeviceProvingKey {
        self.machine().setup_core(program, vk.initial_global_cumulative_sum).0
    }

    fn pk_to_device(&self, pk: &StarkProvingKey<SC>) -> Self::DeviceProvingKey {
        pk.clone()
    }

    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> StarkProvingKey<SC> {
        pk.clone()
    }

    fn commit(
        &self,
        record: &A::Record,
        mut named_traces: Vec<(String, RowMajorMatrix<Val<SC>>)>,
    ) -> ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData> {
        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_traces.sort_by_key(|(name, trace)| (Reverse(trace.height()), name.clone()));

        let pcs = self.config().pcs();

        // On macOS, use MetalDft::coset_lde_batch_multi_ref to do DFT from borrowed
        // traces, then commit the LDEs directly. Avoids cloning ~1.2GB of trace data.
        let t_main_commit = std::time::Instant::now();
        let (main_commit, main_data) = commit_traces_borrowed::<SC>(pcs, &named_traces);
        let main_commit_ms = t_main_commit.elapsed().as_secs_f64() * 1000.0;
        if main_commit_ms > 50.0 {
            tracing::info!("main_commit: {:.0}ms", main_commit_ms);
        }

        // Get the chip ordering.
        let chip_ordering =
            named_traces.iter().enumerate().map(|(i, (name, _))| (name.to_owned(), i)).collect();

        let traces = named_traces.into_iter().map(|(_, trace)| trace).collect::<Vec<_>>();

        ShardMainData {
            traces,
            main_commit,
            main_data,
            chip_ordering,
            public_values: record.public_values(),
        }
    }

    /// Prove the program for the given shard and given a commitment to the main data.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::redundant_closure_for_method_calls)]
    #[allow(clippy::map_unwrap_or)]
    fn open(
        &self,
        pk: &StarkProvingKey<SC>,
        mut data: ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>,
        challenger: &mut <SC as StarkGenericConfig>::Challenger,
    ) -> Result<ShardProof<SC>, Self::Error> {
        let t_open_total = std::time::Instant::now();
        let chips = self.machine().shard_chips_ordered(&data.chip_ordering).collect::<Vec<_>>();
        let traces = data.traces;

        let config = self.machine().config();


        let degrees = traces.iter().map(|trace| trace.height()).collect::<Vec<_>>();

        let log_degrees =
            degrees.iter().map(|degree| log2_strict_usize(*degree)).collect::<Vec<_>>();

        let log_quotient_degrees =
            chips.iter().map(|chip| chip.log_quotient_degree()).collect::<Vec<_>>();

        let pcs = config.pcs();
        let trace_domains =
            degrees.iter().map(|degree| pcs.natural_domain_for_degree(*degree)).collect::<Vec<_>>();

        // Observe the public values and the main commitment.
        challenger.observe_slice(&data.public_values[0..self.num_pv_elts()]);
        challenger.observe(data.main_commit.clone());

        // Obtain the challenges used for the local permutation argument.
        let mut local_permutation_challenges: Vec<SC::Challenge> = Vec::new();
        for _ in 0..2 {
            local_permutation_challenges.push(challenger.sample_ext_element());
        }

        let packed_perm_challenges = local_permutation_challenges
            .iter()
            .map(|c| PackedChallenge::<SC>::from_f(*c))
            .collect::<Vec<_>>();

        // Generate the permutation traces.
        let t_perm_trace = std::time::Instant::now();
        let ((permutation_traces, prep_traces), (global_cumulative_sums, local_cumulative_sums)): (
            (Vec<_>, Vec<_>),
            (Vec<_>, Vec<_>),
        ) = tracing::debug_span!("generate permutation traces").in_scope(|| {
            chips
                .par_iter()
                .zip(traces.par_iter())
                .map(|(chip, main_trace)| {
                    let preprocessed_trace =
                        pk.chip_ordering.get(&chip.name()).map(|&index| &pk.traces[index]);
                    let (perm_trace, local_sum) = chip.generate_permutation_trace(
                        preprocessed_trace,
                        main_trace,
                        &local_permutation_challenges,
                    );
                    let global_sum = if chip.commit_scope() == InteractionScope::Local {
                        SepticDigest::<Val<SC>>::zero()
                    } else {
                        let main_trace_size = main_trace.height() * main_trace.width();
                        let last_row = &main_trace.values[main_trace_size - 14..main_trace_size];
                        SepticDigest(SepticCurve {
                            x: SepticExtension::<Val<SC>>::from_base_fn(|i| last_row[i]),
                            y: SepticExtension::<Val<SC>>::from_base_fn(|i| last_row[i + 7]),
                        })
                    };
                    ((perm_trace, preprocessed_trace), (global_sum, local_sum))
                })
                .unzip()
        });

        // Compute some statistics, then drop main traces to free memory before permutation commit.
        for i in 0..chips.len() {
            let trace_width = traces[i].width();
            let trace_height = traces[i].height();
            let prep_width = prep_traces[i].map_or(0, |x| x.width());
            let permutation_width = permutation_traces[i].width();
            let total_width = trace_width
                + prep_width
                + permutation_width * <SC::Challenge as AbstractExtensionField<SC::Val>>::D;
            tracing::debug!(
                "{:<15} | Main Cols = {:<5} | Pre Cols = {:<5}  | Perm Cols = {:<5} | Rows = {:<5} | Cells = {:<10}",
                chips[i].name(),
                trace_width,
                prep_width,
                permutation_width * <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
                trace_height,
                total_width * trace_height,
            );
        }
        let perm_trace_ms = t_perm_trace.elapsed().as_secs_f64() * 1000.0;
        let use_sequential_open = std::env::var("SEQUENTIAL_OPEN").map_or(true, |v| v != "0");
        drop(traces); // Raw traces no longer needed.

        let t_perm_commit = std::time::Instant::now();
        // Compute quotient domains early (only depends on trace_domains and log_quotient_degrees).
        let quotient_domains = trace_domains
            .iter()
            .zip_eq(log_degrees.iter())
            .zip_eq(log_quotient_degrees.iter())
            .map(|((domain, log_degree), log_quotient_degree)| {
                domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree))
            })
            .collect::<Vec<_>>();

        let domains_and_perm_traces =
            tracing::debug_span!("flatten permutation traces and collect domains").in_scope(|| {
                permutation_traces
                    .into_iter()
                    .zip(trace_domains.iter())
                    .map(|(perm_trace, domain)| {
                        // Zero-copy flatten: EF4 = BinomialExtensionField<BabyBear, 4>
                        // is repr(C) with [BabyBear; 4], and BabyBear is repr(transparent)
                        // wrapping u32. So Vec<EF4> has identical layout to Vec<BabyBear>
                        // with 4× the length. This avoids a full-matrix copy.
                        let ext_width = perm_trace.width();
                        let base_width = ext_width * <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
                        let values = perm_trace.values;
                        let base_len = values.len() * <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
                        let base_cap = values.capacity() * <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
                        let base_ptr = values.as_ptr() as *mut Val<SC>;
                        std::mem::forget(values);
                        let base_values = unsafe { Vec::from_raw_parts(base_ptr, base_len, base_cap) };
                        let trace = RowMajorMatrix::new(base_values, base_width);
                        (*domain, trace)
                    })
                    .collect::<Vec<_>>()
            });

        let pcs = config.pcs();

        let (permutation_commit, mut permutation_data) =
            tracing::debug_span!("commit to permutation traces")
                .in_scope(|| pcs.commit(domains_and_perm_traces));

        // Observe the permutation commitment and cumulative sums.
        challenger.observe(permutation_commit.clone());
        for (local_sum, global_sum) in
            local_cumulative_sums.iter().zip(global_cumulative_sums.iter())
        {
            challenger.observe_slice(local_sum.as_base_slice());
            challenger.observe_slice(&global_sum.0.x.0);
            challenger.observe_slice(&global_sum.0.y.0);
        }

        let perm_commit_ms = t_perm_commit.elapsed().as_secs_f64() * 1000.0;

        // Compute the quotient values.
        let t_quotient = std::time::Instant::now();
        let alpha: SC::Challenge = challenger.sample_ext_element::<SC::Challenge>();
        let parent_span = tracing::debug_span!("compute quotient values");

        let quotient_values = parent_span.in_scope(|| {
            // GPU path for constraint evaluation on macOS with Metal.
            // Batched: encode ALL GPU-eligible chips into one command buffer,
            // process CPU-fallback chips in parallel while GPU executes.
            #[cfg(target_os = "macos")]
            {
                if gpu_constraints_enabled()
                    && std::mem::size_of::<Val<SC>>() == 4
                    && std::mem::size_of::<SC::Challenge>() == 16
                {
                    let metal_state = global_metal();
                    let n_chips = chips.len();

                    // SAFETY: Val<SC> = BabyBear (4 bytes) and SC::Challenge =
                    // BinomialExtensionField<BabyBear, 4> (16 bytes), verified
                    // by size checks above. Types are repr(transparent) or identical.
                    // The transmute helpers reinterpret generic SC types as concrete
                    // BabyBear types for the GPU IR compiler.
                    unsafe {
                        let bb_alpha: p3_field::extension::BinomialExtensionField<
                            p3_baby_bear::BabyBear, 4,
                        > = std::mem::transmute_copy(&alpha);
                        let bb_perm_challenges: &[p3_field::extension::BinomialExtensionField<
                            p3_baby_bear::BabyBear, 4,
                        >] = std::slice::from_raw_parts(
                            local_permutation_challenges.as_ptr() as *const _,
                            local_permutation_challenges.len(),
                        );
                        let bb_public_values: &[p3_baby_bear::BabyBear] =
                            std::slice::from_raw_parts(
                                data.public_values.as_ptr() as *const p3_baby_bear::BabyBear,
                                data.public_values.len(),
                            );

                        // Phase 1: Prepare GPU dispatches using raw LDE data (bitrev path).
                        //
                        // Instead of materializing quotient-domain evaluations (hundreds of
                        // MB per large chip), we pass raw LDE pointers to the GPU kernel
                        // which does bit-reversal indexing internally.
                        //
                        // The LDE data lives in the MMCS prover_data (main_data,
                        // permutation_data, pk.data), which outlives the GPU dispatch.
                        let mut gpu_dispatches: Vec<(
                            usize,
                            metal_ntt::constraints::ChipDispatch,
                        )> = Vec::new();
                        let mut cpu_indices: Vec<usize> = Vec::new();

                        // Get quotient domain sizes for all chips.
                        let quot_sizes: Vec<usize> = quotient_domains
                            .as_slice()
                            .iter()
                            .map(|d| d.size())
                            .collect();

                        // Get raw LDE slices (bit-reversed row order) directly
                        // from MMCS prover data — no materialization.
                        let main_lde_slices =
                            crate::gpu_quotient::get_lde_slices_from_pcs::<SC>(
                                pcs, &data.main_data, &quot_sizes,
                            );
                        let perm_lde_slices =
                            crate::gpu_quotient::get_lde_slices_from_pcs::<SC>(
                                pcs, &permutation_data, &quot_sizes,
                            );
                        // Prep LDE: map chip ordering to quotient sizes.
                        let mut prep_sizes = vec![0usize; pk.chip_ordering.len()];
                        for (i, _) in quotient_domains.as_slice().iter().enumerate() {
                            if let Some(&index) = pk.chip_ordering.get(&chips[i].name()) {
                                prep_sizes[index] = quot_sizes[i];
                            }
                        }
                        let prep_lde_slices =
                            crate::gpu_quotient::get_lde_slices_from_pcs::<SC>(
                                pcs, &pk.data, &prep_sizes,
                            );

                        // Precompute selectors grouped by (trace_log_n, quotient_size).
                        // Chips sharing the same domain pair reuse the same selector data.
                        let mut selector_cache: std::collections::HashMap<(usize, usize), Vec<u32>> =
                            std::collections::HashMap::new();

                        for (i, quotient_domain) in
                            quotient_domains.as_slice().iter().enumerate()
                        {
                            let bb_chip = &*(chips[i] as *const _
                                as *const crate::Chip<p3_baby_bear::BabyBear, A>);
                            let bb_trace_domain = *(&trace_domains[i] as *const _
                                as *const p3_commit::TwoAdicMultiplicativeCoset<
                                    p3_baby_bear::BabyBear,
                                >);
                            let bb_quotient_domain = *(quotient_domain as *const _
                                as *const p3_commit::TwoAdicMultiplicativeCoset<
                                    p3_baby_bear::BabyBear,
                                >);
                            let bb_local_cum_sum = &*(&local_cumulative_sums[i]
                                as *const SC::Challenge
                                as *const p3_field::extension::BinomialExtensionField<
                                    p3_baby_bear::BabyBear,
                                    4,
                                >);
                            let bb_global_cum_sum = &*(&global_cumulative_sums[i]
                                as *const SepticDigest<Val<SC>>
                                as *const SepticDigest<p3_baby_bear::BabyBear>);

                            let (main_lde_data, main_lde_w) = main_lde_slices[i];
                            let (perm_lde_data, perm_lde_w) = perm_lde_slices[i];
                            let prep_lde_info: Option<(&[p3_baby_bear::BabyBear], usize)> =
                                pk.chip_ordering.get(&chips[i].name()).map(|&index| {
                                    prep_lde_slices[index]
                                });

                            // Cache key: (trace_domain.log_n, quotient_domain.size)
                            // These fully determine the selectors since domain.shift is always one().
                            let sel_key = (bb_trace_domain.log_n, bb_quotient_domain.size());
                            let cached_sels = selector_cache
                                .entry(sel_key)
                                .or_insert_with(|| {
                                    crate::gpu_quotient::compute_selector_data(
                                        bb_trace_domain,
                                        bb_quotient_domain,
                                    )
                                });

                            // Try bitrev dispatch (avoids materialization).
                            let dispatch =
                                crate::gpu_quotient::prepare_quotient_dispatch_bitrev(
                                    bb_chip,
                                    chips[i].preprocessed_width(),
                                    <_ as p3_air::BaseAir<Val<SC>>>::width(chips[i]),
                                    chips[i].commit_scope(),
                                    bb_local_cum_sum,
                                    bb_global_cum_sum,
                                    bb_trace_domain,
                                    bb_quotient_domain,
                                    main_lde_data,
                                    main_lde_w,
                                    prep_lde_info.map(|(data, _)| data),
                                    prep_lde_info.map_or(0, |(_, w)| w),
                                    perm_lde_data,
                                    perm_lde_w,
                                    bb_perm_challenges,
                                    bb_alpha,
                                    bb_public_values,
                                    Some(cached_sels.as_slice()),
                                    metal_state,
                                );

                            match dispatch {
                                Some(d) => gpu_dispatches.push((i, d)),
                                None => cpu_indices.push(i),
                            }
                        }

                        tracing::info!(
                            "GPU constraints: {} chips on GPU, {} on CPU",
                            gpu_dispatches.len(), cpu_indices.len()
                        );

                        // Phase 2: Batch GPU dispatch — single command buffer
                        let gpu_cmd = if !gpu_dispatches.is_empty() {
                            let cmd = metal_state.queue.new_command_buffer().to_owned();
                            let encoder = cmd.new_compute_command_encoder();
                            for (_, dispatch) in &gpu_dispatches {
                                dispatch.encode(metal_state, encoder);
                            }
                            encoder.end_encoding();
                            cmd.commit();
                            Some(cmd)
                        } else {
                            None
                        };

                        // Phase 3: CPU fallback chips — parallel while GPU executes
                        let mut results: Vec<Option<Vec<SC::Challenge>>> =
                            (0..n_chips).map(|_| None).collect();

                        let cpu_results: Vec<(usize, Vec<SC::Challenge>)> = cpu_indices
                            .into_par_iter()
                            .map(|i| {
                                let quotient_domain = quotient_domains.as_slice()[i];
                                let prep = pk
                                    .chip_ordering
                                    .get(&chips[i].name())
                                    .map(|&index| {
                                        pcs.get_evaluations_on_domain(
                                            &pk.data,
                                            index,
                                            quotient_domain,
                                        )
                                    });
                                let main = pcs
                                    .get_evaluations_on_domain(
                                        &data.main_data,
                                        i,
                                        quotient_domain,
                                    );
                                let perm = pcs
                                    .get_evaluations_on_domain(
                                        &permutation_data,
                                        i,
                                        quotient_domain,
                                    );
                                let qv = quotient_values(
                                    chips[i],
                                    &local_cumulative_sums[i],
                                    &global_cumulative_sums[i],
                                    trace_domains[i],
                                    quotient_domain,
                                    prep,
                                    main,
                                    perm,
                                    &packed_perm_challenges,
                                    alpha,
                                    &data.public_values,
                                );
                                (i, qv)
                            })
                            .collect();

                        for (i, qv) in cpu_results {
                            results[i] = Some(qv);
                        }

                        // Phase 4: Wait for GPU and collect results
                        if let Some(cmd) = gpu_cmd {
                            cmd.wait_until_completed();
                        }

                        for (i, dispatch) in &gpu_dispatches {
                            let gpu_vals =
                                crate::gpu_quotient::dispatch_to_quotient_values(dispatch);
                            results[*i] =
                                Some(std::mem::transmute::<Vec<_>, Vec<SC::Challenge>>(
                                    gpu_vals,
                                ));
                        }

                        // gpu_dispatches dropped here — safe because GPU has
                        // completed and results are read out. LDE data in
                        // MMCS prover_data outlives this scope.
                        drop(gpu_dispatches);

                        return results.into_iter().map(|r| r.unwrap()).collect();
                    }
                }
            }

            // CPU fallback path (also used on non-macOS).
            // Process chips sequentially to avoid materializing multiple large
            // LDE copies simultaneously. Each chip's quotient_values() is
            // already internally parallelized via par_chunks_mut.
            quotient_domains
                .as_slice()
                .iter()
                .enumerate()
                .map(|(i, quotient_domain)| {
                    tracing::debug_span!(parent: &parent_span, "compute quotient values for domain")
                        .in_scope(|| {
                            let preprocessed_trace_on_quotient_domains =
                                pk.chip_ordering.get(&chips[i].name()).map(|&index| {
                                    pcs.get_evaluations_on_domain(&pk.data, index, *quotient_domain)
                                });
                            let main_trace_on_quotient_domains =
                                pcs.get_evaluations_on_domain(&data.main_data, i, *quotient_domain);
                            let permutation_trace_on_quotient_domains =
                                pcs.get_evaluations_on_domain(&permutation_data, i, *quotient_domain);
                            quotient_values(
                                chips[i],
                                &local_cumulative_sums[i],
                                &global_cumulative_sums[i],
                                trace_domains[i],
                                *quotient_domain,
                                preprocessed_trace_on_quotient_domains,
                                main_trace_on_quotient_domains,
                                permutation_trace_on_quotient_domains,
                                &packed_perm_challenges,
                                alpha,
                                &data.public_values,
                            )
                        })
                })
                .collect::<Vec<_>>()
        });

        // Split the quotient values and commit to them.
        let quotient_domains_and_chunks = quotient_domains
            .into_iter()
            .zip_eq(quotient_values)
            .zip_eq(log_quotient_degrees.iter())
            .flat_map(|((quotient_domain, quotient_values), log_quotient_degree)| {
                let quotient_degree = 1 << *log_quotient_degree;
                // Zero-copy flatten: SC::Challenge = BinomialExtensionField<BabyBear, 4>
                // is repr(C) with [BabyBear; 4]. Reinterpret Vec<Challenge> as Vec<Val>
                // with 4× the length, avoiding the element-by-element copy in flatten_to_base.
                let ext_width = <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
                let n = quotient_values.len();
                let base_len = n * ext_width;
                let base_cap = quotient_values.capacity() * ext_width;
                let base_ptr = quotient_values.as_ptr() as *mut Val<SC>;
                std::mem::forget(quotient_values);
                let base_values = unsafe { Vec::from_raw_parts(base_ptr, base_len, base_cap) };
                let quotient_flat = RowMajorMatrix::new(base_values, ext_width);
                let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
                let qc_domains = quotient_domain.split_domains(quotient_degree);
                qc_domains.into_iter().zip_eq(quotient_chunks)
            })
            .collect::<Vec<_>>();

        let quotient_ms = t_quotient.elapsed().as_secs_f64() * 1000.0;

        // QUOTIENT_DIRECT_HASH mode: compute BLAKE3 on LDE data as a preview of the optimization.
        // Note: Full implementation requires verifier changes to handle empty quotient openings.
        let use_quotient_direct_hash = std::env::var("QUOTIENT_DIRECT_HASH").map_or(false, |v| v == "1");

        if use_quotient_direct_hash {
            // Compute BLAKE3 on the LDE matrices as a commitment preview
            // (The actual optimization would use raw quotient values, but they were consumed
            // in the flat_map above. This still shows the hashing overhead.)
            let t_hash = std::time::Instant::now();
            let mut all_bytes = Vec::new();
            for (_domain, mat) in &quotient_domains_and_chunks {
                for row in mat.rows() {
                    for val in row {
                        all_bytes.extend_from_slice(&val.as_canonical_u32().to_le_bytes());
                    }
                }
            }
            let hash = blake3::hash(&all_bytes);
            let hash_ms = t_hash.elapsed().as_secs_f64() * 1000.0;
            tracing::info!("quotient direct hash preview: {}ms, {} bytes", hash_ms, all_bytes.len());
        }

        // TIERED_QUOTIENT mode: skip committing small chunks to reduce Merkle tree overhead.
        // Small chunks (<256 elements) have high overhead per element for LDE + Merkle commit.
        // The FRI proof will be missing these chunks, so verifier must be updated separately.
        let use_tiered_quotient = std::env::var("TIERED_QUOTIENT").map_or(false, |v| v == "1");
        const TIERED_QUOTIENT_THRESHOLD: usize = 256;

        let (quotient_domains_and_chunks, skipped_small_chunks) = if use_tiered_quotient {
            let t_filter = std::time::Instant::now();
            let mut skipped = 0;
            let filtered: Vec<_> = quotient_domains_and_chunks
                .into_iter()
                .filter_map(|(domain, mat)| {
                    if mat.height() < TIERED_QUOTIENT_THRESHOLD {
                        skipped += 1;
                        None
                    } else {
                        Some((domain, mat))
                    }
                })
                .collect();
            let filter_ms = t_filter.elapsed().as_secs_f64() * 1000.0;
            tracing::info!(
                "tiered quotient: filtered {} small chunks in {}ms, kept {} chunks",
                skipped,
                filter_ms,
                filtered.len()
            );
            (filtered, skipped)
        } else {
            (quotient_domains_and_chunks, 0)
        };

        let num_quotient_chunks = quotient_domains_and_chunks.len();
        if !use_tiered_quotient {
            let expected_chunks = chips.iter().map(|c| 1 << c.log_quotient_degree()).sum::<usize>();
            assert_eq!(
                num_quotient_chunks,
                expected_chunks,
                "expected {} chunks but have {}",
                expected_chunks,
                num_quotient_chunks
            );
        }

        let t_quotient_commit = std::time::Instant::now();
        let (quotient_commit, mut quotient_data) =
            tracing::debug_span!("commit to quotient traces")
                .in_scope(|| pcs.commit(quotient_domains_and_chunks));

        let quotient_commit_ms = t_quotient_commit.elapsed().as_secs_f64() * 1000.0;

        challenger.observe(quotient_commit.clone());
        // Compute the quotient argument.
        let t_pcs_open = std::time::Instant::now();
        let zeta: SC::Challenge = challenger.sample_ext_element();

        let preprocessed_opening_points =
            tracing::debug_span!("compute preprocessed opening points").in_scope(|| {
                pk.traces
                    .iter()
                    .zip(pk.local_only.iter())
                    .map(|(trace, local_only)| {
                        let domain = pcs.natural_domain_for_degree(trace.height());
                        if !local_only {
                            vec![zeta, domain.next_point(zeta).unwrap()]
                        } else {
                            vec![zeta]
                        }
                    })
                    .collect::<Vec<_>>()
            });

        let main_trace_opening_points = tracing::debug_span!("compute main trace opening points")
            .in_scope(|| {
                trace_domains
                    .iter()
                    .zip(chips.iter())
                    .map(|(domain, chip)| {
                        if !chip.local_only() {
                            vec![zeta, domain.next_point(zeta).unwrap()]
                        } else {
                            vec![zeta]
                        }
                    })
                    .collect::<Vec<_>>()
            });

        let permutation_trace_opening_points =
            tracing::debug_span!("compute permutation trace opening points").in_scope(|| {
                trace_domains
                    .iter()
                    .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
                    .collect::<Vec<_>>()
            });

        // Compute quotient opening points, open every chunk at zeta.
        let quotient_opening_points =
            (0..num_quotient_chunks).map(|_| vec![zeta]).collect::<Vec<_>>();

        let (openings, opening_proof) = if use_sequential_open {
            // Memory-efficient path: take LDE leaves from commit data, then restore
            // one round at a time during open. No clone or recompute needed — the LDEs
            // were already computed by commit and are no longer needed for quotient values.
            let main_ldes = data.main_data.take_leaves_rm();
            let perm_ldes = permutation_data.take_leaves_rm();
            let quotient_ldes = quotient_data.take_leaves_rm();

            let saved_ldes = vec![
                Some(main_ldes),    // main: restore on demand
                Some(perm_ldes),    // perm: restore on demand
                Some(quotient_ldes), // quotient: restore on demand
            ];

            let mut rounds: Vec<(&mut PcsProverData<SC>, Vec<Vec<SC::Challenge>>)> = vec![
                (&mut data.main_data, main_trace_opening_points),
                (&mut permutation_data, permutation_trace_opening_points),
                (&mut quotient_data, quotient_opening_points),
            ];

            tracing::debug_span!("open sequential with cached LDEs").in_scope(|| {
                pcs.open_sequential_cached_split(&pk.data, preprocessed_opening_points, &mut rounds, saved_ldes, challenger)
            })
        } else {
            tracing::debug_span!("open multi batches").in_scope(|| {
                pcs.open(
                    vec![
                        (&pk.data, preprocessed_opening_points),
                        (&data.main_data, main_trace_opening_points),
                        (&permutation_data, permutation_trace_opening_points),
                        (&quotient_data, quotient_opening_points),
                    ],
                    challenger,
                )
            })
        };

        // Collect the opened values for each chip.
        let [preprocessed_values, main_values, permutation_values, mut quotient_values] =
            openings.try_into().unwrap();
        assert!(main_values.len() == chips.len());
        let preprocessed_opened_values = preprocessed_values
            .into_iter()
            .zip(pk.local_only.iter())
            .map(|(op, local_only)| {
                if !local_only {
                    let [local, next] = op.try_into().unwrap();
                    AirOpenedValues { local, next }
                } else {
                    let [local] = op.try_into().unwrap();
                    let width = local.len();
                    AirOpenedValues { local, next: vec![SC::Challenge::zero(); width] }
                }
            })
            .collect::<Vec<_>>();

        let main_opened_values = main_values
            .into_iter()
            .zip(chips.iter())
            .map(|(op, chip)| {
                if !chip.local_only() {
                    let [local, next] = op.try_into().unwrap();
                    AirOpenedValues { local, next }
                } else {
                    let [local] = op.try_into().unwrap();
                    let width = local.len();
                    AirOpenedValues { local, next: vec![SC::Challenge::zero(); width] }
                }
            })
            .collect::<Vec<_>>();
        let permutation_opened_values = permutation_values
            .into_iter()
            .map(|op| {
                let [local, next] = op.try_into().unwrap();
                AirOpenedValues { local, next }
            })
            .collect::<Vec<_>>();
        let mut quotient_opened_values = Vec::with_capacity(log_quotient_degrees.len());
        for log_quotient_degree in log_quotient_degrees.iter() {
            let degree = 1 << *log_quotient_degree;
            let slice = quotient_values.drain(0..degree);
            quotient_opened_values.push(slice.map(|mut op| op.pop().unwrap()).collect::<Vec<_>>());
        }

        let opened_values = main_opened_values
            .into_iter()
            .zip_eq(permutation_opened_values)
            .zip_eq(quotient_opened_values)
            .zip_eq(local_cumulative_sums)
            .zip_eq(global_cumulative_sums)
            .zip_eq(log_degrees.iter())
            .enumerate()
            .map(
                |(
                    i,
                    (
                        (
                            (((main, permutation), quotient), local_cumulative_sum),
                            global_cumulative_sum,
                        ),
                        log_degree,
                    ),
                )| {
                    let preprocessed = pk
                        .chip_ordering
                        .get(&chips[i].name())
                        .map(|&index| preprocessed_opened_values[index].clone())
                        .unwrap_or(AirOpenedValues { local: vec![], next: vec![] });
                    ChipOpenedValues {
                        preprocessed,
                        main,
                        permutation,
                        quotient,
                        global_cumulative_sum,
                        local_cumulative_sum,
                        log_degree: *log_degree,
                    }
                },
            )
            .collect::<Vec<_>>();

        let pcs_open_ms = t_pcs_open.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            "open phases: perm_trace={:.0}ms perm_commit={:.0}ms quotient={:.0}ms q_commit={:.0}ms pcs_open={:.0}ms total={:.0}ms",
            perm_trace_ms, perm_commit_ms, quotient_ms, quotient_commit_ms, pcs_open_ms,
            t_open_total.elapsed().as_secs_f64() * 1000.0
        );
        Ok(ShardProof::<SC> {
            commitment: ShardCommitment {
                main_commit: data.main_commit.clone(),
                permutation_commit,
                quotient_commit,
            },
            opened_values: ShardOpenedValues { chips: opened_values },
            opening_proof,
            chip_ordering: data.chip_ordering,
            public_values: data.public_values,
        })
    }

    /// Prove the execution record is valid.
    ///
    /// Given a proving key `pk` and a matching execution record `record`, this function generates
    /// a STARK proof that the execution record is valid.
    ///
    /// When `BATCH_CONSTRAINTS` env var is set > 1, shards are processed in batches to
    /// potentially improve cache locality and amortize trace generation/commit overhead.
    #[allow(clippy::needless_for_each)]
    fn prove(
        &self,
        pk: &StarkProvingKey<SC>,
        mut records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        // Generate dependencies.
        self.machine().generate_dependencies(&mut records, &opts, None);

        // Observe the preprocessed commitment.
        pk.observe_into(challenger);

        // Parse batch size from BATCH_CONSTRAINTS env var (default 1 = fully parallel)
        let batch_size: usize = std::env::var("BATCH_CONSTRAINTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);

        let shard_proofs = if batch_size <= 1 {
            // Default: fully parallel shard processing
            tracing::info_span!("prove_shards").in_scope(|| {
                records
                    .into_par_iter()
                    .map(|record| {
                        let named_traces = self.generate_traces(&record);
                        let shard_data = self.commit(&record, named_traces);
                        self.open(pk, shard_data, &mut challenger.clone())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
        } else {
            // Batched: process shards in batches for better cache locality
            // Generate traces and commit for all shards in parallel upfront,
            // then call open for each shard sequentially.
            let shard_datas: Vec<_> = tracing::debug_span!("generate and commit all traces").in_scope(|| {
                records
                    .into_par_iter()
                    .map(|record| {
                        let named_traces = self.generate_traces(&record);
                        self.commit(&record, named_traces)
                    })
                    .collect()
            });

            // Process open for each shard (this is sequential due to challenger)
            let mut results = Vec::with_capacity(shard_datas.len());
            for shard_data in shard_datas {
                let proof = self.open(pk, shard_data, challenger)?;
                results.push(Ok(proof));
            }
            results.into_iter().collect::<Result<Vec<_>, _>>()?
        };

        Ok(MachineProof { shard_proofs })
    }
}

impl<SC> MachineProvingKey<SC> for StarkProvingKey<SC>
where
    SC: 'static + StarkGenericConfig + Send + Sync,
    PcsProverData<SC>: Send + Sync + Serialize + DeserializeOwned,
    Com<SC>: Send + Sync,
{
    fn preprocessed_commit(&self) -> Com<SC> {
        self.commit.clone()
    }

    fn pc_start(&self) -> Val<SC> {
        self.pc_start
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>> {
        self.initial_global_cumulative_sum
    }

    fn observe_into(&self, challenger: &mut Challenger<SC>) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        let zero = Val::<SC>::zero();
        challenger.observe(zero);
    }
}

impl Display for CpuProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DefaultProverError")
    }
}

impl Error for CpuProverError {}
