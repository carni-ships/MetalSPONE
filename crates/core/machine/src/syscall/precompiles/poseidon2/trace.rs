use core::borrow::BorrowMut;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{ByteLookupEvent, PrecompileEvent, Poseidon2Event},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_stark::air::MachineAir;

use super::columns::{NUM_POSEIDON2_COLS, Poseidon2Cols};
use super::Poseidon2Chip;
use crate::utils::pad_rows_fixed;
use sp1_core_executor::events::ByteRecord;
use crate::operations::poseidon2::populate_perm_deg3;

impl<F: PrimeField32> MachineAir<F> for Poseidon2Chip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Poseidon2".to_string()
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = 8;

        let blu_events: Vec<Vec<ByteLookupEvent>> = input
            .get_precompile_events(SyscallCode::POSEIDON2)
            .par_chunks(chunk_size)
            .map(|ops: &[(sp1_core_executor::events::SyscallEvent, PrecompileEvent)]| {
                let mut blu = Vec::new();
                ops.iter().for_each(|(_, op)| {
                    if let PrecompileEvent::Poseidon2(event) = op {
                        Self::populate_byte_lookup_events(event, &mut blu);
                    }
                });
                blu
            })
            .collect();
        for blu in blu_events {
            output.add_byte_lookup_events(blu);
        }
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();

        for (_, event) in input.get_precompile_events(SyscallCode::POSEIDON2) {
            let event = if let PrecompileEvent::Poseidon2(event) = event {
                event
            } else {
                unreachable!()
            };
            let row = self.event_to_rows(event);
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_POSEIDON2_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_POSEIDON2_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::POSEIDON2).is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl Poseidon2Chip {
    fn event_to_rows<F: PrimeField32>(&self, event: &Poseidon2Event) -> [F; NUM_POSEIDON2_COLS] {
        let mut row = [F::zero(); NUM_POSEIDON2_COLS];
        let cols: &mut Poseidon2Cols<F> = row.as_mut_slice().borrow_mut();

        // Populate basic info.
        cols.shard = F::from_canonical_u32(event.shard);
        cols.clk = F::from_canonical_u32(event.clk);
        cols.input_ptr = F::from_canonical_u32(event.input_ptr);
        cols.output_ptr = F::from_canonical_u32(event.output_ptr);
        cols.is_real = F::one();

        // Populate input memory read records.
        // The input is read at clk, so we populate input_memory with read records.
        for i in 0..16 {
            cols.input_memory[i].populate_read(event.input_memory_records[i], &mut Vec::new());
        }

        // Populate output memory write records.
        // The output is written at clk + 1, so we populate output_memory with write records.
        for i in 0..16 {
            cols.output_memory[i].populate_write(event.output_memory_records[i], &mut Vec::new());
        }

        // Populate permutation columns using the poseidon2 permutation function.
        let input_babybear: [F; 16] =
            core::array::from_fn(|i| F::from_wrapped_u32(event.input[i]));
        let perm_op = populate_perm_deg3(input_babybear, None);
        // Safety: Poseidon2Degree3Cols is #[repr(C)] with exactly 313 elements.
        let perm_slice: &[F] = unsafe {
            std::slice::from_raw_parts(
                &perm_op.permutation as *const _ as *const F,
                313,
            )
        };
        cols.perm.copy_from_slice(perm_slice);

        row
    }

    fn populate_byte_lookup_events(
        event: &Poseidon2Event,
        blu: &mut Vec<ByteLookupEvent>,
    ) {
        // Add byte range checks for the input values.
        for i in 0..16 {
            blu.add_u8_range_checks(&event.input[i].to_le_bytes());
        }
        // Add byte range checks for the output values.
        for i in 0..16 {
            blu.add_u8_range_checks(&event.output[i].to_le_bytes());
        }
    }
}
