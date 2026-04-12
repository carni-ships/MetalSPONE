use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_symmetric::Permutation;
use sp1_primitives::poseidon2_init;

use crate::{
    events::{Poseidon2Event, PrecompileEvent},
    syscalls::{Syscall, SyscallCode, SyscallContext},
};

/// The number of elements in the Poseidon2 state.
const POSEIDON2_WIDTH: usize = 16;

/// A syscall that performs the Poseidon2 permutation.
pub struct Poseidon2Syscall;

impl Poseidon2Syscall {
    pub const fn new() -> Self {
        Self
    }
}

impl Syscall for Poseidon2Syscall {
    fn execute(
        &self,
        rt: &mut SyscallContext,
        syscall_code: SyscallCode,
        arg1: u32,
        arg2: u32,
    ) -> Option<u32> {
        let clk = rt.clk;

        let input_ptr = arg1;
        let output_ptr = arg2;

        // Initialize the Poseidon2 hasher.
        let poseidon2 = poseidon2_init();

        // Read the input state (16 u32 values).
        let (input_memory_records, input): ([_; 16], _) = {
            let mut records = [Default::default(); 16];
            let mut values = [0u32; 16];
            for i in 0..POSEIDON2_WIDTH {
                let (record, value) = rt.mr(input_ptr + (i * 4) as u32);
                records[i] = record;
                values[i] = value;
            }
            (records, values)
        };

        // Convert input to BabyBear values.
        let mut state: [BabyBear; POSEIDON2_WIDTH] =
            core::array::from_fn(|i| BabyBear::from_wrapped_u32(input[i]));

        // Apply the Poseidon2 permutation.
        poseidon2.permute_mut(&mut state);

        // Convert output back to u32 values.
        let output: [u32; POSEIDON2_WIDTH] =
            core::array::from_fn(|i| state[i].as_canonical_u32());

        // Increment clk so that the write is not at the same cycle as the read.
        rt.clk += 1;

        // Write the output state (16 u32 values).
        let output_memory_records: [_; 16] = {
            let mut records = [Default::default(); 16];
            for i in 0..POSEIDON2_WIDTH {
                let record = rt.mw(output_ptr + (i * 4) as u32, output[i]);
                records[i] = record;
            }
            records
        };

        let shard = rt.current_shard();
        let event = PrecompileEvent::Poseidon2(Poseidon2Event {
            shard,
            clk,
            input_ptr,
            output_ptr,
            input,
            output,
            input_memory_records,
            output_memory_records,
            local_mem_access: rt.postprocess(),
        });
        let syscall_event =
            rt.rt.syscall_event(clk, None, None, syscall_code, arg1, arg2, rt.next_pc);
        rt.add_precompile_event(syscall_code, syscall_event, event);

        None
    }

    fn num_extra_cycles(&self) -> u32 {
        1
    }
}
