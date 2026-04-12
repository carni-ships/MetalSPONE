use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_core_executor::syscalls::SyscallCode;
use sp1_stark::{
    air::{InteractionScope, SP1AirBuilder},
};

#[derive(Default)]
pub struct Poseidon2Chip;

use super::columns::{Poseidon2Cols, NUM_POSEIDON2_COLS};
use crate::air::MemoryAirBuilder;

impl<F> BaseAir<F> for Poseidon2Chip {
    fn width(&self) -> usize {
        NUM_POSEIDON2_COLS
    }
}

impl<AB> Air<AB> for Poseidon2Chip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &Poseidon2Cols<AB::Var> = (*local).borrow();
        let next: &Poseidon2Cols<AB::Var> = (*next).borrow();

        // Assert that is_real is a boolean.
        builder.assert_bool(local.is_real);

        // Receive the syscall arguments.
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_canonical_u32(SyscallCode::POSEIDON2.syscall_id()),
            local.input_ptr,
            local.output_ptr,
            local.is_real,
            InteractionScope::Local,
        );

        // Evaluate memory access for input (read at clk).
        for i in 0..16 {
            builder.eval_memory_access(
                local.shard,
                local.clk,
                local.input_ptr + AB::Expr::from_canonical_u32((i * 4) as u32),
                &local.input_memory[i],
                local.is_real,
            );
        }

        // Evaluate memory access for output (write at clk + 1).
        for i in 0..16 {
            builder.eval_memory_access(
                local.shard,
                local.clk + AB::Expr::one(),
                local.output_ptr + AB::Expr::from_canonical_u32((i * 4) as u32),
                &local.output_memory[i],
                local.is_real,
            );
        }

        // Verify that if is_real is true, then next row's is_real should be false (since we have
        // only 1 row per event).
        let mut transition_builder = builder.when_transition();
        transition_builder.when(local.is_real).assert_zero(next.is_real);

        // TODO: Add poseidon2 permutation constraints using the existing poseidon2 constraint
        // evaluation functions from machine/src/operations/poseidon2/air.rs
    }
}
