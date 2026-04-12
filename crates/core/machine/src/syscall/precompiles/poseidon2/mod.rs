mod air;
mod columns;
mod trace;

pub use air::Poseidon2Chip;

#[cfg(test)]
mod tests {
    use sp1_core_executor::{syscalls::SyscallCode, Instruction, Opcode, Program};
    use sp1_stark::CpuProver;

    use crate::{
        io::SP1Stdin,
        utils::{run_test, setup_logger},
    };

    /// Create a simple program that calls the Poseidon2 syscall.
    pub fn poseidon2_program() -> Program {
        let input_ptr = 100;
        let output_ptr = 200;
        let mut instructions = vec![Instruction::new(Opcode::ADD, 29, 0, 5, false, true)];

        // Write input values (16 u32 values = 64 bytes)
        for i in 0..16 {
            instructions.extend(vec![Instruction::new(
                Opcode::ADD,
                30,
                0,
                input_ptr + i * 4,
                false,
                true,
            )]);
            instructions.extend(vec![Instruction::new(
                Opcode::SW,
                29,
                30,
                0,
                false,
                true,
            )]);
        }

        // Call the Poseidon2 syscall
        instructions.extend(vec![
            Instruction::new(Opcode::ADD, 5, 0, SyscallCode::POSEIDON2 as u32, false, true),
            Instruction::new(Opcode::ADD, 10, 0, input_ptr, false, true),
            Instruction::new(Opcode::ADD, 11, 0, output_ptr, false, true),
            Instruction::new(Opcode::ECALL, 5, 10, 11, false, false),
        ]);

        Program::new(instructions, 0, 0)
    }

    #[test]
    fn prove_poseidon2() {
        setup_logger();
        let program = poseidon2_program();
        let stdin = SP1Stdin::new();
        run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }
}