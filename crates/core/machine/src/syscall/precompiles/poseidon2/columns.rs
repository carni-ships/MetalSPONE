use sp1_derive::AlignedBorrow;

use crate::memory::MemoryReadWriteCols;

/// The number of columns in the Poseidon2 chip.
///
/// Poseidon2 permutation state:
/// - External rounds state: 8 * 16 = 128
/// - Internal rounds state: 16
/// - Internal rounds s0: 12
/// - Output state: 16
/// - External rounds sbox: 8 * 16 = 128
/// - Internal rounds sbox: 13
/// Total poseidon2 cols: 313
///
/// Plus: shard, clk, input_ptr, output_ptr, is_real = 5
/// Plus: input_memory[16] and output_memory[16] = 32 MemoryReadWriteCols
pub const NUM_POSEIDON2_COLS: usize = 313 + 5 + 32 * 8;

/// A set of columns for the Poseidon2 precompile operation.
///
/// Each poseidon2 syscall takes 1 row. The input and output are stored in memory
/// and the permutation columns are populated using the existing poseidon2 trace population.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2Cols<T> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clock cycle of the syscall.
    pub clk: T,

    /// The pointer to the input state.
    pub input_ptr: T,

    /// The pointer to the output state.
    pub output_ptr: T,

    /// Memory access for reading the input (16 u32 values).
    pub input_memory: [MemoryReadWriteCols<T>; 16],

    /// Memory access for writing the output (16 u32 values).
    pub output_memory: [MemoryReadWriteCols<T>; 16],

    /// The poseidon2 permutation operation columns (313 u32 values).
    pub perm: [T; 313],

    /// Whether this is a real poseidon2 call.
    pub is_real: T,
}
