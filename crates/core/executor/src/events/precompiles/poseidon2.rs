use serde::{Deserialize, Serialize};

use crate::events::{
    memory::{MemoryReadRecord, MemoryWriteRecord},
    MemoryLocalEvent,
};

/// Poseidon2 Event.
///
/// This event is emitted when a Poseidon2 permutation is performed.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Poseidon2Event {
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u32,
    /// The pointer to the input state (16 u32 values).
    pub input_ptr: u32,
    /// The pointer to the output state (16 u32 values).
    pub output_ptr: u32,
    /// The input state.
    pub input: [u32; 16],
    /// The output state.
    pub output: [u32; 16],
    /// The memory records for reading the input.
    pub input_memory_records: [MemoryReadRecord; 16],
    /// The memory records for writing the output.
    pub output_memory_records: [MemoryWriteRecord; 16],
    /// The local memory access records.
    pub local_mem_access: Vec<MemoryLocalEvent>,
}
