use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, CpuEvent, MemoryReadRecord, MemoryRecordEnum},
    syscalls::SyscallCode,
    ByteOpcode::{self, U16Range},
    ExecutionRecord, Instruction, Program,
};
use sp1_stark::air::MachineAir;
use tracing::instrument;

use super::{columns::NUM_CPU_COLS, CpuChip};
use crate::{cpu::columns::CpuCols, memory::MemoryCols, utils::{zeroed_f_vec, NullByteRecord}};

impl<F: PrimeField32> MachineAir<F> for CpuChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        self.id().to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let n_real_rows = input.cpu_events.len();
        let padded_nb_rows = if let Some(shape) = &input.shape {
            shape.height(&self.id()).unwrap()
        } else if n_real_rows < 16 {
            16
        } else {
            n_real_rows.next_power_of_two()
        };
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_CPU_COLS);

        let chunk_size = std::cmp::max(input.cpu_events.len() / num_cpus::get(), 1);
        values.chunks_mut(chunk_size * NUM_CPU_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_CPU_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut CpuCols<F> = row.borrow_mut();

                    if idx >= input.cpu_events.len() {
                        cols.instruction.imm_b = F::one();
                        cols.instruction.imm_c = F::one();
                        cols.is_syscall = F::one();
                    } else {
                        let event = &input.cpu_events[idx];
                        let instruction = &input.program.fetch(event.pc);
                        self.event_to_row(
                            event,
                            cols,
                            &mut NullByteRecord,
                            input.public_values.execution_shard,
                            instruction,
                        );
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_CPU_COLS)
    }

    #[instrument(name = "generate cpu dependencies", level = "debug", skip_all)]
    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        let chunk_size = std::cmp::max(input.cpu_events.len() / num_cpus::get(), 1);
        let shard = input.public_values.execution_shard;

        let blu_events: Vec<_> = input
            .cpu_events
            .par_chunks(chunk_size)
            .map(|ops: &[CpuEvent]| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                ops.iter().for_each(|op| {
                    let instruction = &input.program.fetch(op.pc);
                    Self::collect_byte_lookups(op, &mut blu, shard, instruction);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            shard.contains_cpu()
        }
    }
}

impl CpuChip {
    /// Collect byte lookup events directly from a CPU event without allocating CpuCols.
    ///
    /// This is a lightweight alternative to `event_to_row` used by `generate_dependencies`.
    /// It produces identical byte lookup events but avoids allocating and populating the
    /// full 180+ field CpuCols struct per event.
    fn collect_byte_lookups(
        event: &CpuEvent,
        output: &mut impl ByteRecord,
        shard: u32,
        instruction: &Instruction,
    ) {
        // Shard/clk range checks (mirrors populate_shard_clk).
        let clk_16bit_limb = (event.clk & 0xffff) as u16;
        let clk_8bit_limb = ((event.clk >> 16) & 0xff) as u8;
        output.add_byte_lookup_event(ByteLookupEvent::new(U16Range, shard as u16, 0, 0, 0));
        output.add_byte_lookup_event(ByteLookupEvent::new(U16Range, clk_16bit_limb, 0, 0, 0));
        output.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U8Range,
            0,
            0,
            0,
            clk_8bit_limb,
        ));

        // Memory access time-diff range checks (mirrors populate_access).
        // For ecall instructions, op_a lookups are discarded (sent to dummy vec in event_to_row).
        if let Some(record) = event.a_record {
            if !instruction.is_ecall_instruction() {
                Self::collect_access_lookups(&record, output);
            }
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.b_record {
            Self::collect_read_access_lookups(&record, output);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.c_record {
            Self::collect_read_access_lookups(&record, output);
        }

        // a_bytes range checks (from op_a_access.access.value after populate).
        let a_value = match event.a_record {
            Some(ref r) => r.current_record().value,
            None => event.a,
        };
        let a_bytes = a_value.to_le_bytes();
        output.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[0],
            c: a_bytes[1],
        });
        output.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[2],
            c: a_bytes[3],
        });
    }

    /// Compute time-diff range checks for a memory record (read or write).
    #[inline(always)]
    fn collect_access_lookups(record: &MemoryRecordEnum, output: &mut impl ByteRecord) {
        let cur = record.current_record();
        let prev = record.previous_record();
        let use_clk = prev.shard == cur.shard;
        let (prev_t, cur_t) = if use_clk {
            (prev.timestamp, cur.timestamp)
        } else {
            (prev.shard, cur.shard)
        };
        let diff_minus_one = cur_t - prev_t - 1;
        output.add_u16_range_check((diff_minus_one & 0xffff) as u16);
        output.add_u8_range_check(0, ((diff_minus_one >> 16) & 0xff) as u8);
    }

    /// Compute time-diff range checks for a read record.
    #[inline(always)]
    fn collect_read_access_lookups(record: &MemoryReadRecord, output: &mut impl ByteRecord) {
        let use_clk = record.prev_shard == record.shard;
        let (prev_t, cur_t) = if use_clk {
            (record.prev_timestamp, record.timestamp)
        } else {
            (record.prev_shard, record.shard)
        };
        let diff_minus_one = cur_t - prev_t - 1;
        output.add_u16_range_check((diff_minus_one & 0xffff) as u16);
        output.add_u8_range_check(0, ((diff_minus_one >> 16) & 0xff) as u8);
    }

    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &CpuEvent,
        cols: &mut CpuCols<F>,
        blu_events: &mut impl ByteRecord,
        shard: u32,
        instruction: &Instruction,
    ) {
        // Populate shard and clk columns.
        self.populate_shard_clk(cols, event, blu_events, shard);

        // Populate basic fields.
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.instruction.populate(instruction);
        cols.op_a_immutable = F::from_bool(
            instruction.is_memory_store_instruction() || instruction.is_branch_instruction(),
        );
        cols.is_memory = F::from_bool(
            instruction.is_memory_load_instruction() || instruction.is_memory_store_instruction(),
        );
        cols.is_syscall = F::from_bool(instruction.is_ecall_instruction());
        *cols.op_a_access.value_mut() = event.a.into();
        *cols.op_b_access.value_mut() = event.b.into();
        *cols.op_c_access.value_mut() = event.c.into();

        cols.shard_to_send = if instruction.is_memory_load_instruction()
            || instruction.is_memory_store_instruction()
            || instruction.is_ecall_instruction()
        {
            cols.shard
        } else {
            F::zero()
        };
        cols.clk_to_send = if instruction.is_memory_load_instruction()
            || instruction.is_memory_store_instruction()
            || instruction.is_ecall_instruction()
        {
            F::from_canonical_u32(event.clk)
        } else {
            F::zero()
        };

        // Populate memory accesses for a, b, and c.
        if let Some(record) = event.a_record {
            if instruction.is_ecall_instruction() {
                // For ecall instructions, pass in a dummy byte lookup vector.  This syscall instruction
                // chip also has a op_a_access field that will be populated and that will contribute
                // to the byte lookup dependencies.
                cols.op_a_access.populate(record, &mut Vec::new());
            } else {
                cols.op_a_access.populate(record, blu_events);
            }
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.b_record {
            cols.op_b_access.populate(record, blu_events);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.c_record {
            cols.op_c_access.populate(record, blu_events);
        }

        if instruction.is_ecall_instruction() {
            let syscall_id = cols.op_a_access.prev_value[0];
            let num_extra_cycles = cols.op_a_access.prev_value[2];
            cols.is_halt =
                F::from_bool(syscall_id == F::from_canonical_u32(SyscallCode::HALT.syscall_id()));
            cols.num_extra_cycles = num_extra_cycles;
        }

        // Populate range checks for a.
        let a_bytes = cols
            .op_a_access
            .access
            .value
            .0
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        blu_events.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[0] as u8,
            c: a_bytes[1] as u8,
        });
        blu_events.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[2] as u8,
            c: a_bytes[3] as u8,
        });

        // Assert that the instruction is not a no-op.
        cols.is_real = F::one();
    }

    /// Populates the shard and clk related rows.
    fn populate_shard_clk<F: PrimeField>(
        &self,
        cols: &mut CpuCols<F>,
        event: &CpuEvent,
        blu_events: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.shard = F::from_canonical_u32(shard);

        let clk_16bit_limb = (event.clk & 0xffff) as u16;
        let clk_8bit_limb = ((event.clk >> 16) & 0xff) as u8;
        cols.clk_16bit_limb = F::from_canonical_u16(clk_16bit_limb);
        cols.clk_8bit_limb = F::from_canonical_u8(clk_8bit_limb);

        blu_events.add_byte_lookup_event(ByteLookupEvent::new(U16Range, shard as u16, 0, 0, 0));
        blu_events.add_byte_lookup_event(ByteLookupEvent::new(U16Range, clk_16bit_limb, 0, 0, 0));
        blu_events.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U8Range,
            0,
            0,
            0,
            clk_8bit_limb as u8,
        ));
    }
}
