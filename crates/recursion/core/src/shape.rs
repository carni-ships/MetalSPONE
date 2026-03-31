#![allow(clippy::never_loop)]

use std::marker::PhantomData;

use hashbrown::HashMap;

use itertools::Itertools;
use p3_field::{extension::BinomiallyExtendable, PrimeField32};
use serde::{Deserialize, Serialize};
use sp1_stark::{air::MachineAir, shape::OrderedShape};

use crate::{
    chips::{
        alu_base::BaseAluChip,
        alu_ext::ExtAluChip,
        batch_fri::BatchFRIChip,
        exp_reverse_bits::ExpReverseBitsLenChip,
        mem::{MemoryConstChip, MemoryVarChip},
        poseidon2_wide::Poseidon2WideChip,
        public_values::{PublicValuesChip, PUB_VALUES_LOG_HEIGHT},
        select::SelectChip,
    },
    machine::RecursionAir,
    RecursionProgram, D,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecursionShape {
    pub(crate) inner: HashMap<String, usize>,
}

impl RecursionShape {
    pub fn clone_into_hash_map(&self) -> HashMap<String, usize> {
        self.inner.clone()
    }
}

impl From<HashMap<String, usize>> for RecursionShape {
    fn from(value: HashMap<String, usize>) -> Self {
        Self { inner: value }
    }
}

pub struct RecursionShapeConfig<F, A> {
    allowed_shapes: Vec<HashMap<String, usize>>,
    _marker: PhantomData<(F, A)>,
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize>
    RecursionShapeConfig<F, RecursionAir<F, DEGREE>>
{
    pub fn fix_shape(&self, program: &mut RecursionProgram<F>) {
        let heights = RecursionAir::<F, DEGREE>::heights(program);

        // Log actual heights for profiling.
        let mut height_strs: Vec<String> = heights.iter()
            .map(|(name, h)| format!("{}={}", name, p3_util::log2_ceil_usize(*h)))
            .collect();
        height_strs.sort();
        tracing::info!("recursion heights: {}", height_strs.join(" "));

        let mut closest_shape = None;

        for shape in self.allowed_shapes.iter() {
            // If any of the heights is greater than the shape, continue.
            let mut valid = true;
            for (name, height) in heights.iter() {
                if *height > (1 << shape.get(name).unwrap()) {
                    valid = false;
                }
            }

            if !valid {
                continue;
            }

            closest_shape = Some(shape.clone());
            break;
        }

        if let Some(shape) = closest_shape {
            let shape = RecursionShape { inner: shape };
            *program.shape_mut() = Some(shape);
        } else {
            panic!("no shape found for heights: {:?}", heights);
        }
    }

    pub fn get_all_shape_combinations(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Vec<OrderedShape>> + '_ {
        (0..batch_size)
            .map(|_| {
                self.allowed_shapes
                    .iter()
                    .cloned()
                    .map(|map| map.into_iter().collect::<OrderedShape>())
            })
            .multi_cartesian_product()
    }

    pub fn union_config_with_extra_room(&self) -> Self {
        let mut map = HashMap::new();
        for shape in self.allowed_shapes.clone() {
            for key in shape.keys() {
                let current = map.get(key).unwrap_or(&0);
                map.insert(key.clone(), *current.max(shape.get(key).unwrap()));
            }
        }
        map.values_mut().for_each(|x| *x += 2);
        map.insert("PublicValues".to_string(), 4);
        Self { allowed_shapes: vec![map], _marker: PhantomData }
    }

    pub fn from_hash_map(hash_map: &HashMap<String, usize>) -> Self {
        Self { allowed_shapes: vec![hash_map.clone()], _marker: PhantomData }
    }

    pub fn first(&self) -> Option<&HashMap<String, usize>> {
        self.allowed_shapes.first()
    }
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> Default
    for RecursionShapeConfig<F, RecursionAir<F, DEGREE>>
{
    fn default() -> Self {
        // Get the names of all the recursion airs to make the shape specification more readable.
        let mem_const = RecursionAir::<F, DEGREE>::MemoryConst(MemoryConstChip::default()).name();
        let mem_var = RecursionAir::<F, DEGREE>::MemoryVar(MemoryVarChip::default()).name();
        let base_alu = RecursionAir::<F, DEGREE>::BaseAlu(BaseAluChip).name();
        let ext_alu = RecursionAir::<F, DEGREE>::ExtAlu(ExtAluChip).name();
        let poseidon2_wide =
            RecursionAir::<F, DEGREE>::Poseidon2Wide(Poseidon2WideChip::<DEGREE>).name();
        let batch_fri = RecursionAir::<F, DEGREE>::BatchFRI(BatchFRIChip::<DEGREE>).name();
        let select = RecursionAir::<F, DEGREE>::Select(SelectChip).name();
        let exp_reverse_bits_len =
            RecursionAir::<F, DEGREE>::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>).name();
        let public_values = RecursionAir::<F, DEGREE>::PublicValues(PublicValuesChip).name();

        // Specify allowed shapes.

        let allowed_shapes = [
            // Micro shape: for FRI_QUERIES≤1 + LOG_BLOWUP≥3.
            [
                (mem_var.clone(), 14),
                (select.clone(), 13),
                (mem_const.clone(), 14),
                (batch_fri.clone(), 12),
                (base_alu.clone(), 13),
                (ext_alu.clone(), 15),
                (exp_reverse_bits_len.clone(), 12),
                (poseidon2_wide.clone(), 12),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Super-minimal shape: for FRI_QUERIES≤4 + LOG_BLOWUP≥3.
            [
                (mem_var.clone(), 15),
                (select.clone(), 15),
                (mem_const.clone(), 14),
                (batch_fri.clone(), 14),
                (base_alu.clone(), 13),
                (ext_alu.clone(), 15),
                (exp_reverse_bits_len.clone(), 14),
                (poseidon2_wide.clone(), 13),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Ultra-minimal shape: for FRI_QUERIES≤8 + LOG_BLOWUP≥2.
            [
                (mem_var.clone(), 16),
                (select.clone(), 16),
                (mem_const.clone(), 15),
                (batch_fri.clone(), 15),
                (base_alu.clone(), 14),
                (ext_alu.clone(), 15),
                (exp_reverse_bits_len.clone(), 15),
                (poseidon2_wide.clone(), 14),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Minimal shape: for low-FRI-query configs (e.g. FRI_QUERIES≤16).
            [
                (mem_var.clone(), 17),
                (select.clone(), 17),
                (mem_const.clone(), 16),
                (batch_fri.clone(), 16),
                (base_alu.clone(), 15),
                (ext_alu.clone(), 15),
                (exp_reverse_bits_len.clone(), 16),
                (poseidon2_wide.clone(), 15),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Tight shape for FRI_QUERIES≤33.
            [
                (mem_var.clone(), 18),
                (select.clone(), 18),
                (mem_const.clone(), 16),
                (batch_fri.clone(), 17),
                (base_alu.clone(), 15),
                (ext_alu.clone(), 16),
                (exp_reverse_bits_len.clone(), 17),
                (poseidon2_wide.clone(), 16),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Fastest shape (small programs).
            [
                (mem_var.clone(), 19),
                (select.clone(), 19),
                (mem_const.clone(), 17),
                (batch_fri.clone(), 19),
                (base_alu.clone(), 16),
                (ext_alu.clone(), 16),
                (exp_reverse_bits_len.clone(), 18),
                (poseidon2_wide.clone(), 17),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Tight shape for bench-sized programs.
            [
                (mem_var.clone(), 19),
                (select.clone(), 20),
                (mem_const.clone(), 18),
                (batch_fri.clone(), 19),
                (base_alu.clone(), 17),
                (ext_alu.clone(), 17),
                (exp_reverse_bits_len.clone(), 18),
                (poseidon2_wide.clone(), 17),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Intermediate shape: tight + Poseidon2Wide=18 + MemoryVar=20.
            [
                (mem_var.clone(), 20),
                (select.clone(), 20),
                (mem_const.clone(), 18),
                (batch_fri.clone(), 19),
                (base_alu.clone(), 17),
                (ext_alu.clone(), 17),
                (exp_reverse_bits_len.clone(), 18),
                (poseidon2_wide.clone(), 18),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Second fastest shape.
            [
                (mem_var.clone(), 20),
                (select.clone(), 20),
                (mem_const.clone(), 18),
                (batch_fri.clone(), 21),
                (base_alu.clone(), 17),
                (ext_alu.clone(), 19),
                (exp_reverse_bits_len.clone(), 18),
                (poseidon2_wide.clone(), 18),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Largest shape (for programs with heavy ALU usage).
            [
                (mem_var.clone(), 21),
                (select.clone(), 21),
                (mem_const.clone(), 19),
                (batch_fri.clone(), 21),
                (base_alu.clone(), 18),
                (ext_alu.clone(), 20),
                (exp_reverse_bits_len.clone(), 19),
                (poseidon2_wide.clone(), 19),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
        ]
        .map(HashMap::from)
        .to_vec();
        Self { allowed_shapes, _marker: PhantomData }
    }
}
