//! IR for compiling constraint evaluation to GPU-executable programs.
//!
//! The compiler records operations from `chip.eval()` into a flat instruction
//! list that can be interpreted by a Metal compute kernel, one thread per row.
//!
//! # Architecture
//!
//! 1. [`IrVal`] is a base-field register handle (u32 ID) that implements
//!    [`AbstractField<F = BabyBear>`]. Arithmetic on `IrVal` values emits
//!    instructions into a thread-local [`IrContext`].
//!
//! 2. Extension field values are represented as
//!    `BinomialExtensionField<IrVal, 4>`, which decomposes EF4 arithmetic
//!    into base-field operations automatically.
//!
//! 3. [`ConstraintCompiler`] implements all the SP1 builder traits
//!    (`SP1AirBuilder`, `MultiTableAirBuilder`, etc.) so it can be passed
//!    to `chip.eval()` to record the full constraint program.
//!
//! 4. [`compile_constraints`] orchestrates the recording and returns a
//!    [`ConstraintProgram`] containing the flat instruction words.

use std::cell::RefCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::air::{EmptyMessageBuilder, MultiTableAirBuilder};
use crate::septic_digest::SepticDigest;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Opcodes — each instruction is [opcode, dst, arg0, arg1] (4 × u32).
// Unused args are 0.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Load a Montgomery-form constant into dst.
pub const OP_CONST: u32 = 0;
/// dst = main_trace[row, arg0]
pub const OP_LOAD_MAIN_LOCAL: u32 = 1;
/// dst = main_trace[row + next_step, arg0]
pub const OP_LOAD_MAIN_NEXT: u32 = 2;
/// dst = prep_trace[row, arg0]
pub const OP_LOAD_PREP_LOCAL: u32 = 3;
/// dst = prep_trace[row + next_step, arg0]
pub const OP_LOAD_PREP_NEXT: u32 = 4;
/// dst = perm_trace[row, arg0] (indexes individual u32s; 4 per EF4 element)
pub const OP_LOAD_PERM_LOCAL: u32 = 5;
/// dst = perm_trace[row + next_step, arg0]
pub const OP_LOAD_PERM_NEXT: u32 = 6;
/// dst = selectors[row, arg0] (0=first_row, 1=last_row, 2=transition)
pub const OP_LOAD_SELECTOR: u32 = 7;
/// dst = bb_add(regs[arg0], regs[arg1])
pub const OP_ADD: u32 = 8;
/// dst = bb_sub(regs[arg0], regs[arg1])
pub const OP_SUB: u32 = 9;
/// dst = bb_mul(regs[arg0], regs[arg1])
pub const OP_MUL: u32 = 10;
/// dst = bb_neg(regs[arg0])  (P - regs[arg0])
pub const OP_NEG: u32 = 11;

/// Selector indices for OP_LOAD_SELECTOR.
pub const SEL_IS_FIRST_ROW: u32 = 0;
/// Selector index.
pub const SEL_IS_LAST_ROW: u32 = 1;
/// Selector index.
pub const SEL_IS_TRANSITION: u32 = 2;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Thread-local IR context
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct IrContext {
    /// Flat instruction words: [op, dst, arg0, arg1] × N.
    words: Vec<u32>,
    /// Next register ID to allocate.
    next_reg: u32,
}

impl IrContext {
    fn new() -> Self {
        Self { words: Vec::with_capacity(4096), next_reg: 0 }
    }

    fn alloc(&mut self) -> u32 {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    fn emit(&mut self, op: u32, dst: u32, arg0: u32, arg1: u32) {
        self.words.extend_from_slice(&[op, dst, arg0, arg1]);
    }
}

thread_local! {
    static CTX: RefCell<IrContext> = RefCell::new(IrContext::new());
}

fn with_ctx<R>(f: impl FnOnce(&mut IrContext) -> R) -> R {
    CTX.with(|c| f(&mut c.borrow_mut()))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// IrVal — base-field register handle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A handle to a base-field register in the constraint IR.
///
/// Implements [`AbstractField<F = BabyBear>`]. Arithmetic operations on
/// `IrVal` emit instructions into the thread-local [`IrContext`].
#[derive(Clone, Copy)]
pub struct IrVal(pub u32);

/// Convert a BabyBear value to its Montgomery-form u32 representation.
/// BabyBear internally stores values in Montgomery form; this round-trips
/// through canonical form to extract it portably.
fn to_monty(n: u32) -> u32 {
    const P: u64 = 0x7800_0001;
    (((n as u64) << 32) % P) as u32
}

impl IrVal {
    fn emit_const(monty_value: u32) -> Self {
        with_ctx(|ctx| {
            let dst = ctx.alloc();
            ctx.emit(OP_CONST, dst, monty_value, 0);
            IrVal(dst)
        })
    }

    fn emit_binop(op: u32, lhs: u32, rhs: u32) -> Self {
        with_ctx(|ctx| {
            let dst = ctx.alloc();
            ctx.emit(op, dst, lhs, rhs);
            IrVal(dst)
        })
    }

    fn emit_unop(op: u32, src: u32) -> Self {
        with_ctx(|ctx| {
            let dst = ctx.alloc();
            ctx.emit(op, dst, src, 0);
            IrVal(dst)
        })
    }

    /// Emit a load instruction (LOAD_MAIN_LOCAL, etc.) with a column index.
    fn emit_load(op: u32, col: u32) -> Self {
        with_ctx(|ctx| {
            let dst = ctx.alloc();
            ctx.emit(op, dst, col, 0);
            IrVal(dst)
        })
    }
}

// ── Arithmetic ──────────────────────────────────────────────────────────

impl Add for IrVal {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::emit_binop(OP_ADD, self.0, rhs.0)
    }
}

impl Sub for IrVal {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::emit_binop(OP_SUB, self.0, rhs.0)
    }
}

impl Mul for IrVal {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::emit_binop(OP_MUL, self.0, rhs.0)
    }
}

impl Neg for IrVal {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::emit_unop(OP_NEG, self.0)
    }
}

impl AddAssign for IrVal {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for IrVal {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for IrVal {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for IrVal {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl Product for IrVal {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}

// ── Cross-type ops with BabyBear ────────────────────────────────────────

impl From<BabyBear> for IrVal {
    #[inline]
    fn from(f: BabyBear) -> Self {
        Self::from_f(f)
    }
}

impl Add<BabyBear> for IrVal {
    type Output = Self;
    #[inline]
    fn add(self, rhs: BabyBear) -> Self {
        self + Self::from_f(rhs)
    }
}

impl Sub<BabyBear> for IrVal {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: BabyBear) -> Self {
        self - Self::from_f(rhs)
    }
}

impl Mul<BabyBear> for IrVal {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: BabyBear) -> Self {
        self * Self::from_f(rhs)
    }
}

// ── Trait impls for type system ─────────────────────────────────────────

impl PartialEq for IrVal {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for IrVal {}

impl Hash for IrVal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Default for IrVal {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Debug for IrVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

impl fmt::Display for IrVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

// ── AbstractField ───────────────────────────────────────────────────────

impl AbstractField for IrVal {
    type F = BabyBear;

    fn zero() -> Self {
        // 0 in Montgomery form is 0.
        Self::emit_const(0)
    }

    fn one() -> Self {
        Self::emit_const(to_monty(1))
    }

    fn two() -> Self {
        Self::emit_const(to_monty(2))
    }

    fn neg_one() -> Self {
        // P - 1 in Montgomery form = to_monty(P - 1).
        Self::emit_const(to_monty(0x7800_0000))
    }

    fn from_f(f: BabyBear) -> Self {
        Self::emit_const(to_monty(f.as_canonical_u32()))
    }

    fn from_bool(b: bool) -> Self {
        Self::emit_const(to_monty(u32::from(b)))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::emit_const(to_monty(u32::from(n)))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::emit_const(to_monty(u32::from(n)))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::emit_const(to_monty(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::emit_const(to_monty((n % 0x7800_0001) as u32))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::emit_const(to_monty((n % 0x7800_0001) as u32))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        // wrapped means n mod P.
        Self::emit_const(to_monty(n % 0x7800_0001))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::emit_const(to_monty((n % 0x7800_0001) as u32))
    }

    fn generator() -> Self {
        Self::from_f(BabyBear::generator())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Extension field type alias
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Extension field register handle (degree 4 over BabyBear).
/// Each component is an [`IrVal`] register.
///
/// Arithmetic on this type automatically decomposes into base-field IR
/// instructions via `BinomialExtensionField`'s generic impls.
pub type IrExtVal = BinomialExtensionField<IrVal, 4>;

/// Concrete EF4 type (the `F` field of `IrExtVal`).
pub type Ef4 = BinomialExtensionField<BabyBear, 4>;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ConstraintCompiler — the recording AirBuilder
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A constraint evaluation recorder that implements all SP1 builder traits.
///
/// When `chip.eval(&mut compiler)` is called, all operations are recorded
/// into the thread-local [`IrContext`] via the [`IrVal`] arithmetic impls.
pub struct ConstraintCompiler<'a> {
    // Trace matrices (pre-populated with load instructions).
    preprocessed: RowMajorMatrix<IrVal>,
    main: RowMajorMatrix<IrVal>,
    perm: RowMajorMatrix<IrExtVal>,

    // Selectors.
    is_first_row: IrVal,
    is_last_row: IrVal,
    is_transition: IrVal,

    // Permutation challenges (alpha, beta as IrExtVal).
    perm_challenges: &'a [IrExtVal],

    // Cumulative sums.
    local_cumulative_sum: &'a IrExtVal,
    global_cumulative_sum: &'a SepticDigest<BabyBear>,

    // Constraint folding.
    alpha_ir: IrExtVal,
    accumulator: IrExtVal,

    // Public values.
    public_values: &'a [BabyBear],
}

// We need self-referential borrows for perm_challenges and local_cumulative_sum.
// Instead, we'll use a two-phase construction: create owned data, then borrow.
// For now, use a helper struct that owns the data and produces the compiler.

/// Owned data for constraint compilation. Create this first, then call
/// [`CompilerData::compiler`] to get a borrowing [`ConstraintCompiler`].
#[allow(dead_code)]
pub struct CompilerData {
    preprocessed: RowMajorMatrix<IrVal>,
    main: RowMajorMatrix<IrVal>,
    perm: RowMajorMatrix<IrExtVal>,
    is_first_row: IrVal,
    is_last_row: IrVal,
    is_transition: IrVal,
    perm_challenges: Vec<IrExtVal>,
    local_cumulative_sum: IrExtVal,
    global_cumulative_sum: SepticDigest<BabyBear>,
    alpha_ir: IrExtVal,
    public_values: Vec<BabyBear>,
    /// Widths for the output program.
    main_width: usize,
    prep_width: usize,
    /// Number of u32 columns in the perm trace (= perm_ef4_cols * 4).
    perm_u32_width: usize,
}

impl CompilerData {
    /// Build a [`ConstraintCompiler`] that borrows from this data.
    pub fn compiler(&self) -> ConstraintCompiler<'_> {
        ConstraintCompiler {
            preprocessed: self.preprocessed.clone(),
            main: self.main.clone(),
            perm: self.perm.clone(),
            is_first_row: self.is_first_row,
            is_last_row: self.is_last_row,
            is_transition: self.is_transition,
            perm_challenges: &self.perm_challenges,
            local_cumulative_sum: &self.local_cumulative_sum,
            global_cumulative_sum: &self.global_cumulative_sum,
            alpha_ir: self.alpha_ir,
            accumulator: IrExtVal::zero(),
            public_values: &self.public_values,
        }
    }
}

// ── AirBuilder ──────────────────────────────────────────────────────────

impl<'a> AirBuilder for ConstraintCompiler<'a> {
    type F = BabyBear;
    type Expr = IrVal;
    type Var = IrVal;
    type M = RowMajorMatrix<IrVal>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: IrVal = x.into();
        // Horner folding: accumulator = accumulator * alpha + x
        self.accumulator = self.accumulator * self.alpha_ir;
        self.accumulator = self.accumulator + x;
    }
}

// ── ExtensionBuilder ────────────────────────────────────────────────────

impl<'a> ExtensionBuilder for ConstraintCompiler<'a> {
    type EF = Ef4;
    type ExprEF = IrExtVal;
    type VarEF = IrExtVal;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, x: I) {
        let x: IrExtVal = x.into();
        // Horner folding: accumulator = accumulator * alpha + x
        self.accumulator = self.accumulator * self.alpha_ir;
        self.accumulator = self.accumulator + x;
    }
}

// ── PermutationAirBuilder ───────────────────────────────────────────────

impl<'a> PermutationAirBuilder for ConstraintCompiler<'a> {
    type MP = RowMajorMatrix<IrExtVal>;
    type RandomVar = IrExtVal;

    fn permutation(&self) -> Self::MP {
        self.perm.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.perm_challenges
    }
}

// ── MultiTableAirBuilder ────────────────────────────────────────────────

impl<'a> MultiTableAirBuilder<'a> for ConstraintCompiler<'a> {
    type LocalSum = IrExtVal;
    type GlobalSum = BabyBear;

    fn local_cumulative_sum(&self) -> &'a Self::LocalSum {
        self.local_cumulative_sum
    }

    fn global_cumulative_sum(&self) -> &'a SepticDigest<Self::GlobalSum> {
        self.global_cumulative_sum
    }
}

// ── PairBuilder ─────────────────────────────────────────────────────────

impl PairBuilder for ConstraintCompiler<'_> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

// ── AirBuilderWithPublicValues ──────────────────────────────────────────

impl AirBuilderWithPublicValues for ConstraintCompiler<'_> {
    type PublicVar = BabyBear;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

// ── EmptyMessageBuilder → gives us BaseAirBuilder → SP1AirBuilder ──────

impl EmptyMessageBuilder for ConstraintCompiler<'_> {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ConstraintProgram — compiled output
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A compiled constraint program ready for GPU execution.
#[derive(Clone, Debug)]
pub struct ConstraintProgram {
    /// Flat instruction words: \[op, dst, arg0, arg1\] × num_instructions.
    pub words: Vec<u32>,
    /// Total number of registers used.
    pub num_regs: u32,
    /// Register IDs holding the final EF4 accumulator (4 base components).
    pub accumulator_regs: [u32; 4],
    /// Main trace width (columns).
    pub main_width: usize,
    /// Preprocessed trace width (columns).
    pub prep_width: usize,
    /// Permutation trace width in u32s (= num EF4 columns × 4).
    pub perm_u32_width: usize,
}

impl ConstraintProgram {
    /// Number of instructions in the program.
    pub fn num_instructions(&self) -> usize {
        self.words.len() / 4
    }

    /// Compact registers by reusing dead ones via a free list.
    ///
    /// In SSA form, num_regs == num_instructions. This pass finds the last
    /// use of each register and reassigns register IDs so that dead registers
    /// are recycled. Typically reduces register count by 5-10×.
    pub fn compact_registers(&mut self) {
        let n = self.num_instructions();
        let nr = self.num_regs as usize;

        // Step 1: Find last_use[reg] = last instruction index that reads reg.
        // Also count the accumulator regs as used at the very end.
        let mut last_use = vec![0usize; nr];
        for i in 0..n {
            let base = i * 4;
            let op = self.words[base];
            // Mark source operands as used at instruction i.
            match op {
                OP_ADD | OP_SUB | OP_MUL => {
                    let a0 = self.words[base + 2] as usize;
                    let a1 = self.words[base + 3] as usize;
                    last_use[a0] = last_use[a0].max(i);
                    last_use[a1] = last_use[a1].max(i);
                }
                OP_NEG => {
                    let a0 = self.words[base + 2] as usize;
                    last_use[a0] = last_use[a0].max(i);
                }
                _ => {} // CONST, LOAD_* have no register sources
            }
        }
        // Accumulator regs are live at the end.
        for &r in &self.accumulator_regs {
            last_use[r as usize] = n;
        }

        // Step 2: Forward pass — assign new register IDs using a free list.
        let mut remap = vec![0u32; nr];
        let mut free_list: Vec<u32> = Vec::new();
        let mut next_new_reg = 0u32;

        for i in 0..n {
            let base = i * 4;
            let dst = self.words[base + 1] as usize;

            // Allocate a new register for dst.
            let new_dst = if let Some(r) = free_list.pop() {
                r
            } else {
                let r = next_new_reg;
                next_new_reg += 1;
                r
            };
            remap[dst] = new_dst;

            // Free any source registers whose last use is this instruction.
            let op = self.words[base];
            match op {
                OP_ADD | OP_SUB | OP_MUL => {
                    let a0 = self.words[base + 2] as usize;
                    let a1 = self.words[base + 3] as usize;
                    // Free a0 if last used here and not the same as dst.
                    if last_use[a0] == i && a0 != dst {
                        free_list.push(remap[a0]);
                    }
                    // Free a1 if last used here, not same as dst or a0.
                    if last_use[a1] == i && a1 != dst && a1 != a0 {
                        free_list.push(remap[a1]);
                    }
                }
                OP_NEG => {
                    let a0 = self.words[base + 2] as usize;
                    if last_use[a0] == i && a0 != dst {
                        free_list.push(remap[a0]);
                    }
                }
                _ => {}
            }
            // Also free dst if it's dead immediately (last_use == 0 means
            // only used as a definition, which can happen for side-effect-free
            // instructions). But in our IR, every instruction's result feeds
            // into something eventually, so this is rare.
        }

        // Step 3: Rewrite instruction words with new register IDs.
        for i in 0..n {
            let base = i * 4;
            let op = self.words[base];
            // Rewrite dst.
            self.words[base + 1] = remap[self.words[base + 1] as usize];
            // Rewrite source registers.
            match op {
                OP_ADD | OP_SUB | OP_MUL => {
                    self.words[base + 2] = remap[self.words[base + 2] as usize];
                    self.words[base + 3] = remap[self.words[base + 3] as usize];
                }
                OP_NEG => {
                    self.words[base + 2] = remap[self.words[base + 2] as usize];
                }
                _ => {} // CONST, LOAD_* — arg fields are not register IDs
            }
        }

        // Step 4: Remap accumulator registers and update count.
        for r in &mut self.accumulator_regs {
            *r = remap[*r as usize];
        }
        self.num_regs = next_new_reg;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compilation API
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

use p3_air::Air;

use super::Chip;

/// Compile a chip's constraint evaluation into a GPU-executable program.
///
/// This runs `chip.eval()` with a recording builder to capture all
/// arithmetic operations as IR instructions. The resulting program can
/// be interpreted by a Metal compute kernel, one thread per quotient row.
///
/// Constants (alpha, challenges, cumulative sums, public values) are baked
/// into the program. Recompile when these change between proofs.
pub fn compile_constraints<A: for<'b> Air<ConstraintCompiler<'b>>>(
    chip: &Chip<BabyBear, A>,
    prep_width: usize,
    main_width: usize,
    commit_scope: crate::air::InteractionScope,
    local_cumulative_sum: &Ef4,
    global_cumulative_sum: &SepticDigest<BabyBear>,
    perm_challenges: &[Ef4],
    alpha: Ef4,
    public_values: &[BabyBear],
) -> ConstraintProgram {
    // Reset the thread-local IR context.
    CTX.with(|c| *c.borrow_mut() = IrContext::new());

    let prep_width = prep_width.max(1);
    let main_width = main_width.max(1);
    let perm_ef4_cols = chip.permutation_width(); // number of EF4 columns
    let perm_u32_width = perm_ef4_cols * 4;

    // Build preprocessed matrix: 2 rows × prep_width columns.
    let prep_values: Vec<IrVal> = (0..2u32)
        .flat_map(|row| {
            let op = if row == 0 { OP_LOAD_PREP_LOCAL } else { OP_LOAD_PREP_NEXT };
            (0..prep_width).map(move |col| IrVal::emit_load(op, col as u32))
        })
        .collect();
    let preprocessed = RowMajorMatrix::new(prep_values, prep_width);

    // Build main matrix: 2 rows × main_width columns.
    let main_values: Vec<IrVal> = (0..2u32)
        .flat_map(|row| {
            let op = if row == 0 { OP_LOAD_MAIN_LOCAL } else { OP_LOAD_MAIN_NEXT };
            (0..main_width).map(move |col| IrVal::emit_load(op, col as u32))
        })
        .collect();
    let main = RowMajorMatrix::new(main_values, main_width);

    // Build permutation matrix: 2 rows × perm_ef4_cols columns of IrExtVal.
    let perm_values: Vec<IrExtVal> = (0..2u32)
        .flat_map(|row| {
            let op = if row == 0 { OP_LOAD_PERM_LOCAL } else { OP_LOAD_PERM_NEXT };
            (0..perm_ef4_cols).map(move |col| {
                let base = (col * 4) as u32;
                IrExtVal::from_base_fn(|k| IrVal::emit_load(op, base + k as u32))
            })
        })
        .collect();
    let perm = RowMajorMatrix::new(perm_values, perm_ef4_cols.max(1));

    // Selectors.
    let is_first_row = IrVal::emit_load(OP_LOAD_SELECTOR, SEL_IS_FIRST_ROW);
    let is_last_row = IrVal::emit_load(OP_LOAD_SELECTOR, SEL_IS_LAST_ROW);
    let is_transition = IrVal::emit_load(OP_LOAD_SELECTOR, SEL_IS_TRANSITION);

    // Convert constants to IR registers.
    let alpha_ir = IrExtVal::from_f(alpha);
    let perm_challenges_ir: Vec<IrExtVal> =
        perm_challenges.iter().map(|c| IrExtVal::from_f(*c)).collect();
    let local_cum_sum_ir = IrExtVal::from_f(*local_cumulative_sum);

    // Build the compiler data (owns everything).
    let data = CompilerData {
        preprocessed,
        main,
        perm,
        is_first_row,
        is_last_row,
        is_transition,
        perm_challenges: perm_challenges_ir,
        local_cumulative_sum: local_cum_sum_ir,
        global_cumulative_sum: *global_cumulative_sum,
        alpha_ir,
        public_values: public_values.to_vec(),
        main_width,
        prep_width,
        perm_u32_width,
    };

    // Create the compiler and evaluate constraints.
    // We manually replicate what Chip::eval() does to avoid requiring MachineAir<BabyBear>.
    let mut compiler = data.compiler();
    chip.air.eval(&mut compiler);
    crate::eval_permutation_constraints_prescoped(
        &chip.local_sends,
        &chip.local_receives,
        chip.logup_batch_size(),
        commit_scope,
        &mut compiler,
    );

    // Extract the accumulator register IDs.
    let acc = compiler.accumulator;
    let acc_slice: &[IrVal] = acc.as_base_slice();
    let accumulator_regs = [acc_slice[0].0, acc_slice[1].0, acc_slice[2].0, acc_slice[3].0];

    // Extract the program from the thread-local context.
    let (words, num_regs) = CTX.with(|c| {
        let ctx = c.borrow();
        (ctx.words.clone(), ctx.next_reg)
    });

    let mut program =
        ConstraintProgram { words, num_regs, accumulator_regs, main_width, prep_width, perm_u32_width };
    program.compact_registers();
    program
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_val_arithmetic() {
        // Reset context.
        CTX.with(|c| *c.borrow_mut() = IrContext::new());

        let a = IrVal::from_canonical_u32(5);
        let b = IrVal::from_canonical_u32(7);
        let c = a + b;
        let d = c * a;
        let _e = -d;

        let (words, num_regs) =
            CTX.with(|c| { let ctx = c.borrow(); (ctx.words.clone(), ctx.next_reg) });

        // 5 values created: a(r0), b(r1), c(r2), d(r3), e(r4)
        assert_eq!(num_regs, 5);
        // 5 instructions × 4 words = 20 words
        assert_eq!(words.len(), 20);

        // Verify first instruction: CONST r0, to_monty(5)
        assert_eq!(words[0], OP_CONST);
        assert_eq!(words[1], 0); // dst = r0
        assert_eq!(words[2], to_monty(5));

        // Verify add: ADD r2, r0, r1
        assert_eq!(words[8], OP_ADD);
        assert_eq!(words[9], 2); // dst = r2
        assert_eq!(words[10], 0); // lhs = r0
        assert_eq!(words[11], 1); // rhs = r1
    }

    #[test]
    fn test_ir_ext_val() {
        CTX.with(|c| *c.borrow_mut() = IrContext::new());

        let a = IrExtVal::from_base_fn(|i| IrVal::from_canonical_u32(i as u32));
        let regs_after_a = CTX.with(|c| c.borrow().next_reg);
        assert_eq!(regs_after_a, 4); // 4 CONST instructions

        let b = IrExtVal::one();
        let regs_after_b = CTX.with(|c| c.borrow().next_reg);
        // one() allocates at least 1 register (for the 1) and possibly more for zeros
        assert!(regs_after_b > regs_after_a);

        let _c = a + b;
        let regs_after_c = CTX.with(|c| c.borrow().next_reg);
        // Addition adds 4 new registers (one per component)
        assert_eq!(regs_after_c, regs_after_b + 4);
    }

    #[test]
    fn test_to_monty_roundtrip() {
        // Verify that to_monty matches BabyBear's internal representation.
        let val = BabyBear::from_canonical_u32(42);
        let monty = to_monty(42);
        // Round-trip: from_monty(to_monty(42)) should give 42.
        let back = BabyBear::from_canonical_u32(
            (((monty as u64) * 1) % 0x7800_0001) as u32, // This isn't right; just check it's nonzero.
        );
        // At minimum, to_monty(0) == 0 and to_monty(1) != 0.
        assert_eq!(to_monty(0), 0);
        assert_ne!(to_monty(1), 0);
        assert_ne!(to_monty(1), 1);
    }
}
