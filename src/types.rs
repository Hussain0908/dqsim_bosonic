use std::collections::HashMap;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Register {
    pub name: String,
    pub size: usize,
    pub base: usize,
}

#[derive(Deserialize)]
pub struct Circuit {
    pub qregs: HashMap<String, Register>,
    pub instructions: Vec<Instruction>,
}

impl Circuit {
    pub fn num_qubits(&self) -> usize {
        self.qregs
            .values()
            .map(|r| r.base + r.size)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Deserialize, Clone)]
pub struct Condition {
    pub creg_base: usize,
    pub creg_size: usize,
    pub creg_value: u64,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Instruction {
    // -----------------------------------------------------------------------
    // Single-qubit fixed
    // -----------------------------------------------------------------------
    Id     { qubit: usize },
    X      { qubit: usize },
    Y      { qubit: usize },
    Z      { qubit: usize },
    H      { qubit: usize },
    S      { qubit: usize },
    Sdg    { qubit: usize },
    T      { qubit: usize },
    Tdg    { qubit: usize },
    Sx     { qubit: usize },
    Sxdg   { qubit: usize },
    // -----------------------------------------------------------------------
    // Single-qubit parametric
    // -----------------------------------------------------------------------
    U3  { qubit: usize, theta: f64, phi: f64, lam: f64 },
    U2  { qubit: usize, phi: f64, lam: f64 },
    U1  { qubit: usize, lam: f64 },
    U   { qubit: usize, theta: f64, phi: f64, lam: f64 },
    P   { qubit: usize, lam: f64 },
    Rx  { qubit: usize, theta: f64 },
    Ry  { qubit: usize, theta: f64 },
    Rz  { qubit: usize, phi: f64 },
    U0  { qubit: usize },
    // -----------------------------------------------------------------------
    // Two-qubit fixed
    // -----------------------------------------------------------------------
    Cx   { control: usize, target: usize },
    Cz   { control: usize, target: usize },
    Cy   { control: usize, target: usize },
    Ch   { control: usize, target: usize },
    Swap { a: usize, b: usize },
    Csx  { control: usize, target: usize },
    // -----------------------------------------------------------------------
    // Two-qubit parametric
    // -----------------------------------------------------------------------
    Crx { control: usize, target: usize, theta: f64 },
    Cry { control: usize, target: usize, theta: f64 },
    Crz { control: usize, target: usize, lam: f64 },
    Cu1 { control: usize, target: usize, lam: f64 },
    Cp  { control: usize, target: usize, lam: f64 },
    Cu3 { control: usize, target: usize, theta: f64, phi: f64, lam: f64 },
    Cu  { control: usize, target: usize, theta: f64, phi: f64, lam: f64, gamma: f64 },
    Rxx { a: usize, b: usize, theta: f64 },
    Rzz { a: usize, b: usize, theta: f64 },
    // -----------------------------------------------------------------------
    // Three-qubit
    // -----------------------------------------------------------------------
    Ccx     { control1: usize, control2: usize, target: usize },
    Cswap   { control: usize, target1: usize, target2: usize },
    Rccx    { control1: usize, control2: usize, target: usize },
    Rc3x    { control1: usize, control2: usize, control3: usize, target: usize },
    C3x     { control1: usize, control2: usize, control3: usize, target: usize },
    C3sqrtx { control1: usize, control2: usize, control3: usize, target: usize },
    C4x     { control1: usize, control2: usize, control3: usize, control4: usize, target: usize },
    // -----------------------------------------------------------------------
    // Generic / cross-node
    // -----------------------------------------------------------------------
    Gate { name: String, params: Vec<f64>, qubits: Vec<usize> },
    // -----------------------------------------------------------------------
    // Measurement and classical control
    // -----------------------------------------------------------------------
    Measure    { qubit: usize, cbit: usize },
    Reset      { qubit: usize },
    Conditional { condition: Condition, op: Box<Instruction> },
    // -----------------------------------------------------------------------
    // No-ops
    // -----------------------------------------------------------------------
    Barrier,
    Classical  { name: String },
}

impl Instruction {
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Instruction::X { qubit }
            | Instruction::Y { qubit }
            | Instruction::Z { qubit }
            | Instruction::H { qubit }
            | Instruction::S { qubit }
            | Instruction::Sdg { qubit }
            | Instruction::T { qubit }
            | Instruction::Tdg { qubit }
            | Instruction::Sx { qubit }
            | Instruction::Sxdg { qubit }
            | Instruction::U0 { qubit }
            | Instruction::Id { qubit }
            | Instruction::Reset { qubit } => vec![*qubit],

            Instruction::U3 { qubit, .. }
            | Instruction::U2 { qubit, .. }
            | Instruction::U1 { qubit, .. }
            | Instruction::U { qubit, .. }
            | Instruction::P { qubit, .. }
            | Instruction::Rx { qubit, .. }
            | Instruction::Ry { qubit, .. }
            | Instruction::Rz { qubit, .. } => vec![*qubit],

            Instruction::Measure { qubit, .. } => vec![*qubit],

            Instruction::Cx { control, target }
            | Instruction::Cz { control, target }
            | Instruction::Cy { control, target }
            | Instruction::Ch { control, target }
            | Instruction::Csx { control, target }
            | Instruction::Crx { control, target, .. }
            | Instruction::Cry { control, target, .. }
            | Instruction::Crz { control, target, .. }
            | Instruction::Cu1 { control, target, .. }
            | Instruction::Cp { control, target, .. }
            | Instruction::Cu3 { control, target, .. }
            | Instruction::Cu { control, target, .. } => vec![*control, *target],

            Instruction::Swap { a, b }
            | Instruction::Rxx { a, b, .. }
            | Instruction::Rzz { a, b, .. } => vec![*a, *b],

            Instruction::Ccx { control1, control2, target }
            | Instruction::Rccx { control1, control2, target } => {
                vec![*control1, *control2, *target]
            }

            Instruction::Cswap { control, target1, target2 } => {
                vec![*control, *target1, *target2]
            }

            Instruction::Rc3x { control1, control2, control3, target }
            | Instruction::C3x { control1, control2, control3, target }
            | Instruction::C3sqrtx { control1, control2, control3, target } => {
                vec![*control1, *control2, *control3, *target]
            }

            Instruction::C4x { control1, control2, control3, control4, target } => {
                vec![*control1, *control2, *control3, *control4, *target]
            }

            Instruction::Gate { qubits, .. } => qubits.clone(),

            Instruction::Conditional { op, .. } => op.qubits(),

            Instruction::Barrier | Instruction::Classical { .. } => vec![],
        }
    }
}
