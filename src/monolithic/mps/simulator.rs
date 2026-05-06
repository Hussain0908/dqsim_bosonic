use std::collections::HashMap;

use nalgebra::DMatrix;
use num_complex::Complex64;
use pyo3::prelude::*;

use crate::gates;
use crate::monolithic::statevector::SimulationResult;
use crate::types::{Circuit, Instruction};

type C = Complex64;

#[derive(Clone)]
struct Tensor {
    left: usize,
    right: usize,
    data: Vec<C>,
}

impl Tensor {
    fn zero(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![C::new(0.0, 0.0); left * 2 * right],
        }
    }

    #[inline]
    fn idx(&self, l: usize, s: usize, r: usize) -> usize {
        (l * 2 + s) * self.right + r
    }

    #[inline]
    fn get(&self, l: usize, s: usize, r: usize) -> C {
        self.data[self.idx(l, s, r)]
    }

    #[inline]
    fn set(&mut self, l: usize, s: usize, r: usize, v: C) {
        let idx = self.idx(l, s, r);
        self.data[idx] = v;
    }
}

struct Mps {
    tensors: Vec<Tensor>,
}

impl Mps {
    fn new(num_qubits: usize) -> Self {
        let mut tensors = Vec::with_capacity(num_qubits);
        for _ in 0..num_qubits {
            let mut tensor = Tensor::zero(1, 1);
            tensor.set(0, 0, 0, C::new(1.0, 0.0));
            tensors.push(tensor);
        }
        Self { tensors }
    }

    fn apply_1q(&mut self, qubit: usize, mat: &[[C; 2]; 2]) {
        let old = self.tensors[qubit].clone();
        let mut new = Tensor::zero(old.left, old.right);
        for l in 0..old.left {
            for r in 0..old.right {
                for s_out in 0..2 {
                    let mut acc = C::new(0.0, 0.0);
                    for s_in in 0..2 {
                        acc += mat[s_out][s_in] * old.get(l, s_in, r);
                    }
                    new.set(l, s_out, r, acc);
                }
            }
        }
        self.tensors[qubit] = new;
    }

    fn apply_2q(&mut self, a: usize, b: usize, mat: &[[C; 4]; 4]) -> PyResult<()> {
        if a == b {
            return Ok(());
        }
        let (lo, hi, routed) = if a < b { (a, b, false) } else { (b, a, true) };

        for pos in ((lo + 1)..hi).rev() {
            self.apply_adjacent_2q(pos, &gates::swap())?;
        }

        let gate = if routed { routed_2q_matrix(mat) } else { *mat };
        self.apply_adjacent_2q(lo, &gate)?;

        for pos in (lo + 1)..hi {
            self.apply_adjacent_2q(pos, &gates::swap())?;
        }
        Ok(())
    }

    fn apply_adjacent_2q(&mut self, q: usize, mat: &[[C; 4]; 4]) -> PyResult<()> {
        let left_tensor = self.tensors[q].clone();
        let right_tensor = self.tensors[q + 1].clone();
        if left_tensor.right != right_tensor.left {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Invalid MPS bond dimensions",
            ));
        }

        let la = left_tensor.left;
        let bond = left_tensor.right;
        let rb = right_tensor.right;
        let rows = la * 2;
        let cols = 2 * rb;
        let mut matrix = DMatrix::<C>::zeros(rows, cols);

        for l in 0..la {
            for s0_out in 0..2 {
                for s1_out in 0..2 {
                    let row_gate = s0_out * 2 + s1_out;
                    let mut acc = C::new(0.0, 0.0);
                    for s0_in in 0..2 {
                        for s1_in in 0..2 {
                            let col_gate = s0_in * 2 + s1_in;
                            let mut theta = C::new(0.0, 0.0);
                            for x in 0..bond {
                                theta += left_tensor.get(l, s0_in, x)
                                    * right_tensor.get(x, s1_in, 0);
                            }
                            acc += mat[row_gate][col_gate] * theta;
                        }
                    }
                    matrix[(l * 2 + s0_out, s1_out * rb)] = acc;
                }
            }
        }

        if rb > 1 {
            matrix.fill(C::new(0.0, 0.0));
            for l in 0..la {
                for r in 0..rb {
                    for s0_out in 0..2 {
                        for s1_out in 0..2 {
                            let row_gate = s0_out * 2 + s1_out;
                            let mut acc = C::new(0.0, 0.0);
                            for s0_in in 0..2 {
                                for s1_in in 0..2 {
                                    let col_gate = s0_in * 2 + s1_in;
                                    let mut theta = C::new(0.0, 0.0);
                                    for x in 0..bond {
                                        theta += left_tensor.get(l, s0_in, x)
                                            * right_tensor.get(x, s1_in, r);
                                    }
                                    acc += mat[row_gate][col_gate] * theta;
                                }
                            }
                            matrix[(l * 2 + s0_out, s1_out * rb + r)] = acc;
                        }
                    }
                }
            }
        }

        let svd = matrix.svd(true, true);
        let u = svd.u.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("MPS SVD did not return U")
        })?;
        let vt = svd.v_t.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("MPS SVD did not return Vt")
        })?;
        let keep = svd
            .singular_values
            .iter()
            .filter(|&&s| s > 1e-12)
            .count()
            .max(1);

        let mut new_left = Tensor::zero(la, keep);
        let mut new_right = Tensor::zero(keep, rb);
        for l in 0..la {
            for s in 0..2 {
                let row = l * 2 + s;
                for k in 0..keep {
                    new_left.set(l, s, k, u[(row, k)]);
                }
            }
        }
        for k in 0..keep {
            let sigma = C::new(svd.singular_values[k], 0.0);
            for s in 0..2 {
                for r in 0..rb {
                    let col = s * rb + r;
                    new_right.set(k, s, r, sigma * vt[(k, col)]);
                }
            }
        }

        self.tensors[q] = new_left;
        self.tensors[q + 1] = new_right;
        Ok(())
    }

    fn to_statevector(&self) -> Vec<C> {
        let mut state = vec![C::new(1.0, 0.0)];
        let mut right_dim = 1;
        for tensor in &self.tensors {
            let basis_count = state.len() / tensor.left;
            let mut next = vec![C::new(0.0, 0.0); basis_count * 2 * tensor.right];
            for basis in 0..basis_count {
                for l in 0..tensor.left {
                    let amp = state[basis * right_dim + l];
                    for s in 0..2 {
                        for r in 0..tensor.right {
                            let idx = ((basis + s * basis_count) * tensor.right) + r;
                            next[idx] += amp * tensor.get(l, s, r);
                        }
                    }
                }
            }
            state = next;
            right_dim = tensor.right;
        }
        state
    }
}

#[pyclass]
pub struct MpsSimulator {
    _seed: Option<u64>,
}

#[pymethods]
impl MpsSimulator {
    #[new]
    #[pyo3(signature = (seed=None))]
    pub fn new(seed: Option<u64>) -> Self {
        Self { _seed: seed }
    }

    pub fn simulate(&self, _py: Python, circuit: &Bound<PyAny>) -> PyResult<SimulationResult> {
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let mut mps = Mps::new(rust_circuit.num_qubits());
        for inst in &rust_circuit.instructions {
            run_instruction(&mut mps, inst)?;
        }
        Ok(SimulationResult::new(
            mps.to_statevector(),
            rust_circuit.num_qubits(),
            HashMap::new(),
            None,
        ))
    }
}

fn run_instruction(mps: &mut Mps, inst: &Instruction) -> PyResult<()> {
    match inst {
        Instruction::Id { .. } | Instruction::U0 { .. } | Instruction::Barrier | Instruction::Classical { .. } => {}
        Instruction::X { qubit } => mps.apply_1q(*qubit, &gates::X),
        Instruction::Y { qubit } => mps.apply_1q(*qubit, &gates::Y),
        Instruction::Z { qubit } => mps.apply_1q(*qubit, &gates::Z),
        Instruction::H { qubit } => mps.apply_1q(*qubit, &gates::h()),
        Instruction::S { qubit } => mps.apply_1q(*qubit, &gates::s_gate()),
        Instruction::Sdg { qubit } => mps.apply_1q(*qubit, &gates::sdg()),
        Instruction::T { qubit } => mps.apply_1q(*qubit, &gates::t_gate()),
        Instruction::Tdg { qubit } => mps.apply_1q(*qubit, &gates::tdg()),
        Instruction::Sx { qubit } => mps.apply_1q(*qubit, &gates::sx()),
        Instruction::Sxdg { qubit } => mps.apply_1q(*qubit, &gates::sxdg()),
        Instruction::U3 { qubit, theta, phi, lam } | Instruction::U { qubit, theta, phi, lam } => {
            mps.apply_1q(*qubit, &gates::u3(*theta, *phi, *lam));
        }
        Instruction::U2 { qubit, phi, lam } => mps.apply_1q(*qubit, &gates::u2(*phi, *lam)),
        Instruction::U1 { qubit, lam } | Instruction::P { qubit, lam } => {
            mps.apply_1q(*qubit, &gates::u1(*lam));
        }
        Instruction::Rx { qubit, theta } => mps.apply_1q(*qubit, &gates::rx(*theta)),
        Instruction::Ry { qubit, theta } => mps.apply_1q(*qubit, &gates::ry(*theta)),
        Instruction::Rz { qubit, phi } => mps.apply_1q(*qubit, &gates::rz(*phi)),
        Instruction::Cx { control, target } => mps.apply_2q(*control, *target, &gates::cnot())?,
        Instruction::Cz { control, target } => mps.apply_2q(*control, *target, &gates::cz())?,
        Instruction::Cy { control, target } => mps.apply_2q(*control, *target, &gates::cy())?,
        Instruction::Ch { control, target } => mps.apply_2q(*control, *target, &gates::ch())?,
        Instruction::Swap { a, b } => mps.apply_2q(*a, *b, &gates::swap())?,
        Instruction::Csx { control, target } => mps.apply_2q(*control, *target, &gates::csx())?,
        Instruction::Crx { control, target, theta } => mps.apply_2q(*control, *target, &gates::crx(*theta))?,
        Instruction::Cry { control, target, theta } => mps.apply_2q(*control, *target, &gates::cry(*theta))?,
        Instruction::Crz { control, target, lam } => mps.apply_2q(*control, *target, &gates::crz(*lam))?,
        Instruction::Cu1 { control, target, lam } | Instruction::Cp { control, target, lam } => {
            mps.apply_2q(*control, *target, &gates::cu1(*lam))?;
        }
        Instruction::Cu3 { control, target, theta, phi, lam } => {
            mps.apply_2q(*control, *target, &gates::cu3(*theta, *phi, *lam))?;
        }
        Instruction::Cu { control, target, theta, phi, lam, gamma } => {
            mps.apply_2q(*control, *target, &gates::cu(*theta, *phi, *lam, *gamma))?;
        }
        Instruction::Rxx { a, b, theta } => mps.apply_2q(*a, *b, &gates::rxx(*theta))?,
        Instruction::Rzz { a, b, theta } => mps.apply_2q(*a, *b, &gates::rzz(*theta))?,
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "MPS simulator only supports unitary one- and two-qubit instructions"
            )));
        }
    }
    Ok(())
}

fn routed_2q_matrix(mat: &[[C; 4]; 4]) -> [[C; 4]; 4] {
    let mut out = [[C::new(0.0, 0.0); 4]; 4];
    for a_out in 0..2 {
        for b_out in 0..2 {
            for a_in in 0..2 {
                for b_in in 0..2 {
                    let row = a_out * 2 + b_out;
                    let col = a_in * 2 + b_in;
                    let swapped_row = b_out * 2 + a_out;
                    let swapped_col = b_in * 2 + a_in;
                    out[row][col] = mat[swapped_row][swapped_col];
                }
            }
        }
    }
    out
}
