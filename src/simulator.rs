use std::collections::HashMap;
use std::time::Instant;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::engine::{apply_n_qubit, apply_one_qubit, marginal_probs, measure_qubit, sample_counts};
use crate::gates;
use crate::types::{Circuit, Instruction};

type C = Complex64;

// ---------------------------------------------------------------------------
// Profiling accumulator (internal, not a pyclass)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ProfileAcc {
    oq_calls: u64,
    oq_time: f64,
    nq_calls: u64,
    nq_time: f64,
    mq_calls: u64,
    mq_time: f64,
}

// ---------------------------------------------------------------------------
// SimulationProfile
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SimulationProfile {
    #[pyo3(get)]
    pub apply_one_qubit_calls: u64,
    #[pyo3(get)]
    pub apply_one_qubit_time: f64,
    #[pyo3(get)]
    pub apply_n_qubit_calls: u64,
    #[pyo3(get)]
    pub apply_n_qubit_time: f64,
    #[pyo3(get)]
    pub measure_qubit_calls: u64,
    #[pyo3(get)]
    pub measure_qubit_time: f64,
    #[pyo3(get)]
    pub total_time: f64,
}

#[pymethods]
impl SimulationProfile {
    fn __repr__(&self) -> String {
        let total = self.total_time.max(1e-9);
        format!(
            "SimulationProfile (total: {:.2} ms)\n  apply_one_qubit : {:4} calls  {:8.2} ms  ({:.1}%)\n  apply_n_qubit   : {:4} calls  {:8.2} ms  ({:.1}%)\n  measure_qubit   : {:4} calls  {:8.2} ms  ({:.1}%)",
            self.total_time * 1000.0,
            self.apply_one_qubit_calls,
            self.apply_one_qubit_time * 1000.0,
            100.0 * self.apply_one_qubit_time / total,
            self.apply_n_qubit_calls,
            self.apply_n_qubit_time * 1000.0,
            100.0 * self.apply_n_qubit_time / total,
            self.measure_qubit_calls,
            self.measure_qubit_time * 1000.0,
            100.0 * self.measure_qubit_time / total,
        )
    }
}

// ---------------------------------------------------------------------------
// SimulationResult
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SimulationResult {
    sv: Vec<C>,
    #[pyo3(get)]
    pub num_qubits: usize,
    cbits: HashMap<usize, i32>,
    prof: Option<Py<SimulationProfile>>,
}

#[pymethods]
impl SimulationResult {
    /// Raw complex amplitudes as a NumPy array of shape (2^n,).
    #[getter]
    fn statevector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C>> {
        self.sv.clone().into_pyarray_bound(py)
    }

    /// Final classical register state. Keys are absolute cbit indices.
    #[getter]
    fn classical_bits(&self, py: Python) -> PyObject {
        let d = PyDict::new_bound(py);
        for (&k, &v) in &self.cbits {
            d.set_item(k, v).unwrap();
        }
        d.into()
    }

    /// Profiling data, or None if the simulator was not run with profile=True.
    #[getter]
    fn profile(&self, py: Python) -> PyObject {
        match &self.prof {
            None => py.None(),
            Some(p) => p.clone_ref(py).into_py(py),
        }
    }

    /// Full or marginal probability distribution. Keys are integer basis states.
    /// qubits[0] is MSB of the output index. If None, all qubits in descending order.
    #[pyo3(signature = (qubits=None))]
    fn probabilities(&self, py: Python, qubits: Option<Vec<usize>>) -> PyObject {
        let qs: Vec<usize> = qubits.unwrap_or_else(|| (0..self.num_qubits).rev().collect());
        let probs = marginal_probs(&self.sv, self.num_qubits, &qs);
        let d = PyDict::new_bound(py);
        for (j, p) in probs.iter().enumerate() {
            if *p > 0.0 {
                d.set_item(j, p).unwrap();
            }
        }
        d.into()
    }

    /// Sample the distribution. Bitstrings have qubits[0] as the leftmost (MSB) character.
    #[pyo3(signature = (shots=1000, qubits=None, seed=None))]
    fn counts(
        &self,
        py: Python,
        shots: usize,
        qubits: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyObject {
        let qs: Vec<usize> = qubits.unwrap_or_else(|| (0..self.num_qubits).rev().collect());
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        let c = sample_counts(&self.sv, self.num_qubits, shots, &mut rng, Some(&qs));
        let d = PyDict::new_bound(py);
        for (k, v) in c {
            d.set_item(k, v).unwrap();
        }
        d.into()
    }

    /// Compute |<self|other>|^2.
    fn fidelity(&self, other: PyReadonlyArray1<C>) -> f64 {
        let arr = other.as_array();
        if arr.len() != self.sv.len() {
            return 0.0;
        }
        let dot: C = self.sv.iter().zip(arr.iter()).map(|(a, b)| a.conj() * b).sum();
        dot.norm_sqr()
    }
}

// ---------------------------------------------------------------------------
// StatevectorSimulator
// ---------------------------------------------------------------------------

#[pyclass]
pub struct StatevectorSimulator {
    seed: Option<u64>,
    profile: bool,
}

#[pymethods]
impl StatevectorSimulator {
    #[new]
    #[pyo3(signature = (seed=None, profile=false))]
    pub fn new(seed: Option<u64>, profile: bool) -> Self {
        Self { seed, profile }
    }

    /// Run the circuit and return a SimulationResult.
    /// Accepts both Circuit and DistributedCircuit.
    pub fn simulate(&self, py: Python, circuit: &Bound<PyAny>) -> PyResult<SimulationResult> {
        // 1. Get monolithic circuit (one boundary crossing if DistributedCircuit)
        let monolithic = if circuit.hasattr("as_monolithic_circuit")? {
            circuit.call_method0("as_monolithic_circuit")?
        } else {
            circuit.clone()
        };

        // 2. Serialize entire circuit to JSON (one boundary crossing)
        let json_str: String = monolithic
            .call_method0("model_dump_json")?
            .extract()?;

        // 3. Deserialize in Rust — no more Python calls until we return
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        // 4. Initialise statevector |0...0⟩
        let n = rust_circuit.num_qubits();
        let mut state = vec![C::new(0.0, 0.0); 1 << n];
        state[0] = C::new(1.0, 0.0);
        let mut cbits: HashMap<usize, i32> = HashMap::new();

        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut acc: Option<ProfileAcc> = if self.profile {
            Some(ProfileAcc::default())
        } else {
            None
        };

        let t0 = Instant::now();

        // 5. Execute all instructions natively in Rust
        for inst in &rust_circuit.instructions {
            run_instruction(&mut state, inst, n, &mut cbits, &mut rng, &mut acc)?;
        }

        let total_time = t0.elapsed().as_secs_f64();

        // 6. Build SimulationProfile if requested
        let prof = acc
            .map(|a| {
                Py::new(
                    py,
                    SimulationProfile {
                        apply_one_qubit_calls: a.oq_calls,
                        apply_one_qubit_time: a.oq_time,
                        apply_n_qubit_calls: a.nq_calls,
                        apply_n_qubit_time: a.nq_time,
                        measure_qubit_calls: a.mq_calls,
                        measure_qubit_time: a.mq_time,
                        total_time,
                    },
                )
            })
            .transpose()?;

        Ok(SimulationResult {
            sv: state,
            num_qubits: n,
            cbits,
            prof,
        })
    }
}

// ---------------------------------------------------------------------------
// Instruction dispatcher
// ---------------------------------------------------------------------------

fn run_instruction(
    state: &mut Vec<C>,
    inst: &Instruction,
    n: usize,
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
    acc: &mut Option<ProfileAcc>,
) -> PyResult<()> {
    match inst {
        // -- Single-qubit fixed ------------------------------------------
        Instruction::Id { .. } | Instruction::U0 { .. } => {}

        Instruction::X { qubit } => do_oq(state, &gates::X, *qubit, n, acc),
        Instruction::Y { qubit } => do_oq(state, &gates::Y, *qubit, n, acc),
        Instruction::Z { qubit } => do_oq(state, &gates::Z, *qubit, n, acc),
        Instruction::H { qubit } => {
            let m = gates::h();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::S { qubit } => {
            let m = gates::s_gate();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Sdg { qubit } => {
            let m = gates::sdg();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::T { qubit } => {
            let m = gates::t_gate();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Tdg { qubit } => {
            let m = gates::tdg();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Sx { qubit } => {
            let m = gates::sx();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Sxdg { qubit } => {
            let m = gates::sxdg();
            do_oq(state, &m, *qubit, n, acc);
        }

        // -- Single-qubit parametric -------------------------------------
        Instruction::U3 { qubit, theta, phi, lam } => {
            let m = gates::u3(*theta, *phi, *lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::U2 { qubit, phi, lam } => {
            let m = gates::u2(*phi, *lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::U1 { qubit, lam } => {
            let m = gates::u1(*lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::U { qubit, theta, phi, lam } => {
            let m = gates::u(*theta, *phi, *lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::P { qubit, lam } => {
            let m = gates::p(*lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Rx { qubit, theta } => {
            let m = gates::rx(*theta);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Ry { qubit, theta } => {
            let m = gates::ry(*theta);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Rz { qubit, phi } => {
            let m = gates::rz(*phi);
            do_oq(state, &m, *qubit, n, acc);
        }

        // -- Two-qubit fixed ---------------------------------------------
        Instruction::Cx { control, target } => {
            do_nq(state, &m4(gates::cnot()), &[*control, *target], n, acc);
        }
        Instruction::Cz { control, target } => {
            do_nq(state, &m4(gates::cz()), &[*control, *target], n, acc);
        }
        Instruction::Cy { control, target } => {
            do_nq(state, &m4(gates::cy()), &[*control, *target], n, acc);
        }
        Instruction::Ch { control, target } => {
            do_nq(state, &m4(gates::ch()), &[*control, *target], n, acc);
        }
        Instruction::Swap { a, b } => {
            do_nq(state, &m4(gates::swap()), &[*a, *b], n, acc);
        }
        Instruction::Csx { control, target } => {
            do_nq(state, &m4(gates::csx()), &[*control, *target], n, acc);
        }

        // -- Two-qubit parametric ----------------------------------------
        Instruction::Crx { control, target, theta } => {
            do_nq(state, &m4(gates::crx(*theta)), &[*control, *target], n, acc);
        }
        Instruction::Cry { control, target, theta } => {
            do_nq(state, &m4(gates::cry(*theta)), &[*control, *target], n, acc);
        }
        Instruction::Crz { control, target, lam } => {
            do_nq(state, &m4(gates::crz(*lam)), &[*control, *target], n, acc);
        }
        Instruction::Cu1 { control, target, lam } => {
            do_nq(state, &m4(gates::cu1(*lam)), &[*control, *target], n, acc);
        }
        Instruction::Cp { control, target, lam } => {
            do_nq(state, &m4(gates::cp(*lam)), &[*control, *target], n, acc);
        }
        Instruction::Cu3 { control, target, theta, phi, lam } => {
            do_nq(state, &m4(gates::cu3(*theta, *phi, *lam)), &[*control, *target], n, acc);
        }
        Instruction::Cu { control, target, theta, phi, lam, gamma } => {
            do_nq(
                state,
                &m4(gates::cu(*theta, *phi, *lam, *gamma)),
                &[*control, *target],
                n,
                acc,
            );
        }
        Instruction::Rxx { a, b, theta } => {
            do_nq(state, &m4(gates::rxx(*theta)), &[*a, *b], n, acc);
        }
        Instruction::Rzz { a, b, theta } => {
            do_nq(state, &m4(gates::rzz(*theta)), &[*a, *b], n, acc);
        }

        // -- Three-qubit -------------------------------------------------
        Instruction::Ccx { control1, control2, target } => {
            do_nq(state, &m8(gates::ccx()), &[*control1, *control2, *target], n, acc);
        }
        Instruction::Cswap { control, target1, target2 } => {
            do_nq(state, &m8(gates::cswap()), &[*control, *target1, *target2], n, acc);
        }
        Instruction::Rccx { control1, control2, target } => {
            do_nq(state, &m8(gates::rccx()), &[*control1, *control2, *target], n, acc);
        }

        // -- Four-qubit --------------------------------------------------
        Instruction::Rc3x { control1, control2, control3, target } => {
            do_nq(
                state,
                &m16(gates::rc3x()),
                &[*control1, *control2, *control3, *target],
                n,
                acc,
            );
        }
        Instruction::C3x { control1, control2, control3, target } => {
            do_nq(
                state,
                &m16(gates::c3x()),
                &[*control1, *control2, *control3, *target],
                n,
                acc,
            );
        }
        Instruction::C3sqrtx { control1, control2, control3, target } => {
            do_nq(
                state,
                &m16(gates::c3sqrtx()),
                &[*control1, *control2, *control3, *target],
                n,
                acc,
            );
        }

        // -- Five-qubit --------------------------------------------------
        Instruction::C4x { control1, control2, control3, control4, target } => {
            do_nq(
                state,
                &m32(gates::c4x()),
                &[*control1, *control2, *control3, *control4, *target],
                n,
                acc,
            );
        }

        // -- Cross-node / generic ----------------------------------------
        Instruction::Gate { name, qubits, .. } => {
            match name.to_lowercase().as_str() {
                "remote_link_phi_plus" => {
                    do_nq(state, &m4(gates::phi_plus()), qubits, n, acc);
                }
                "remote_link_psi_minus" => {
                    do_nq(state, &m4(gates::psi_minus()), qubits, n, acc);
                }
                "remote_link_psi_plus" => {
                    do_nq(state, &m4(gates::psi_plus()), qubits, n, acc);
                }
                "nonlocal_cz" => {
                    do_nq(state, &m4(gates::NONLOCAL_CZ), qubits, n, acc);
                }
                other => {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                        "Unsupported generic gate: {other:?}. Decompose it before simulating."
                    )));
                }
            }
        }

        // -- Measurement -------------------------------------------------
        Instruction::Measure { qubit, cbit } => {
            let outcome = do_mq(state, *qubit, n, rng, acc);
            cbits.insert(*cbit, outcome as i32);
        }

        // -- Classical control -------------------------------------------
        Instruction::Conditional { condition, op } => {
            let mut actual: u64 = 0;
            for bit in 0..condition.creg_size {
                let val = *cbits.get(&(condition.creg_base + bit)).unwrap_or(&0) as u64;
                actual |= val << bit;
            }
            if actual == condition.creg_value {
                run_instruction(state, op, n, cbits, rng, acc)?;
            }
        }

        // -- Reset -------------------------------------------------------
        Instruction::Reset { qubit } => {
            let outcome = do_mq(state, *qubit, n, rng, acc);
            if outcome == 1 {
                do_oq(state, &gates::X, *qubit, n, acc);
            }
        }

        // -- No-ops ------------------------------------------------------
        Instruction::Barrier | Instruction::Classical { .. } => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Timed engine wrappers
// ---------------------------------------------------------------------------

#[inline]
fn do_oq(
    state: &mut Vec<C>,
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    acc: &mut Option<ProfileAcc>,
) {
    match acc {
        None => apply_one_qubit(state, u, target, n, &[]),
        Some(a) => {
            let t = Instant::now();
            apply_one_qubit(state, u, target, n, &[]);
            a.oq_time += t.elapsed().as_secs_f64();
            a.oq_calls += 1;
        }
    }
}

#[inline]
fn do_nq(
    state: &mut Vec<C>,
    u: &[Vec<C>],
    qubits: &[usize],
    n: usize,
    acc: &mut Option<ProfileAcc>,
) {
    match acc {
        None => apply_n_qubit(state, u, qubits, n),
        Some(a) => {
            let t = Instant::now();
            apply_n_qubit(state, u, qubits, n);
            a.nq_time += t.elapsed().as_secs_f64();
            a.nq_calls += 1;
        }
    }
}

#[inline]
fn do_mq(
    state: &mut Vec<C>,
    qubit: usize,
    n: usize,
    rng: &mut impl Rng,
    acc: &mut Option<ProfileAcc>,
) -> u8 {
    match acc {
        None => measure_qubit(state, qubit, n, rng),
        Some(a) => {
            let t = Instant::now();
            let outcome = measure_qubit(state, qubit, n, rng);
            a.mq_time += t.elapsed().as_secs_f64();
            a.mq_calls += 1;
            outcome
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix-to-Vec converters (fixed-size arrays → Vec<Vec<C>> for apply_n_qubit)
// ---------------------------------------------------------------------------

fn m4(m: [[C; 4]; 4]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
fn m8(m: [[C; 8]; 8]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
fn m16(m: [[C; 16]; 16]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
fn m32(m: [[C; 32]; 32]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
