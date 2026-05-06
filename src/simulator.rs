use pyo3::prelude::*;

use crate::distributed::pblock::PBlockSimulator;
use crate::monolithic::statevector::StatevectorSimulator;

enum MonolithicSimulationMode {
    StateVector,
}

enum DistributedSimulationMode {
    PBlock,
}

fn parse_monolithic_mode(mode: &str) -> PyResult<MonolithicSimulationMode> {
    match mode.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "state_vector" => Ok(MonolithicSimulationMode::StateVector),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported monolithic simulation mode {other:?}; expected 'state_vector'"
        ))),
    }
}

fn parse_distributed_mode(mode: &str) -> PyResult<DistributedSimulationMode> {
    match mode.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "p_block" => Ok(DistributedSimulationMode::PBlock),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported distributed simulation mode {other:?}; expected 'p_block'"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (circuit, mode="state_vector", seed=None, profile=false))]
pub fn simulate_monolithic(
    py: Python,
    circuit: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    seed: Option<u64>,
    profile: bool,
) -> PyResult<PyObject> {
    match parse_monolithic_mode(mode)? {
        MonolithicSimulationMode::StateVector => {
            let sim = StatevectorSimulator::new(seed, profile);
            Ok(sim.simulate(py, circuit)?.into_py(py))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (distributed, mode="p_block", seed=None))]
pub fn simulate_distributed(
    py: Python,
    distributed: &Bound<PyAny>,  // todo: tighten this
    mode: &str,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    match parse_distributed_mode(mode)? {
        DistributedSimulationMode::PBlock => {
            let sim = PBlockSimulator::new(seed);
            Ok(sim.simulate(py, distributed)?.into_py(py))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (circuit, mode="state_vector", shots=1000, seed=None))]
pub fn simulate_monolithic_shots(
    py: Python,
    circuit: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    shots: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    match parse_monolithic_mode(mode)? {
        MonolithicSimulationMode::StateVector => {
            let sim = StatevectorSimulator::new(seed, false);
            sim.simulate_shots(py, circuit, shots)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (distributed, mode="p_block", shots=1000, seed=None))]
pub fn simulate_distributed_shots(
    py: Python,
    distributed: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    shots: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    match parse_distributed_mode(mode)? {
        DistributedSimulationMode::PBlock => {
            let sim = PBlockSimulator::new(seed);
            sim.simulate_shots(py, distributed, shots)
        }
    }
}
