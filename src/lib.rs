mod engine;
mod gates;
mod pblock;
mod simulator;
mod types;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<simulator::StatevectorSimulator>()?;
    m.add_class::<simulator::SimulationResult>()?;
    m.add_class::<simulator::SimulationProfile>()?;
    m.add_class::<pblock::CompositeSimulator>()?;
    m.add_class::<pblock::CompositeResult>()?;
    Ok(())
}
