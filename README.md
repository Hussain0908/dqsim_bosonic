# dqsim

`dqsim` is a native Rust statevector simulator for distributed quantum circuits.
It exposes a small Python API and is designed to run circuits from the
[`bosonic-model`](https://pypi.org/project/bosonic-model/) ecosystem without
calling back into Python for every gate.

The simulator accepts a `bosonic_model.Circuit` or `bosonic_model.DistributedCircuit`,
serializes it once with `model_dump_json()`, deserializes the instruction stream
in Rust, and executes the full simulation in native code.

## Features

- Python package backed by a Rust/PyO3 extension module.
- Statevector simulation for monolithic and distributed circuit models.
- Single-qubit, two-qubit, and selected multi-qubit gates.
- Measurement, reset, and simple classical conditionals.
- Native support for distributed-circuit gates:
  - `remote_link_psi_minus`
  - `remote_link_psi_plus`
  - `nonlocal_cz`
- Probability queries, sampled counts, statevector access, fidelity checks, and
  optional execution profiling.
- Parallel statevector kernels using Rayon for larger simulations.

## Installation

For local development, install the package from the repository root:

```bash
python -m pip install .
```

For an editable native-extension workflow, use Maturin:

```bash
python -m pip install maturin
maturin develop
```

The package requires Python 3.10 or newer and depends on:

- `bosonic-model`
- `bosonic-converters`
- `numpy`

## Quick Start

```python
from bosonic_model import Circuit, Register, HInstruction, CxInstruction
from dqsim import StatevectorSimulator

circuit = Circuit(
    qregs={"q": Register(name="q", size=2, base=0)},
    instructions=[
        HInstruction(qubit=0),
        CxInstruction(control=0, target=1),
    ],
)

sim = StatevectorSimulator(seed=7, profile=True)
result = sim.simulate(circuit)

print(result.num_qubits)
print(result.statevector)
print(result.probabilities())
print(result.counts(shots=1000, seed=7))
print(result.profile)
```

Example probability output for the Bell state:

```python
{0: 0.5000000000000001, 3: 0.5000000000000001}
```

## Python API

`dqsim` exports three classes:

```python
from dqsim import StatevectorSimulator, SimulationResult, SimulationProfile
```

### `StatevectorSimulator`

```python
StatevectorSimulator(seed=None, profile=False)
```

- `seed`: optional integer seed used for measurements during simulation.
- `profile`: when `True`, records time spent in gate and measurement kernels.

Run a circuit:

```python
result = StatevectorSimulator(seed=123).simulate(circuit)
```

`simulate()` accepts:

- `bosonic_model.Circuit`
- `bosonic_model.DistributedCircuit`

If the input object has `as_monolithic_circuit()`, `dqsim` calls it before
simulation. This is how distributed circuits are converted into a native
instruction stream.

### `SimulationResult`

Properties:

- `statevector`: NumPy array of complex amplitudes with shape `(2 ** n,)`.
- `num_qubits`: number of qubits in the simulated circuit.
- `classical_bits`: dictionary of final measured classical bit values keyed by
  absolute classical bit index.
- `profile`: `SimulationProfile` when profiling is enabled, otherwise `None`.

Methods:

```python
result.probabilities(qubits=None)
```

Returns a dictionary of nonzero probabilities keyed by integer basis state. When
`qubits` is omitted, all qubits are returned in descending order. When `qubits`
is provided, `qubits[0]` is the most significant bit of the returned index.

```python
result.counts(shots=1000, qubits=None, seed=None)
```

Samples bitstrings from the final state distribution. The leftmost bit in each
bitstring corresponds to `qubits[0]`.

```python
result.fidelity(other_statevector)
```

Computes `|<result|other>|^2` against another complex statevector.

### `SimulationProfile`

When profiling is enabled, `result.profile` includes:

- `apply_one_qubit_calls`
- `apply_one_qubit_time`
- `apply_n_qubit_calls`
- `apply_n_qubit_time`
- `measure_qubit_calls`
- `measure_qubit_time`
- `total_time`

Printing the profile gives a compact timing summary.

## Supported Instructions

The Rust instruction parser currently supports these `bosonic-model`
instruction kinds:

- Single-qubit fixed gates: `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`,
  `sx`, `sxdg`
- Single-qubit parameterized gates: `u3`, `u2`, `u1`, `u`, `p`, `rx`, `ry`,
  `rz`, `u0`
- Two-qubit fixed gates: `cx`, `cz`, `cy`, `ch`, `swap`, `csx`
- Two-qubit parameterized gates: `crx`, `cry`, `crz`, `cu1`, `cp`, `cu3`,
  `cu`, `rxx`, `rzz`
- Multi-qubit gates: `ccx`, `cswap`, `rccx`, `rc3x`, `c3x`, `c3sqrtx`, `c4x`
- Generic distributed gates: `remote_link_psi_minus`, `remote_link_psi_plus`,
  `nonlocal_cz`
- Measurement and control: `measure`, `reset`, `conditional`
- No-ops: `barrier`, `classical`

Unsupported generic gates raise `NotImplementedError` and should be decomposed
before simulation.

## Development

The repository is intentionally small:

```text
dqsim/__init__.py    Python exports for the native extension
src/lib.rs           PyO3 module registration
src/simulator.rs     Python-facing simulator and instruction dispatcher
src/engine.rs        Statevector kernels, measurement, probabilities, counts
src/gates.rs         Gate matrices
src/types.rs         Rust structs matching bosonic-model JSON
```

Run the Rust checks:

```bash
cargo test
```

Build the Python source distribution and wheel:

```bash
make build
```

`make build` removes `dist/`, builds the package through the `pyproject.toml`
build backend, and validates the artifacts with Twine.

Publish to TestPyPI:

```bash
make publish
```

Publish to PyPI:

```bash
make publish REPOSITORY=pypi
```

The `Makefile` uses `uvx` to run build and Twine tools without requiring them
to be installed in the active environment.

## Implementation Notes

- The Rust crate builds a `cdylib` named `_core`, exposed in Python as
  `dqsim._core`.
- Statevectors are initialized to `|0...0>`.
- Measurement uses a ChaCha8 RNG. Passing a simulator seed makes in-circuit
  measurement deterministic; passing a seed to `counts()` makes sampling
  deterministic.
- For larger statevectors, gate application and measurement kernels use Rayon
  parallel iterators.
- Fixed-size matrices are used for gates up to five qubits.
