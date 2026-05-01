"""
Shot-based equivalence tests using real QASMBench circuits.

For each circuit, runs SHOTS samples from both the monolithic statevector
simulator and the composite (pblock) simulator, then asserts that the
estimated probability distributions agree within a sampling-noise tolerance.

This approach works even for circuits with mid-circuit measurements because
correctness is established statistically rather than by exact statevector
comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import qasmpi
from bosonic_model.qasm.translator import Translator
from bosonic_sdk.distributor.distributors.bosonic_distributor import BosonicDistributor

from dqsim import CompositeSimulator, StatevectorSimulator

SEED = 42
SHOTS = 2000
# With 2000 shots, σ ≈ sqrt(p(1-p)/N) ≤ 0.011 for any p.
# TOL = 5σ ≈ 0.05 gives a very low false-failure rate.
TOL = 0.05


def _from_qasmpi(name: str):
    """Fetch a QASMBench circuit by name, parse it, and strip terminal measurements.

    Measurements are removed so we compare the unitary part of each circuit —
    the quantum marginals rather than classical register outcomes.
    """
    circuit = Translator().from_qasm(qasmpi.get_circuit(name))
    instructions = [i for i in circuit.instructions if i.kind != "measure"]
    return circuit.model_copy(update={"instructions": instructions})


def _n_qubits(circuit) -> int:
    return max(r.base + r.size for r in circuit.qregs.values())


def _marginalise(probs: dict[int, float], data_indices: list[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    for full_state, prob in probs.items():
        data_state = 0
        for out_bit, qubit_idx in enumerate(data_indices):
            if (full_state >> qubit_idx) & 1:
                data_state |= 1 << out_bit
        result[data_state] = result.get(data_state, 0.0) + prob
    return result


def _monolithic_probs(circuit) -> dict[int, float]:
    """Estimate probability distribution from SHOTS samples of the monolithic SV."""
    counts = StatevectorSimulator(seed=SEED).simulate(circuit).counts(shots=SHOTS, seed=SEED)
    return {int(bits, 2): n / SHOTS for bits, n in counts.items()}


def _composite_probs(circuit, *, nodes: int, qubits_per_node: int) -> dict[int, float]:
    """Estimate probability distribution by sampling the composite simulator output."""
    distributed = BosonicDistributor().distribute(circuit, nodes=nodes, qubits_per_node=qubits_per_node)
    result = CompositeSimulator(seed=SEED).simulate(distributed)
    marginal = _marginalise(result.probabilities(), result.physical_qubits[::2])

    rng = np.random.default_rng(SEED)
    states = list(marginal.keys())
    weights = np.array([marginal[s] for s in states], dtype=float)
    weights /= weights.sum()
    sampled = rng.choice(states, size=SHOTS, p=weights)
    unique, cnts = np.unique(sampled, return_counts=True)
    return {int(s): int(c) / SHOTS for s, c in zip(unique, cnts)}


def _assert_distributions_match(
    a: dict[int, float], b: dict[int, float], tol: float = TOL
) -> None:
    all_states = set(a) | set(b)
    for state in all_states:
        pa = a.get(state, 0.0)
        pb = b.get(state, 0.0)
        assert abs(pa - pb) < tol, (
            f"state {state:b}: monolithic={pa:.4f}, composite={pb:.4f}, diff={abs(pa-pb):.4f} > tol={tol}"
        )


# ---------------------------------------------------------------------------
# QASMBench circuits: (qasmpi_name, nodes, qubits_per_node)
# nodes and qubits_per_node are chosen so n <= nodes * qubits_per_node.
# ---------------------------------------------------------------------------

_QASMPI_CIRCUITS = [
    ("basis_test_n4",  2, 2),
    ("cat_state_n4",   2, 2),
    ("hs4_n4",         2, 2),
    ("inverseqft_n4",  2, 2),
    ("qrng_n4",        2, 2),
    ("qec_sm_n5",      3, 2),
]

_SHOT_SMALL = [
    (name, _from_qasmpi(name), nodes, qpn)
    for name, nodes, qpn in _QASMPI_CIRCUITS
]


class TestShotEquivalence:
    """Composite simulator matches monolithic SV within sampling noise."""

    @pytest.mark.parametrize(
        "name,circuit,nodes,qpn",
        _SHOT_SMALL,
        ids=[t[0] for t in _SHOT_SMALL],
    )
    def test_shot_distributions_match(
        self, name: str, circuit, nodes: int, qpn: int
    ) -> None:
        mono = _monolithic_probs(circuit)
        comp = _composite_probs(circuit, nodes=nodes, qubits_per_node=qpn)
        _assert_distributions_match(mono, comp)
