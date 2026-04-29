"""
E2E tests: monolithic circuit → BosonicDistributor → dqsim.

Verifies that the data-qubit marginals of the distributed+simulated circuit
match a direct simulation of the original circuit.
"""

from __future__ import annotations

import pytest
from bosonic_model import Circuit, Register
from bosonic_model.instructions import CxInstruction, HInstruction, XInstruction
from bosonic_sdk.distributor.distributors.bosonic_distributor import BosonicDistributor

from dqsim import StatevectorSimulator

SEED = 42


def _circuit(n: int, instructions: list) -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=n, base=0)},
        cregs={},
        instructions=instructions,
    )


def _marginalise(probs: dict[int, float], data_indices: list[int]) -> dict[int, float]:
    """Reduce a full probability dict to marginals over the given qubit indices."""
    result: dict[int, float] = {}
    for full_state, prob in probs.items():
        data_state = 0
        for out_bit, qubit_idx in enumerate(data_indices):
            if (full_state >> qubit_idx) & 1:
                data_state |= 1 << out_bit
        result[data_state] = result.get(data_state, 0.0) + prob
    return result


def _simulate(circuit: Circuit) -> dict[int, float]:
    return StatevectorSimulator(seed=SEED).simulate(circuit).probabilities()


def _distribute_and_simulate(circuit: Circuit, *, nodes: int, qubits_per_node: int) -> dict[int, float]:
    distributed = BosonicDistributor().distribute(circuit, nodes=nodes, qubits_per_node=qubits_per_node)
    monolithic = distributed.as_monolithic_circuit()
    probs = StatevectorSimulator(seed=SEED).simulate(monolithic).probabilities()
    # BosonicDistributor interleaves data (even) and comm (odd) qubits.
    total = monolithic.qubits()
    data_indices = list(range(0, total, 2))
    return _marginalise(probs, data_indices)


def _assert_marginals_match(original: dict[int, float], distributed: dict[int, float], tol: float = 1e-6) -> None:
    all_states = set(original) | set(distributed)
    for state in all_states:
        p_orig = original.get(state, 0.0)
        p_dist = distributed.get(state, 0.0)
        assert abs(p_orig - p_dist) < tol, (
            f"state {state:b}: original={p_orig:.8f}, distributed={p_dist:.8f}"
        )


class TestBosonicDistributorE2E:
    def test_x_cx_deterministic(self) -> None:
        """X q[0]; CX q[0]→q[1] produces |11⟩. Cross-node CX forces teleportation."""
        circuit = _circuit(2, [
            XInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
        ])

        original = _simulate(circuit)
        assert abs(original.get(3, 0.0) - 1.0) < 1e-6, "original should be |11⟩"

        distributed = _distribute_and_simulate(circuit, nodes=2, qubits_per_node=1)
        assert abs(distributed.get(3, 0.0) - 1.0) < 1e-6, "distributed data qubits should be |11⟩"

    def test_bell_pair(self) -> None:
        """H q[0]; CX q[0]→q[1] prepares |Φ+⟩. Marginals: P[00]=P[11]=0.5."""
        circuit = _circuit(2, [
            HInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
        ])

        original = _simulate(circuit)
        distributed = _distribute_and_simulate(circuit, nodes=2, qubits_per_node=1)
        _assert_marginals_match(original, distributed)

    def test_local_only_circuit(self) -> None:
        """Circuit with no cross-node gates: distribution is trivial, marginals must match exactly."""
        circuit = _circuit(2, [
            HInstruction(qubit=0, qubits=[0]),
            HInstruction(qubit=1, qubits=[1]),
        ])

        original = _simulate(circuit)
        distributed = _distribute_and_simulate(circuit, nodes=2, qubits_per_node=1)
        _assert_marginals_match(original, distributed)
