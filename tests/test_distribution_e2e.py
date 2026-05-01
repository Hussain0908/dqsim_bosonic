"""
E2E tests: monolithic circuit → BosonicDistributor → dqsim.

Verifies that the data-qubit marginals of the distributed+simulated circuit
match a direct simulation of the original circuit.
"""

from __future__ import annotations

import math

import pytest
from bosonic_model import Circuit, Register
from bosonic_model.instructions import (
    CcxInstruction,
    CxInstruction,
    CzInstruction,
    HInstruction,
    RxInstruction,
    RyInstruction,
    SwapInstruction,
    XInstruction,
    YInstruction,
    ZInstruction,
)
from bosonic_sdk.distributor.distributors.bosonic_distributor import BosonicDistributor

from dqsim import CompositeSimulator, StatevectorSimulator

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


def _composite_simulate(circuit: Circuit, *, nodes: int, qubits_per_node: int) -> dict[int, float]:
    distributed = BosonicDistributor().distribute(circuit, nodes=nodes, qubits_per_node=qubits_per_node)
    result = CompositeSimulator(seed=SEED).simulate(distributed)
    probs = result.probabilities()
    data_indices = result.physical_qubits[::2]  # even physical qubits are data
    return _marginalise(probs, data_indices)


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


class TestCompositeSimulatorE2E:

    def test_x_cx_deterministic(self) -> None:
        circuit = _circuit(2, [
            XInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
        ])
        original = _simulate(circuit)
        result = _composite_simulate(circuit, nodes=2, qubits_per_node=1)
        _assert_marginals_match(original, result)
        assert abs(result.get(3, 0.0) - 1.0) < 1e-6

    def test_bell_pair(self) -> None:
        circuit = _circuit(2, [
            HInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
        ])
        original = _simulate(circuit)
        result = _composite_simulate(circuit, nodes=2, qubits_per_node=1)
        _assert_marginals_match(original, result)

    def test_local_only_circuit(self) -> None:
        circuit = _circuit(2, [
            HInstruction(qubit=0, qubits=[0]),
            HInstruction(qubit=1, qubits=[1]),
        ])
        original = _simulate(circuit)
        result = _composite_simulate(circuit, nodes=2, qubits_per_node=1)
        _assert_marginals_match(original, result)

    def test_composite_matches_monolithic_marginals(self) -> None:
        """CompositeSimulator data-qubit marginals must equal monolithic marginals exactly."""
        circuit = _circuit(2, [
            HInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
        ])
        monolithic_marginals = _distribute_and_simulate(circuit, nodes=2, qubits_per_node=1)
        composite_marginals = _composite_simulate(circuit, nodes=2, qubits_per_node=1)
        _assert_marginals_match(monolithic_marginals, composite_marginals)


# ---------------------------------------------------------------------------
# _ALL_SMALL — parametrised suite: (id, circuit, nodes, qubits_per_node)
# ---------------------------------------------------------------------------

_ALL_SMALL: list[tuple[str, Circuit, int, int]] = [
    # -- 2-qubit, cross-node -------------------------------------------------
    ("bell_pair",
     _circuit(2, [HInstruction(qubit=0, qubits=[0]),
                  CxInstruction(control=0, target=1, qubits=[0, 1], params=[])]),
     2, 1),
    ("x_cx_deterministic",
     _circuit(2, [XInstruction(qubit=0, qubits=[0]),
                  CxInstruction(control=0, target=1, qubits=[0, 1], params=[])]),
     2, 1),
    ("cz_pair",
     _circuit(2, [HInstruction(qubit=0, qubits=[0]),
                  HInstruction(qubit=1, qubits=[1]),
                  CzInstruction(control=0, target=1, qubits=[0, 1], params=[])]),
     2, 1),
    ("ry_cx",
     _circuit(2, [RyInstruction(qubit=0, qubits=[0], theta=math.pi / 3, params=[math.pi / 3]),
                  CxInstruction(control=0, target=1, qubits=[0, 1], params=[])]),
     2, 1),
    ("rx_ry_cx",
     _circuit(2, [RxInstruction(qubit=0, qubits=[0], theta=math.pi / 3, params=[math.pi / 3]),
                  RyInstruction(qubit=1, qubits=[1], theta=math.pi / 4, params=[math.pi / 4]),
                  CxInstruction(control=0, target=1, qubits=[0, 1], params=[])]),
     2, 1),
    ("xyz_single",
     _circuit(2, [XInstruction(qubit=0, qubits=[0]),
                  YInstruction(qubit=1, qubits=[1]),
                  ZInstruction(qubit=0, qubits=[0])]),
     2, 1),
    # -- 2-qubit, local only -------------------------------------------------
    ("local_hh",
     _circuit(2, [HInstruction(qubit=0, qubits=[0]),
                  HInstruction(qubit=1, qubits=[1])]),
     2, 1),
    # -- 4-qubit, cross-node -------------------------------------------------
    ("ghz4",
     _circuit(4, [HInstruction(qubit=0, qubits=[0]),
                  CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
                  CxInstruction(control=1, target=2, qubits=[1, 2], params=[]),
                  CxInstruction(control=2, target=3, qubits=[2, 3], params=[])]),
     2, 2),
    ("chain_cx4",
     _circuit(4, [HInstruction(qubit=0, qubits=[0]),
                  CxInstruction(control=0, target=2, qubits=[0, 2], params=[]),
                  CxInstruction(control=1, target=3, qubits=[1, 3], params=[])]),
     2, 2),
    ("local_4h",
     _circuit(4, [HInstruction(qubit=i, qubits=[i]) for i in range(4)]),
     2, 2),
]


class TestAllSmall:
    """Parametrised: monolithic SV == CompositeSimulator marginals for every circuit in _ALL_SMALL."""

    @pytest.mark.parametrize("name,circuit,nodes,qpn", _ALL_SMALL, ids=[t[0] for t in _ALL_SMALL])
    def test_composite_matches_monolithic(
        self, name: str, circuit: Circuit, nodes: int, qpn: int
    ) -> None:
        monolithic = _simulate(circuit)
        composite = _composite_simulate(circuit, nodes=nodes, qubits_per_node=qpn)
        _assert_marginals_match(monolithic, composite)
