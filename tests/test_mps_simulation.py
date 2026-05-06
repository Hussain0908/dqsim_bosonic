from __future__ import annotations

import math

import pytest
import qasmpi
from bosonic_model import Circuit, Register
from bosonic_model.qasm import Translator
from bosonic_model.instructions import (
    CxInstruction,
    HInstruction,
    RxInstruction,
    RyInstruction,
    RzInstruction,
    SwapInstruction,
    XInstruction,
)

from dqsim import simulate_monolithic


SEED = 42
TOL = 1e-8


def _circuit(n: int, instructions: list) -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=n, base=0)},
        cregs={},
        instructions=instructions,
    )


def _assert_probs_close(a: dict[int, float], b: dict[int, float], tol: float = TOL) -> None:
    for state in set(a) | set(b):
        assert abs(a.get(state, 0.0) - b.get(state, 0.0)) < tol


def _assert_mps_matches_statevector(circuit: Circuit) -> None:
    sv = simulate_monolithic(circuit, mode="state_vector", seed=SEED).probabilities()
    mps = simulate_monolithic(circuit, mode="mps", seed=SEED).probabilities()
    _assert_probs_close(sv, mps)


def _from_qasmpi(name: str) -> Circuit:
    circuit = Translator().from_qasm(qasmpi.get_circuit(name))
    return circuit.model_copy(
        update={"instructions": [i for i in circuit.instructions if i.kind != "measure"]}
    )


class TestMpsSimulation:
    def test_product_state(self) -> None:
        _assert_mps_matches_statevector(
            _circuit(3, [
                HInstruction(qubit=0, qubits=[0]),
                RxInstruction(qubit=1, qubits=[1], theta=math.pi / 3, params=[math.pi / 3]),
                RyInstruction(qubit=2, qubits=[2], theta=math.pi / 5, params=[math.pi / 5]),
                RzInstruction(qubit=0, qubits=[0], phi=math.pi / 7, params=[math.pi / 7]),
            ])
        )

    def test_bell_pair(self) -> None:
        _assert_mps_matches_statevector(
            _circuit(2, [
                HInstruction(qubit=0, qubits=[0]),
                CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
            ])
        )

    def test_non_adjacent_cx(self) -> None:
        _assert_mps_matches_statevector(
            _circuit(3, [
                HInstruction(qubit=0, qubits=[0]),
                CxInstruction(control=0, target=2, qubits=[0, 2], params=[]),
            ])
        )

    def test_swap(self) -> None:
        _assert_mps_matches_statevector(
            _circuit(3, [
                XInstruction(qubit=1, qubits=[1]),
                SwapInstruction(a=1, b=2, qubits=[1, 2], params=[]),
            ])
        )

    def test_unsupported_multi_qubit_gate_raises(self) -> None:
        from bosonic_model.instructions import CcxInstruction

        circuit = _circuit(3, [
            HInstruction(qubit=0, qubits=[0]),
            CcxInstruction(control1=0, control2=1, target=2, qubits=[0, 1, 2], params=[]),
        ])

        with pytest.raises(NotImplementedError, match="one- and two-qubit"):
            simulate_monolithic(circuit, mode="mps", seed=SEED)

    @pytest.mark.parametrize("name", ["deutsch_n2", "bell_n4", "qft_n4", "qaoa_n6"])
    def test_qasmpi_circuit_matches_statevector(self, name: str) -> None:
        _assert_mps_matches_statevector(_from_qasmpi(name))
