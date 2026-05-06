from __future__ import annotations

import math

import pytest
import qasmpi
from bosonic_model import Circuit, Register
from bosonic_model.qasm import Translator
from bosonic_model.instructions import (
    Condition,
    ConditionalInstruction,
    CxInstruction,
    HInstruction,
    MeasureInstruction,
    RxInstruction,
    RyInstruction,
    RzInstruction,
    ResetInstruction,
    SwapInstruction,
    XInstruction,
)
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor

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

    def test_mid_circuit_measurement_and_conditional(self) -> None:
        circuit = Circuit(
            qregs={"q": Register(name="q", size=2, base=0)},
            cregs={"c": Register(name="c", size=1, base=0)},
            instructions=[
                XInstruction(qubit=0, qubits=[0]),
                MeasureInstruction(qubit=0, cbit=0, qubits=[0]),
                ConditionalInstruction(
                    condition=Condition(creg_base=0, creg_size=1, creg_value=1),
                    op=XInstruction(qubit=1, qubits=[1]),
                    qubits=[1],
                ),
            ],
        )
        result = simulate_monolithic(circuit, mode="mps", seed=SEED)
        assert result.classical_bits == {0: 1}
        assert result.probabilities() == {3: 1.0}

    def test_reset(self) -> None:
        circuit = _circuit(1, [
            XInstruction(qubit=0, qubits=[0]),
            ResetInstruction(qubit=0, qubits=[0]),
        ])
        result = simulate_monolithic(circuit, mode="mps", seed=SEED)
        assert result.probabilities() == {0: 1.0}

    def test_lowered_distributed_circuit_runs_as_monolithic(self) -> None:
        circuit = _from_qasmpi("deutsch_n2")
        distributed = DisqcoDistributor().distribute(
            circuit,
            nodes=2,
            qubits_per_node=1,
            lowered=True,
        )
        monolithic = distributed.as_monolithic_circuit()
        result = simulate_monolithic(monolithic, mode="mps", seed=SEED)
        assert abs(sum(result.probabilities().values()) - 1.0) < TOL

    @pytest.mark.parametrize("name", ["deutsch_n2", "bell_n4", "qft_n4", "qaoa_n6"])
    def test_qasmpi_circuit_matches_statevector(self, name: str) -> None:
        _assert_mps_matches_statevector(_from_qasmpi(name))
