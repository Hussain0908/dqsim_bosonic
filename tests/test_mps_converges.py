from __future__ import annotations

from bosonic_model import Circuit, Register
from bosonic_model.instructions import CxInstruction, HInstruction, MeasureInstruction, RyInstruction

from dqsim import simulate_monolithic, simulate_monolithic_shots


SEED = 42


def _circuit(n: int, instructions: list, cregs: dict | None = None) -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=n, base=0)},
        cregs=cregs or {},
        instructions=instructions,
    )


def _l1_error(a: dict[int, float], b: dict[int, float]) -> float:
    return sum(abs(a.get(state, 0.0) - b.get(state, 0.0)) for state in set(a) | set(b))


def _frequency_l1_error(a: dict[str, int], b: dict[str, int], shots: int) -> float:
    return sum(
        abs(a.get(state, 0) / shots - b.get(state, 0) / shots)
        for state in set(a) | set(b)
    )


def _high_entanglement_circuit() -> Circuit:
    instructions = [HInstruction(qubit=q, qubits=[q]) for q in range(6)]
    pairs = [(0, 5), (1, 4), (2, 3), (0, 2), (3, 5), (1, 3)]

    for layer in range(3):
        for control, target in pairs:
            instructions.append(
                CxInstruction(control=control, target=target, qubits=[control, target], params=[])
            )
        for qubit in range(6):
            theta = 0.2 * (layer + 1) * (qubit + 1)
            instructions.append(
                RyInstruction(qubit=qubit, qubits=[qubit], theta=theta, params=[theta])
            )

    return _circuit(6, instructions)


def _measured_entanglement_circuit() -> Circuit:
    instructions = [HInstruction(qubit=q, qubits=[q]) for q in range(4)]
    pairs = [(0, 3), (1, 2), (0, 2), (3, 1)]

    for layer in range(2):
        for control, target in pairs:
            instructions.append(
                CxInstruction(control=control, target=target, qubits=[control, target], params=[])
            )
        for qubit in range(4):
            theta = 0.3 * (layer + 1) * (qubit + 1)
            instructions.append(
                RyInstruction(qubit=qubit, qubits=[qubit], theta=theta, params=[theta])
            )

    for qubit in range(4):
        instructions.append(MeasureInstruction(qubit=qubit, cbit=qubit, qubits=[qubit]))

    return _circuit(4, instructions, cregs={"c": Register(name="c", size=4, base=0)})


class TestMpsConvergence:
    def test_full_distribution_converges_to_statevector_as_bond_dimension_increases(self) -> None:
        circuit = _high_entanglement_circuit()
        statevector_probs = simulate_monolithic(
            circuit, mode="state_vector", seed=SEED
        ).probabilities()

        errors = []
        for bond_dimension in [2, 3, 4, 8]:
            mps_probs = simulate_monolithic(
                circuit,
                mode="mps",
                seed=SEED,
                max_bond_dimension=bond_dimension,
            ).probabilities()
            errors.append(_l1_error(statevector_probs, mps_probs))

        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        assert errors[3] < 1e-10

    def test_marginals_converge_for_ordered_and_reordered_qubit_lists(self) -> None:
        circuit = _high_entanglement_circuit()
        statevector_result = simulate_monolithic(circuit, mode="state_vector", seed=SEED)
        mps_results = {
            bond_dimension: simulate_monolithic(
                circuit,
                mode="mps",
                seed=SEED,
                max_bond_dimension=bond_dimension,
            )
            for bond_dimension in [2, 3, 4, 8]
        }

        for qubits in ([1, 3], [3, 1], [5, 2, 0]):
            statevector_probs = statevector_result.probabilities(list(qubits))
            errors = [
                _l1_error(
                    statevector_probs,
                    mps_results[bond_dimension].probabilities(list(qubits)),
                )
                for bond_dimension in [2, 3, 4, 8]
            ]

            assert errors[1] < errors[0]
            assert errors[2] < errors[1]
            assert errors[3] < 1e-10

    def test_measurement_shot_distribution_converges_with_bond_dimension(self) -> None:
        circuit = _measured_entanglement_circuit()
        shots = 300
        statevector_counts = simulate_monolithic_shots(
            circuit, mode="state_vector", shots=shots, seed=SEED
        )

        errors = []
        for bond_dimension in [1, 2, 4]:
            mps_counts = simulate_monolithic_shots(
                circuit,
                mode="mps",
                shots=shots,
                seed=SEED,
                max_bond_dimension=bond_dimension,
            )
            assert sum(mps_counts.values()) == shots
            errors.append(_frequency_l1_error(statevector_counts, mps_counts, shots))

        assert errors[1] < errors[0]
        assert errors[2] < 1e-10
