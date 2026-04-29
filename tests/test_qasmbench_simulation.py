"""
Smoke tests: fetch small QASMBench circuits via qasmpi, simulate with
StatevectorSimulator, and print the marginal probability distribution.

Circuits that cannot be parsed by the current translator (multi-bit
conditionals, unknown qregs from includes) are skipped automatically.
"""

import math

import pytest
import qasmpi
from bosonic_model.qasm import QasmError, Translator

from dqsim import StatevectorSimulator

_ALL_SMALL = [
    "adder_n10",
    "adder_n4",
    "basis_change_n3",
    "basis_test_n4",
    "basis_trotter_n4",
    "bb84_n8",
    "bell_n4",
    "cat_state_n4",
    "deutsch_n2",
    "dnn_n2",
    "dnn_n8",
    "error_correctiond3_n5",
    "fredkin_n3",
    "grover_n2",
    # "hhl_n10", -- slow, 186,801 gate ops
    "hhl_n7",
    "hs4_n4",
    "inverseqft_n4",
    "ipea_n2",
    "ising_n10",
    "iswap_n2",
    "linearsolver_n3",
    "lpn_n5",
    "pea_n5",
    "qaoa_n3",
    "qaoa_n6",
    "qec_en_n5",
    "qec_sm_n5",
    "qft_n4",
    "qpe_n9",
    "qrng_n4",
    "quantumwalks_n2",
    "sat_n7",
    "shor_n5",
    "simon_n6",
    "teleportation_n3",
    "toffoli_n3",
    "variational_n4",
    "vqe_n4",
    "wstate_n3",
]

SEED = 42


def _simulate(name: str):
    qasm_text = qasmpi.get_circuit(name)
    circuit = Translator().from_qasm(qasm_text)
    sim = StatevectorSimulator(seed=SEED)
    return sim.simulate(circuit)


def _format_probs(probs: dict, n: int) -> str:
    entries = sorted(probs.items(), key=lambda x: -x[1])
    lines = [f"  |{k:0{n}b}⟩  {v:.6f}" for k, v in entries]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parametrized smoke test — all small circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_SMALL)
def test_small_circuit_marginal_distribution(name: str, capsys) -> None:
    try:
        result = _simulate(name)
    except (QasmError, NotImplementedError) as exc:
        pytest.skip(f"circuit not supported by current translator: {exc}")

    probs = result.probabilities()
    n = result.num_qubits

    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-6, f"probabilities sum to {total}, expected 1.0"
    assert all(p >= 0.0 for p in probs.values()), "negative probability found"
    assert len(probs) >= 1, "empty probability distribution"

    print(f"\n{name} ({n} qubits) — marginal distribution:")
    print(_format_probs(probs, n))


# ---------------------------------------------------------------------------
# Targeted tests for circuits with analytically known outcomes
# ---------------------------------------------------------------------------


class TestKnownOutcomes:
    """
    These circuits produce a deterministic post-measurement state, so we can
    assert the exact dominant basis state regardless of shots.
    """

    def _dominant(self, name: str) -> tuple[int, float]:
        result = _simulate(name)
        probs = result.probabilities()
        return max(probs.items(), key=lambda x: x[1])

    def test_deutsch_n2(self) -> None:
        # Deutsch algorithm on a constant function: answer qubit ends in |1⟩
        # Full state index 1 = 0b01 means qubit 0 = 1
        state, prob = self._dominant("deutsch_n2")
        assert prob > 0.99
        assert state == 1

    def test_grover_n2(self) -> None:
        # 2-qubit Grover search: both qubits marked → |11⟩ = state 3
        state, prob = self._dominant("grover_n2")
        assert prob > 0.99
        assert state == 3

    def test_toffoli_n3(self) -> None:
        # Toffoli on |110⟩ → |111⟩, state 7 = 0b111
        state, prob = self._dominant("toffoli_n3")
        assert prob > 0.99
        assert state == 7

    def test_fredkin_n3(self) -> None:
        # Fredkin (controlled-SWAP); result is a single basis state
        state, prob = self._dominant("fredkin_n3")
        assert prob > 0.99

    def test_teleportation_n3(self) -> None:
        # Teleportation protocol ends in a definite qubit state
        state, prob = self._dominant("teleportation_n3")
        assert prob > 0.99


class TestNormalisationAndSupport:
    """
    Property checks that must hold for every circuit regardless of its physics.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "qft_n4",
            "ising_n10",
            "adder_n10",
            "qaoa_n6",
            "simon_n6",
            "qpe_n9",
            "sat_n7",
        ],
    )
    def test_probabilities_are_normalised(self, name: str) -> None:
        try:
            result = _simulate(name)
        except (QasmError, NotImplementedError) as exc:
            pytest.skip(str(exc))

        probs = result.probabilities()
        total = sum(probs.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    @pytest.mark.parametrize(
        "name",
        [
            "qft_n4",
            "ising_n10",
            "adder_n10",
            "qaoa_n6",
            "simon_n6",
            "qpe_n9",
            "sat_n7",
        ],
    )
    def test_probabilities_are_non_negative(self, name: str) -> None:
        try:
            result = _simulate(name)
        except (QasmError, NotImplementedError) as exc:
            pytest.skip(str(exc))

        probs = result.probabilities()
        for state, p in probs.items():
            assert p >= 0.0, f"state {state} has negative probability {p}"
