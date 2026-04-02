from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from bosonic_model.circuit import Circuit
from bosonic_model.distributed_circuit import DistributedCircuit
from bosonic_model.instructions import (
    BarrierInstruction,
    C3sqrtxInstruction,
    C3xInstruction,
    C4xInstruction,
    CcxInstruction,
    ChInstruction,
    ClassicalInstruction,
    ConditionalInstruction,
    CpInstruction,
    CrxInstruction,
    CryInstruction,
    CrzInstruction,
    CswapInstruction,
    CsxInstruction,
    Cu1Instruction,
    Cu3Instruction,
    CuInstruction,
    CxInstruction,
    CyInstruction,
    CzInstruction,
    GateInstruction,
    HInstruction,
    IdInstruction,
    InstructionType,
    MeasureInstruction,
    PInstruction,
    Rc3xInstruction,
    RccxInstruction,
    ResetInstruction,
    RxInstruction,
    RxxInstruction,
    RyInstruction,
    RzInstruction,
    RzzInstruction,
    SdgInstruction,
    SInstruction,
    SwapInstruction,
    SxdgInstruction,
    SxInstruction,
    TdgInstruction,
    TInstruction,
    U0Instruction,
    U1Instruction,
    U2Instruction,
    U3Instruction,
    UInstruction,
    XInstruction,
    YInstruction,
    ZInstruction,
)

from . import gates as g
from .engine import apply_n_qubit, apply_one_qubit, marginal_probs, measure_qubit, sample_counts


class SimulationResult:
    """Holds the statevector and classical bit state after a completed simulation."""

    def __init__(
        self,
        statevector: NDArray[np.complex128],
        num_qubits: int,
        classical_bits: dict[int, int],
    ) -> None:
        self._sv = statevector
        self._n = num_qubits
        self._cbits = classical_bits

    @property
    def statevector(self) -> NDArray[np.complex128]:
        """Raw complex amplitudes, shape (2^n,). Read-only view."""
        return self._sv

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def classical_bits(self) -> dict[int, int]:
        """Final classical register state. Keys are absolute cbit indices."""
        return dict(self._cbits)

    def probabilities(self, qubits: list[int] | None = None) -> dict[int, float]:
        """Full or marginal probability distribution. Keys are integer basis states.

        If `qubits` is None, returns all 2^n basis states.
        qubits[0] is MSB of the output index.
        """
        if qubits is None:
            qubits = list(range(self._n - 1, -1, -1))
        probs_arr = marginal_probs(self._sv, self._n, qubits)
        return {j: float(probs_arr[j]) for j in range(len(probs_arr)) if probs_arr[j] > 0}

    def counts(
        self,
        shots: int = 1000,
        qubits: list[int] | None = None,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Sample the distribution.

        Bitstrings have qubits[0] as the leftmost (MSB) character.
        If `qubits` is None, all qubits are included (qubit n-1 leftmost).
        Pass `seed` for reproducible sampling independent of the simulator seed.
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if qubits is None:
            qubits = list(range(self._n - 1, -1, -1))
        return sample_counts(self._sv, self._n, shots, rng, qubits)

    def fidelity(self, other: NDArray[np.complex128]) -> float:
        """Compute |<psi|other>|^2."""
        if other.shape != self._sv.shape:
            return 0.0
        return float(abs(np.vdot(self._sv, other)) ** 2)


class StatevectorSimulator:
    """Pure-NumPy statevector simulator for Circuit and DistributedCircuit."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng: Generator = np.random.default_rng(seed)

    def simulate(self, circuit: DistributedCircuit | Circuit) -> SimulationResult:
        """Run the circuit to completion and return the result."""
        if isinstance(circuit, DistributedCircuit):
            monolithic = circuit.as_monolithic_circuit()
        else:
            monolithic = circuit

        n = _num_qubits(monolithic)
        state = np.zeros(1 << n, dtype=np.complex128)
        state[0] = 1.0
        cbits: dict[int, int] = {}
        all_idx = np.arange(1 << n, dtype=np.intp)

        for inst in monolithic.instructions:
            self._apply(state, inst, n, cbits, all_idx)

        return SimulationResult(state, n, cbits)

    def _apply(
        self,
        state: NDArray[np.complex128],
        inst: InstructionType,
        n: int,
        cbits: dict[int, int],
        all_idx: np.ndarray,
    ) -> None:
        match inst:
            # ----------------------------------------------------------------
            # Single-qubit fixed
            # ----------------------------------------------------------------
            case IdInstruction():
                pass
            case XInstruction():
                apply_one_qubit(state, g.X, inst.qubit, n, all_idx=all_idx)
            case YInstruction():
                apply_one_qubit(state, g.Y, inst.qubit, n, all_idx=all_idx)
            case ZInstruction():
                apply_one_qubit(state, g.Z, inst.qubit, n, all_idx=all_idx)
            case HInstruction():
                apply_one_qubit(state, g.H, inst.qubit, n, all_idx=all_idx)
            case SInstruction():
                apply_one_qubit(state, g.S, inst.qubit, n, all_idx=all_idx)
            case SdgInstruction():
                apply_one_qubit(state, g.Sdg, inst.qubit, n, all_idx=all_idx)
            case TInstruction():
                apply_one_qubit(state, g.T, inst.qubit, n, all_idx=all_idx)
            case TdgInstruction():
                apply_one_qubit(state, g.Tdg, inst.qubit, n, all_idx=all_idx)
            case SxInstruction():
                apply_one_qubit(state, g.SX, inst.qubit, n, all_idx=all_idx)
            case SxdgInstruction():
                apply_one_qubit(state, g.SXdg, inst.qubit, n, all_idx=all_idx)
            # ----------------------------------------------------------------
            # Single-qubit parametric
            # ----------------------------------------------------------------
            case U3Instruction():
                apply_one_qubit(state, g.u3(inst.theta, inst.phi, inst.lam), inst.qubit, n, all_idx=all_idx)
            case U2Instruction():
                apply_one_qubit(state, g.u2(inst.phi, inst.lam), inst.qubit, n, all_idx=all_idx)
            case U1Instruction():
                apply_one_qubit(state, g.u1(inst.lam), inst.qubit, n, all_idx=all_idx)
            case UInstruction():
                apply_one_qubit(state, g.u(inst.theta, inst.phi, inst.lam), inst.qubit, n, all_idx=all_idx)
            case PInstruction():
                apply_one_qubit(state, g.p(inst.lam), inst.qubit, n, all_idx=all_idx)
            case RxInstruction():
                apply_one_qubit(state, g.rx(inst.theta), inst.qubit, n, all_idx=all_idx)
            case RyInstruction():
                apply_one_qubit(state, g.ry(inst.theta), inst.qubit, n, all_idx=all_idx)
            case RzInstruction():
                apply_one_qubit(state, g.rz(inst.phi), inst.qubit, n, all_idx=all_idx)
            case U0Instruction():
                pass  # identity / delay
            # ----------------------------------------------------------------
            # Two-qubit fixed
            # ----------------------------------------------------------------
            case CxInstruction():
                apply_n_qubit(state, g.CNOT, [inst.control, inst.target], n, all_idx=all_idx)
            case CzInstruction():
                apply_n_qubit(state, g.CZ, [inst.control, inst.target], n, all_idx=all_idx)
            case CyInstruction():
                apply_n_qubit(state, g.CY, [inst.control, inst.target], n, all_idx=all_idx)
            case ChInstruction():
                apply_n_qubit(state, g.CH, [inst.control, inst.target], n, all_idx=all_idx)
            case SwapInstruction():
                apply_n_qubit(state, g.SWAP, [inst.a, inst.b], n, all_idx=all_idx)
            case CsxInstruction():
                apply_n_qubit(state, g.CSX, [inst.control, inst.target], n, all_idx=all_idx)
            # ----------------------------------------------------------------
            # Two-qubit parametric
            # ----------------------------------------------------------------
            case CrxInstruction():
                apply_n_qubit(state, g.crx(inst.theta), [inst.control, inst.target], n, all_idx=all_idx)
            case CryInstruction():
                apply_n_qubit(state, g.cry(inst.theta), [inst.control, inst.target], n, all_idx=all_idx)
            case CrzInstruction():
                apply_n_qubit(state, g.crz(inst.lam), [inst.control, inst.target], n, all_idx=all_idx)
            case Cu1Instruction():
                apply_n_qubit(state, g.cu1(inst.lam), [inst.control, inst.target], n, all_idx=all_idx)
            case CpInstruction():
                apply_n_qubit(state, g.cp(inst.lam), [inst.control, inst.target], n, all_idx=all_idx)
            case Cu3Instruction():
                apply_n_qubit(
                    state, g.cu3(inst.theta, inst.phi, inst.lam), [inst.control, inst.target], n, all_idx=all_idx
                )
            case CuInstruction():
                apply_n_qubit(
                    state,
                    g.cu(inst.theta, inst.phi, inst.lam, inst.gamma),
                    [inst.control, inst.target],
                    n, all_idx=all_idx,
                )
            case RxxInstruction():
                apply_n_qubit(state, g.rxx(inst.theta), [inst.a, inst.b], n, all_idx=all_idx)
            case RzzInstruction():
                apply_n_qubit(state, g.rzz(inst.theta), [inst.a, inst.b], n, all_idx=all_idx)
            # ----------------------------------------------------------------
            # Three-qubit
            # ----------------------------------------------------------------
            case CcxInstruction():
                apply_n_qubit(
                    state, g.CCX, [inst.control1, inst.control2, inst.target], n, all_idx=all_idx
                )
            case CswapInstruction():
                apply_n_qubit(
                    state, g.CSWAP, [inst.control, inst.target1, inst.target2], n, all_idx=all_idx
                )
            case RccxInstruction():
                apply_n_qubit(
                    state, g.RCCX, [inst.control1, inst.control2, inst.target], n, all_idx=all_idx
                )
            case Rc3xInstruction():
                apply_n_qubit(
                    state,
                    g.RC3X,
                    [inst.control1, inst.control2, inst.control3, inst.target],
                    n, all_idx=all_idx,
                )
            case C3xInstruction():
                apply_n_qubit(
                    state,
                    g.C3X,
                    [inst.control1, inst.control2, inst.control3, inst.target],
                    n, all_idx=all_idx,
                )
            case C3sqrtxInstruction():
                apply_n_qubit(
                    state,
                    g.C3SQRTX,
                    [inst.control1, inst.control2, inst.control3, inst.target],
                    n, all_idx=all_idx,
                )
            case C4xInstruction():
                apply_n_qubit(
                    state,
                    g.C4X,
                    [inst.control1, inst.control2, inst.control3, inst.control4, inst.target],
                    n, all_idx=all_idx,
                )
            # ----------------------------------------------------------------
            # Cross-node / generic named gates
            # ----------------------------------------------------------------
            case GateInstruction():
                name = inst.name.lower()
                if name == "remote_link_psi_minus":
                    apply_n_qubit(state, g.PSI_MINUS, list(inst.qubits), n, all_idx=all_idx)
                elif name == "remote_link_psi_plus":
                    apply_n_qubit(state, g.PSI_PLUS, list(inst.qubits), n, all_idx=all_idx)
                elif name == "nonlocal_cz":
                    apply_n_qubit(state, g.NONLOCAL_CZ, list(inst.qubits), n, all_idx=all_idx)
                else:
                    raise NotImplementedError(
                        f"Unsupported generic gate: {inst.name!r}. "
                        "Decompose it before simulating."
                    )
            # ----------------------------------------------------------------
            # Measurement and classical control
            # ----------------------------------------------------------------
            case MeasureInstruction():
                outcome, _ = measure_qubit(state, inst.qubit, n, self._rng, all_idx=all_idx)
                cbits[inst.cbit] = outcome

            case ConditionalInstruction():
                cbit_val = cbits.get(inst.condition.cbit, 0)
                if bool(cbit_val) == inst.condition.value:
                    self._apply(state, inst.op, n, cbits, all_idx)

            case ResetInstruction():
                outcome, _ = measure_qubit(state, inst.qubit, n, self._rng, all_idx=all_idx)
                if outcome == 1:
                    apply_one_qubit(state, g.X, inst.qubit, n, all_idx=all_idx)

            # ----------------------------------------------------------------
            # No-ops
            # ----------------------------------------------------------------
            case BarrierInstruction() | ClassicalInstruction():
                pass

            case _:
                raise NotImplementedError(
                    f"Unsupported instruction type: {type(inst).__name__}"
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_qubits(circuit: Circuit) -> int:
    """Infer the number of qubits from the circuit's quantum registers."""
    if not circuit.qregs:
        return 0
    return max(reg.base + reg.size for reg in circuit.qregs.values())
