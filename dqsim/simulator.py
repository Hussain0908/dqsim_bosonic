from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

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


@dataclass
class SimulationProfile:
    """Timing and call-count data for a single simulate() call."""

    apply_one_qubit_calls: int = 0
    apply_one_qubit_time: float = 0.0
    apply_n_qubit_calls: int = 0
    apply_n_qubit_time: float = 0.0
    measure_qubit_calls: int = 0
    measure_qubit_time: float = 0.0
    total_time: float = 0.0

    # internal scratch — not part of the public interface
    _start: float = field(default=0.0, repr=False)

    def __repr__(self) -> str:
        total = self.total_time or 1e-9
        lines = [
            f"SimulationProfile (total: {self.total_time * 1000:.2f} ms)",
            f"  apply_one_qubit : {self.apply_one_qubit_calls:4d} calls  "
            f"{self.apply_one_qubit_time * 1000:8.2f} ms  "
            f"({100 * self.apply_one_qubit_time / total:.1f}%)",
            f"  apply_n_qubit   : {self.apply_n_qubit_calls:4d} calls  "
            f"{self.apply_n_qubit_time * 1000:8.2f} ms  "
            f"({100 * self.apply_n_qubit_time / total:.1f}%)",
            f"  measure_qubit   : {self.measure_qubit_calls:4d} calls  "
            f"{self.measure_qubit_time * 1000:8.2f} ms  "
            f"({100 * self.measure_qubit_time / total:.1f}%)",
        ]
        return "\n".join(lines)


class SimulationResult:
    """Holds the statevector and classical bit state after a completed simulation."""

    def __init__(
        self,
        statevector: NDArray[np.complex128],
        num_qubits: int,
        classical_bits: dict[int, int],
        profile: SimulationProfile | None = None,
    ) -> None:
        self._sv = statevector
        self._n = num_qubits
        self._cbits = classical_bits
        self._profile = profile

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

    @property
    def profile(self) -> SimulationProfile | None:
        """Profiling data, or None if the simulator was not run with profile=True."""
        return self._profile

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

    def __init__(self, seed: int | None = None, profile: bool = False) -> None:
        self._rng: Generator = np.random.default_rng(seed)
        self._profile = profile
        # Engine function references — swapped for timed wrappers when profiling.
        self._oq: Callable = apply_one_qubit
        self._nq: Callable = apply_n_qubit
        self._mq: Callable = measure_qubit

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

        prof: SimulationProfile | None = None
        if self._profile:
            prof = SimulationProfile()
            self._oq, self._nq, self._mq = _timed_wrappers(prof)

        t0 = time.perf_counter()
        for inst in monolithic.instructions:
            self._apply(state, inst, n, cbits, all_idx)

        if prof is not None:
            prof.total_time = time.perf_counter() - t0
            self._oq, self._nq, self._mq = apply_one_qubit, apply_n_qubit, measure_qubit

        return SimulationResult(state, n, cbits, profile=prof)

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
                self._oq(state, g.X, inst.qubit, n, all_idx=all_idx)
            case YInstruction():
                self._oq(state, g.Y, inst.qubit, n, all_idx=all_idx)
            case ZInstruction():
                self._oq(state, g.Z, inst.qubit, n, all_idx=all_idx)
            case HInstruction():
                self._oq(state, g.H, inst.qubit, n, all_idx=all_idx)
            case SInstruction():
                self._oq(state, g.S, inst.qubit, n, all_idx=all_idx)
            case SdgInstruction():
                self._oq(state, g.Sdg, inst.qubit, n, all_idx=all_idx)
            case TInstruction():
                self._oq(state, g.T, inst.qubit, n, all_idx=all_idx)
            case TdgInstruction():
                self._oq(state, g.Tdg, inst.qubit, n, all_idx=all_idx)
            case SxInstruction():
                self._oq(state, g.SX, inst.qubit, n, all_idx=all_idx)
            case SxdgInstruction():
                self._oq(state, g.SXdg, inst.qubit, n, all_idx=all_idx)
            # ----------------------------------------------------------------
            # Single-qubit parametric
            # ----------------------------------------------------------------
            case U3Instruction():
                self._oq(state, g.u3(inst.theta, inst.phi, inst.lam), inst.qubit, n, all_idx=all_idx)
            case U2Instruction():
                self._oq(state, g.u2(inst.phi, inst.lam), inst.qubit, n, all_idx=all_idx)
            case U1Instruction():
                self._oq(state, g.u1(inst.lam), inst.qubit, n, all_idx=all_idx)
            case UInstruction():
                self._oq(state, g.u(inst.theta, inst.phi, inst.lam), inst.qubit, n, all_idx=all_idx)
            case PInstruction():
                self._oq(state, g.p(inst.lam), inst.qubit, n, all_idx=all_idx)
            case RxInstruction():
                self._oq(state, g.rx(inst.theta), inst.qubit, n, all_idx=all_idx)
            case RyInstruction():
                self._oq(state, g.ry(inst.theta), inst.qubit, n, all_idx=all_idx)
            case RzInstruction():
                self._oq(state, g.rz(inst.phi), inst.qubit, n, all_idx=all_idx)
            case U0Instruction():
                pass  # identity / delay
            # ----------------------------------------------------------------
            # Two-qubit fixed
            # ----------------------------------------------------------------
            case CxInstruction():
                self._nq(state, g.CNOT, [inst.control, inst.target], n, all_idx=all_idx)
            case CzInstruction():
                self._nq(state, g.CZ, [inst.control, inst.target], n, all_idx=all_idx)
            case CyInstruction():
                self._nq(state, g.CY, [inst.control, inst.target], n, all_idx=all_idx)
            case ChInstruction():
                self._nq(state, g.CH, [inst.control, inst.target], n, all_idx=all_idx)
            case SwapInstruction():
                self._nq(state, g.SWAP, [inst.a, inst.b], n, all_idx=all_idx)
            case CsxInstruction():
                self._nq(state, g.CSX, [inst.control, inst.target], n, all_idx=all_idx)
            # ----------------------------------------------------------------
            # Two-qubit parametric
            # ----------------------------------------------------------------
            case CrxInstruction():
                self._nq(state, g.crx(inst.theta), [inst.control, inst.target], n, all_idx=all_idx)
            case CryInstruction():
                self._nq(state, g.cry(inst.theta), [inst.control, inst.target], n, all_idx=all_idx)
            case CrzInstruction():
                self._nq(state, g.crz(inst.lam), [inst.control, inst.target], n, all_idx=all_idx)
            case Cu1Instruction():
                self._nq(state, g.cu1(inst.lam), [inst.control, inst.target], n, all_idx=all_idx)
            case CpInstruction():
                self._nq(state, g.cp(inst.lam), [inst.control, inst.target], n, all_idx=all_idx)
            case Cu3Instruction():
                self._nq(
                    state, g.cu3(inst.theta, inst.phi, inst.lam), [inst.control, inst.target], n, all_idx=all_idx
                )
            case CuInstruction():
                self._nq(
                    state,
                    g.cu(inst.theta, inst.phi, inst.lam, inst.gamma),
                    [inst.control, inst.target],
                    n, all_idx=all_idx,
                )
            case RxxInstruction():
                self._nq(state, g.rxx(inst.theta), [inst.a, inst.b], n, all_idx=all_idx)
            case RzzInstruction():
                self._nq(state, g.rzz(inst.theta), [inst.a, inst.b], n, all_idx=all_idx)
            # ----------------------------------------------------------------
            # Three-qubit
            # ----------------------------------------------------------------
            case CcxInstruction():
                self._nq(
                    state, g.CCX, [inst.control1, inst.control2, inst.target], n, all_idx=all_idx
                )
            case CswapInstruction():
                self._nq(
                    state, g.CSWAP, [inst.control, inst.target1, inst.target2], n, all_idx=all_idx
                )
            case RccxInstruction():
                self._nq(
                    state, g.RCCX, [inst.control1, inst.control2, inst.target], n, all_idx=all_idx
                )
            case Rc3xInstruction():
                self._nq(
                    state,
                    g.RC3X,
                    [inst.control1, inst.control2, inst.control3, inst.target],
                    n, all_idx=all_idx,
                )
            case C3xInstruction():
                self._nq(
                    state,
                    g.C3X,
                    [inst.control1, inst.control2, inst.control3, inst.target],
                    n, all_idx=all_idx,
                )
            case C3sqrtxInstruction():
                self._nq(
                    state,
                    g.C3SQRTX,
                    [inst.control1, inst.control2, inst.control3, inst.target],
                    n, all_idx=all_idx,
                )
            case C4xInstruction():
                self._nq(
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
                    self._nq(state, g.PSI_MINUS, list(inst.qubits), n, all_idx=all_idx)
                elif name == "remote_link_psi_plus":
                    self._nq(state, g.PSI_PLUS, list(inst.qubits), n, all_idx=all_idx)
                elif name == "nonlocal_cz":
                    self._nq(state, g.NONLOCAL_CZ, list(inst.qubits), n, all_idx=all_idx)
                else:
                    raise NotImplementedError(
                        f"Unsupported generic gate: {inst.name!r}. "
                        "Decompose it before simulating."
                    )
            # ----------------------------------------------------------------
            # Measurement and classical control
            # ----------------------------------------------------------------
            case MeasureInstruction():
                outcome, _ = self._mq(state, inst.qubit, n, self._rng, all_idx=all_idx)
                cbits[inst.cbit] = outcome

            case ConditionalInstruction():
                cbit_val = cbits.get(inst.condition.cbit, 0)
                if bool(cbit_val) == inst.condition.value:
                    self._apply(state, inst.op, n, cbits, all_idx)

            case ResetInstruction():
                outcome, _ = self._mq(state, inst.qubit, n, self._rng, all_idx=all_idx)
                if outcome == 1:
                    self._oq(state, g.X, inst.qubit, n, all_idx=all_idx)

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
# Profiling helpers
# ---------------------------------------------------------------------------

def _timed_wrappers(prof: SimulationProfile):
    """Return timed versions of the three engine functions that accumulate into prof."""

    def _oq(state, u2, target, n, controls=None, all_idx=None):
        t0 = time.perf_counter()
        apply_one_qubit(state, u2, target, n, controls, all_idx)
        prof.apply_one_qubit_time += time.perf_counter() - t0
        prof.apply_one_qubit_calls += 1

    def _nq(state, u, qubits, n, all_idx=None):
        t0 = time.perf_counter()
        apply_n_qubit(state, u, qubits, n, all_idx)
        prof.apply_n_qubit_time += time.perf_counter() - t0
        prof.apply_n_qubit_calls += 1

    def _mq(state, qubit, n, rng, all_idx=None):
        t0 = time.perf_counter()
        result = measure_qubit(state, qubit, n, rng, all_idx)
        prof.measure_qubit_time += time.perf_counter() - t0
        prof.measure_qubit_calls += 1
        return result

    return _oq, _nq, _mq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_qubits(circuit: Circuit) -> int:
    """Infer the number of qubits from the circuit's quantum registers."""
    if not circuit.qregs:
        return 0
    return max(reg.base + reg.size for reg in circuit.qregs.values())
