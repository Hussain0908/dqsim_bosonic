from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def apply_one_qubit(
    state: NDArray[np.complex128],
    u2: NDArray[np.complex128],
    target: int,
    n: int,
    controls: list[tuple[int, bool]] | None = None,
    all_idx: NDArray[np.intp] | None = None,
) -> None:
    """Apply a 2x2 unitary to `target` qubit, in-place.

    `controls` is a list of (wire_index, is_control) pairs.
    is_control=True means the wire must be |1> to activate the gate.
    is_control=False means the wire must be |0> (anti-control).
    """
    inclusion_mask = 0
    desired_mask = 0
    for wire, flag in (controls or []):
        bit = 1 << wire
        inclusion_mask |= bit
        if flag:
            desired_mask |= bit

    if all_idx is None:
        all_idx = np.arange(1 << n, dtype=np.intp)

    idx0 = all_idx[(all_idx & (1 << target)) == 0]

    # Filter by control conditions
    if inclusion_mask:
        idx0 = idx0[(idx0 & inclusion_mask) == desired_mask]

    idx1 = idx0 | (1 << target)

    # Gather, apply 2x2 unitary, scatter — all vectorised
    a = state[idx0]
    b = state[idx1]
    state[idx0] = u2[0, 0] * a + u2[0, 1] * b
    state[idx1] = u2[1, 0] * a + u2[1, 1] * b


def apply_n_qubit(
    state: NDArray[np.complex128],
    u: NDArray[np.complex128],
    qubits: list[int],
    n: int,
    all_idx: NDArray[np.intp] | None = None,
) -> None:
    """Apply a 2^k x 2^k unitary to the given qubits (arbitrary positions), in-place.

    qubits[0] is the MSB of the gate's index space (e.g. control qubit for CNOT).
    qubits[-1] is the LSB (e.g. target qubit for CNOT).
    """
    k = len(qubits)
    dim = 1 << k

    if all_idx is None:
        all_idx = np.arange(1 << n, dtype=np.intp)

    # All state indices where every target qubit is 0
    mask = 0
    for q in qubits:
        mask |= 1 << q
    base = all_idx[(all_idx & mask) == 0]  # shape: (2^(n-k),)

    # Build index table: idx[j] = base + offset for gate-row j
    idx = np.empty((dim, len(base)), dtype=np.intp)
    for j in range(dim):
        offset = 0
        for i, q in enumerate(qubits):
            if (j >> (k - 1 - i)) & 1:
                offset += 1 << q
        idx[j] = base + offset

    # Gather amplitudes, apply gate matrix, scatter back
    v = state[idx]          # (dim, num_base) complex128
    w = u @ v               # (dim, num_base)
    for j in range(dim):
        state[idx[j]] = w[j]


def measure_qubit(
    state: NDArray[np.complex128],
    qubit: int,
    n: int,
    rng: Generator,
    all_idx: NDArray[np.intp] | None = None,
) -> tuple[int, NDArray[np.complex128]]:
    """Measure a single qubit, collapse the state, renormalize.

    Returns (outcome, collapsed_state). The original state array is modified in-place
    and also returned for convenience.
    """
    if all_idx is None:
        all_idx = np.arange(1 << n, dtype=np.intp)
    ones_mask = all_idx[(all_idx & (1 << qubit)) != 0]

    p1 = float(np.sum(np.abs(state[ones_mask]) ** 2))
    outcome = int(rng.random() < p1)

    if outcome == 1:
        # Zero out |qubit=0> amplitudes
        zeros_mask = all_idx[(all_idx & (1 << qubit)) == 0]
        state[zeros_mask] = 0.0
        norm = np.sqrt(p1) if p1 > 0 else 1.0
    else:
        # Zero out |qubit=1> amplitudes
        state[ones_mask] = 0.0
        norm = np.sqrt(max(1.0 - p1, 0.0))
        if norm == 0.0:
            norm = 1.0

    state /= norm
    return outcome, state


def marginal_probs(
    state: NDArray[np.complex128],
    n: int,
    qubits: list[int],
) -> NDArray[np.float64]:
    """Compute marginal probability distribution over the specified qubits.

    Returns a 1D array of length 2^len(qubits) where index j corresponds to
    the basis state of the specified qubits (qubits[0]=MSB, qubits[-1]=LSB).
    """
    k = len(qubits)
    dim = 1 << k
    probs = np.zeros(dim, dtype=np.float64)

    all_idx = np.arange(1 << n, dtype=np.intp)
    mask = 0
    for q in qubits:
        mask |= 1 << q

    # Group full state indices by their marginal pattern
    for j in range(dim):
        # Compute which full-state indices map to marginal index j
        desired = 0
        for i, q in enumerate(qubits):
            if (j >> (k - 1 - i)) & 1:
                desired |= 1 << q
        sub_idx = all_idx[(all_idx & mask) == desired]
        probs[j] = float(np.sum(np.abs(state[sub_idx]) ** 2))

    return probs


def sample_counts(
    state: NDArray[np.complex128],
    n: int,
    shots: int,
    rng: Generator,
    qubits: list[int] | None = None,
) -> dict[str, int]:
    """Sample `shots` outcomes from the state distribution.

    If `qubits` is given, marginalize to that subset before sampling.
    qubits[0] is MSB of the output bitstring.
    Returns {bitstring: count}.
    """
    if qubits is None:
        qubits = list(range(n - 1, -1, -1))  # all qubits, MSB first

    k = len(qubits)
    probs = marginal_probs(state, n, qubits)

    # Build CDF and sample all shots at once
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0  # guard against floating-point drift

    r = rng.random(shots)
    outcomes = np.searchsorted(cdf, r)
    outcomes = np.clip(outcomes, 0, (1 << k) - 1)

    counts: dict[str, int] = {}
    for outcome in outcomes:
        bits = format(int(outcome), f"0{k}b")
        counts[bits] = counts.get(bits, 0) + 1

    return counts
