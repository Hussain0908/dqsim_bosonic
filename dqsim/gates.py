from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bosonic_converters.remote_link import RemoteLinkGatePsiMinus, RemoteLinkGatePsiPlus

C = np.complex128

# ---------------------------------------------------------------------------
# Fixed single-qubit gates
# ---------------------------------------------------------------------------

I2: NDArray[C] = np.eye(2, dtype=C)

X: NDArray[C] = np.array([[0, 1], [1, 0]], dtype=C)
Y: NDArray[C] = np.array([[0, -1j], [1j, 0]], dtype=C)
Z: NDArray[C] = np.array([[1, 0], [0, -1]], dtype=C)
H: NDArray[C] = np.array([[1, 1], [1, -1]], dtype=C) / np.sqrt(2)

S: NDArray[C] = np.diag([1, 1j]).astype(C)
Sdg: NDArray[C] = np.diag([1, -1j]).astype(C)
T: NDArray[C] = np.diag([1, np.exp(1j * np.pi / 4)]).astype(C)
Tdg: NDArray[C] = np.diag([1, np.exp(-1j * np.pi / 4)]).astype(C)

SX: NDArray[C] = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=C) / 2
SXdg: NDArray[C] = SX.conj().T.copy()

# ---------------------------------------------------------------------------
# Parametric single-qubit gates
# ---------------------------------------------------------------------------


def u3(theta: float, phi: float, lam: float) -> NDArray[C]:
    """General SU(2) gate (IBM u3)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=C,
    )


def u2(phi: float, lam: float) -> NDArray[C]:
    return u3(np.pi / 2, phi, lam)


def u1(lam: float) -> NDArray[C]:
    return np.diag([1, np.exp(1j * lam)]).astype(C)


def u(theta: float, phi: float, lam: float) -> NDArray[C]:
    """IBM U gate (same as u3, no global phase difference at this level)."""
    return u3(theta, phi, lam)


def p(lam: float) -> NDArray[C]:
    """Phase gate: diag(1, e^{i*lam})."""
    return np.diag([1, np.exp(1j * lam)]).astype(C)


def rx(theta: float) -> NDArray[C]:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=C)


def ry(theta: float) -> NDArray[C]:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=C)


def rz(phi: float) -> NDArray[C]:
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)]).astype(C)


def u0(_gamma: float) -> NDArray[C]:
    """u0 is a delay / identity."""
    return I2.copy()


# ---------------------------------------------------------------------------
# Fixed two-qubit gates  (4x4, qubit ordering: qubits[0]=MSB, qubits[1]=LSB)
# ---------------------------------------------------------------------------

CNOT: NDArray[C] = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=C
)

CZ: NDArray[C] = np.diag([1, 1, 1, -1]).astype(C)

CY: NDArray[C] = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=C
)

_h = 1 / np.sqrt(2)
CH: NDArray[C] = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, _h, _h],
        [0, 0, _h, -_h],
    ],
    dtype=C,
)

SWAP: NDArray[C] = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=C
)

# Controlled-SX
_sx = (1 + 1j) / 2
_sxc = (1 - 1j) / 2
CSX: NDArray[C] = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, _sx, _sxc],
        [0, 0, _sxc, _sx],
    ],
    dtype=C,
)

# ---------------------------------------------------------------------------
# Parametric two-qubit gates
# ---------------------------------------------------------------------------


def _controlled(u2x2: NDArray[C]) -> NDArray[C]:
    """Embed a 2x2 unitary as the lower-right block of a 4x4 controlled gate."""
    m = np.eye(4, dtype=C)
    m[2:, 2:] = u2x2
    return m


def crx(theta: float) -> NDArray[C]:
    return _controlled(rx(theta))


def cry(theta: float) -> NDArray[C]:
    return _controlled(ry(theta))


def crz(lam: float) -> NDArray[C]:
    return _controlled(rz(lam))


def cu1(lam: float) -> NDArray[C]:
    return _controlled(u1(lam))


def cp(lam: float) -> NDArray[C]:
    return _controlled(p(lam))


def cu3(theta: float, phi: float, lam: float) -> NDArray[C]:
    return _controlled(u3(theta, phi, lam))


def cu(theta: float, phi: float, lam: float, gamma: float) -> NDArray[C]:
    """IBM CU gate — adds global phase e^{i*gamma} to the U block."""
    inner = np.exp(1j * gamma) * u3(theta, phi, lam)
    return _controlled(inner)


def rxx(theta: float) -> NDArray[C]:
    """exp(-i theta/2 * X⊗X)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]],
        dtype=C,
    )


def rzz(theta: float) -> NDArray[C]:
    """exp(-i theta/2 * Z⊗Z)."""
    ep = np.exp(1j * theta / 2)
    em = np.exp(-1j * theta / 2)
    return np.diag([em, ep, ep, em]).astype(C)


# ---------------------------------------------------------------------------
# Fixed three-qubit gates  (8x8)
# ---------------------------------------------------------------------------

def _toffoli() -> NDArray[C]:
    m = np.eye(8, dtype=C)
    m[6, 6] = m[7, 7] = 0
    m[6, 7] = m[7, 6] = 1
    return m


CCX: NDArray[C] = _toffoli()


def _fredkin() -> NDArray[C]:
    m = np.eye(8, dtype=C)
    m[5, 5] = m[6, 6] = 0
    m[5, 6] = m[6, 5] = 1
    return m


CSWAP: NDArray[C] = _fredkin()


def _rccx() -> NDArray[C]:
    """Relative-phase Toffoli (simplified Toffoli)."""
    m = np.zeros((8, 8), dtype=C)
    m[0, 0] = 1
    m[1, 1] = 1
    m[2, 2] = 1
    m[3, 3] = 1
    m[4, 4] = 1
    m[5, 5] = 1
    # On |110> and |111>:
    m[6, 7] = 1j
    m[7, 6] = -1j
    return m


RCCX: NDArray[C] = _rccx()


def _rc3x() -> NDArray[C]:
    """Simplified 3-controlled-X (relative phase)."""
    m = np.eye(16, dtype=C)
    # Acts on |1110> ↔ |1111> with phase
    m[14, 14] = 1j
    m[14, 15] = 0
    m[15, 14] = 0
    m[15, 15] = -1j
    return m


RC3X: NDArray[C] = _rc3x()


def _c3x() -> NDArray[C]:
    """4-qubit CCCx (3 controls, 1 target)."""
    m = np.eye(16, dtype=C)
    m[14, 14] = m[15, 15] = 0
    m[14, 15] = m[15, 14] = 1
    return m


C3X: NDArray[C] = _c3x()


def _c3sqrtx() -> NDArray[C]:
    """CCCSqrtX."""
    m = np.eye(16, dtype=C)
    m[14:, 14:] = SX
    return m


C3SQRTX: NDArray[C] = _c3sqrtx()


def _c4x() -> NDArray[C]:
    """5-qubit CCCCx (4 controls, 1 target)."""
    m = np.eye(32, dtype=C)
    m[30, 30] = m[31, 31] = 0
    m[30, 31] = m[31, 30] = 1
    return m


C4X: NDArray[C] = _c4x()

# ---------------------------------------------------------------------------
# Cross-node gates
# ---------------------------------------------------------------------------

PSI_MINUS: NDArray[C] = RemoteLinkGatePsiMinus().to_matrix().astype(C)
PSI_PLUS: NDArray[C] = RemoteLinkGatePsiPlus().to_matrix().astype(C)
NONLOCAL_CZ: NDArray[C] = np.diag([1, 1, 1, -1]).astype(C)
