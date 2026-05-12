"""
Compare dqsim statevector and MPS simulation on an MPS-friendly circuit.

Run with:
    uv run --python 3.11 maturin develop --release
    uv run --python 3.11 python examples/compare_sv_mps.py
    uv run --python 3.11 python examples/compare_sv_mps.py --compare-qubits 3,7,12

This is intentionally an example script, not a pytest test. It builds an
18-qubit nearest-neighbor brickwork circuit, times statevector once, then times
MPS with several max bond dimensions and reports total variation distance from
the statevector probabilities.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from bosonic_model import Circuit, Register
from bosonic_model.instructions import (
    CxInstruction,
    CzInstruction,
    RxInstruction,
    RyInstruction,
    RzInstruction,
)

from dqsim import simulate_monolithic


SEED = 42


def _elapsed_ms(fn: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1_000


def _total_variation_distance(a: dict[int, float], b: dict[int, float]) -> float:
    states = set(a) | set(b)
    return 0.5 * sum(abs(a.get(state, 0.0) - b.get(state, 0.0)) for state in states)


def build_nearest_neighbor_brickwork(n_qubits: int = 18, depth: int = 150) -> Circuit:
    instructions = []

    for layer in range(depth):
        for qubit in range(n_qubits):
            theta = 0.011 * (layer + qubit + 1)
            phi = 0.017 * (layer + 2 * qubit + 1)
            instructions.append(
                RyInstruction(qubit=qubit, qubits=[qubit], theta=theta, params=[theta])
            )
            instructions.append(
                RzInstruction(qubit=qubit, qubits=[qubit], phi=phi, params=[phi])
            )

        for control in range(0, n_qubits - 1, 2):
            target = control + 1
            instructions.append(
                CxInstruction(
                    control=control,
                    target=target,
                    qubits=[control, target],
                    params=[],
                )
            )

        for qubit in range(n_qubits):
            theta = 0.013 * (layer + 3 * qubit + 1)
            instructions.append(
                RxInstruction(qubit=qubit, qubits=[qubit], theta=theta, params=[theta])
            )

        for control in range(1, n_qubits - 1, 2):
            target = control + 1
            instructions.append(
                CzInstruction(
                    control=control,
                    target=target,
                    qubits=[control, target],
                    params=[],
                )
            )

    return Circuit(
        qregs={"q": Register(name="q", size=n_qubits, base=0)},
        cregs={},
        instructions=instructions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=18)
    parser.add_argument("--depth", type=int, default=150)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--truncation-threshold", type=float, default=1e-12)
    parser.add_argument(
        "--bond-dimensions",
        type=str,
        default="4,8,16,32,64,none",
        help="Comma-separated max bond dimensions. Use 'unlimited' for exact MPS.",
    )
    parser.add_argument(
        "--compare-qubits",
        type=str,
        default=None,
        help=(
            "Comma-separated qubit indices for marginal comparison. "
            "Defaults to the full distribution."
        ),
    )
    return parser.parse_args()


def _parse_bond_dimensions(value: str) -> list[int | None]:
    dimensions = []
    for token in value.split(","):
        token = token.strip().lower()
        if token in {"", "none", "unlimited"}:
            dimensions.append(None)
        else:
            dimensions.append(int(token))
    return dimensions


def _parse_compare_qubits(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def main() -> None:
    args = parse_args()
    circuit = build_nearest_neighbor_brickwork(args.qubits, args.depth)
    bond_dimensions = _parse_bond_dimensions(args.bond_dimensions)
    compare_qubits = _parse_compare_qubits(args.compare_qubits)

    print(
        f"circuit: {args.qubits} qubits, "
        f"depth={args.depth}, instructions={len(circuit.instructions)}"
    )

    statevector_result, statevector_ms = _elapsed_ms(
        lambda: simulate_monolithic(circuit, mode="state_vector", seed=args.seed)
    )
    statevector_probs = statevector_result.probabilities(compare_qubits)

    print(f"state_vector  time_ms={statevector_ms:10.2f}")
    if compare_qubits is None:
        print("comparison: full distribution")
    else:
        print(f"comparison: marginal qubits {compare_qubits}")
    print("mps           bond_dim     time_ms      tvd_vs_sv")

    for bond_dimension in bond_dimensions:
        options = {
            "mode": "mps",
            "seed": args.seed,
            "truncation_threshold": args.truncation_threshold,
        }
        if bond_dimension is not None:
            options["max_bond_dimension"] = bond_dimension

        mps_result, mps_ms = _elapsed_ms(lambda: simulate_monolithic(circuit, **options))
        tvd = _total_variation_distance(
            statevector_probs,
            mps_result.probabilities(compare_qubits),
        )
        label = "unlimited" if bond_dimension is None else str(bond_dimension)
        print(f"mps           {label:>8}  {mps_ms:10.2f}  {tvd:13.6e}")


if __name__ == "__main__":
    main()
