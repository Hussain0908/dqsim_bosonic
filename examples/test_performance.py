"""
Performance benchmark: dqsim vs Qiskit Aer.

Run with:  pytest examples/test_performance.py -v -s

Two timing axes are reported for each circuit:

  per-run (ms)
    dqsim-sv   : StatevectorSimulator.simulate()
    dqsim-mps  : simulate_monolithic(distributed.as_monolithic_circuit(), mode="mps")
    dqsim-pblock : HypergraphDistributor.distribute() + PBlockSimulator.simulate()
    aer-sv     : AerSimulator(statevector).run(qc, shots=1) on a measurement-free circuit

  per-shot (µs)
    dqsim-sv   : (simulate() + result.counts(SHOTS)) / SHOTS
    dqsim-mps  : true MPS trajectories on distributed.as_monolithic_circuit()
    dqsim-pblock : (distribute() + simulate() + sample SHOTS) / SHOTS
    aer        : AerSimulator().run(qc_with_meas, shots=SHOTS) / SHOTS

All timings are from a single call.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import qasmpi
from bosonic_model.qasm import QasmError, Translator
from bosonic_converters import CircuitConverters
from bosonic_sdk.distributor.distributors.hypergraph_distributor import HypergraphDistributor
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor
from bosonic_sdk.simulation.simulator import Simulator as BosonicSimulator

from dqsim import (
    PBlockSimulator,
    StatevectorSimulator,
    simulate_monolithic,
    simulate_monolithic_shots,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
SHOTS = 100 # usually higher for good values - i'm keeping it small for quick local iterations

# (qasmpi_name, nodes, qubits_per_node)
# nodes × qubits_per_node must be >= circuit qubit count.
_BENCH_CIRCUITS = [
    ("deutsch_n2",    2, 1),
    ("toffoli_n3",    2, 3),
    ("adder_n4",      2, 2),
    ("qft_n4",        2, 2),
    ("bell_n4",       2, 2),
    ("qaoa_n6",       3, 3),
    ("qpe_n9",        3, 4),
    ("ising_n10",     2, 5),
    # ("qft_n18",       2, 9),
    # ("square_root_n18", 2, 9),
    # ("dnn_n16",       2, 8),
    # ("cc_n12",        2, 6),
    # ("bv_n14",         2, 7),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed_ms(fn) -> float:
    """Return wall-clock time of one fn() call, in milliseconds."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1_000


def _strip_measurements(circuit):
    return circuit.model_copy(
        update={"instructions": [i for i in circuit.instructions if i.kind != "measure"]}
    )


def _add_all_measurements(circuit):
    from bosonic_model.instructions import MeasureInstruction
    n = max(r.base + r.size for r in circuit.qregs.values())
    cregs = {}
    if not circuit.cregs:
        from bosonic_model import Register
        cregs = {"m": Register(name="m", size=n, base=0)}
    else:
        cregs = circuit.cregs
    meas = [MeasureInstruction(qubit=q, cbit=q) for q in range(n)]
    return circuit.model_copy(update={
        "cregs": cregs,
        "instructions": list(circuit.instructions) + meas,
    })


def _n_qubits(circuit) -> int:
    return max(r.base + r.size for r in circuit.qregs.values())


def _instruction_name(inst) -> str:
    inner = inst.op if getattr(inst, "kind", "") == "conditional" else inst
    return str(getattr(inner, "name", getattr(inner, "kind", "")))


def _instruction_names(distributed) -> list[str]:
    return [_instruction_name(inst) for inst in distributed.as_monolithic_circuit().instructions]


def _symbolic_stats(distributed) -> dict[str, int]:
    names = _instruction_names(distributed)
    remote_cz = sum(name == "remote_cz" for name in names)
    remote_swap = sum(name == "remote_swap" for name in names)
    return {
        "remote_cz": remote_cz,
        "remote_swap": remote_swap,
        "ebits": remote_cz + 2 * remote_swap,
    }


def _lowered_ebits(distributed) -> int:
    return sum(name == "remote_link_phi_plus" for name in _instruction_names(distributed))


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _bench_dqsim_sv_run(circuit) -> float:
    """Time (ms) for a single StatevectorSimulator.simulate() call."""
    sim = StatevectorSimulator(seed=SEED)
    return _elapsed_ms(lambda: sim.simulate(circuit))


def _bench_dqsim_sv_shots(circuit) -> float:
    """Time (ms) for simulate_shots() — true per-shot trajectories."""
    sim = StatevectorSimulator(seed=SEED)
    return _elapsed_ms(lambda: sim.simulate_shots(circuit, shots=SHOTS))


def _bench_dqsim_mps_run(circuit) -> float:
    """Time (ms) for MPS simulation on a monolithic circuit."""
    return _elapsed_ms(lambda: simulate_monolithic(circuit, mode="mps", seed=SEED))


def _bench_dqsim_mps_shots(circuit) -> float:
    """Time (ms) for true MPS shot trajectories."""
    return _elapsed_ms(
        lambda: simulate_monolithic_shots(circuit, mode="mps", shots=SHOTS, seed=SEED)
    )


def _bench_dqsim_pblock_run(distributed) -> float:
    """Time (ms) for PBlockSimulator.simulate() on a pre-distributed circuit."""
    sim = PBlockSimulator(seed=SEED)
    return _elapsed_ms(lambda: sim.simulate(distributed))


def _bench_dqsim_pblock_shots(distributed) -> float:
    """Time (ms) for simulate_shots() — true per-shot trajectories."""
    sim = PBlockSimulator(seed=SEED)
    return _elapsed_ms(lambda: sim.simulate_shots(distributed, shots=SHOTS))


def _bench_aer_sv_run(circuit) -> float:
    """Time (ms) for Aer statevector run — preprocessing and backend init excluded."""
    qc = CircuitConverters.to_qiskit(circuit)
    if qc.num_clbits == 0:
        qc.measure_all()
    sim = BosonicSimulator()
    qc = sim.prepare(qc)
    backend = sim.build_backend("statevector")
    return _elapsed_ms(lambda: sim.simulate(qc, backend, shots=1))


def _bench_aer_shots(circuit) -> float:
    """Time (ms) for Aer statevector shots — preprocessing and backend init excluded."""
    qc = CircuitConverters.to_qiskit(circuit)
    if qc.num_clbits == 0:
        qc.measure_all()
    sim = BosonicSimulator()
    qc = sim.prepare(qc)
    backend = sim.build_backend("statevector")
    return _elapsed_ms(lambda: sim.simulate(qc, backend, shots=SHOTS))


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_W = {
    "name":             16,
    "qb":                4,
    "sv_run":           12,
    "pblock_run":       13,
    "aer_run":          10,
    "sv_shot":          14,
    "pblock_shot":      15,
    "aer_shot":         11,
    "sv_speedup":       12,
    "pblock_speedup":   13,
}

_HEADER = (
    f"{'Circuit':<16}  {'Qb':>4}  "
    f"{'sv run(ms)':>12}  {'mps run(ms)':>12}  {'pblock(L=T) run':>16}  {'pblock(L=F) run':>16}  {'aer run(ms)':>11}  "
    f"{'sv shot(µs)':>12}  {'mps shot(µs)':>13}  {'pblock(L=T) shot':>17}  {'pblock(L=F) shot':>17}  {'aer shot(µs)':>12}  "
    f"{'sv speedup':>11}  {'mps speedup':>12}  {'L=T speedup':>12}  {'L=F speedup':>12}"
)

_SEP = "-" * len(_HEADER)


def _fmt_ms(v: float | None, w: int) -> str:
    return f"{v:>{w}.2f}" if v is not None else f"{'N/A':>{w}}"

def _fmt_speedup(aer: float, dqsim: float | None, w: int) -> str:
    return f"{aer / dqsim:>{w}.1f}x" if dqsim is not None else f"{'N/A':>{w}}"


def _row(
    name: str,
    qb: int,
    sv_run: float,
    mps_run: float | None,
    pblock_run: float,
    raw_run: float | None,
    aer_run: float,
    sv_shot_total: float,
    mps_shot_total: float | None,
    pblock_shot_total: float,
    raw_shot_total: float | None,
    aer_shot_total: float,
) -> str:
    sv_shot_us = sv_shot_total / SHOTS * 1_000
    mps_shot_us = mps_shot_total / SHOTS * 1_000 if mps_shot_total is not None else None
    pblock_shot_us = pblock_shot_total / SHOTS * 1_000
    raw_shot_us = raw_shot_total / SHOTS * 1_000 if raw_shot_total is not None else None
    aer_shot_us = aer_shot_total / SHOTS * 1_000

    return (
        f"{name:<16}  {qb:>4}  "
        f"{sv_run:>12.2f}  {_fmt_ms(mps_run, 12)}  {pblock_run:>16.2f}  {_fmt_ms(raw_run, 16)}  {aer_run:>11.2f}  "
        f"{sv_shot_us:>12.2f}  {_fmt_ms(mps_shot_us, 13)}  {pblock_shot_us:>17.2f}  {_fmt_ms(raw_shot_us, 17)}  {aer_shot_us:>12.2f}  "
        f"{aer_run / sv_run:>11.1f}x  {_fmt_speedup(aer_run, mps_run, 12)}  {aer_run / pblock_run:>12.1f}x  {_fmt_speedup(aer_run, raw_run, 12)}"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_benchmark_table(self) -> None:
        rows = []
        dist = DisqcoDistributor()

        total = len(_BENCH_CIRCUITS)
        for idx, (name, nodes, qpn) in enumerate(_BENCH_CIRCUITS, 1):
            print(f"  [{idx}/{total}] {name} ...", flush=True)

            raw_circuit = Translator().from_qasm(qasmpi.get_circuit(name))
            circuit_no_meas = _strip_measurements(raw_circuit)
            n = _n_qubits(circuit_no_meas)

            distributed_lowered = dist.distribute(
                circuit_no_meas,
                nodes=nodes,
                qubits_per_node=qpn,
                lowered=True,
            )
            actual_qpn = {n: len(qs) for n, qs in distributed_lowered.qubits_per_node.items()}
            print(f"         nodes={nodes}, qubits_per_node={qpn} → actual: {actual_qpn}", flush=True)
            
            try:
                distributed_symbolic = dist.distribute(
                    circuit_no_meas,
                    nodes=nodes,
                    qubits_per_node=qpn,
                    lowered=False,
                )
            except ValueError as exc:
                print(f"         lowered=False benchmark unavailable: {exc}", flush=True)
                distributed_symbolic = None

            print(f"         dqsim-sv run ...", end="", flush=True)
            sv_run = _bench_dqsim_sv_run(circuit_no_meas)
            print(f" {sv_run:.2f} ms", flush=True)

            try:
                print(f"         dqsim-mps run ...", end="", flush=True)
                mps_run = _bench_dqsim_mps_run(distributed_lowered.as_monolithic_circuit())
                print(f" {mps_run:.2f} ms", flush=True)
            except NotImplementedError as exc:
                print(f" unavailable: {exc}", flush=True)
                mps_run = None

            print(f"         dqsim-pblock run (lowered=True) ...", end="", flush=True)
            pblock_run = _bench_dqsim_pblock_run(distributed_lowered)
            print(f" {pblock_run:.2f} ms", flush=True)

            if distributed_symbolic is not None:
                print(f"         dqsim-pblock run (lowered=False) ...", end="", flush=True)
                raw_run = _bench_dqsim_pblock_run(distributed_symbolic)
                print(f" {raw_run:.2f} ms", flush=True)
            else:
                raw_run = None
            

            print(f"         bosonic (aer) sv run ...", end="", flush=True)
            aer_run = _bench_aer_sv_run(distributed_lowered.as_monolithic_circuit())
            print(f" {aer_run:.2f} ms", flush=True)

            print(f"         dqsim-sv {SHOTS} shots (monolithic) ...", end="", flush=True)
            sv_shot_total = _bench_dqsim_sv_shots(distributed_lowered.as_monolithic_circuit())
            print(f" {sv_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)

            if mps_run is not None:
                print(f"         dqsim-mps {SHOTS} shots (monolithic) ...", end="", flush=True)
                mps_shot_total = _bench_dqsim_mps_shots(distributed_lowered.as_monolithic_circuit())
                print(f" {mps_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)
            else:
                mps_shot_total = None

            print(f"         dqsim-pblock {SHOTS} shots (lowered=True) ...", end="", flush=True)
            pblock_shot_total = _bench_dqsim_pblock_shots(distributed_lowered)
            print(f" {pblock_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)

            if distributed_symbolic is not None:
                print(f"         dqsim-pblock {SHOTS} shots (lowered=False) ...", end="", flush=True)
                raw_shot_total = _bench_dqsim_pblock_shots(distributed_symbolic)
                print(f" {raw_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)
            else:
                raw_shot_total = None

            print(f"         bosonic (aer) {SHOTS} shots ...", end="", flush=True)
            aer_shot_total = _bench_aer_shots(distributed_lowered.as_monolithic_circuit())
            print(f" {aer_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)

            rows.append((name, n, sv_run, mps_run, pblock_run, raw_run, aer_run, sv_shot_total, mps_shot_total, pblock_shot_total, raw_shot_total, aer_shot_total))
            print(f"         done.", flush=True)

        print(f"\n\nPerformance: dqsim vs Qiskit Aer  (SHOTS={SHOTS}, single-call timing)\n")
        print(_HEADER)
        print(_SEP)
        for name, qb, sv_run, mps_run, pblock_run, raw_run, aer_run, sv_shot, mps_shot, pblock_shot, raw_shot, aer_shot in rows:
            print(_row(name, qb, sv_run, mps_run, pblock_run, raw_run, aer_run, sv_shot, mps_shot, pblock_shot, raw_shot, aer_shot))
        print(_SEP)
        print(
            "  sv run(ms)   : dqsim StatevectorSimulator.simulate() — one statevector evolution\n"
            "  mps run(ms)  : dqsim MPS simulation, materialized to dense SimulationResult\n"
            "  pblock run(ms): PBlockSimulator.simulate() on lowered=True distributed circuit\n"
            "  raw run(ms)   : PBlockSimulator.simulate() on lowered=False distributed circuit\n"
            "  aer run(ms)  : AerSimulator(statevector).run(shots=1) — one statevector evolution\n"
            "  *  shot(µs)  : total shot-batch time / SHOTS  (dqsim: simulate+counts; aer: run(shots=SHOTS))\n"
            "  *  speedup   : aer_run / dqsim_run  (>1 = dqsim faster)\n"
        )
