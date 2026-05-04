"""
Performance benchmark: dqsim vs Qiskit Aer.

Run with:  pytest tests/test_performance.py -v -s

Two timing axes are reported for each circuit:

  per-run (ms)
    dqsim-sv   : StatevectorSimulator.simulate()
    dqsim-comp : DisqcoDistributor.distribute() + CompositeSimulator.simulate()
    aer-sv     : AerSimulator(statevector).run(qc, shots=1) on a measurement-free circuit

  per-shot (µs)
    dqsim-sv   : (simulate() + result.counts(SHOTS)) / SHOTS
    dqsim-comp : (distribute() + simulate() + sample SHOTS) / SHOTS
    aer        : AerSimulator().run(qc_with_meas, shots=SHOTS) / SHOTS

All per-run numbers are the median of REPS independent calls.
All per-shot numbers are the median of REPS full shot-batch calls.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest
import qasmpi
from bosonic_model.qasm import QasmError, Translator
from bosonic_converters import CircuitConverters
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor
from qiskit_aer import AerSimulator as _AerSimulator

from dqsim import CompositeSimulator, StatevectorSimulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
SHOTS = 1000
REPS = 5  # timing repetitions per metric; median is reported

# (qasmpi_name, nodes, qubits_per_node)
# nodes × qubits_per_node must be >= circuit qubit count.
_BENCH_CIRCUITS = [
    ("deutsch_n2",    2, 1),
    ("toffoli_n3",    2, 2),
    ("adder_n4",      2, 2),
    ("qft_n4",        2, 2),
    ("bell_n4",       2, 2),
    ("qaoa_n6",       2, 3),
    ("qpe_n9",        2, 5),   # too slow for composite sim
    ("ising_n10",     2, 5),   # too slow for composite sim
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median_ms(fn, reps: int = REPS) -> float:
    """Return median wall-clock time of fn() over `reps` calls, in milliseconds."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1_000)
    return statistics.median(times)


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


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _bench_dqsim_sv_run(circuit) -> float:
    """Median time (ms) for a single StatevectorSimulator.simulate() call."""
    sim = StatevectorSimulator(seed=SEED)
    return _median_ms(lambda: sim.simulate(circuit))


def _bench_dqsim_sv_shots(circuit) -> float:
    """Median time (ms) for simulate() + counts(SHOTS)."""
    sim = StatevectorSimulator(seed=SEED)

    def run():
        result = sim.simulate(circuit)
        result.counts(SHOTS, seed=SEED)

    return _median_ms(run)


def _bench_dqsim_comp_run(distributed) -> float:
    """Median time (ms) for CompositeSimulator.simulate() on a pre-distributed circuit."""
    sim = CompositeSimulator(seed=SEED)
    return _median_ms(lambda: sim.simulate(distributed))


def _bench_dqsim_comp_shots(distributed) -> float:
    """Median time (ms) for CompositeSimulator.simulate() + SHOTS samples."""
    sim = CompositeSimulator(seed=SEED)
    rng = np.random.default_rng(SEED)

    def run():
        result = sim.simulate(distributed)
        probs = result.probabilities()
        if probs:
            states = list(probs.keys())
            weights = np.array([probs[s] for s in states], dtype=float)
            weights /= weights.sum()
            rng.choice(states, size=SHOTS, p=weights)

    return _median_ms(run)


def _bench_aer_sv_run(circuit) -> float:
    """Median time (ms) for AerSimulator(statevector).run(qc_no_meas, shots=1)."""
    no_meas = _strip_measurements(circuit)
    qc = CircuitConverters.to_qiskit(no_meas)
    aer = _AerSimulator(method="statevector")

    def run():
        aer.run(qc, shots=1).result()

    return _median_ms(run)


def _bench_aer_shots(circuit) -> float:
    """Median time (ms) for AerSimulator.run(qc_with_meas, shots=SHOTS)."""
    qc = CircuitConverters.to_qiskit(circuit)
    # Ensure there are measurements; add them if missing.
    if qc.num_clbits == 0:
        qc.measure_all()
    aer = _AerSimulator()

    def run():
        aer.run(qc, shots=SHOTS, seed_simulator=SEED).result()

    return _median_ms(run)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_W = {
    "name":             16,
    "qb":                4,
    "sv_run":           12,
    "comp_run":         13,
    "aer_run":          10,
    "sv_shot":          14,
    "comp_shot":        15,
    "aer_shot":         11,
    "sv_speedup":       12,
    "comp_speedup":     13,
}

_HEADER = (
    f"{'Circuit':<16}  {'Qb':>4}  "
    f"{'sv run(ms)':>12}  {'comp(L=T) run':>14}  {'comp(L=F) run':>14}  {'aer run(ms)':>11}  "
    f"{'sv shot(µs)':>12}  {'comp(L=T) shot':>15}  {'comp(L=F) shot':>15}  {'aer shot(µs)':>12}  "
    f"{'sv speedup':>11}  {'L=T speedup':>12}  {'L=F speedup':>12}"
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
    comp_run: float,
    raw_run: float | None,
    aer_run: float,
    sv_shot_total: float,
    comp_shot_total: float,
    raw_shot_total: float | None,
    aer_shot_total: float,
) -> str:
    sv_shot_us = sv_shot_total / SHOTS * 1_000
    comp_shot_us = comp_shot_total / SHOTS * 1_000
    raw_shot_us = raw_shot_total / SHOTS * 1_000 if raw_shot_total is not None else None
    aer_shot_us = aer_shot_total / SHOTS * 1_000

    return (
        f"{name:<16}  {qb:>4}  "
        f"{sv_run:>12.2f}  {comp_run:>14.2f}  {_fmt_ms(raw_run, 14)}  {aer_run:>11.2f}  "
        f"{sv_shot_us:>12.2f}  {comp_shot_us:>15.2f}  {_fmt_ms(raw_shot_us, 15)}  {aer_shot_us:>12.2f}  "
        f"{aer_run / sv_run:>11.1f}x  {aer_run / comp_run:>12.1f}x  {_fmt_speedup(aer_run, raw_run, 12)}"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_benchmark_table(self) -> None:
        rows = []

        total = len(_BENCH_CIRCUITS)
        for idx, (name, nodes, qpn) in enumerate(_BENCH_CIRCUITS, 1):
            print(f"  [{idx}/{total}] {name} ...", flush=True)

            raw_circuit = Translator().from_qasm(qasmpi.get_circuit(name))
            circuit_no_meas = _strip_measurements(raw_circuit)
            n = _n_qubits(circuit_no_meas)
            dist = DisqcoDistributor()
            
            distributed_lowered = dist.distribute(circuit_no_meas, nodes=nodes, qubits_per_node=qpn, lowered=True)
            print(f"qubits_per_node: { {n: len(qs) for n, qs in distributed_lowered.qubits_per_node.items()} }", flush=True)
            remote_count = sum(
                1 for c in distributed_lowered.circuits.values()
                for inst in c.instructions
                if getattr(inst, 'name', '').startswith('remote_')
            )
            print(f"remote gates: {remote_count}", flush=True)

            try:
                distributed_raw = dist.distribute(circuit_no_meas, nodes=nodes, qubits_per_node=qpn, lowered=False)
            except ValueError as exc:
                print(f"         lowered=False unavailable: {exc}", flush=True)
                distributed_raw = None

            print(f"         dqsim-sv run ...", end="", flush=True)
            sv_run = _bench_dqsim_sv_run(circuit_no_meas)
            print(f" {sv_run:.2f} ms", flush=True)

            print(f"         dqsim-comp run (lowered=True) ...", end="", flush=True)
            comp_run = _bench_dqsim_comp_run(distributed_lowered)
            print(f" {comp_run:.2f} ms", flush=True)

            if distributed_raw is not None:
                print(f"         dqsim-comp run (lowered=False) ...", end="", flush=True)
                raw_run = _bench_dqsim_comp_run(distributed_raw)
                print(f" {raw_run:.2f} ms", flush=True)
            else:
                raw_run = None

            print(f"         aer-sv run ...", end="", flush=True)
            aer_run = _bench_aer_sv_run(circuit_no_meas)
            print(f" {aer_run:.2f} ms", flush=True)

            print(f"         dqsim-sv {SHOTS} shots ...", end="", flush=True)
            sv_shot_total = _bench_dqsim_sv_shots(circuit_no_meas)
            print(f" {sv_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)

            print(f"         dqsim-comp {SHOTS} shots (lowered=True) ...", end="", flush=True)
            comp_shot_total = _bench_dqsim_comp_shots(distributed_lowered)
            print(f" {comp_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)

            if distributed_raw is not None:
                print(f"         dqsim-comp {SHOTS} shots (lowered=False) ...", end="", flush=True)
                raw_shot_total = _bench_dqsim_comp_shots(distributed_raw)
                print(f" {raw_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)
            else:
                raw_shot_total = None

            print(f"         aer {SHOTS} shots ...", end="", flush=True)
            aer_shot_total = _bench_aer_shots(circuit_no_meas)
            print(f" {aer_shot_total / SHOTS * 1_000:.2f} µs/shot", flush=True)

            rows.append((name, n, sv_run, comp_run, raw_run, aer_run, sv_shot_total, comp_shot_total, raw_shot_total, aer_shot_total))
            print(f"         done.", flush=True)

        print(f"\n\nPerformance: dqsim vs Qiskit Aer  (SHOTS={SHOTS}, REPS={REPS}, median timing)\n")
        print(_HEADER)
        print(_SEP)
        for name, qb, sv_run, comp_run, raw_run, aer_run, sv_shot, comp_shot, raw_shot, aer_shot in rows:
            print(_row(name, qb, sv_run, comp_run, raw_run, aer_run, sv_shot, comp_shot, raw_shot, aer_shot))
        print(_SEP)
        print(
            "  sv run(ms)   : dqsim StatevectorSimulator.simulate() — one statevector evolution\n"
            "  comp run(ms) : CompositeSimulator.simulate() on lowered=True distributed circuit\n"
            "  raw run(ms)  : CompositeSimulator.simulate() on lowered=False distributed circuit\n"
            "  aer run(ms)  : AerSimulator(statevector).run(shots=1) — one statevector evolution\n"
            "  *  shot(µs)  : total shot-batch time / SHOTS  (dqsim: simulate+counts; aer: run(shots=SHOTS))\n"
            "  *  speedup   : aer_run / dqsim_run  (>1 = dqsim faster)\n"
        )
