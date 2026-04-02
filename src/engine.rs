use num_complex::Complex64;
use rand::Rng;

type C = Complex64;

// ---------------------------------------------------------------------------
// apply_one_qubit
// Apply a 2x2 unitary to `target` qubit in-place.
// `controls` is a slice of (wire, must_be_one) pairs.
// ---------------------------------------------------------------------------

pub fn apply_one_qubit(
    state: &mut Vec<C>,
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
) {
    let inclusion_mask: usize = controls.iter().fold(0, |m, &(w, _)| m | (1 << w));
    let desired_mask:   usize = controls.iter().fold(0, |m, &(w, flag)| if flag { m | (1 << w) } else { m });
    let half = 1 << target;
    let dim  = 1 << n;

    let mut i = 0;
    while i < dim {
        if (i & half) == 0 {
            if inclusion_mask == 0 || (i & inclusion_mask) == desired_mask {
                let j = i | half;
                let a = state[i];
                let b = state[j];
                state[i] = u[0][0] * a + u[0][1] * b;
                state[j] = u[1][0] * a + u[1][1] * b;
            }
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// apply_n_qubit
// Apply a 2^k × 2^k unitary to k arbitrary qubits in-place.
// qubits[0] = MSB of the gate's index space, qubits[k-1] = LSB.
// ---------------------------------------------------------------------------

pub fn apply_n_qubit(state: &mut Vec<C>, u: &[Vec<C>], qubits: &[usize], n: usize) {
    let k   = qubits.len();
    let dim = 1 << k;
    let n_states = 1 << n;

    // Mask of all target qubit positions
    let mask: usize = qubits.iter().fold(0, |m, &q| m | (1 << q));

    // Scratch buffers — allocated once, reused every iteration
    let mut v   = vec![C::new(0.0, 0.0); dim];
    let mut w   = vec![C::new(0.0, 0.0); dim];
    let mut idx = vec![0usize; dim];

    // Iterate over base indices (all target qubits = 0)
    for base in (0..n_states).filter(|&i| (i & mask) == 0) {
        // Build index table for this base
        for j in 0..dim {
            let mut offset = 0usize;
            for (bit, &q) in qubits.iter().enumerate() {
                if (j >> (k - 1 - bit)) & 1 == 1 {
                    offset |= 1 << q;
                }
            }
            idx[j] = base + offset;
        }

        // Gather
        for j in 0..dim {
            v[j] = state[idx[j]];
        }

        // Apply gate matrix
        for row in 0..dim {
            let mut acc = C::new(0.0, 0.0);
            for col in 0..dim {
                acc += u[row][col] * v[col];
            }
            w[row] = acc;
        }

        // Scatter
        for j in 0..dim {
            state[idx[j]] = w[j];
        }
    }
}

// ---------------------------------------------------------------------------
// measure_qubit
// Collapse the statevector on `qubit`, renormalise, return outcome (0 or 1).
// ---------------------------------------------------------------------------

pub fn measure_qubit<R: Rng>(state: &mut Vec<C>, qubit: usize, n: usize, rng: &mut R) -> u8 {
    let n_states = 1 << n;
    let bit = 1 << qubit;

    // P(outcome=1)
    let p1: f64 = (0..n_states)
        .filter(|&i| (i & bit) != 0)
        .map(|i| state[i].norm_sqr())
        .sum();

    let outcome = if rng.gen::<f64>() < p1 { 1u8 } else { 0u8 };

    if outcome == 1 {
        let norm = p1.sqrt().max(1e-15);
        for i in (0..n_states).filter(|&i| (i & bit) == 0) {
            state[i] = C::new(0.0, 0.0);
        }
        for i in (0..n_states).filter(|&i| (i & bit) != 0) {
            state[i] /= norm;
        }
    } else {
        let norm = (1.0 - p1).max(0.0).sqrt().max(1e-15);
        for i in (0..n_states).filter(|&i| (i & bit) != 0) {
            state[i] = C::new(0.0, 0.0);
        }
        for i in (0..n_states).filter(|&i| (i & bit) == 0) {
            state[i] /= norm;
        }
    }

    outcome
}

// ---------------------------------------------------------------------------
// marginal_probs
// Returns a Vec of length 2^k giving probabilities for each basis state of
// the specified qubits. qubits[0] = MSB of output index.
// ---------------------------------------------------------------------------

pub fn marginal_probs(state: &[C], n: usize, qubits: &[usize]) -> Vec<f64> {
    let k   = qubits.len();
    let dim = 1 << k;
    let n_states = 1 << n;

    let mask: usize = qubits.iter().fold(0, |m, &q| m | (1 << q));
    let mut probs = vec![0.0f64; dim];

    for j in 0..dim {
        // Which full-state bit pattern corresponds to marginal index j?
        let mut desired = 0usize;
        for (bit, &q) in qubits.iter().enumerate() {
            if (j >> (k - 1 - bit)) & 1 == 1 {
                desired |= 1 << q;
            }
        }
        probs[j] = (0..n_states)
            .filter(|&i| (i & mask) == desired)
            .map(|i| state[i].norm_sqr())
            .sum();
    }

    probs
}

// ---------------------------------------------------------------------------
// sample_counts
// Sample `shots` outcomes from the state distribution.
// Returns a Vec of (bitstring, count) pairs.
// ---------------------------------------------------------------------------

pub fn sample_counts<R: Rng>(
    state: &[C],
    n: usize,
    shots: usize,
    rng: &mut R,
    qubits: Option<&[usize]>,
) -> std::collections::HashMap<String, usize> {
    let all_qubits: Vec<usize>;
    let q = match qubits {
        Some(qs) => qs,
        None => {
            all_qubits = (0..n).rev().collect();
            &all_qubits
        }
    };

    let k     = q.len();
    let probs = marginal_probs(state, n, q);

    // Build CDF
    let mut cdf = vec![0.0f64; probs.len()];
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        cdf[i] = acc;
    }
    *cdf.last_mut().unwrap() = 1.0;

    let mut counts = std::collections::HashMap::new();
    for _ in 0..shots {
        let r: f64 = rng.gen();
        let idx = cdf.partition_point(|&c| c < r).min((1 << k) - 1);
        let bits = format!("{:0>width$b}", idx, width = k);
        *counts.entry(bits).or_insert(0) += 1;
    }
    counts
}
