//! Poseidon over Goldilocks without 128-bit integers.
//!
//! wasm32 has no 64x64->128 multiply: every `u128` product is a call to `__multi3`, which costs
//! the CPU its parallelism across independent multiplications. Here each product is four
//! 32x32->64 multiplications and all sums are carried in `u64` words. The permutation is the
//! same function as `Poseidon::poseidon`; state words are any `u64` (not canonical).

use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon::{Poseidon, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS};
use crate::hash::poseidon_goldilocks::poseidon12_mds::mds_multiply_freq;

type F = GoldilocksField;
const EPSILON: u64 = 0xffff_ffff;

/// The 128-bit product as `(high, low)`.
#[inline(always)]
const fn wide_mul(a: u64, b: u64) -> (u64, u64) {
    let (a0, a1) = (a & EPSILON, a >> 32);
    let (b0, b1) = (b & EPSILON, b >> 32);
    let p00 = a0 * b0;
    let mid = a0 * b1 + (p00 >> 32);
    let mid2 = a1 * b0 + (mid & EPSILON);
    (a1 * b1 + (mid >> 32) + (mid2 >> 32), (mid2 << 32) | (p00 & EPSILON))
}

/// `high * 2^64 + low` reduced to one word (2^64 = 2^32 - 1 and 2^96 = -1 in the field).
#[inline(always)]
const fn reduce128(high: u64, low: u64) -> u64 {
    let (mut t0, borrow) = low.overflowing_sub(high >> 32);
    if borrow {
        t0 = t0.wrapping_sub(EPSILON);
    }
    let (sum, carry) = t0.overflowing_add((high & EPSILON) * EPSILON);
    sum.wrapping_add(EPSILON * carry as u64)
}

#[inline(always)]
const fn mul(a: u64, b: u64) -> u64 {
    let (high, low) = wide_mul(a, b);
    reduce128(high, low)
}

/// `a + b` for any two words.
#[inline(always)]
const fn add(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    let (sum, carry) = sum.overflowing_add(EPSILON * carry as u64);
    sum.wrapping_add(EPSILON * carry as u64)
}

/// `s + a * b`.
#[inline(always)]
const fn mul_add(s: u64, a: u64, b: u64) -> u64 {
    let (high, low) = wide_mul(a, b);
    let (low, carry) = low.overflowing_add(s);
    reduce128(high + carry as u64, low)
}

#[inline(always)]
const fn sbox(x: u64) -> u64 {
    let x2 = mul(x, x);
    let x4 = mul(x2, x2);
    let x3 = mul(x, x2);
    mul(x3, x4)
}

/// A sum of up to 2^32 products, kept in three words and reduced once.
struct ProductSum {
    low: u64,
    high: u64,
    top: u64,
}

impl ProductSum {
    #[inline(always)]
    const fn new() -> Self {
        Self { low: 0, high: 0, top: 0 }
    }

    #[inline(always)]
    fn add_product(&mut self, a: u64, b: u64) {
        let (high, low) = wide_mul(a, b);
        let (low, carry) = self.low.overflowing_add(low);
        let (high, carry_a) = self.high.overflowing_add(high);
        let (high, carry_b) = high.overflowing_add(carry as u64);
        self.low = low;
        self.high = high;
        self.top += carry_a as u64 + carry_b as u64;
    }

    /// 2^128 = -2^32 in the field.
    #[inline(always)]
    fn reduce(&self) -> u64 {
        let (mut t, borrow) = reduce128(self.high, self.low).overflowing_sub(self.top << 32);
        if borrow {
            t = t.wrapping_sub(EPSILON);
        }
        t
    }
}

#[inline(always)]
fn full_round(state: &mut [u64; 12], round: usize) {
    for i in 0..12 {
        // Round constants are canonical, so one carry fix is enough after the first add.
        state[i] = sbox(add(state[i], ALL_ROUND_CONSTANTS[i + 12 * round]));
    }
    // The MDS matrix has small entries: multiply the low and high halves of every word
    // separately (no overflow) and recombine, as `Poseidon::mds_layer` does for Goldilocks.
    let mut low = [0u64; 12];
    let mut high = [0u64; 12];
    for i in 0..12 {
        low[i] = state[i] & EPSILON;
        high[i] = state[i] >> 32;
    }
    let s0 = state[0];
    let low = mds_multiply_freq(low);
    let high = mds_multiply_freq(high);
    for i in 0..12 {
        let (sum, carry) = low[i].overflowing_add(high[i] << 32);
        state[i] = reduce128((high[i] >> 32) + carry as u64, sum);
    }
    state[0] = mul_add(state[0], s0, <F as Poseidon>::MDS_MATRIX_DIAG[0]);
}

#[inline(always)]
fn partial_rounds(state: &mut [u64; 12]) {
    for i in 0..12 {
        state[i] = add(state[i], <F as Poseidon>::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
    }
    let mut next = [0u64; 12];
    next[0] = state[0];
    for c in 1..12 {
        let mut sum = ProductSum::new();
        for r in 1..12 {
            sum.add_product(state[r], <F as Poseidon>::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1]);
        }
        next[c] = sum.reduce();
    }
    *state = next;

    let mds_0_0 = <F as Poseidon>::MDS_MATRIX_CIRC[0] + <F as Poseidon>::MDS_MATRIX_DIAG[0];
    for round in 0..N_PARTIAL_ROUNDS {
        let s0 = add(sbox(state[0]), <F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[round]);
        let w_hats = &<F as Poseidon>::FAST_PARTIAL_ROUND_W_HATS[round];
        let vs = &<F as Poseidon>::FAST_PARTIAL_ROUND_VS[round];
        let mut d = ProductSum::new();
        d.add_product(s0, mds_0_0);
        for i in 1..12 {
            d.add_product(state[i], w_hats[i - 1]);
        }
        for i in 1..12 {
            state[i] = mul_add(state[i], s0, vs[i - 1]);
        }
        state[0] = d.reduce();
    }
}

/// `Poseidon::mds_partial_layer_fast` for Goldilocks (used by the Poseidon gate's evaluation).
#[inline]
pub fn mds_partial_layer_fast(state: &[F; 12], round: usize) -> [F; 12] {
    let mds_0_0 = <F as Poseidon>::MDS_MATRIX_CIRC[0] + <F as Poseidon>::MDS_MATRIX_DIAG[0];
    let w_hats = &<F as Poseidon>::FAST_PARTIAL_ROUND_W_HATS[round];
    let vs = &<F as Poseidon>::FAST_PARTIAL_ROUND_VS[round];
    let s0 = state[0].0;
    let mut d = ProductSum::new();
    d.add_product(s0, mds_0_0);
    for i in 1..12 {
        d.add_product(state[i].0, w_hats[i - 1]);
    }
    let mut result = [GoldilocksField(0); 12];
    result[0] = GoldilocksField(d.reduce());
    for i in 1..12 {
        result[i] = GoldilocksField(mul_add(state[i].0, s0, vs[i - 1]));
    }
    result
}

/// `Poseidon::poseidon` for Goldilocks.
#[inline]
pub fn poseidon(input: [F; 12]) -> [F; 12] {
    let mut state = [0u64; 12];
    for i in 0..12 {
        state[i] = input[i].0;
    }
    for round in 0..HALF_N_FULL_ROUNDS {
        full_round(&mut state, round);
    }
    partial_rounds(&mut state);
    for round in HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS..2 * HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS {
        full_round(&mut state, round);
    }
    let mut output = [GoldilocksField(0); 12];
    for i in 0..12 {
        output[i] = GoldilocksField(state[i]);
    }
    output
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::field::types::{Field, PrimeField64};

    const ORDER: u64 = 0xffff_ffff_0000_0001;

    #[test]
    fn matches_the_reference_permutation() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x706f7365);
        let edge = [0u64, 1, EPSILON, ORDER - 1, ORDER, ORDER + 1, u64::MAX, u64::MAX - EPSILON, 1 << 32, 1 << 63];
        for case in 0..20_000 {
            let input: [F; 12] = core::array::from_fn(|i| {
                // Non-canonical words and boundary values as well as uniform ones.
                if case < 200 { GoldilocksField(edge[(case + i * 3) % edge.len()]) } else { GoldilocksField(rng.gen::<u64>()) }
            });
            let expected = <F as Poseidon>::poseidon_naive(input);
            let actual = poseidon(input);
            for i in 0..12 {
                assert_eq!(actual[i].to_canonical_u64(), expected[i].to_canonical_u64(), "case {case} word {i}");
            }
        }
    }

    #[test]
    fn partial_layer_matches_the_reference() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        for case in 0..20_000 {
            let state: [F; 12] = core::array::from_fn(|_| GoldilocksField(rng.gen::<u64>()));
            let round = case % N_PARTIAL_ROUNDS;
            let expected = <F as Poseidon>::mds_partial_layer_fast_field::<F, 1>(&state, round);
            let actual = mds_partial_layer_fast(&state, round);
            for i in 0..12 {
                assert_eq!(actual[i].to_canonical_u64(), expected[i].to_canonical_u64(), "case {case} word {i}");
            }
        }
    }

    #[test]
    fn mds_layer_field_matches_the_row_formula() {
        use crate::field::extension::quadratic::QuadraticExtension;
        use crate::field::types::Sample;
        type FE = QuadraticExtension<F>;
        for _ in 0..2_000 {
            let state: [FE; 12] = core::array::from_fn(|_| FE::rand());
            let actual = <F as Poseidon>::mds_layer_field::<FE, 2>(&state);
            for r in 0..12 {
                assert_eq!(actual[r], <F as Poseidon>::mds_row_shf_field::<FE, 2>(r, &state), "row {r}");
            }
        }
    }

    #[test]
    fn arithmetic_matches_the_field() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        for _ in 0..100_000 {
            let (a, b, s): (u64, u64, u64) = (rng.gen(), rng.gen(), rng.gen());
            assert_eq!(GoldilocksField(mul(a, b)).to_canonical_u64(), (GoldilocksField(a) * GoldilocksField(b)).to_canonical_u64());
            assert_eq!(GoldilocksField(add(a, b)).to_canonical_u64(), (F::from_noncanonical_u64(a) + F::from_noncanonical_u64(b)).to_canonical_u64());
            assert_eq!(GoldilocksField(mul_add(s, a, b)).to_canonical_u64(), (F::from_noncanonical_u64(s) + GoldilocksField(a) * GoldilocksField(b)).to_canonical_u64());
        }
    }
}
