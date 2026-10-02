use alloc::vec::Vec;

use crate::packed::PackedField;
use crate::types::Field;

/// Precomputations of the evaluation of `Z_H(X) = X^n - 1` on a coset `gK` with `H <= K`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize_rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serialize_speedy", derive(speedy::Readable, speedy::Writable))]
#[repr(C)]
pub struct ZeroPolyOnCoset<F: Field> {
    /// `n = |H|`.
    n: F,
    /// `rate = |K|/|H|`.
    rate: usize,
    /// Holds `g^n * (w^n)^i - 1 = g^n * v^i - 1` for `i in 0..rate`, with `w` a generator of `K` and `v` a
    /// `rate`-primitive root of unity.
    evals: Vec<F>,
    /// Holds the multiplicative inverses of `evals`.
    inverses: Vec<F>,
}

impl<F: Field> ZeroPolyOnCoset<F> {
    pub fn new(n_log: usize, rate_bits: usize) -> Self {
        let g_pow_n = F::coset_shift().exp_power_of_2(n_log);
        let evals = F::two_adic_subgroup(rate_bits)
            .into_iter()
            .map(|x| g_pow_n * x - F::ONE)
            .collect::<Vec<_>>();
        let inverses = F::batch_multiplicative_inverse(&evals);
        Self {
            n: F::from_canonical_usize(1 << n_log),
            rate: 1 << rate_bits,
            evals,
            inverses,
        }
    }

    /// Returns `Z_H(g * w^i)`.
    pub fn eval(&self, i: usize) -> F {
        self.evals[i % self.rate]
    }

    /// Returns `1 / Z_H(g * w^i)`.
    pub fn eval_inverse(&self, i: usize) -> F {
        self.inverses[i % self.rate]
    }

    /// Like `eval_inverse`, but for a range of indices starting with `i_start`.
    pub fn eval_inverse_packed<P: PackedField<Scalar = F>>(&self, i_start: usize) -> P {
        let mut packed = P::ZEROS;
        packed
            .as_slice_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(j, packed_j)| *packed_j = self.eval_inverse(i_start + j));
        packed
    }

    /// Returns `L_0(x) = Z_H(x)/(n * (x - 1))` with `x = w^i`.
    pub fn eval_l_0(&self, i: usize, x: F) -> F {
        // Could also precompute the inverses using Montgomery.
        self.eval(i) * (self.n * (x - F::ONE)).inverse()
    }

    /// `eval_l_0(indices[k], xs[k])` for every `k`, with one batch inversion. Panics if some
    /// `xs[k]` is one, like `eval_l_0`.
    pub fn eval_l_0_batch(&self, indices: &[usize], xs: &[F]) -> Vec<F> {
        assert_eq!(indices.len(), xs.len());
        let denominators: Vec<F> = xs.iter().map(|&x| self.n * (x - F::ONE)).collect();
        assert!(
            denominators.iter().all(|d| d.is_nonzero()),
            "L_0 is undefined at x = 1"
        );
        F::batch_multiplicative_inverse(&denominators)
            .into_iter()
            .zip(indices)
            .map(|(inv, &i)| self.eval(i) * inv)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::ZeroPolyOnCoset;
    use crate::goldilocks_field::GoldilocksField as F;
    use crate::types::Field;

    #[test]
    fn eval_l_0_batch_matches_eval_l_0() {
        let (n_log, rate_bits) = (5, 3);
        let z_h = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);
        let points = F::two_adic_subgroup(n_log + rate_bits);
        let indices: Vec<usize> = (3..3 + 40).collect();
        let xs: Vec<F> = indices.iter().map(|&i| F::coset_shift() * points[i]).collect();
        let batch = z_h.eval_l_0_batch(&indices, &xs);
        for (k, (&i, &x)) in indices.iter().zip(&xs).enumerate() {
            assert_eq!(batch[k], z_h.eval_l_0(i, x));
        }
    }
}
