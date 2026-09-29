use alloc::vec::Vec;
use core::cmp::{max, min};

use plonky2_util::{log2_strict, reverse_index_bits_in_place};
use unroll::unroll_for_loops;

use crate::packable::Packable;
use crate::packed::PackedField;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::Field;

pub type FftRootTable<F> = Vec<Vec<F>>;

pub fn fft_root_table<F: Field>(n: usize) -> FftRootTable<F> {
    let lg_n = log2_strict(n);
    // bases[i] = g^2^i, for i = 0, ..., lg_n - 1
    let mut bases = Vec::with_capacity(lg_n);
    let mut base = F::primitive_root_of_unity(lg_n);
    bases.push(base);
    for _ in 1..lg_n {
        base = base.square(); // base = g^2^_
        bases.push(base);
    }

    let mut root_table = Vec::with_capacity(lg_n);
    for lg_m in 1..=lg_n {
        let half_m = 1 << (lg_m - 1);
        let base = bases[lg_n - lg_m];
        let root_row = base.powers().take(half_m.max(2)).collect();
        root_table.push(root_row);
    }
    root_table
}

#[inline]
fn fft_dispatch<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    let computed_root_table = root_table.is_none().then(|| fft_root_table(input.len()));
    let used_root_table = root_table.or(computed_root_table.as_ref()).unwrap();

    fft_classic(input, zero_factor.unwrap_or(0), used_root_table);
}

#[inline]
pub fn fft<F: Field>(poly: PolynomialCoeffs<F>) -> PolynomialValues<F> {
    fft_with_options(poly, None, None)
}

#[inline]
pub fn fft_with_options<F: Field>(
    poly: PolynomialCoeffs<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    let PolynomialCoeffs { coeffs: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);
    PolynomialValues::new(buffer)
}

#[inline]
pub fn ifft<F: Field>(poly: PolynomialValues<F>) -> PolynomialCoeffs<F> {
    ifft_with_options(poly, None, None)
}

pub fn ifft_with_options<F: Field>(
    poly: PolynomialValues<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialCoeffs<F> {
    let n = poly.len();
    let lg_n = log2_strict(n);
    let n_inv = F::inverse_2exp(lg_n);

    let PolynomialValues { values: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);

    // We reverse all values except the first, and divide each by n.
    buffer[0] *= n_inv;
    buffer[n / 2] *= n_inv;
    for i in 1..(n / 2) {
        let j = n - i;
        let coeffs_i = buffer[j] * n_inv;
        let coeffs_j = buffer[i] * n_inv;
        buffer[i] = coeffs_i;
        buffer[j] = coeffs_j;
    }
    PolynomialCoeffs { coeffs: buffer }
}

// ---------------------------------------------------------------------------------------------
// Cache-blocked kernel.
//
// After the bit reversal, butterfly layer `l` (half-size `2^l`) pairs index `x` with `x + 2^l`
// for every `x` whose bit `l` is clear, using twiddle `root_table[l][x mod 2^l]`. The previous
// kernel swept the whole array once per layer, i.e. `lg_n` read+write passes over data far
// larger than L1/L2. This kernel performs exactly the same butterflies (same operands, same
// packed operations, same operand order), only in a different order, so the output is
// bit-identical in its raw `u64` representation:
//
// * Fused group (`fused_group`): consecutive layers `p0 .. p0 + g` (g <= MAX_FUSED_LAYERS)
//   only touch index bits `p0 .. p0 + g`, so for a fixed low offset `j < 2^p0` the `2^g`
//   elements `base + a * 2^p0 + j` form an independent radix-`2^g` sub-FFT. Offsets are
//   processed in chunks of `chunk` consecutive vectors, and all `g` layers are applied to the
//   `2^g x chunk` tile while it is resident in L1: one pass per group instead of `g`.
// * Phase 1 walks the array in L2-sized blocks, touching main memory once. Inside each L2
//   block, every L1-sized block gets the zero-factor fill, the sub-width layers and all packed
//   layers local to it (1a); then the layers local to the L2 block run as fused groups (1b).
// * Phase 2 runs the remaining large-stride layers as fused groups over the whole array.
//
// For lg_n = 20 with AVX2 packing that is 1 + 2 passes over memory instead of 1 (fill) + 20,
// plus the unchanged bit-reversal permutation in `fft_classic`.
// ---------------------------------------------------------------------------------------------

/// Bytes per L1 block (phase 1a). 16 KiB of data plus up to 16 KiB of twiddles fits the 32 KiB
/// L1D of Zen 3 (48 KiB on Zen 5).
const LOG_L1_BLOCK_BYTES: usize = 14;

/// Bytes per L2 block (phase 1). 128 KiB leaves room for twiddles and the SMT sibling in a
/// 512 KiB (Zen 3) or 1 MiB (Zen 5) L2.
const LOG_L2_BLOCK_BYTES: usize = 17;

/// Maximum number of consecutive large-stride layers fused into one pass (radix `2^g`).
const MAX_FUSED_LAYERS: usize = 3;

/// Bytes per row of a phase-2 tile. A tile is `2^g` rows of this size.
const LOG_CHUNK_BYTES: usize = 10;

#[inline]
fn floor_log2(x: usize) -> usize {
    (usize::BITS - 1 - x.leading_zeros()) as usize
}

/// `(xs[i], ys[i]) := (xs[i] + ws[i] * ys[i], xs[i] - ws[i] * ys[i])`, with the exact operation
/// sequence of the original layer loop.
#[inline(always)]
fn butterfly_rows<P: PackedField>(xs: &mut [P], ys: &mut [P], ws: &[P]) {
    let len = xs.len();
    let (ys, ws) = (&mut ys[..len], &ws[..len]);
    for j in 0..len {
        let omega = ws[j];
        let t = omega * ys[j];
        let u = xs[j];
        xs[j] = u + t;
        ys[j] = u - t;
    }
}

/// The layers with half-size below the packing width, applied to one block. Identical to the
/// first loop of the original kernel restricted to `block`.
#[unroll_for_loops]
#[inline]
fn sub_width_layers<P: PackedField>(
    block: &mut [P],
    r: usize,
    lg_n: usize,
    lg_packed_width: usize,
    omegas: &[P; 4],
) {
    for lg_half_m in 0..4 {
        if (r..min(lg_n, lg_packed_width)).contains(&lg_half_m) {
            let half_m = 1 << lg_half_m;
            let omega = omegas[lg_half_m];
            for k in (0..block.len()).step_by(2) {
                let (u, v) = block[k].interleave(block[k + 1], half_m);
                let t = omega * v;
                (block[k], block[k + 1]) = (u + t).interleave(u - t, half_m);
            }
        }
    }
}

/// Generic FFT implementation that works with both scalar and packed inputs.
///
/// `values` must already be in bit-reversed order. Unlike the original kernel, this one also
/// performs the zero-factor fill (`values[i] = values[i & !(2^r - 1)]`) itself, so that it can be
/// done block by block instead of as a separate pass. The fill is idempotent, so callers that
/// already did it get the same result.
pub(crate) fn fft_classic_simd<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    lg_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let lg_packed_width = log2_strict(P::WIDTH); // 0 when P is a scalar.
    assert!(lg_packed_width <= 4);
    debug_assert!(values.len() == 1 << lg_n && lg_n >= lg_packed_width);
    let lg_packed_n = lg_n - lg_packed_width;

    // First packed layer to perform. Packed layer `p` is field layer `p + lg_packed_width`.
    let s = max(r, lg_packed_width);
    let p_start = s - lg_packed_width;

    let lg_vec_bytes = floor_log2(core::mem::size_of::<P>().max(1));
    let bits_for = |lg_bytes: usize| lg_bytes.saturating_sub(lg_vec_bytes);
    // Blocks hold at least two vectors (sub-width layers pair adjacent vectors).
    let l2_bits = bits_for(LOG_L2_BLOCK_BYTES).max(1).min(lg_packed_n);
    let l1_bits = bits_for(LOG_L1_BLOCK_BYTES).max(1).min(l2_bits);
    let chunk = 1usize << bits_for(LOG_CHUNK_BYTES).min(l1_bits);

    // The fill only copies inside aligned groups of 2^r elements, so it can be done per L1
    // block when such a group fits in one.
    let fill_per_block = r > 0 && r <= l1_bits + lg_packed_width;
    if r > 0 && !fill_per_block {
        fill_zero_factor(values, r);
    }

    // ---- Phase 1: every layer local to an L2 block, one L2 block at a time. ----
    let has_sub_width = r < min(lg_n, lg_packed_width);
    if fill_per_block || has_sub_width || p_start < l2_bits {
        // Set omegas[l] to root_table[l][0..2^l] but repeated (only used for l < lg_packed_width).
        let mut omegas = [P::default(); 4];
        for (lg_half_m, omega) in omegas.iter_mut().enumerate().take(lg_packed_width) {
            let half_m = 1 << lg_half_m;
            for (j, omega_j) in omega.as_slice_mut().iter_mut().enumerate() {
                *omega_j = root_table[lg_half_m][j % half_m];
            }
        }
        let l1_len = 1usize << (l1_bits + lg_packed_width); // in field elements
        let l2_len = 1usize << (l2_bits + lg_packed_width);
        for l2_block in values.chunks_exact_mut(l2_len) {
            // Phase 1a: the fill, the sub-width layers and packed layers below l1_bits, per
            // L1-sized block.
            for l1_block in l2_block.chunks_exact_mut(l1_len) {
                if fill_per_block {
                    fill_zero_factor(l1_block, r);
                }
                let block = P::pack_slice_mut(l1_block);
                if has_sub_width {
                    sub_width_layers(block, r, lg_n, lg_packed_width, &omegas);
                }
                for p in p_start..l1_bits {
                    let half = 1usize << p;
                    let omega_table = &P::pack_slice(&root_table[p + lg_packed_width][..])[..half];
                    for sub in block.chunks_exact_mut(2 * half) {
                        let (lo, hi) = sub.split_at_mut(half);
                        butterfly_rows(lo, hi, omega_table);
                    }
                }
            }
            // Phase 1b: packed layers l1_bits..l2_bits while the L2 block is still cached.
            fused_layers(
                P::pack_slice_mut(l2_block),
                max(p_start, l1_bits),
                l2_bits,
                chunk,
                lg_packed_width,
                root_table,
            );
        }
    }

    // ---- Phase 2: the remaining large-stride layers over the whole array. ----
    fused_layers(
        P::pack_slice_mut(values),
        max(p_start, l2_bits),
        lg_packed_n,
        chunk,
        lg_packed_width,
        root_table,
    );
}

/// `values[i] = values[i & !(2^r - 1)]`, with indices relative to `values` (which must start at
/// a multiple of `2^r`). `chunks_mut` also covers `2^r > values.len()` like the original loop.
#[inline]
fn fill_zero_factor<F: Copy>(values: &mut [F], r: usize) {
    for group in values.chunks_mut(1 << r) {
        let head = group[0];
        group.fill(head);
    }
}

/// Apply packed layers `from..to` to `packed_values`, in passes of up to `MAX_FUSED_LAYERS`
/// layers each.
#[inline]
fn fused_layers<P: PackedField>(
    packed_values: &mut [P],
    from: usize,
    to: usize,
    chunk: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let mut p0 = from;
    while p0 < to {
        let remaining = to - p0;
        let groups = remaining.div_ceil(MAX_FUSED_LAYERS);
        let g = remaining.div_ceil(groups);
        fused_group(packed_values, p0, g, chunk, lg_packed_width, root_table);
        p0 += g;
    }
}

/// Apply packed layers `p0 .. p0 + g` in one pass (see the module comment above).
#[inline]
fn fused_group<P: PackedField>(
    packed_values: &mut [P],
    p0: usize,
    g: usize,
    chunk: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let stride = 1usize << p0; // distance between the rows of a tile, in vectors
    let rows = 1usize << g;
    let super_len = stride << g;
    let chunk = min(chunk, stride);

    let mut omega_tables: [&[P]; MAX_FUSED_LAYERS] = [&[]; MAX_FUSED_LAYERS];
    for (t, table) in omega_tables.iter_mut().enumerate().take(g) {
        *table = &P::pack_slice(&root_table[p0 + t + lg_packed_width][..])[..stride << t];
    }

    for sb in packed_values.chunks_exact_mut(super_len) {
        for j0 in (0..stride).step_by(chunk) {
            for (t, omega_table) in omega_tables.iter().enumerate().take(g) {
                let half = stride << t; // butterfly half-size of layer p0 + t
                let low_mask = (1usize << t) - 1;
                for a in 0..rows {
                    if a & (1 << t) != 0 {
                        continue;
                    }
                    let x = a * stride + j0;
                    let tw = (a & low_mask) * stride + j0;
                    let (lo, hi) = sb.split_at_mut(x + half);
                    butterfly_rows(
                        &mut lo[x..x + chunk],
                        &mut hi[..chunk],
                        &omega_table[tw..tw + chunk],
                    );
                }
            }
        }
    }
}

/// FFT implementation based on Section 32.3 of "Introduction to
/// Algorithms" by Cormen et al.
///
/// The parameter r signifies that the first 1/2^r of the entries of
/// input may be non-zero, but the last 1 - 1/2^r entries are
/// definitely zero.
pub(crate) fn fft_classic<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
    reverse_index_bits_in_place(values);

    let n = values.len();
    let lg_n = log2_strict(n);

    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    // After reverse_index_bits, the only non-zero elements of values
    // are at indices i*2^r for i = 0..n/2^r.  The loop below copies
    // the value at i*2^r to the positions [i*2^r + 1, i*2^r + 2, ...,
    // (i+1)*2^r - 1]; i.e. it replaces the 2^r - 1 zeros following
    // element i*2^r with the value at i*2^r.  This corresponds to the
    // first r rounds of the FFT when there are 2^r zeros at the end
    // of the original input.
    //
    // The fill is performed by `fft_classic_simd`, fused into its first pass.

    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    if lg_n <= lg_packed_width {
        // Need the slice to be at least the width of two packed vectors for the vectorized version
        // to work. Do this tiny problem in scalar.
        fft_classic_simd::<F>(values, r, lg_n, root_table);
    } else {
        fft_classic_simd::<<F as Packable>::Packing>(values, r, lg_n, root_table);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use plonky2_util::{log2_ceil, log2_strict};

    use crate::fft::{fft, fft_with_options, ifft};
    use crate::goldilocks_field::GoldilocksField;
    use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
    use crate::types::Field;

    #[test]
    fn fft_and_ifft() {
        type F = GoldilocksField;
        let degree = 200usize;
        let degree_padded = degree.next_power_of_two();

        // Create a vector of coeffs; the first degree of them are
        // "random", the last degree_padded-degree of them are zero.
        let coeffs = (0..degree)
            .map(|i| F::from_canonical_usize(i * 1337 % 100))
            .chain(core::iter::repeat_n(F::ZERO, degree_padded - degree))
            .collect::<Vec<_>>();
        assert_eq!(coeffs.len(), degree_padded);
        let coefficients = PolynomialCoeffs { coeffs };

        let points = fft(coefficients.clone());
        assert_eq!(points, evaluate_naive(&coefficients));

        let interpolated_coefficients = ifft(points);
        for i in 0..degree {
            assert_eq!(interpolated_coefficients.coeffs[i], coefficients.coeffs[i]);
        }
        for i in degree..degree_padded {
            assert_eq!(interpolated_coefficients.coeffs[i], F::ZERO);
        }

        for r in 0..4 {
            // expand coefficients by factor 2^r by filling with zeros
            let zero_tail = coefficients.lde(r);
            assert_eq!(
                fft(zero_tail.clone()),
                fft_with_options(zero_tail, Some(r), None)
            );
        }
    }

    fn evaluate_naive<F: Field>(coefficients: &PolynomialCoeffs<F>) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_padded = 1 << log2_ceil(degree);

        let coefficients_padded = coefficients.padded(degree_padded);
        evaluate_naive_power_of_2(&coefficients_padded)
    }

    fn evaluate_naive_power_of_2<F: Field>(
        coefficients: &PolynomialCoeffs<F>,
    ) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_log = log2_strict(degree);

        let subgroup = F::two_adic_subgroup(degree_log);

        let values = subgroup
            .into_iter()
            .map(|x| evaluate_at_naive(coefficients, x))
            .collect();
        PolynomialValues::new(values)
    }

    fn evaluate_at_naive<F: Field>(coefficients: &PolynomialCoeffs<F>, point: F) -> F {
        let mut sum = F::ZERO;
        let mut point_power = F::ONE;
        for &c in &coefficients.coeffs {
            sum += c * point_power;
            point_power *= point;
        }
        sum
    }
}
