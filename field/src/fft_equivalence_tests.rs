//! Bit-exactness tests: the fused FFT kernel in `crate::fft` against the verbatim copy of the
//! original layer-by-layer kernel in `crate::fft_reference`.
//!
//! Every comparison is on the raw `u64` representation (`GoldilocksField.0`), which is stronger
//! than comparing canonical values.

use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use plonky2_util::{log2_strict, reverse_index_bits_in_place};

use crate::extension::quadratic::QuadraticExtension;
use crate::extension::quintic::QuinticExtension;
use crate::extension::FieldExtension;
use crate::fft::{self, fft_root_table, FftRootTable};
use crate::fft_reference;
use crate::goldilocks_field::GoldilocksField;
use crate::ops::Square;
use crate::packed::PackedField;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::Field;

type F = GoldilocksField;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Largest size exercised. The full range is only affordable in optimised builds.
const MAX_LG_N: usize = if cfg!(debug_assertions) { 16 } else { 22 };

/// SplitMix64: deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    /// Uniform canonical values.
    Canonical,
    /// Uniform raw u64, so about 2^-32 of them non-canonical.
    RawAny,
    /// Only non-canonical representations (values in [p, 2^64)).
    NonCanonical,
    /// Each element drawn from 0, 1, p-1, p, p+1, 2^64-1, 2^32, 2^32-1 or a random value >= p.
    EdgeMix,
    AllZero,
    AllOne,
    AllPMinusOne,
}

const PATTERNS: [Pattern; 7] = [
    Pattern::Canonical,
    Pattern::RawAny,
    Pattern::NonCanonical,
    Pattern::EdgeMix,
    Pattern::AllZero,
    Pattern::AllOne,
    Pattern::AllPMinusOne,
];

fn gen_value(rng: &mut Rng, pattern: Pattern) -> u64 {
    match pattern {
        Pattern::Canonical => rng.next() % P,
        Pattern::RawAny => rng.next(),
        Pattern::NonCanonical => P + rng.next() % (u64::MAX - P + 1),
        Pattern::EdgeMix => match rng.next() % 9 {
            0 => 0,
            1 => 1,
            2 => P - 1,
            3 => P,
            4 => P + 1,
            5 => u64::MAX,
            6 => 1 << 32,
            7 => (1 << 32) - 1,
            _ => P + rng.next() % (u64::MAX - P + 1),
        },
        Pattern::AllZero => 0,
        Pattern::AllOne => 1,
        Pattern::AllPMinusOne => P - 1,
    }
}

/// `n` values; when `zero_tail_bits > 0` only the first `n >> zero_tail_bits` are non-zero, as
/// `fft_with_options(.., Some(r), ..)` callers guarantee.
fn gen_input(rng: &mut Rng, pattern: Pattern, lg_n: usize, zero_tail_bits: usize) -> Vec<F> {
    let n = 1usize << lg_n;
    let nonzero = (n >> zero_tail_bits).max(1);
    (0..n)
        .map(|i| {
            if i < nonzero {
                GoldilocksField(gen_value(rng, pattern))
            } else {
                F::ZERO
            }
        })
        .collect()
}

fn raw(values: &[F]) -> Vec<u64> {
    values.iter().map(|x| x.0).collect()
}

fn assert_raw_eq(new: &[F], old: &[F], ctx: &dyn Fn() -> alloc::string::String) {
    assert_eq!(new.len(), old.len());
    if let Some(i) = (0..new.len()).find(|&i| new[i].0 != old[i].0) {
        panic!(
            "raw mismatch at index {i}: new {:#x} old {:#x} ({})",
            new[i].0,
            old[i].0,
            ctx()
        );
    }
}

fn zero_factors(lg_n: usize) -> Vec<usize> {
    // r = lg_n + 1 is outside the documented contract but must still behave like the original.
    let mut rs: Vec<usize> = [
        0,
        1,
        2,
        3,
        4,
        lg_n / 2,
        lg_n.saturating_sub(1),
        lg_n,
        lg_n + 1,
    ]
    .into_iter()
    .filter(|&r| r <= lg_n + 1)
    .collect();
    rs.sort_unstable();
    rs.dedup();
    rs
}

fn random_root_table(rng: &mut Rng, lg_n: usize) -> FftRootTable<F> {
    (1..=lg_n)
        .map(|lg_m| {
            let len = (1usize << (lg_m - 1)).max(2);
            (0..len).map(|_| GoldilocksField(rng.next())).collect()
        })
        .collect()
}

#[test]
fn fft_and_ifft_bit_identical_to_reference() {
    let mut rng = Rng(0x5eed_f0f7);
    for lg_n in 0..=MAX_LG_N {
        let table = fft_root_table::<F>(1 << lg_n);
        for pattern in PATTERNS {
            // Large sizes: fewer patterns to keep the runtime reasonable.
            if lg_n >= 20 && matches!(pattern, Pattern::AllOne | Pattern::RawAny) {
                continue;
            }
            for r in zero_factors(lg_n) {
                let input = gen_input(&mut rng, pattern, lg_n, r);
                let ctx = || alloc::format!("lg_n={lg_n} pattern={pattern:?} r={r}");

                // fft / fft_with_options, with and without a caller-supplied root table.
                let new =
                    fft::fft_with_options(PolynomialCoeffs::new(input.clone()), Some(r), None);
                let old = fft_reference::fft_with_options(
                    PolynomialCoeffs::new(input.clone()),
                    Some(r),
                    None,
                );
                assert_raw_eq(&new.values, &old.values, &ctx);
                let new_t = fft::fft_with_options(
                    PolynomialCoeffs::new(input.clone()),
                    Some(r),
                    Some(&table),
                );
                assert_raw_eq(&new_t.values, &old.values, &ctx);
                if r == 0 {
                    let plain = fft::fft(PolynomialCoeffs::new(input.clone()));
                    assert_raw_eq(&plain.values, &old.values, &ctx);
                }

                // ifft / ifft_with_options.
                let new =
                    fft::ifft_with_options(PolynomialValues::new(input.clone()), Some(r), None);
                let old = fft_reference::ifft_with_options(
                    PolynomialValues::new(input.clone()),
                    Some(r),
                    None,
                );
                assert_raw_eq(&new.coeffs, &old.coeffs, &ctx);
                let new_t = fft::ifft_with_options(
                    PolynomialValues::new(input.clone()),
                    Some(r),
                    Some(&table),
                );
                assert_raw_eq(&new_t.coeffs, &old.coeffs, &ctx);
                if r == 0 {
                    let plain = fft::ifft(PolynomialValues::new(input));
                    assert_raw_eq(&plain.coeffs, &old.coeffs, &ctx);
                }
            }
        }
    }
}

/// Arbitrary (non-root-of-unity) twiddles: every twiddle must be consumed at exactly the same
/// butterfly as in the original kernel, or the outputs diverge.
#[test]
fn caller_supplied_random_root_table_bit_identical() {
    let mut rng = Rng(0x7ab1e);
    for lg_n in 0..=MAX_LG_N {
        let table = random_root_table(&mut rng, lg_n);
        for r in zero_factors(lg_n) {
            let input = gen_input(&mut rng, Pattern::EdgeMix, lg_n, 0);
            let ctx = || alloc::format!("lg_n={lg_n} r={r} random table");
            let new =
                fft::fft_with_options(PolynomialCoeffs::new(input.clone()), Some(r), Some(&table));
            let old = fft_reference::fft_with_options(
                PolynomialCoeffs::new(input),
                Some(r),
                Some(&table),
            );
            assert_raw_eq(&new.values, &old.values, &ctx);
        }
    }
}

#[test]
fn coset_fft_and_ifft_bit_identical() {
    let mut rng = Rng(0xc05e7);
    let shift = F::coset_shift();
    for lg_n in 0..=MAX_LG_N.min(20) {
        let table = fft_root_table::<F>(1 << lg_n);
        for r in [0, 1, 3].into_iter().filter(|&r| r <= lg_n) {
            let input = gen_input(&mut rng, Pattern::Canonical, lg_n, r);
            let ctx = || alloc::format!("coset lg_n={lg_n} r={r}");

            let new = PolynomialCoeffs::new(input.clone()).coset_fft_with_options(
                shift,
                Some(r),
                Some(&table),
            );
            // Same pre-scaling as PolynomialCoeffs::coset_fft_with_options, then the reference.
            let scaled: Vec<F> = shift.powers().zip(&input).map(|(s, &c)| s * c).collect();
            let old = fft_reference::fft_with_options(
                PolynomialCoeffs::new(scaled),
                Some(r),
                Some(&table),
            );
            assert_raw_eq(&new.values, &old.values, &ctx);

            let new = PolynomialValues::new(input.clone()).coset_ifft(shift);
            let old_coeffs = fft_reference::ifft(PolynomialValues::new(input));
            let old: Vec<F> = shift
                .inverse()
                .powers()
                .zip(old_coeffs.coeffs)
                .map(|(s, c)| s * c)
                .collect();
            assert_raw_eq(&new.coeffs, &old, &ctx);
        }
    }
}

/// Extension fields use the scalar (width 1) path with vectors whose size is not 8 bytes
/// (16 bytes for quadratic, 40 bytes for quintic).
#[test]
fn extension_field_fft_bit_identical() {
    fn check<E: Field + FieldExtension<D, BaseField = F>, const D: usize>(
        rng: &mut Rng,
        max: usize,
    ) {
        for lg_n in 0..=max {
            for r in [0, 2].into_iter().filter(|&r| r <= lg_n) {
                let n = 1usize << lg_n;
                let input: Vec<E> = (0..n)
                    .map(|i| {
                        if i < (n >> r) {
                            let limbs: [F; D] = core::array::from_fn(|_| {
                                GoldilocksField(gen_value(rng, Pattern::EdgeMix))
                            });
                            E::from_basefield_array(limbs)
                        } else {
                            E::ZERO
                        }
                    })
                    .collect();
                let new =
                    fft::fft_with_options(PolynomialCoeffs::new(input.clone()), Some(r), None);
                let old =
                    fft_reference::fft_with_options(PolynomialCoeffs::new(input), Some(r), None);
                for (i, (a, b)) in new.values.iter().zip(&old.values).enumerate() {
                    let (a, b) = (a.to_basefield_array(), b.to_basefield_array());
                    assert_eq!(
                        raw(&a),
                        raw(&b),
                        "extension D={D} lg_n={lg_n} r={r} index {i}"
                    );
                }
            }
        }
    }
    let mut rng = Rng(0xe7);
    check::<QuadraticExtension<F>, 2>(&mut rng, MAX_LG_N.min(18));
    check::<QuinticExtension<F>, 5>(&mut rng, MAX_LG_N.min(16));
}

// ---------------------------------------------------------------------------------------------
// Width-generic check: a portable packed type of width W built from scalar Goldilocks ops. This
// exercises the kernel's index arithmetic for widths 2, 8 (the AVX-512 width) and 16 on any host.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
struct Portable<const W: usize>([F; W]);

impl<const W: usize> Portable<W> {
    fn map2(self, rhs: Self, f: impl Fn(F, F) -> F) -> Self {
        Self(core::array::from_fn(|i| f(self.0[i], rhs.0[i])))
    }
}

impl<const W: usize> Default for Portable<W> {
    fn default() -> Self {
        Self([F::ZERO; W])
    }
}
impl<const W: usize> From<F> for Portable<W> {
    fn from(x: F) -> Self {
        Self([x; W])
    }
}
impl<const W: usize> Add for Portable<W> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.map2(rhs, |a, b| a + b)
    }
}
impl<const W: usize> Add<F> for Portable<W> {
    type Output = Self;
    fn add(self, rhs: F) -> Self {
        self + Self::from(rhs)
    }
}
impl<const W: usize> Add<Portable<W>> for F {
    type Output = Portable<W>;
    fn add(self, rhs: Portable<W>) -> Portable<W> {
        Portable::from(self) + rhs
    }
}
impl<const W: usize> AddAssign for Portable<W> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<const W: usize> AddAssign<F> for Portable<W> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}
impl<const W: usize> Sub for Portable<W> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.map2(rhs, |a, b| a - b)
    }
}
impl<const W: usize> Sub<F> for Portable<W> {
    type Output = Self;
    fn sub(self, rhs: F) -> Self {
        self - Self::from(rhs)
    }
}
impl<const W: usize> Sub<Portable<W>> for F {
    type Output = Portable<W>;
    fn sub(self, rhs: Portable<W>) -> Portable<W> {
        Portable::from(self) - rhs
    }
}
impl<const W: usize> SubAssign for Portable<W> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<const W: usize> SubAssign<F> for Portable<W> {
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}
impl<const W: usize> Mul for Portable<W> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.map2(rhs, |a, b| a * b)
    }
}
impl<const W: usize> Mul<F> for Portable<W> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        self * Self::from(rhs)
    }
}
impl<const W: usize> Mul<Portable<W>> for F {
    type Output = Portable<W>;
    fn mul(self, rhs: Portable<W>) -> Portable<W> {
        Portable::from(self) * rhs
    }
}
impl<const W: usize> MulAssign for Portable<W> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<const W: usize> MulAssign<F> for Portable<W> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}
impl<const W: usize> Div<F> for Portable<W> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: F) -> Self {
        self * rhs.inverse()
    }
}
impl<const W: usize> Neg for Portable<W> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.map(|x| -x))
    }
}
impl<const W: usize> Square for Portable<W> {
    fn square(&self) -> Self {
        *self * *self
    }
}
impl<const W: usize> Sum for Portable<W> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}
impl<const W: usize> Product for Portable<W> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from(F::ONE), |a, b| a * b)
    }
}

unsafe impl<const W: usize> PackedField for Portable<W> {
    type Scalar = F;
    const WIDTH: usize = W;
    const ZEROS: Self = Self([F::ZERO; W]);
    const ONES: Self = Self([F::ONE; W]);

    fn from_slice(slice: &[F]) -> &Self {
        assert_eq!(slice.len(), W);
        unsafe { &*slice.as_ptr().cast() }
    }
    fn from_slice_mut(slice: &mut [F]) -> &mut Self {
        assert_eq!(slice.len(), W);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }
    fn as_slice(&self) -> &[F] {
        &self.0
    }
    fn as_slice_mut(&mut self) -> &mut [F] {
        &mut self.0
    }

    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        assert!(block_len.is_power_of_two() && block_len <= W);
        if block_len == W {
            return (*self, other);
        }
        let (a, b) = (self.0, other.0);
        let mut r0 = [F::ZERO; W];
        let mut r1 = [F::ZERO; W];
        for pair in 0..W / (2 * block_len) {
            let lo = 2 * pair * block_len;
            let hi = lo + block_len;
            r0[lo..hi].copy_from_slice(&a[lo..hi]);
            r0[hi..hi + block_len].copy_from_slice(&b[lo..hi]);
            r1[lo..hi].copy_from_slice(&a[hi..hi + block_len]);
            r1[hi..hi + block_len].copy_from_slice(&b[hi..hi + block_len]);
        }
        (Self(r0), Self(r1))
    }
}

/// The zero-factor fill that callers of the reference kernel do after the bit reversal.
fn prepare_fill(values: &mut [F], r: usize) {
    if r > 0 {
        let mask = !((1 << r) - 1);
        for i in 0..values.len() {
            values[i] = values[i & mask];
        }
    }
}

fn check_width<const W: usize>(rng: &mut Rng, max_lg_n: usize) {
    let lg_w = log2_strict(W);
    for lg_n in (lg_w + 1)..=max_lg_n {
        let table = fft_root_table::<F>(1 << lg_n);
        for r in zero_factors(lg_n) {
            for pattern in [Pattern::Canonical, Pattern::EdgeMix] {
                let mut input = gen_input(rng, pattern, lg_n, r);
                // The new kernel does the zero-factor fill itself, so it gets the bit-reversed
                // input without the fill; the reference kernel expects the caller to fill.
                reverse_index_bits_in_place(&mut input);
                let mut new = input.clone();
                prepare_fill(&mut input, r);
                let mut old = input.clone();
                let mut scalar = input;
                fft::fft_classic_simd::<Portable<W>>(&mut new, r, lg_n, &table);
                fft_reference::fft_classic_simd::<Portable<W>>(&mut old, r, lg_n, &table);
                fft_reference::fft_classic_simd::<F>(&mut scalar, r, lg_n, &table);
                let ctx = || alloc::format!("width {W} lg_n={lg_n} r={r} {pattern:?}");
                assert_raw_eq(&new, &old, &ctx);
                // Sanity check of the portable packing itself: lane-wise scalar ops must agree
                // with the scalar kernel.
                assert_raw_eq(&new, &scalar, &ctx);
            }
        }
    }
}

#[test]
fn width_generic_portable_packing_bit_identical() {
    let mut rng = Rng(0x9ac4);
    let max = MAX_LG_N.min(20);
    check_width::<2>(&mut rng, max);
    check_width::<4>(&mut rng, max);
    check_width::<8>(&mut rng, max);
    check_width::<16>(&mut rng, max);
}

/// The fused kernel still computes the DFT (independent of the reference).
#[test]
fn fused_fft_matches_naive_dft() {
    let mut rng = Rng(0xd1f7);
    for lg_n in 0..=10 {
        let n = 1usize << lg_n;
        let coeffs = gen_input(&mut rng, Pattern::Canonical, lg_n, 0);
        let values = fft::fft(PolynomialCoeffs::new(coeffs.clone())).values;
        let g = F::primitive_root_of_unity(lg_n);
        for (i, x) in g.powers().take(n).enumerate() {
            let mut acc = F::ZERO;
            let mut xp = F::ONE;
            for &c in &coeffs {
                acc += c * xp;
                xp *= x;
            }
            assert_eq!(values[i], acc, "lg_n={lg_n} i={i}");
        }
        let back = fft::ifft(PolynomialValues::new(values)).coeffs;
        assert_eq!(back, coeffs);
    }
}
