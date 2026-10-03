// GPU evaluation of plonky2 quotient polynomials (lab).
//
// One thread per point of the quotient domain (the coset LDE of size N = n << rate_bits). Each
// thread evaluates exactly what `eval_vanishing_poly_base_batch` evaluates at its point: the
// `L_0(x) (Z(x) - 1)` terms, the permutation partial-product checks, and the filtered gate
// constraints, combined with powers of each alpha and divided by `Z_H(x)`. Field arithmetic is
// exact, so the result equals the CPU result element for element (all values are kept canonical).
//
// Inputs are the LDEs of the prover's polynomial batches, column-major, in natural order (column
// `c` at point `i` is `lde[c * N + i]`, the value at `x_i = shift * omega^i`) or, with
// `leaf_order`, in the commitments' leaf order (point `i` at `lde[c * N + bitrev(i)]`). The output
// is in natural order either way.

#include <cstdint>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>

typedef unsigned long long u64;
typedef unsigned int u32;

#define GL_P 0xFFFFFFFF00000001ull
#define GL_EPS 0xFFFFFFFFull
#define EXT_W 7ull  // GoldilocksField quadratic extension: X^2 = 7
#define LDE_SMEM_LG 13   // coset LDE: NTT blocks of up to 2^13 elements (64 KiB) in shared memory
#define LDE_MAX_LG_N 18
#define LDE_MAX_RATE_BITS 4
#ifndef LDE_THREADS
#define LDE_THREADS 512
#endif

// ---- Goldilocks field, canonical representatives ----------------------------------------------

__device__ __forceinline__ u64 gadd(u64 a, u64 b) {
    u64 s = a + b;
    if (s < a) return s + GL_EPS;  // 2^64 = EPS (mod p); cannot overflow again
    return s >= GL_P ? s - GL_P : s;
}

__device__ __forceinline__ u64 gsub(u64 a, u64 b) {
    u64 d = a - b;
    return a < b ? d - GL_EPS : d;  // a - b + 2^64 - EPS = a - b + p
}

// Same reduction as plonky2's `reduce128`, then canonicalised.
__device__ __forceinline__ u64 gmul(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 hi_hi = hi >> 32;
    u64 hi_lo = hi & GL_EPS;
    u64 t0 = lo - hi_hi;
    if (lo < hi_hi) t0 -= GL_EPS;
    u64 t1 = hi_lo * GL_EPS;
    u64 t2 = t0 + t1;
    if (t2 < t0) t2 += GL_EPS;
    return t2 >= GL_P ? t2 - GL_P : t2;
}

__device__ __forceinline__ u64 gpow(u64 b, u64 e) {
    u64 r = 1;
    while (e) {
        if (e & 1) r = gmul(r, b);
        b = gmul(b, b);
        e >>= 1;
    }
    return r;
}

__device__ __forceinline__ u64 ginv(u64 a) { return gpow(a, GL_P - 2); }

__device__ __forceinline__ u64 gsbox(u64 x) {  // x^7
    u64 x2 = gmul(x, x);
    u64 x3 = gmul(x2, x);
    u64 x4 = gmul(x2, x2);
    return gmul(x3, x4);
}

// ---- Quadratic extension ----------------------------------------------------------------------

struct E2 {
    u64 a, b;  // a + b X
};

__device__ __forceinline__ E2 eadd(E2 x, E2 y) { return {gadd(x.a, y.a), gadd(x.b, y.b)}; }
__device__ __forceinline__ E2 esub(E2 x, E2 y) { return {gsub(x.a, y.a), gsub(x.b, y.b)}; }
__device__ __forceinline__ E2 emul(E2 x, E2 y) {
    return {gadd(gmul(x.a, y.a), gmul(EXT_W, gmul(x.b, y.b))), gadd(gmul(x.a, y.b), gmul(x.b, y.a))};
}
__device__ __forceinline__ E2 escal(E2 x, u64 s) { return {gmul(x.a, s), gmul(x.b, s)}; }

// ---- Poseidon constants -----------------------------------------------------------------------

#define SPONGE_WIDTH 12
#define HALF_N_FULL_ROUNDS 4
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS (2 * HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS)

#define POS_RC 0                                       // ALL_ROUND_CONSTANTS [12 * N_ROUNDS]
#define POS_CIRC (POS_RC + SPONGE_WIDTH * N_ROUNDS)    // MDS_MATRIX_CIRC [12]
#define POS_DIAG (POS_CIRC + 12)                       // MDS_MATRIX_DIAG [12]
#define POS_FIRST (POS_DIAG + 12)                      // FAST_PARTIAL_FIRST_ROUND_CONSTANT [12]
#define POS_FAST_RC (POS_FIRST + 12)                   // FAST_PARTIAL_ROUND_CONSTANTS [22]
#define POS_VS (POS_FAST_RC + N_PARTIAL_ROUNDS)        // FAST_PARTIAL_ROUND_VS [22][11]
#define POS_WHATS (POS_VS + N_PARTIAL_ROUNDS * 11)     // FAST_PARTIAL_ROUND_W_HATS [22][11]
#define POS_INIT (POS_WHATS + N_PARTIAL_ROUNDS * 11)   // FAST_PARTIAL_ROUND_INITIAL_MATRIX [11][11]
#define POS_LEN (POS_INIT + 11 * 11)

__constant__ u64 c_pos[POS_LEN];

__device__ __forceinline__ void pos_mds(u64* s) {
    u64 r[SPONGE_WIDTH];
#pragma unroll
    for (int row = 0; row < SPONGE_WIDTH; row++) {
        u64 acc = gmul(s[row], c_pos[POS_DIAG + row]);
#pragma unroll
        for (int k = 0; k < SPONGE_WIDTH; k++) acc = gadd(acc, gmul(s[(k + row) % SPONGE_WIDTH], c_pos[POS_CIRC + k]));
        r[row] = acc;
    }
#pragma unroll
    for (int k = 0; k < SPONGE_WIDTH; k++) s[k] = r[k];
}

__device__ __forceinline__ void pos_mds_ext(const E2* s, E2* r) {
#pragma unroll
    for (int row = 0; row < SPONGE_WIDTH; row++) {
        E2 acc = escal(s[row], c_pos[POS_DIAG + row]);
#pragma unroll
        for (int k = 0; k < SPONGE_WIDTH; k++) acc = eadd(acc, escal(s[(k + row) % SPONGE_WIDTH], c_pos[POS_CIRC + k]));
        r[row] = acc;
    }
}

// ---- Gates ------------------------------------------------------------------------------------

enum GateKind : u32 {
    G_NOOP = 0,
    G_CONSTANT = 1,       // p0 = num_consts
    G_PUBLIC_INPUT = 2,
    G_ARITHMETIC = 3,     // p0 = num_ops
    G_BASE_SUM = 4,       // p0 = num_limbs, p1 = base
    G_ARITH_EXT = 5,      // p0 = num_ops
    G_MUL_EXT = 6,        // p0 = num_ops
    G_REDUCING = 7,       // p0 = num_coeffs
    G_REDUCING_EXT = 8,   // p0 = num_coeffs
    G_RANDOM_ACCESS = 9,  // p0 = bits, p1 = num_copies, p2 = num_extra_constants
    G_POSEIDON_MDS = 10,
    G_POSEIDON = 11,
    G_COSET_INTERP = 12,  // p0 = subgroup_bits, p1 = degree; data: domain[2^bits], weights[2^bits]
    G_EXPONENTIATION = 13,    // p0 = num_power_bits
    // psy u32 gates (psy_common_circuit / psy_plonky2_basic_helpers; same constraints in both)
    G_COMPARISON = 14,        // p0 = num_bits, p1 = num_chunks
    G_U32_ADD_MANY = 15,      // p0 = num_addends, p1 = num_ops, p2 = limbs per op (2-bit limbs)
    G_U32_ARITHMETIC = 16,    // p0 = num_ops
    G_U32_SUBTRACTION = 17,   // p0 = num_ops
    G_U32_INTERLEAVE = 18,    // p0 = num_ops
    G_UNINTERLEAVE_B32 = 19,  // p0 = num_ops
    G_UNINTERLEAVE_U32 = 20,  // p0 = num_ops
    G_U32_RANGE_CHECK = 21,   // p0 = num_input_limbs (2-bit aux limbs, 16 per input)
};

struct GateDesc {
    u32 kind, row, selector, group_lo, group_hi, p0, p1, p2, data_off, num_constraints;
};

#define NC 2                // num_challenges supported
#define UNUSED_SELECTOR 4294967295ull
#define MAX_RA_ITEMS 32     // RandomAccessGate: 2^bits <= 32
#define MAX_CI_POINTS 16    // CosetInterpolationGate: 2^subgroup_bits <= 16

struct QuotientArgs {
    const u64* cs;     // constants then sigmas, [num_constants + num_routed][N]
    const u64* wires;  // [num_wires][N]
    const u64* zs;     // Z per challenge, then partial products challenge-major, [..][N]
    const u64* aux;    // k_is | betas | gammas | alpha powers [NC][num_terms] | gate data
    const GateDesc* gates;
    u64* out;          // [NC][N]
    u64* debug;        // [num_gates][debug_stride]: unfiltered constraints at `debug_point`
    u64 coset_shift;
    u64 omega;         // primitive N-th root of unity
    u64 n_field;       // n = |H| as a field element
    u64 zh_evals[8];   // Z_H(x_i) depends only on i mod rate
    u64 zh_invs[8];
    u64 pih[4];        // public inputs hash
    u32 lg_lde;
    u32 rate_mask;     // rate - 1
    u32 next_step;     // index distance from x to g x
    u32 num_constants;
    u32 const_prefix;  // num_selectors + num_lookup_selectors
    u32 many_selectors;
    u32 num_routed;
    u32 num_wires;
    u32 num_prods;
    u32 max_degree;
    u32 off_kis, off_betas, off_gammas, off_alpha, num_terms, off_gate_data;
    u32 num_gates;
    u32 debug_stride;
    u32 leaf_order;    // inputs in leaf order (bit-reversed point index), see above
    int debug_point;   // -1: off
};

// LDE values from the NTT are reduced, but canonicalise on load anyway: the field functions above
// assume canonical inputs.
__device__ __forceinline__ u64 gcanon(u64 x) { return x >= GL_P ? x - GL_P : x; }

// Loads of one point; `i` is the point's index in the LDE buffers.
struct Point {
    const QuotientArgs& a;
    u32 N, i;
    __device__ __forceinline__ u64 wire(u32 j) const { return gcanon(a.wires[(size_t)j * N + i]); }
    __device__ __forceinline__ E2 wire_ext(u32 j) const { return {wire(j), wire(j + 1)}; }
    __device__ __forceinline__ u64 cons(u32 c) const { return gcanon(a.cs[(size_t)(a.const_prefix + c) * N + i]); }
    __device__ __forceinline__ u64 selector(u32 s) const { return gcanon(a.cs[(size_t)s * N + i]); }
    __device__ __forceinline__ u64 sigma(u32 j) const { return gcanon(a.cs[(size_t)(a.num_constants + j) * N + i]); }
    __device__ __forceinline__ u64 z(u32 col, u32 idx) const { return gcanon(a.zs[(size_t)col * N + idx]); }
};

// Accumulates sum_m term_m * alpha_k^m for each challenge k.
struct Acc {
    u64 v[NC];
    const u64* ap;
    u32 stride;
    __device__ __forceinline__ void add(u32 m, u64 t) {
#pragma unroll
        for (int k = 0; k < NC; k++) v[k] = gadd(v[k], gmul(t, ap[k * stride + m]));
    }
};

// Receives one gate's constraints in order and accumulates them, unfiltered, into `g` (the
// gate's filter multiplies the sum once, in the caller).
struct Sink {
    Acc& g;
    u32 m;        // index of the gate's next constraint among all vanishing terms
    u64* dbg;     // non-null: also store the unfiltered value
    u32 c;
    __device__ __forceinline__ void one(u64 v) {
        if (dbg) dbg[c] = v;
        c++;
        g.add(m++, v);
    }
    __device__ __forceinline__ void ext(E2 v) {
        one(v.a);
        one(v.b);
    }
};

__device__ void g_poseidon(const Point& p, Sink& s) {
    const u32 WIRE_SWAP = 2 * SPONGE_WIDTH, START_DELTA = 2 * SPONGE_WIDTH + 1, START_FULL_0 = START_DELTA + 4;
    const u32 START_PARTIAL = START_FULL_0 + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1);
    const u32 START_FULL_1 = START_PARTIAL + N_PARTIAL_ROUNDS;

    u64 swap = p.wire(WIRE_SWAP);
    s.one(gmul(swap, gsub(swap, 1)));
    for (u32 k = 0; k < 4; k++) {
        u64 lhs = p.wire(k), rhs = p.wire(k + 4), delta = p.wire(START_DELTA + k);
        s.one(gsub(gmul(swap, gsub(rhs, lhs)), delta));
    }
    u64 st[SPONGE_WIDTH];
    for (u32 k = 0; k < 4; k++) {
        u64 delta = p.wire(START_DELTA + k);
        st[k] = gadd(p.wire(k), delta);
        st[k + 4] = gsub(p.wire(k + 4), delta);
    }
    for (u32 k = 8; k < SPONGE_WIDTH; k++) st[k] = p.wire(k);

    u32 round_ctr = 0;
    for (u32 r = 0; r < HALF_N_FULL_ROUNDS; r++) {
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gadd(st[k], c_pos[POS_RC + k + SPONGE_WIDTH * round_ctr]);
        if (r != 0) {
            for (u32 k = 0; k < SPONGE_WIDTH; k++) {
                u64 sbox_in = p.wire(START_FULL_0 + SPONGE_WIDTH * (r - 1) + k);
                s.one(gsub(st[k], sbox_in));
                st[k] = sbox_in;
            }
        }
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gsbox(st[k]);
        pos_mds(st);
        round_ctr++;
    }

    // Partial rounds: partial_first_constant_layer, mds_partial_layer_init, then the fast rounds.
    for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gadd(st[k], c_pos[POS_FIRST + k]);
    {
        u64 r0[SPONGE_WIDTH];
        r0[0] = st[0];
        for (u32 c = 1; c < SPONGE_WIDTH; c++) r0[c] = 0;
        for (u32 r = 1; r < SPONGE_WIDTH; r++)
            for (u32 c = 1; c < SPONGE_WIDTH; c++)
                r0[c] = gadd(r0[c], gmul(st[r], c_pos[POS_INIT + (r - 1) * 11 + (c - 1)]));
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = r0[k];
    }
    const u64 mds0to0 = c_pos[POS_CIRC] + c_pos[POS_DIAG];  // small constants, no reduction needed
    for (u32 r = 0; r < N_PARTIAL_ROUNDS; r++) {
        u64 sbox_in = p.wire(START_PARTIAL + r);
        s.one(gsub(st[0], sbox_in));
        st[0] = gsbox(sbox_in);
        if (r < N_PARTIAL_ROUNDS - 1) st[0] = gadd(st[0], c_pos[POS_FAST_RC + r]);
        // mds_partial_layer_fast(state, r)
        u64 d = gmul(st[0], mds0to0);
        for (u32 k = 1; k < SPONGE_WIDTH; k++) d = gadd(d, gmul(st[k], c_pos[POS_WHATS + r * 11 + (k - 1)]));
        u64 s0 = st[0];
        st[0] = d;
        for (u32 k = 1; k < SPONGE_WIDTH; k++) st[k] = gadd(gmul(s0, c_pos[POS_VS + r * 11 + (k - 1)]), st[k]);
    }
    round_ctr += N_PARTIAL_ROUNDS;

    for (u32 r = 0; r < HALF_N_FULL_ROUNDS; r++) {
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gadd(st[k], c_pos[POS_RC + k + SPONGE_WIDTH * round_ctr]);
        for (u32 k = 0; k < SPONGE_WIDTH; k++) {
            u64 sbox_in = p.wire(START_FULL_1 + SPONGE_WIDTH * r + k);
            s.one(gsub(st[k], sbox_in));
            st[k] = sbox_in;
        }
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gsbox(st[k]);
        pos_mds(st);
        round_ctr++;
    }
    for (u32 k = 0; k < SPONGE_WIDTH; k++) s.one(gsub(st[k], p.wire(SPONGE_WIDTH + k)));
}

__device__ void g_poseidon_mds(const Point& p, Sink& s) {
    E2 in[SPONGE_WIDTH], out[SPONGE_WIDTH];
    for (u32 k = 0; k < SPONGE_WIDTH; k++) in[k] = p.wire_ext(2 * k);
    pos_mds_ext(in, out);
    for (u32 k = 0; k < SPONGE_WIDTH; k++) s.ext(esub(p.wire_ext(2 * (SPONGE_WIDTH + k)), out[k]));
}

// `partial_interpolate` from coset_interpolation.rs.
__device__ void partial_interpolate(const u64* domain, const E2* values, const u64* weights, u32 lo, u32 hi, E2 x,
                                    E2& eval, E2& prod) {
    for (u32 k = lo; k < hi; k++) {
        E2 val = escal(values[k], weights[k]);
        E2 term = {gsub(x.a, domain[k]), x.b};
        E2 next_eval = eadd(emul(eval, term), emul(val, prod));
        prod = emul(prod, term);
        eval = next_eval;
    }
}

__device__ void g_coset_interp(const Point& p, const GateDesc& g, const u64* data, Sink& s) {
    const u32 num_points = 1u << g.p0, degree = g.p1;
    const u64* domain = data;
    const u64* weights = data + num_points;
    const u32 start_values = 1;
    const u32 start_eval_point = start_values + 2 * num_points;
    const u32 start_eval_value = start_eval_point + 2;
    const u32 start_inter = start_eval_value + 2;
    const u32 num_inter = (num_points - 2) / (degree - 1);
    const u32 shifted_point = start_inter + 2 * 2 * num_inter;

    u64 shift = p.wire(0);
    E2 eval_point = p.wire_ext(start_eval_point);
    E2 shifted = p.wire_ext(shifted_point);
    s.ext(esub(eval_point, escal(shifted, shift)));

    E2 values[MAX_CI_POINTS];
    for (u32 k = 0; k < num_points; k++) values[k] = p.wire_ext(start_values + 2 * k);

    E2 ce = {0, 0}, cp = {1, 0};
    partial_interpolate(domain, values, weights, 0, degree, shifted, ce, cp);
    for (u32 k = 0; k < num_inter; k++) {
        E2 ie = p.wire_ext(start_inter + 2 * k);
        E2 ip = p.wire_ext(start_inter + 2 * (num_inter + k));
        s.ext(esub(ie, ce));
        s.ext(esub(ip, cp));
        u32 start = 1 + (degree - 1) * (k + 1);
        u32 end = min(start + degree - 1, num_points);
        ce = ie;
        cp = ip;
        partial_interpolate(domain, values, weights, start, end, shifted, ce, cp);
    }
    s.ext(esub(p.wire_ext(start_eval_value), ce));
}

__device__ void g_random_access(const Point& p, const GateDesc& g, Sink& s) {
    const u32 bits = g.p0, num_copies = g.p1, num_extra = g.p2, vec = 1u << bits;
    const u32 start_extra = (2 + vec) * num_copies;
    const u32 num_routed = start_extra + num_extra;
    for (u32 copy = 0; copy < num_copies; copy++) {
        u32 base = (2 + vec) * copy;
        u64 access_index = p.wire(base);
        u64 claimed = p.wire(base + 1);
        u64 items[MAX_RA_ITEMS];
        for (u32 k = 0; k < vec; k++) items[k] = p.wire(base + 2 + k);
        for (u32 k = 0; k < bits; k++) {
            u64 b = p.wire(num_routed + copy * bits + k);
            s.one(gmul(b, gsub(b, 1)));
        }
        u64 rec = 0;
        for (int k = (int)bits - 1; k >= 0; k--) rec = gadd(gadd(rec, rec), p.wire(num_routed + copy * bits + k));
        s.one(gsub(rec, access_index));
        u32 len = vec;
        for (u32 k = 0; k < bits; k++) {
            u64 b = p.wire(num_routed + copy * bits + k);
            for (u32 t = 0; t < len / 2; t++) {
                u64 x = items[2 * t], y = items[2 * t + 1];
                items[t] = gadd(x, gmul(b, gsub(y, x)));
            }
            len /= 2;
        }
        s.one(gsub(items[0], claimed));
    }
    for (u32 k = 0; k < num_extra; k++) s.one(gsub(p.cons(k), p.wire(start_extra + k)));
}

// prod_{x < 4} (l - x): range check of a 2-bit limb.
__device__ __forceinline__ u64 limb2_check(u64 l) {
    return gmul(gmul(l, gsub(l, 1)), gmul(gsub(l, 2), gsub(l, 3)));
}

__device__ void g_exponentiation(const Point& p, const GateDesc& g, Sink& s) {
    const u32 n = g.p0;
    u64 base = p.wire(0);
    for (u32 k = 0; k < n; k++) {
        u64 prev = 1;
        if (k > 0) {
            u64 v = p.wire(2 + n + k - 1);
            prev = gmul(v, v);
        }
        u64 bit = p.wire(1 + (n - k - 1));
        u64 computed = gmul(prev, gadd(gmul(bit, base), gsub(1, bit)));
        s.one(gsub(computed, p.wire(2 + n + k)));
    }
    s.one(gsub(p.wire(1 + n), p.wire(2 + n + n - 1)));
}

__device__ void g_comparison(const Point& p, const GateDesc& g, Sink& s) {
    const u32 num_bits = g.p0, nc = g.p1;
    const u32 chunk_bits = (num_bits + nc - 1) / nc;
    const u64 chunk_base = 1ull << chunk_bits;
    const u32 chunk_size = 1u << chunk_bits;
    u64 first = 0, second = 0;
    for (int k = (int)nc - 1; k >= 0; k--) {
        first = gadd(gmul(first, chunk_base), p.wire(4 + k));
        second = gadd(gmul(second, chunk_base), p.wire(4 + nc + k));
    }
    s.one(gsub(first, p.wire(0)));
    s.one(gsub(second, p.wire(1)));
    u64 msd = 0;
    for (u32 k = 0; k < nc; k++) {
        u64 fc = p.wire(4 + k), sc = p.wire(4 + nc + k);
        u64 fp = 1, sp = 1;
        for (u32 x = 0; x < chunk_size; x++) {
            fp = gmul(fp, gsub(fc, x));
            sp = gmul(sp, gsub(sc, x));
        }
        s.one(fp);
        s.one(sp);
        u64 diff = gsub(sc, fc);
        u64 eq_dummy = p.wire(4 + 2 * nc + k);
        u64 equal = p.wire(4 + 3 * nc + k);
        s.one(gsub(gmul(diff, eq_dummy), gsub(1, equal)));
        s.one(gmul(equal, diff));
        u64 inter = p.wire(4 + 4 * nc + k);
        s.one(gsub(inter, gmul(equal, msd)));
        msd = gadd(inter, gmul(gsub(1, equal), diff));
    }
    u64 msd_wire = p.wire(3);
    s.one(gsub(msd_wire, msd));
    u64 bits_combined = 0;
    for (int k = (int)chunk_bits; k >= 0; k--) bits_combined = gadd(gadd(bits_combined, bits_combined), p.wire(4 + 5 * nc + k));
    for (u32 k = 0; k <= chunk_bits; k++) {
        u64 b = p.wire(4 + 5 * nc + k);
        s.one(gmul(b, gsub(1, b)));
    }
    s.one(gsub(gadd(msd_wire, chunk_base), bits_combined));
    s.one(gsub(p.wire(2), p.wire(4 + 5 * nc + chunk_bits)));
}

__device__ void g_u32_add_many(const Point& p, const GateDesc& g, Sink& s) {
    const u32 na = g.p0, num_ops = g.p1, num_limbs = g.p2, result_limbs = 16;
    for (u32 op = 0; op < num_ops; op++) {
        const u32 w = (na + 3) * op;
        u64 computed = 0;
        for (u32 j = 0; j < na; j++) computed = gadd(computed, p.wire(w + j));
        computed = gadd(computed, p.wire(w + na));
        u64 out_result = p.wire(w + na + 1), out_carry = p.wire(w + na + 2);
        s.one(gsub(gadd(gmul(out_carry, 1ull << 32), out_result), computed));
        u64 cr = 0, cc = 0;
        const u32 limbs = (na + 3) * num_ops + num_limbs * op;
        for (int j = (int)num_limbs - 1; j >= 0; j--) {
            u64 l = p.wire(limbs + j);
            s.one(limb2_check(l));
            if ((u32)j < result_limbs) cr = gadd(gmul(cr, 4), l);
            else cc = gadd(gmul(cc, 4), l);
        }
        s.one(gsub(cr, out_result));
        s.one(gsub(cc, out_carry));
    }
}

__device__ void g_u32_arithmetic(const Point& p, const GateDesc& g, Sink& s) {
    const u32 num_ops = g.p0;
    for (u32 op = 0; op < num_ops; op++) {
        const u32 w = 6 * op;
        u64 computed = gadd(gmul(p.wire(w), p.wire(w + 1)), p.wire(w + 2));
        u64 out_low = p.wire(w + 3), out_high = p.wire(w + 4), inverse = p.wire(w + 5);
        u64 diff = gsub(0xFFFFFFFFull, out_high);
        u64 hi_not_max = gsub(gmul(inverse, diff), 1);
        s.one(gmul(hi_not_max, out_low));
        s.one(gsub(gadd(gmul(out_high, 1ull << 32), out_low), computed));
        u64 cl = 0, ch = 0;
        const u32 limbs = 6 * num_ops + 32 * op;
        for (int j = 31; j >= 0; j--) {
            u64 l = p.wire(limbs + j);
            s.one(limb2_check(l));
            if (j < 16) cl = gadd(gmul(cl, 4), l);
            else ch = gadd(gmul(ch, 4), l);
        }
        s.one(gsub(cl, out_low));
        s.one(gsub(ch, out_high));
    }
}

__device__ void g_u32_subtraction(const Point& p, const GateDesc& g, Sink& s) {
    const u32 num_ops = g.p0;
    for (u32 op = 0; op < num_ops; op++) {
        const u32 w = 5 * op;
        u64 initial = gsub(gsub(p.wire(w), p.wire(w + 1)), p.wire(w + 2));
        u64 out_result = p.wire(w + 3), out_borrow = p.wire(w + 4);
        s.one(gsub(out_result, gadd(initial, gmul(out_borrow, 1ull << 32))));
        u64 c = 0;
        const u32 limbs = 5 * num_ops + 16 * op;
        for (int j = 15; j >= 0; j--) {
            u64 l = p.wire(limbs + j);
            s.one(limb2_check(l));
            c = gadd(gmul(c, 4), l);
        }
        s.one(gsub(c, out_result));
        s.one(gmul(out_borrow, gsub(1, out_borrow)));
    }
}

__device__ void g_u32_interleave(const Point& p, const GateDesc& g, Sink& s) {
    const u32 num_ops = g.p0;
    for (u32 op = 0; op < num_ops; op++) {
        const u32 bits = 2 * num_ops + 32 * op;
        u64 x = 0, xi = 0;
        for (u32 j = 0; j < 32; j++) {
            u64 b = p.wire(bits + j);
            x = gadd(gadd(x, x), b);
            xi = gadd(gmul(xi, 4), b);
        }
        s.one(gsub(x, p.wire(2 * op)));
        s.one(gsub(xi, p.wire(2 * op + 1)));
        for (u32 j = 0; j < 32; j++) {
            u64 b = p.wire(bits + j);
            s.one(gmul(b, gsub(b, 1)));
        }
    }
}

__device__ void g_u32_range_check(const Point& p, const GateDesc& g, Sink& s) {
    const u32 n = g.p0;
    for (u32 k = 0; k < n; k++) {
        const u32 aux = n + 16 * k;
        u64 computed = 0;
        for (int j = 15; j >= 0; j--) computed = gadd(gmul(computed, 4), p.wire(aux + j));
        s.one(gsub(computed, p.wire(k)));
        for (u32 j = 0; j < 16; j++) s.one(limb2_check(p.wire(aux + j)));
    }
}

// UninterleaveToB32 (`to_u32` false) and UninterleaveToU32 (`to_u32` true).
__device__ void g_uninterleave(const Point& p, const GateDesc& g, Sink& s, bool to_u32) {
    const u32 num_ops = g.p0;
    for (u32 op = 0; op < num_ops; op++) {
        const u32 bits = 3 * num_ops + 64 * op;
        u64 xi = 0;
        for (u32 j = 0; j < 64; j++) xi = gadd(gadd(xi, xi), p.wire(bits + j));
        s.one(gsub(xi, p.wire(3 * op)));
        u64 evens = 0, odds = 0;
        for (u32 k = 0; k < 32; k++) {
            u64 coeff = to_u32 ? (1ull << (31 - k)) : (1ull << (2 * (31 - k)));
            evens = gadd(evens, gmul(p.wire(bits + 2 * k), coeff));
            odds = gadd(odds, gmul(p.wire(bits + 2 * k + 1), coeff));
        }
        s.one(gsub(evens, p.wire(3 * op + 1)));
        s.one(gsub(odds, p.wire(3 * op + 2)));
        for (u32 j = 0; j < 64; j++) {
            u64 b = p.wire(bits + j);
            s.one(gmul(b, gsub(b, 1)));
        }
    }
}

__device__ void eval_gate(const Point& p, const GateDesc& g, const u64* gate_data, Sink& s) {
    switch (g.kind) {
    case G_NOOP:
        break;
    case G_CONSTANT:
        for (u32 k = 0; k < g.p0; k++) s.one(gsub(p.cons(k), p.wire(k)));
        break;
    case G_PUBLIC_INPUT:
        for (u32 k = 0; k < 4; k++) s.one(gsub(p.wire(k), p.a.pih[k]));
        break;
    case G_ARITHMETIC: {
        u64 c0 = p.cons(0), c1 = p.cons(1);
        for (u32 k = 0; k < g.p0; k++) {
            u64 m0 = p.wire(4 * k), m1 = p.wire(4 * k + 1), ad = p.wire(4 * k + 2), out = p.wire(4 * k + 3);
            s.one(gsub(out, gadd(gmul(gmul(m0, m1), c0), gmul(ad, c1))));
        }
        break;
    }
    case G_BASE_SUM: {
        const u32 num_limbs = g.p0;
        const u64 base = g.p1;
        u64 sum = p.wire(0), computed = 0;
        for (int k = (int)num_limbs - 1; k >= 0; k--) computed = gadd(gmul(computed, base), p.wire(1 + k));
        s.one(gsub(computed, sum));
        for (u32 k = 0; k < num_limbs; k++) {
            u64 limb = p.wire(1 + k), prod = 1;
            for (u64 d = 0; d < base; d++) prod = gmul(prod, gsub(limb, d));
            s.one(prod);
        }
        break;
    }
    case G_ARITH_EXT: {
        u64 c0 = p.cons(0), c1 = p.cons(1);
        for (u32 k = 0; k < g.p0; k++) {
            E2 m0 = p.wire_ext(8 * k), m1 = p.wire_ext(8 * k + 2), ad = p.wire_ext(8 * k + 4), out = p.wire_ext(8 * k + 6);
            s.ext(esub(out, eadd(escal(emul(m0, m1), c0), escal(ad, c1))));
        }
        break;
    }
    case G_MUL_EXT: {
        u64 c0 = p.cons(0);
        for (u32 k = 0; k < g.p0; k++) {
            E2 m0 = p.wire_ext(6 * k), m1 = p.wire_ext(6 * k + 2), out = p.wire_ext(6 * k + 4);
            s.ext(esub(out, escal(emul(m0, m1), c0)));
        }
        break;
    }
    case G_REDUCING: {
        const u32 nc = g.p0, start_accs = 6 + nc;
        E2 alpha = p.wire_ext(2), acc = p.wire_ext(4);
        for (u32 k = 0; k < nc; k++) {
            E2 acc_k = (k == nc - 1) ? p.wire_ext(0) : p.wire_ext(start_accs + 2 * k);
            E2 t = emul(acc, alpha);
            t.a = gadd(t.a, p.wire(6 + k));
            s.ext(esub(t, acc_k));
            acc = acc_k;
        }
        break;
    }
    case G_REDUCING_EXT: {
        const u32 nc = g.p0, start_accs = 6 + 2 * nc;
        E2 alpha = p.wire_ext(2), acc = p.wire_ext(4);
        for (u32 k = 0; k < nc; k++) {
            E2 acc_k = (k == nc - 1) ? p.wire_ext(0) : p.wire_ext(start_accs + 2 * k);
            s.ext(esub(eadd(emul(acc, alpha), p.wire_ext(6 + 2 * k)), acc_k));
            acc = acc_k;
        }
        break;
    }
    case G_RANDOM_ACCESS:
        g_random_access(p, g, s);
        break;
    case G_POSEIDON_MDS:
        g_poseidon_mds(p, s);
        break;
    case G_POSEIDON:
        g_poseidon(p, s);
        break;
    case G_COSET_INTERP:
        g_coset_interp(p, g, gate_data + g.data_off, s);
        break;
    case G_EXPONENTIATION:
        g_exponentiation(p, g, s);
        break;
    case G_COMPARISON:
        g_comparison(p, g, s);
        break;
    case G_U32_ADD_MANY:
        g_u32_add_many(p, g, s);
        break;
    case G_U32_ARITHMETIC:
        g_u32_arithmetic(p, g, s);
        break;
    case G_U32_SUBTRACTION:
        g_u32_subtraction(p, g, s);
        break;
    case G_U32_INTERLEAVE:
        g_u32_interleave(p, g, s);
        break;
    case G_UNINTERLEAVE_B32:
        g_uninterleave(p, g, s, false);
        break;
    case G_UNINTERLEAVE_U32:
        g_uninterleave(p, g, s, true);
        break;
    case G_U32_RANGE_CHECK:
        g_u32_range_check(p, g, s);
        break;
    }
}

#ifndef QK_THREADS
#define QK_THREADS 128
#endif
#ifndef QK_MIN_BLOCKS
#define QK_MIN_BLOCKS 5
#endif

// Register cap from the launch bounds: about 100 registers per thread measured best (occupancy over
// spills) on sm_120.
__global__ void __launch_bounds__(QK_THREADS, QK_MIN_BLOCKS) quotient_kernel(const QuotientArgs a) {
    const u32 N = 1u << a.lg_lde;
    // `t` indexes the LDE buffers; `i` is the point x_i.
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    const u32 i = a.leaf_order ? __brev(t) >> (32 - a.lg_lde) : t;
    Point p{a, N, t};
    const u64* k_is = a.aux + a.off_kis;
    const u64* betas = a.aux + a.off_betas;
    const u64* gammas = a.aux + a.off_gammas;
    const u64* gate_data = a.aux + a.off_gate_data;
    Acc acc;
    for (int k = 0; k < NC; k++) acc.v[k] = 0;
    acc.ap = a.aux + a.off_alpha;
    acc.stride = a.num_terms;

    const u64 x = gmul(a.coset_shift, gpow(a.omega, i));
    const u32 r = i & a.rate_mask;
    // L_0(x) = Z_H(x) / (n (x - 1)).
    const u64 l0 = gmul(a.zh_evals[r], ginv(gmul(a.n_field, gsub(x, 1))));
    u32 m = 0;
    for (u32 c = 0; c < NC; c++) acc.add(m++, gmul(l0, gsub(p.z(c, t), 1)));

    const u32 i_next = (i + a.next_step) & (N - 1);
    const u32 t_next = a.leaf_order ? __brev(i_next) >> (32 - a.lg_lde) : i_next;
    const u32 num_chunks = (a.num_routed + a.max_degree - 1) / a.max_degree;
    for (u32 c = 0; c < NC; c++) {
        const u64 beta = betas[c], gamma = gammas[c];
        u64 prev = p.z(c, t);
        for (u32 chunk = 0; chunk < num_chunks; chunk++) {
            u64 num = 1, den = 1;
            u32 hi = min((chunk + 1) * a.max_degree, a.num_routed);
            for (u32 j = chunk * a.max_degree; j < hi; j++) {
                u64 w = p.wire(j);
                num = gmul(num, gadd(gadd(w, gmul(beta, gmul(k_is[j], x))), gamma));
                den = gmul(den, gadd(gadd(w, gmul(beta, p.sigma(j))), gamma));
            }
            u64 next = chunk < a.num_prods ? p.z(NC + c * a.num_prods + chunk, t) : p.z(c, t_next);
            acc.add(m++, gsub(gmul(prev, num), gmul(next, den)));
            prev = next;
        }
    }

    // Gate constraints, each filtered by its selector polynomial, summed by constraint index.
    const bool dbg = a.debug && (int)i == a.debug_point;
    for (u32 gi = 0; gi < a.num_gates; gi++) {
        const GateDesc& g = a.gates[gi];
        u64 s = p.selector(g.selector);
        u64 filter = 1;
        for (u32 k = g.group_lo; k < g.group_hi; k++)
            if (k != g.row) filter = gmul(filter, gsub(k, s));
        if (a.many_selectors) filter = gmul(filter, gsub(UNUSED_SELECTOR % GL_P, s));
        Acc gacc;
        for (int k = 0; k < NC; k++) gacc.v[k] = 0;
        gacc.ap = acc.ap;
        gacc.stride = acc.stride;
        Sink sink{gacc, m, dbg ? a.debug + (size_t)gi * a.debug_stride : nullptr, 0};
        eval_gate(p, g, gate_data, sink);
        for (int k = 0; k < NC; k++) acc.v[k] = gadd(acc.v[k], gmul(gacc.v[k], filter));
    }

    const u64 zh_inv = a.zh_invs[r];
    for (u32 c = 0; c < NC; c++) a.out[(size_t)c * N + i] = gmul(acc.v[c], zh_inv);
}

// ---- Poseidon permutation and the FRI proof-of-work search ------------------------------------

// 160-bit accumulator lo + mid 2^64 + hi 2^128, for sums of 64x64-bit products reduced once.
struct Acc160 {
    u64 lo, mid;
    u32 hi;
};

__device__ __forceinline__ void acc_mul(Acc160& a, u64 x, u64 y) {
    u64 plo = x * y, phi = __umul64hi(x, y);
    u64 t = a.lo + plo;
    u64 c = t < plo;
    a.lo = t;
    u64 m = a.mid + phi;
    u32 c2 = m < phi;
    u64 m2 = m + c;
    c2 += m2 < c;
    a.mid = m2;
    a.hi += c2;
}

// Canonical (mid + hi 2^64) 2^64 + lo mod p, with 2^64 = EPS (mod p).
__device__ __forceinline__ u64 reduce160(const Acc160& a) {
    u64 h = (u64)a.hi * GL_EPS;  // hi < 2^32
    u64 t = a.mid + h;
    if (t < h) t += GL_EPS;
    u64 hi_hi = t >> 32, hi_lo = t & GL_EPS;
    u64 t0 = a.lo - hi_hi;
    if (a.lo < hi_hi) t0 -= GL_EPS;
    u64 t1 = hi_lo * GL_EPS;
    u64 t2 = t0 + t1;
    if (t2 < t0) t2 += GL_EPS;
    return t2 >= GL_P ? t2 - GL_P : t2;
}

// plonky2's Goldilocks Poseidon MDS matrix (MDS_MATRIX_CIRC; MDS_MATRIX_DIAG is MDS_DIAG0 at index
// 0 and zero elsewhere), compiled in so the products become 32-bit multiplications by immediates.
// `plonky2_quotient_init` rejects constants that disagree.
#define MDS_CIRC_VALUES {17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20}
#define MDS_DIAG0 8u

// Row ROW of the MDS layer: each element split into 32-bit halves, the products by the small
// entries summed per half (both below 2^42), combined and reduced once.
template <u32 ROW>
__device__ __forceinline__ u64 mds_row(const u64* s) {
    constexpr u32 C[SPONGE_WIDTH] = MDS_CIRC_VALUES;
    u64 lo_sum = 0, hi_sum = 0;
#pragma unroll
    for (u32 k = 0; k < SPONGE_WIDTH; k++) {
        const u32 c = C[k] + (k == 0 && ROW == 0 ? MDS_DIAG0 : 0u);
        const u64 v = s[(k + ROW) % SPONGE_WIDTH];
        lo_sum += (u64)(u32)v * c;
        hi_sum += (v >> 32) * c;
    }
    const u64 lo = lo_sum + (hi_sum << 32);
    const u64 hi = (hi_sum >> 32) + (lo < lo_sum);  // < 2^11
    const u64 h = (hi << 32) - hi;                  // hi 2^64 = hi EPS (mod p)
    u64 r = lo + h;
    if (r < h) r += GL_EPS;
    return r >= GL_P ? r - GL_P : r;
}

template <u32 ROW>
__device__ __forceinline__ void mds_rows(const u64* s, u64* t) {
    t[ROW] = mds_row<ROW>(s);
    if constexpr (ROW + 1 < SPONGE_WIDTH) mds_rows<ROW + 1>(s, t);
}

// `Poseidon::poseidon`: full rounds, fast partial rounds, full rounds. Same values as plonky2;
// sums of products are accumulated wide and reduced once.
__device__ void poseidon_permute(u64* st) {
    u32 round_ctr = 0;
#pragma unroll
    for (u32 r = 0; r < HALF_N_FULL_ROUNDS; r++) {
#pragma unroll
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gsbox(gadd(st[k], c_pos[POS_RC + k + SPONGE_WIDTH * round_ctr]));
        u64 t[SPONGE_WIDTH];
        mds_rows<0>(st, t);
#pragma unroll
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = t[k];
        round_ctr++;
    }
#pragma unroll
    for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gadd(st[k], c_pos[POS_FIRST + k]);
    {
        u64 r0[SPONGE_WIDTH];
        r0[0] = st[0];
#pragma unroll
        for (u32 c = 1; c < SPONGE_WIDTH; c++) {
            Acc160 a{0, 0, 0};
#pragma unroll
            for (u32 r = 1; r < SPONGE_WIDTH; r++) acc_mul(a, st[r], c_pos[POS_INIT + (r - 1) * 11 + (c - 1)]);
            r0[c] = reduce160(a);
        }
#pragma unroll
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = r0[k];
    }
    constexpr u32 C[SPONGE_WIDTH] = MDS_CIRC_VALUES;
    const u64 mds0to0 = C[0] + MDS_DIAG0;
#pragma unroll
    for (u32 r = 0; r < N_PARTIAL_ROUNDS; r++) {
        st[0] = gadd(gsbox(st[0]), c_pos[POS_FAST_RC + r]);
        Acc160 a{0, 0, 0};
        acc_mul(a, st[0], mds0to0);
#pragma unroll
        for (u32 k = 1; k < SPONGE_WIDTH; k++) acc_mul(a, st[k], c_pos[POS_WHATS + r * 11 + (k - 1)]);
        u64 s0 = st[0];
        st[0] = reduce160(a);
#pragma unroll
        for (u32 k = 1; k < SPONGE_WIDTH; k++) st[k] = gadd(gmul(s0, c_pos[POS_VS + r * 11 + (k - 1)]), st[k]);
    }
    round_ctr += N_PARTIAL_ROUNDS;
#pragma unroll
    for (u32 r = 0; r < HALF_N_FULL_ROUNDS; r++) {
#pragma unroll
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = gsbox(gadd(st[k], c_pos[POS_RC + k + SPONGE_WIDTH * round_ctr]));
        u64 t[SPONGE_WIDTH];
        mds_rows<0>(st, t);
#pragma unroll
        for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = t[k];
        round_ctr++;
    }
}

// ---- Merkle trees in plonky2's layout ---------------------------------------------------------

// Index in a subtree's digest buffer of node `index` on `layer` (0 = leaf digests); see
// `subtree_digest_index` in merkle_tree.rs.
__device__ __forceinline__ u32 subtree_digest_index(u32 layer, u32 index) {
    return 2 * (((index >> 1) << (layer + 1)) + (1u << layer) - 1) + (index & 1);
}

// `hash_n_to_hash_no_pad` of each row-major leaf (cols > 4): zero state, overwrite RATE elements
// per permutation, first 4 elements out. Leaves of subtree s go to its layer 0, or straight to
// the cap when subtrees are single leaves.
// COL_MAJOR: leaf element (r, k) at leaves[k * rows + r] (a commitment's column-major LDE in leaf
// order) instead of leaves[r * cols + k].
template <bool COL_MAJOR>
__global__ void __launch_bounds__(128) merkle_leaves_kernel(u64* digests, u64* cap, const u64* leaves, u32 rows, u32 cols,
                                                          u32 lg_subtree_leaves, u32 per_subtree) {
    const u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    u64 st[SPONGE_WIDTH];
#pragma unroll
    for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = 0;
    for (u32 start = 0; start < cols; start += 8) {
        const u32 len = min(8u, cols - start);
        for (u32 k = 0; k < len; k++)
            st[k] = gcanon(COL_MAJOR ? leaves[(size_t)(start + k) * rows + r] : leaves[(size_t)r * cols + start + k]);
        poseidon_permute(st);
    }
    u64* out;
    if (per_subtree == 0) {
        out = cap + (size_t)r * 4;
    } else {
        const u32 s = r >> lg_subtree_leaves, idx = r & ((1u << lg_subtree_leaves) - 1);
        out = digests + ((size_t)s * per_subtree + subtree_digest_index(0, idx)) * 4;
    }
    out[0] = st[0];
    out[1] = st[1];
    out[2] = st[2];
    out[3] = st[3];
}

// `compress` (two_to_one) of the children of every node on `layer` (>= 1) of every subtree; the
// subtree roots (layer == lg_subtree_leaves) go to the cap.
__global__ void __launch_bounds__(128) merkle_layer_kernel(u64* digests, u64* cap, u32 layer, u32 lg_subtree_leaves,
                                                         u32 per_subtree, u32 num_subtrees) {
    const u32 nodes = 1u << (lg_subtree_leaves - layer);
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nodes * num_subtrees) return;
    const u32 s = t / nodes, idx = t % nodes;
    const u64* base = digests + (size_t)s * per_subtree * 4;
    const u64* l = base + (size_t)subtree_digest_index(layer - 1, 2 * idx) * 4;  // right sibling follows
    u64 st[SPONGE_WIDTH];
#pragma unroll
    for (u32 k = 0; k < 8; k++) st[k] = l[k];
#pragma unroll
    for (u32 k = 8; k < SPONGE_WIDTH; k++) st[k] = 0;
    poseidon_permute(st);
    u64* out = layer == lg_subtree_leaves ? cap + (size_t)s * 4
                                          : digests + ((size_t)s * per_subtree + subtree_digest_index(layer, idx)) * 4;
    out[0] = st[0];
    out[1] = st[1];
    out[2] = st[2];
    out[3] = st[3];
}

static int merkle(u64* digests, u64* cap, const u64* leaves, u32 lg_rows, u32 cols, u32 cap_height, bool col_major,
                  cudaStream_t stream) {
    const u32 rows = 1u << lg_rows, lg_sub = lg_rows - cap_height, subtrees = 1u << cap_height;
    const u32 per_subtree = lg_sub == 0 ? 0 : 2 * ((1u << lg_sub) - 1);
    if (col_major)
        merkle_leaves_kernel<true><<<(rows + 127) / 128, 128, 0, stream>>>(digests, cap, leaves, rows, cols, lg_sub, per_subtree);
    else
        merkle_leaves_kernel<false><<<(rows + 127) / 128, 128, 0, stream>>>(digests, cap, leaves, rows, cols, lg_sub, per_subtree);
    for (u32 layer = 1; layer <= lg_sub; layer++) {
        const u32 n = (1u << (lg_sub - layer)) * subtrees;
        merkle_layer_kernel<<<(n + 127) / 128, 128, 0, stream>>>(digests, cap, layer, lg_sub, per_subtree, subtrees);
    }
    return (int)cudaGetLastError();
}

// Merkle tree of `rows` row-major leaves with 2^cap_height subtrees, digests in plonky2's
// layout (2 * (subtree leaves - 1) per subtree) and the cap, on `stream`.
extern "C" int plonky2_merkle(u64* digests, u64* cap, const u64* leaves, u32 lg_rows, u32 cols, u32 cap_height,
                              cudaStream_t stream) {
    return merkle(digests, cap, leaves, lg_rows, cols, cap_height, false, stream);
}

// The same tree from column-major leaves (`cols` columns of 2^lg_rows, as the coset LDE leaves
// them): no transpose needed.
extern "C" int plonky2_merkle_cols(u64* digests, u64* cap, const u64* leaves, u32 lg_rows, u32 cols, u32 cap_height,
                                   cudaStream_t stream) {
    return merkle(digests, cap, leaves, lg_rows, cols, cap_height, true, stream);
}

struct PowArgs {
    u64 state[SPONGE_WIDTH];  // duplex state before the candidate is written (canonical)
    u64 base;                 // first candidate of this launch
    u64* best;                // smallest accepting candidate so far (atomicMin)
    u32 pos;                  // index the candidate overwrites
    u32 response;             // index of the PoW response after the permutation
    u32 min_leading_zeros;
    u32 count;                // candidates in this launch
};

__global__ void __launch_bounds__(128) pow_kernel(const PowArgs a) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= a.count) return;
    const u64 candidate = a.base + t;
    u64 st[SPONGE_WIDTH];
    for (u32 k = 0; k < SPONGE_WIDTH; k++) st[k] = a.state[k];
    st[a.pos] = candidate;
    poseidon_permute(st);
    if ((u32)__clzll((long long)st[a.response]) >= a.min_leading_zeros) atomicMin(a.best, candidate);
}

extern "C" int plonky2_pow_launch(const PowArgs* args, cudaStream_t stream) {
    const u32 threads = 128;
    pow_kernel<<<(args->count + threads - 1) / threads, threads, 0, stream>>>(*args);
    return (int)cudaGetLastError();
}

__global__ void permute_kernel(u64* states, u32 n) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) poseidon_permute(states + (size_t)t * SPONGE_WIDTH);
}

// Test hook: permutes `n` device-resident states in place.
extern "C" int plonky2_poseidon_permute_states(u64* states, u32 n) {
    permute_kernel<<<(n + 127) / 128, 128>>>(states, n);
    return (int)cudaGetLastError();
}

// ---- Merkle digest layout -------------------------------------------------------------------

// zeknox fills each cap subtree's digests in heap order (node k has children 2k+2 and 2k+3, so
// level d holds k in [2^d - 2, 2^(d+1) - 2)); plonky2 stores node `index` of `layer` (0 = leaf
// digests) at 2 * (((index >> 1) << (layer + 1)) + 2^layer - 1) + (index & 1)
// (`subtree_digest_index` in merkle_tree.rs). Moves every 4-element digest to its plonky2 slot.
__global__ void digest_layout_kernel(u64* dst, const u64* src, u32 per_subtree, u32 lg_leaves, u32 total) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const u32 s = t / per_subtree, k = t % per_subtree;
    const u32 d = 31 - __clz(k + 2);
    const u32 index = k + 2 - (1u << d);
    const u32 layer = lg_leaves - d;
    const u32 pos = 2 * (((index >> 1) << (layer + 1)) + (1u << layer) - 1) + (index & 1);
    const u64* sp = src + ((size_t)s * per_subtree + k) * 4;
    u64* dp = dst + ((size_t)s * per_subtree + pos) * 4;
    dp[0] = sp[0];
    dp[1] = sp[1];
    dp[2] = sp[2];
    dp[3] = sp[3];
}

extern "C" int plonky2_digest_layout(u64* dst, const u64* src, u32 per_subtree, u32 lg_leaves, u32 num_subtrees) {
    const u32 total = per_subtree * num_subtrees;
    if (total == 0) return 0;
    digest_layout_kernel<<<(total + 255) / 256, 256>>>(dst, src, per_subtree, lg_leaves, total);
    return (int)cudaGetLastError();
}

extern "C" int plonky2_quotient_init(const u64* poseidon_constants, u32 len) {
    if (len != POS_LEN) return -1;
    constexpr u32 C[SPONGE_WIDTH] = MDS_CIRC_VALUES;
    for (u32 k = 0; k < SPONGE_WIDTH; k++) {
        if (poseidon_constants[POS_CIRC + k] != C[k] || poseidon_constants[POS_DIAG + k] != (k == 0 ? MDS_DIAG0 : 0u)) return -2;
    }
    int e = (int)cudaMemcpyToSymbol(c_pos, poseidon_constants, sizeof(u64) * POS_LEN);
    // From pageable memory the copy may still be in flight when this returns; kernels on
    // non-blocking streams (PoW, quotient, FRI) do not wait for the legacy stream.
    return e ? e : (int)cudaDeviceSynchronize();
}

extern "C" u32 plonky2_quotient_poseidon_constants_len() { return POS_LEN; }

extern "C" int plonky2_quotient_launch(const QuotientArgs* args, cudaStream_t stream) {
    const u32 N = 1u << args->lg_lde;
    const u32 threads = QK_THREADS;
    quotient_kernel<<<(N + threads - 1) / threads, threads, 0, stream>>>(*args);
    return (int)cudaGetLastError();
}

// ---- Coset LDE in leaf order ------------------------------------------------------------------

// With R = 2^rate_bits and N = R n, the value of a polynomial with coefficients c_k (k < n) at
// x_i = s w_N^i, i = R m + j, is NTT_n(c_k (s w_N^j)^k)[m]: R size-n NTTs of scaled coefficients
// instead of one size-N NTT of a zero-padded input. A commitment's leaf r is the point
// i = bitrev_{lg N}(r) = R bitrev_{lg n}(p) + j for r = bitrev_{rate_bits}(j) n + p, so a
// decimation-in-frequency NTT (natural input, bit-reversed output) of coset j fills leaf positions
// bitrev_{rate_bits}(j) n .. + n in order. The output is column-major in leaf order:
// column c at leaf r is dst[c N + r].

// Larger n (up to 2^18) take two passes over the same in-place DIF: the first lg_n - 13 stages
// combine elements 2^13 or more apart, so each offset o < 2^13 runs them in registers over
// o + k 2^13; every stage after that stays inside one contiguous block of 2^13, which finishes in
// shared memory.

// Host Goldilocks multiplication for the tables (same reduction as `gmul`).
static u64 hmul(u64 a, u64 b) {
    const unsigned __int128 p = (unsigned __int128)a * b;
    const u64 lo = (u64)p, hi = (u64)(p >> 64);
    const u64 hi_hi = hi >> 32, hi_lo = hi & GL_EPS;
    u64 t0 = lo - hi_hi;
    if (lo < hi_hi) t0 -= GL_EPS;
    const u64 t1 = hi_lo * GL_EPS;
    u64 t2 = t0 + t1;
    if (t2 < t0) t2 += GL_EPS;
    return t2 >= GL_P ? t2 - GL_P : t2;
}

// Per (lg_n, rate_bits): scale[j][k] = (s w_N^j)^k for j < R, k < n; tw[k] = w_n^k for k < n / 2.
struct LdeTables {
    u64* scale = nullptr;
    u64* tw = nullptr;
};

static std::mutex lde_tables_mu;
static LdeTables lde_tables[LDE_MAX_LG_N + 1][LDE_MAX_RATE_BITS + 1];

__global__ void coset_lde_kernel(u64*, const u64*, const u64*, const u64*, u32, u32);
__global__ void coset_lde_pass2(u64*, const u64*, u32);

static int lde_tables_get(u32 lg_n, u32 rate_bits, u64 omega_big, u64 shift, LdeTables* out) {
    std::lock_guard<std::mutex> g(lde_tables_mu);
    static bool smem_attr = false;
    if (!smem_attr) {
        int e = (int)cudaFuncSetAttribute(coset_lde_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 << LDE_SMEM_LG);
        if (!e) e = (int)cudaFuncSetAttribute(coset_lde_pass2, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 << LDE_SMEM_LG);
        if (e) return e;
        smem_attr = true;
    }
    LdeTables& t = lde_tables[lg_n][rate_bits];
    if (!t.scale) {
        const u32 n = 1u << lg_n, R = 1u << rate_bits;
        std::vector<u64> scale((size_t)R * n), tw(n / 2 > 0 ? n / 2 : 1);
        u64 sj = shift;  // s w_N^j
        for (u32 j = 0; j < R; j++) {
            u64 p = 1;
            for (u32 k = 0; k < n; k++) {
                scale[(size_t)j * n + k] = p;
                p = hmul(p, sj);
            }
            sj = hmul(sj, omega_big);
        }
        u64 omega_n = omega_big;
        for (u32 b = 0; b < rate_bits; b++) omega_n = hmul(omega_n, omega_n);
        u64 p = 1;
        for (u32 k = 0; k < n / 2; k++) {
            tw[k] = p;
            p = hmul(p, omega_n);
        }
        LdeTables nt;
        int e = (int)cudaMalloc(&nt.scale, scale.size() * 8);
        if (!e) e = (int)cudaMalloc(&nt.tw, tw.size() * 8);
        if (!e) e = (int)cudaMemcpy(nt.scale, scale.data(), scale.size() * 8, cudaMemcpyHostToDevice);
        if (!e) e = (int)cudaMemcpy(nt.tw, tw.data(), tw.size() * 8, cudaMemcpyHostToDevice);
        // Pageable sources: wait for the copies before any stream may read the tables.
        if (!e) e = (int)cudaDeviceSynchronize();
        if (e) {
            cudaFree(nt.scale);
            cudaFree(nt.tw);
            return e;
        }
        t = nt;
    }
    *out = t;
    return 0;
}

// DIF stages with half-size h = 2^(lg_size - 1) .. 1 on `x` (2^lg_size elements in shared memory)
// of a size-2^lg_n NTT: (u, v) -> (u + v, (u - v) w_{2h}^lo) with w_{2h} = w_n^(n / 2h).
__device__ __forceinline__ void dif_stages(u64* x, u32 lg_size, u32 lg_n, const u64* tw) {
    const u32 half = 1u << (lg_size - 1);
    for (u32 lg_h = lg_size; lg_h-- > 0;) {
        const u32 h = 1u << lg_h, shift = lg_n - 1 - lg_h;
        for (u32 b = threadIdx.x; b < half; b += blockDim.x) {
            const u32 lo = b & (h - 1);
            const u32 i = ((b >> lg_h) << (lg_h + 1)) | lo;
            const u64 u = x[i], v = x[i + h];
            x[i] = gadd(u, v);
            x[i + h] = gmul(gsub(u, v), tw[lo << shift]);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ u64* lde_coset_out(u64* dst, u32 col, u32 j, u32 lg_n, u32 rate_bits) {
    const u32 jr = rate_bits == 0 ? 0 : __brev(j) >> (32 - rate_bits);
    return dst + ((size_t)col << (lg_n + rate_bits)) + ((size_t)jr << lg_n);
}

// lg_n <= 13. One block per (coset j, column): scale, size-n DIF NTT in shared memory, store.
__global__ void __launch_bounds__(LDE_THREADS) coset_lde_kernel(u64* dst, const u64* src, const u64* scale,
                                                               const u64* tw, u32 lg_n, u32 rate_bits) {
    extern __shared__ u64 x[];
    const u32 n = 1u << lg_n, j = blockIdx.x, col = blockIdx.y;
    const u64* c = src + (size_t)col * n;
    const u64* sc = scale + (size_t)j * n;
    for (u32 k = threadIdx.x; k < n; k += blockDim.x) x[k] = gmul(gcanon(c[k]), sc[k]);
    __syncthreads();
    dif_stages(x, lg_n, lg_n, tw);
    u64* out = lde_coset_out(dst, col, j, lg_n, rate_bits);
    for (u32 p = threadIdx.x; p < n; p += blockDim.x) out[p] = x[p];
}

// lg_n > 13, pass 1: scale, then the first L1 = lg_n - 13 stages in registers. One thread per
// (offset o < 2^13, coset j, column) over the elements o + k 2^13, k < 2^L1; written in place of
// the coset's output.
template <u32 L1>
__global__ void __launch_bounds__(256) coset_lde_pass1(u64* dst, const u64* src, const u64* scale, const u64* tw,
                                                      u32 rate_bits) {
    constexpr u32 SUB = 1u << LDE_SMEM_LG, K = 1u << L1, LG_N = LDE_SMEM_LG + L1;
    const u32 o = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y, col = blockIdx.z;
    const u64* c = src + ((size_t)col << LG_N);
    const u64* sc = scale + ((size_t)j << LG_N);
    u64 v[K];
#pragma unroll
    for (u32 k = 0; k < K; k++) v[k] = gmul(gcanon(c[o + k * SUB]), sc[o + k * SUB]);
#pragma unroll
    for (int st = L1 - 1; st >= 0; st--) {
        // lg_h = 13 + st; element o + k SUB has lo = (o + k SUB) mod h = o + (k mod 2^st) SUB.
        const u32 shift = LG_N - 1 - (LDE_SMEM_LG + st);
#pragma unroll
        for (u32 k = 0; k < K; k++) {
            if (k & (1u << st)) continue;
            const u32 lo = o + (k & ((1u << st) - 1)) * SUB;
            const u64 u = v[k], w = v[k + (1u << st)];
            v[k] = gadd(u, w);
            v[k + (1u << st)] = gmul(gsub(u, w), tw[lo << shift]);
        }
    }
    u64* out = lde_coset_out(dst, col, j, LG_N, rate_bits);
#pragma unroll
    for (u32 k = 0; k < K; k++) out[o + k * SUB] = v[k];
}

// lg_n > 13, pass 2: the remaining 13 stages of every contiguous block of 2^13, in place.
__global__ void __launch_bounds__(LDE_THREADS) coset_lde_pass2(u64* dst, const u64* tw, u32 lg_n) {
    extern __shared__ u64 x[];
    constexpr u32 SUB = 1u << LDE_SMEM_LG;
    u64* blk = dst + (size_t)blockIdx.x * SUB;
    for (u32 k = threadIdx.x; k < SUB; k += blockDim.x) x[k] = blk[k];
    __syncthreads();
    dif_stages(x, LDE_SMEM_LG, lg_n, tw);
    for (u32 k = threadIdx.x; k < SUB; k += blockDim.x) blk[k] = x[k];
}

// Coset LDE of `cols` canonical coefficient columns (column-major, n each) into `dst` (column-major,
// leaf order, N each). `omega_big` is the primitive N-th root of unity and `shift` the coset shift.
extern "C" int plonky2_coset_lde(u64* dst, const u64* src, u32 lg_n, u32 rate_bits, u32 cols, u64 omega_big, u64 shift,
                                 cudaStream_t stream) {
    if (lg_n < 1 || lg_n > LDE_MAX_LG_N || rate_bits > LDE_MAX_RATE_BITS || cols == 0 || cols > 65535) return -1;
    LdeTables t;
    int e = lde_tables_get(lg_n, rate_bits, omega_big, shift, &t);
    if (e) return e;
    const u32 R = 1u << rate_bits;
    if (lg_n <= LDE_SMEM_LG) {
        coset_lde_kernel<<<dim3(R, cols), LDE_THREADS, (size_t)8 << lg_n, stream>>>(dst, src, t.scale, t.tw, lg_n, rate_bits);
        return (int)cudaGetLastError();
    }
    const dim3 g1((1u << LDE_SMEM_LG) / 256, R, cols);
    switch (lg_n - LDE_SMEM_LG) {
        case 1: coset_lde_pass1<1><<<g1, 256, 0, stream>>>(dst, src, t.scale, t.tw, rate_bits); break;
        case 2: coset_lde_pass1<2><<<g1, 256, 0, stream>>>(dst, src, t.scale, t.tw, rate_bits); break;
        case 3: coset_lde_pass1<3><<<g1, 256, 0, stream>>>(dst, src, t.scale, t.tw, rate_bits); break;
        case 4: coset_lde_pass1<4><<<g1, 256, 0, stream>>>(dst, src, t.scale, t.tw, rate_bits); break;
        case 5: coset_lde_pass1<5><<<g1, 256, 0, stream>>>(dst, src, t.scale, t.tw, rate_bits); break;
        default: return -1;
    }
    // Every column's N = R n outputs are consecutive, so the blocks of all columns are one range.
    const size_t blocks = ((size_t)cols << (lg_n + rate_bits)) >> LDE_SMEM_LG;
    coset_lde_pass2<<<dim3((u32)blocks), LDE_THREADS, (size_t)8 << LDE_SMEM_LG, stream>>>(dst, t.tw, lg_n);
    return (int)cudaGetLastError();
}

// ---- Inverse NTT (values -> coefficients) -----------------------------------------------------

// Natural-order values v_m = p(w_n^m) to natural-order coefficients c_k = n^-1 sum_m v_m w_n^-mk:
// the coset LDE's DIF stages with inverse twiddles (natural in, bit-reversed out), then a
// bit-reversed store scaled by n^-1. lg_n > 13 runs the stages in two passes as the LDE does.

static u64 hpow(u64 b, u64 e) {
    u64 r = 1;
    while (e) {
        if (e & 1) r = hmul(r, b);
        b = hmul(b, b);
        e >>= 1;
    }
    return r;
}

static std::mutex intt_mu;
static u64* intt_tw[LDE_MAX_LG_N + 1];  // w_n^-k, k < n / 2

__global__ void intt_kernel(u64*, const u64*, const u64*, u32, u64);

static int intt_tables_get(u32 lg_n, u64 omega_n, u64** out) {
    std::lock_guard<std::mutex> g(intt_mu);
    static bool smem_attr = false;
    if (!smem_attr) {
        int e = (int)cudaFuncSetAttribute(intt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 << LDE_SMEM_LG);
        if (!e) e = (int)cudaFuncSetAttribute(coset_lde_pass2, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 << LDE_SMEM_LG);
        if (e) return e;
        smem_attr = true;
    }
    if (!intt_tw[lg_n]) {
        const u32 n = 1u << lg_n;
        const u64 inv = hpow(omega_n, n - 1);
        std::vector<u64> tw(n / 2 > 0 ? n / 2 : 1);
        u64 p = 1;
        for (u32 k = 0; k < n / 2; k++) {
            tw[k] = p;
            p = hmul(p, inv);
        }
        u64* d = nullptr;
        int e = (int)cudaMalloc(&d, tw.size() * 8);
        if (!e) e = (int)cudaMemcpy(d, tw.data(), tw.size() * 8, cudaMemcpyHostToDevice);
        if (!e) e = (int)cudaDeviceSynchronize();  // pageable source, see lde_tables_get
        if (e) {
            cudaFree(d);
            return e;
        }
        intt_tw[lg_n] = d;
    }
    *out = intt_tw[lg_n];
    return 0;
}

// lg_n <= 13: one block per column.
__global__ void __launch_bounds__(LDE_THREADS) intt_kernel(u64* dst, const u64* src, const u64* tw, u32 lg_n, u64 n_inv) {
    extern __shared__ u64 x[];
    const u32 n = 1u << lg_n;
    const u64* s = src + ((size_t)blockIdx.x << lg_n);
    for (u32 k = threadIdx.x; k < n; k += blockDim.x) x[k] = gcanon(s[k]);
    __syncthreads();
    dif_stages(x, lg_n, lg_n, tw);
    u64* d = dst + ((size_t)blockIdx.x << lg_n);
    for (u32 k = threadIdx.x; k < n; k += blockDim.x) d[k] = gmul(x[__brev(k) >> (32 - lg_n)], n_inv);
}

// lg_n > 13, pass 1: the first lg_n - 13 stages in registers, src -> dst.
template <u32 L1>
__global__ void __launch_bounds__(256) intt_pass1(u64* dst, const u64* src, const u64* tw) {
    constexpr u32 SUB = 1u << LDE_SMEM_LG, K = 1u << L1, LG_N = LDE_SMEM_LG + L1;
    const u32 o = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    const u64* c = src + ((size_t)col << LG_N);
    u64 v[K];
#pragma unroll
    for (u32 k = 0; k < K; k++) v[k] = gcanon(c[o + k * SUB]);
#pragma unroll
    for (int st = L1 - 1; st >= 0; st--) {
        const u32 shift = LG_N - 1 - (LDE_SMEM_LG + st);
#pragma unroll
        for (u32 k = 0; k < K; k++) {
            if (k & (1u << st)) continue;
            const u32 lo = o + (k & ((1u << st) - 1)) * SUB;
            const u64 u = v[k], w = v[k + (1u << st)];
            v[k] = gadd(u, w);
            v[k + (1u << st)] = gmul(gsub(u, w), tw[lo << shift]);
        }
    }
    u64* out = dst + ((size_t)col << LG_N);
#pragma unroll
    for (u32 k = 0; k < K; k++) out[o + k * SUB] = v[k];
}

// lg_n > 13, last step: in-place bit-reversal permutation (an involution: swap k with bitrev(k)
// once) scaled by n^-1.
__global__ void bitrev_scale_kernel(u64* d, u32 lg_n, size_t total, u64 n_inv) {
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    const u32 k = (u32)(g & ((1u << lg_n) - 1));
    const u32 r = __brev(k) >> (32 - lg_n);
    if (k > r) return;
    u64* col = d + (g - k);
    const u64 a = col[k], b = col[r];
    col[k] = gmul(b, n_inv);
    if (k != r) col[r] = gmul(a, n_inv);
}

// Inverse NTT of `cols` columns of 2^lg_n canonical values (column-major) into `dst`; `omega_n`
// is the primitive n-th root of unity. `dst` must not overlap `src`.
extern "C" int plonky2_intt(u64* dst, const u64* src, u32 lg_n, u32 cols, u64 omega_n, cudaStream_t stream) {
    if (lg_n < 1 || lg_n > LDE_MAX_LG_N || cols == 0 || cols > 65535) return -1;
    u64* tw;
    int e = intt_tables_get(lg_n, omega_n, &tw);
    if (e) return e;
    const u64 n_inv = hpow(1ull << lg_n, GL_P - 2);
    if (lg_n <= LDE_SMEM_LG) {
        intt_kernel<<<cols, LDE_THREADS, (size_t)8 << lg_n, stream>>>(dst, src, tw, lg_n, n_inv);
        return (int)cudaGetLastError();
    }
    const dim3 g1((1u << LDE_SMEM_LG) / 256, cols);
    switch (lg_n - LDE_SMEM_LG) {
        case 1: intt_pass1<1><<<g1, 256, 0, stream>>>(dst, src, tw); break;
        case 2: intt_pass1<2><<<g1, 256, 0, stream>>>(dst, src, tw); break;
        case 3: intt_pass1<3><<<g1, 256, 0, stream>>>(dst, src, tw); break;
        case 4: intt_pass1<4><<<g1, 256, 0, stream>>>(dst, src, tw); break;
        case 5: intt_pass1<5><<<g1, 256, 0, stream>>>(dst, src, tw); break;
        default: return -1;
    }
    const size_t blocks = ((size_t)cols << lg_n) >> LDE_SMEM_LG;
    coset_lde_pass2<<<dim3((u32)blocks), LDE_THREADS, (size_t)8 << LDE_SMEM_LG, stream>>>(dst, tw, lg_n);
    const size_t total = (size_t)cols << lg_n;
    bitrev_scale_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(dst, lg_n, total, n_inv);
    return (int)cudaGetLastError();
}

// Coset inverse NTT: natural-order values at x_i = s w_n^i to coefficients. The inverse NTT gives
// c_k s^k; each column is then scaled by s^-k (table per lg_n, built once).
static u64* coset_unscale[LDE_MAX_LG_N + 1];

__global__ void scale_columns_kernel(u64* d, const u64* tab, u32 lg_n, size_t total) {
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    d[g] = gmul(d[g], tab[g & ((1u << lg_n) - 1)]);
}

extern "C" int plonky2_coset_intt(u64* dst, const u64* src, u32 lg_n, u32 cols, u64 omega_n, u64 shift_inv,
                                  cudaStream_t stream) {
    int e = plonky2_intt(dst, src, lg_n, cols, omega_n, stream);
    if (e) return e;
    u64* tab;
    {
        std::lock_guard<std::mutex> g(intt_mu);
        if (!coset_unscale[lg_n]) {
            const u32 n = 1u << lg_n;
            std::vector<u64> t(n);
            u64 p = 1;
            for (u32 k = 0; k < n; k++) {
                t[k] = p;
                p = hmul(p, shift_inv);
            }
            u64* d = nullptr;
            e = (int)cudaMalloc(&d, (size_t)n * 8);
            if (!e) e = (int)cudaMemcpy(d, t.data(), (size_t)n * 8, cudaMemcpyHostToDevice);
            if (!e) e = (int)cudaDeviceSynchronize();  // pageable source, see lde_tables_get
            if (e) {
                cudaFree(d);
                return e;
            }
            coset_unscale[lg_n] = d;
        }
        tab = coset_unscale[lg_n];
    }
    const size_t total = (size_t)cols << lg_n;
    scale_columns_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(dst, tab, lg_n, total);
    return (int)cudaGetLastError();
}

// Column-major [cols][rows] -> row-major [rows][cols], 32 x 32 tiles.
__global__ void transpose_kernel(u64* dst, const u64* src, u32 rows, u32 cols) {
    __shared__ u64 tile[32][33];
    const u32 r0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
    for (u32 k = threadIdx.y; k < 32; k += blockDim.y) {
        const u32 c = c0 + k, r = r0 + threadIdx.x;
        if (c < cols && r < rows) tile[k][threadIdx.x] = src[(size_t)c * rows + r];
    }
    __syncthreads();
    for (u32 k = threadIdx.y; k < 32; k += blockDim.y) {
        const u32 r = r0 + k, c = c0 + threadIdx.x;
        if (c < cols && r < rows) dst[(size_t)r * cols + c] = tile[threadIdx.x][k];
    }
}

extern "C" int plonky2_transpose(u64* dst, const u64* src, u32 rows, u32 cols, cudaStream_t stream) {
    transpose_kernel<<<dim3((rows + 31) / 32, (cols + 31) / 32), dim3(32, 8), 0, stream>>>(dst, src, rows, cols);
    return (int)cudaGetLastError();
}

// ---- FRI final polynomial ---------------------------------------------------------------------

// plonky2's `prove_openings` reduces the opened polynomials of each batch b with powers of alpha
// (C_b = sum_j alpha^j p_j), divides by X - z_b and sums the batches, earlier ones scaled by
// alpha^(L_b) (L_b: polynomials in later batches). At every LDE point this is
//   V(x) = sum_b alpha^(L_b) (C_b(x) - C_b(z_b)) / (x - z_b),
// evaluated here from the LDE columns (leaf order) with C_b(z_b) from the opening values. One
// thread per leaf position t (point i = bitrev(t)); the output is in leaf order, (a, b) per value.
struct FriArgs {
    const u64* const* cols;  // each term's LDE column
    const u64* coef;         // each term's alpha^j, (a, b)
    u64* out;                // [N][2]
    u64 shift;               // coset shift
    u64 omega;               // primitive N-th root of unity
    u64 z[2][2];             // opening point of each batch
    u64 cz[2][2];            // C_b(z_b)
    u64 scale[2][2];         // alpha^(L_b)
    u32 start[3];            // terms of batch b: start[b] .. start[b + 1]
    u32 lg_lde;
    u32 num_batches;
};

__global__ void __launch_bounds__(128) fri_values_kernel(const FriArgs a) {
    const u32 N = 1u << a.lg_lde;
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    const u32 i = __brev(t) >> (32 - a.lg_lde);
    const u64 x = gmul(a.shift, gpow(a.omega, i));
    E2 v = {0, 0};
    for (u32 b = 0; b < a.num_batches; b++) {
        Acc160 sa{0, 0, 0}, sb{0, 0, 0};
        for (u32 k = a.start[b]; k < a.start[b + 1]; k++) {
            const u64 p = gcanon(a.cols[k][t]);
            acc_mul(sa, p, a.coef[2 * k]);
            acc_mul(sb, p, a.coef[2 * k + 1]);
        }
        const E2 num = esub(E2{reduce160(sa), reduce160(sb)}, E2{a.cz[b][0], a.cz[b][1]});
        // 1 / (c + d X) = (c - d X) / (c^2 - 7 d^2); d = -z.b != 0 (checked by the host).
        const E2 den = {gsub(x, a.z[b][0]), gsub(0, a.z[b][1])};
        const u64 ninv = ginv(gsub(gmul(den.a, den.a), gmul(EXT_W, gmul(den.b, den.b))));
        const E2 inv = {gmul(den.a, ninv), gmul(gsub(0, den.b), ninv)};
        v = eadd(v, emul(emul(num, inv), E2{a.scale[b][0], a.scale[b][1]}));
    }
    a.out[2 * (size_t)t] = v.a;
    a.out[2 * (size_t)t + 1] = v.b;
}

extern "C" int plonky2_fri_values_launch(const FriArgs* args, cudaStream_t stream) {
    if (args->num_batches > 2 || args->lg_lde < 1) return -1;
    const u32 N = 1u << args->lg_lde;
    fri_values_kernel<<<(N + 127) / 128, 128, 0, stream>>>(*args);
    return (int)cudaGetLastError();
}
// ---- Permutation partial products --------------------------------------------------------------

// `wires_permutation_partial_products_and_zs` for both challenges. Row i (x = w_n^i) splits the
// routed wires into chunks of `degree`; chunk k's quotient is
//   q_k(i) = prod_j (w_j(i) + beta k_j x + gamma) / prod_j (w_j(i) + beta sigma_j(i) + gamma).
// Z(0) = 1 and Z(i + 1) = Z(i) prod_k q_k(i); the partial products are Z(i) prod_{t <= k} q_t(i)
// for k < num_prods. Field arithmetic is exact, so the values equal the CPU's.
struct PpArgs {
    const u64* wires;   // [num_routed][n] subgroup values
    const u64* sigmas;  // [n][num_routed]
    const u64* k_is;    // [num_routed]
    u64* q;             // scratch [NC][num_chunks][n]: chunk quotients
    u64* row;           // scratch [NC][n]: row products, then Z
    u64* block;         // scratch [NC][blocks]: per-block products for the scan
    u64* out;           // [NC + NC num_prods][n]: Z per challenge, then partial products by challenge
    u64 beta[2], gamma[2];
    u64 omega;          // primitive n-th root of unity
    u32 lg_n, num_routed, degree, num_chunks, num_prods;
};

#define PP_MAX_CHUNKS 32

__global__ void __launch_bounds__(128) pp_quotients_kernel(const PpArgs a) {
    const u32 n = 1u << a.lg_n;
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (i >= n) return;
    const u64 x = gpow(a.omega, i);
    const u64 beta = a.beta[c], gamma = a.gamma[c];
    const u64* sig = a.sigmas + (size_t)i * a.num_routed;
    u64 num[PP_MAX_CHUNKS], den[PP_MAX_CHUNKS];
    for (u32 k = 0; k < a.num_chunks; k++) {
        u64 np = 1, dp = 1;
        const u32 hi = min((k + 1) * a.degree, a.num_routed);
        for (u32 j = k * a.degree; j < hi; j++) {
            const u64 w = gcanon(a.wires[(size_t)j * n + i]);
            np = gmul(np, gadd(gadd(w, gmul(beta, gmul(a.k_is[j], x))), gamma));
            dp = gmul(dp, gadd(gadd(w, gmul(beta, gcanon(sig[j]))), gamma));
        }
        num[k] = np;
        den[k] = dp;
    }
    // Montgomery's trick: one inversion for the row's denominators.
    u64 pre[PP_MAX_CHUNKS];
    u64 acc = 1;
    for (u32 k = 0; k < a.num_chunks; k++) {
        pre[k] = acc;
        acc = gmul(acc, den[k]);
    }
    u64 inv = ginv(acc);
    u64 r = 1;
    for (u32 k = a.num_chunks; k-- > 0;) {
        const u64 qk = gmul(num[k], gmul(inv, pre[k]));
        inv = gmul(inv, den[k]);
        a.q[((size_t)c * a.num_chunks + k) * n + i] = qk;
        r = gmul(r, qk);
    }
    a.row[(size_t)c * n + i] = r;
}

// Exclusive prefix product of row[c][0..n) in blocks of PP_SCAN elements: each block scans its
// slice in place (exclusive) and records its total; the totals are scanned by one block; then each
// block multiplies its slice by its prefix.
#define PP_SCAN 1024
__device__ void block_exclusive_scan(u64* s, u32 len) {
    // Hillis-Steele inclusive scan in shared memory, then shifted.
    for (u32 off = 1; off < len; off <<= 1) {
        u64 v = 1;
        if (threadIdx.x < len && threadIdx.x >= off) v = s[threadIdx.x - off];
        __syncthreads();
        if (threadIdx.x < len && threadIdx.x >= off) s[threadIdx.x] = gmul(s[threadIdx.x], v);
        __syncthreads();
    }
}

__global__ void __launch_bounds__(PP_SCAN) pp_scan_blocks(u64* row, u64* block, u32 lg_n, u32 blocks) {
    __shared__ u64 s[PP_SCAN];
    const u32 n = 1u << lg_n, c = blockIdx.y, b = blockIdx.x;
    const u32 len = min((u32)PP_SCAN, n);
    u64* r = row + (size_t)c * n + (size_t)b * PP_SCAN;
    if (threadIdx.x < len) s[threadIdx.x] = r[threadIdx.x];
    __syncthreads();
    block_exclusive_scan(s, len);  // inclusive
    if (threadIdx.x < len) r[threadIdx.x] = threadIdx.x == 0 ? 1 : s[threadIdx.x - 1];
    if (threadIdx.x == len - 1) block[(size_t)c * blocks + b] = s[len - 1];
}

__global__ void __launch_bounds__(PP_SCAN) pp_scan_totals(u64* block, u32 blocks) {
    __shared__ u64 s[PP_SCAN];
    const u32 c = blockIdx.x;
    u64* t = block + (size_t)c * blocks;
    if (threadIdx.x < blocks) s[threadIdx.x] = t[threadIdx.x];
    __syncthreads();
    block_exclusive_scan(s, blocks);
    if (threadIdx.x < blocks) t[threadIdx.x] = threadIdx.x == 0 ? 1 : s[threadIdx.x - 1];
}

// Z(i) = block prefix * in-block exclusive prefix; partial products from Z and the quotients.
__global__ void __launch_bounds__(128) pp_outputs_kernel(const PpArgs a, u32 blocks) {
    const u32 n = 1u << a.lg_n;
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (i >= n) return;
    u64 z = a.row[(size_t)c * n + i];
    if (blocks > 1) z = gmul(z, a.block[(size_t)c * blocks + i / PP_SCAN]);
    a.out[(size_t)c * n + i] = z;
    u64 acc = z;
    for (u32 k = 0; k < a.num_prods; k++) {
        acc = gmul(acc, a.q[((size_t)c * a.num_chunks + k) * n + i]);
        a.out[((size_t)2 + (size_t)c * a.num_prods + k) * n + i] = acc;
    }
}

// Runs the kernels on `stream`; scratch pointers in `args` must hold the sizes noted there, with
// `block` holding NC * ceil(n / PP_SCAN) values. n / PP_SCAN must be at most PP_SCAN (n <= 2^20).
extern "C" int plonky2_partial_products(const PpArgs* args, cudaStream_t stream) {
    const PpArgs& a = *args;
    if (a.num_chunks > PP_MAX_CHUNKS || a.num_chunks == 0 || a.num_prods + 1 != a.num_chunks || a.lg_n < 1) return -1;
    const u32 n = 1u << a.lg_n;
    const u32 blocks = (n + PP_SCAN - 1) / PP_SCAN;
    if (blocks > PP_SCAN) return -1;
    pp_quotients_kernel<<<dim3((n + 127) / 128, 2), 128, 0, stream>>>(a);
    pp_scan_blocks<<<dim3(blocks, 2), PP_SCAN, 0, stream>>>(a.row, a.block, a.lg_n, blocks);
    if (blocks > 1) pp_scan_totals<<<2, PP_SCAN, 0, stream>>>(a.block, blocks);
    pp_outputs_kernel<<<dim3((n + 127) / 128, 2), 128, 0, stream>>>(a, blocks);
    return (int)cudaGetLastError();
}
// ---- Openings from the LDE's first coset -------------------------------------------------------

// A polynomial p of degree < n is determined by its values on the coset sH, which are the first n
// leaf positions of its leaf-order LDE: position t holds p(x_t), x_t = s w_n^bitrev(t). Lagrange
// interpolation on sH gives
//   p(z) = sum_t p(x_t) W_t,   W_t = (z^n - s^n) / (n s^n) * x_t / (z - x_t),
// exact for z outside sH (the host checks z is not in the base field). The weights depend only on
// z, so each polynomial costs one dot product.
struct OpenArgs {
    const u64* const* cols;  // each term's LDE column (leaf order)
    const u32* point;        // each term's point index (0 or 1)
    u64* weights;            // [2][n][2]
    u64* out;                // [terms][2]
    u64 z[2][2];             // the points
    u64 c[2][2];             // (z^n - s^n) / (n s^n) per point
    u64 shift, omega;        // coset shift, primitive n-th root of unity
    u32 lg_n, num_points, num_terms;
};

__global__ void __launch_bounds__(256) open_weights_kernel(const OpenArgs a) {
    const u32 n = 1u << a.lg_n;
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x, p = blockIdx.y;
    if (t >= n) return;
    const u64 x = gmul(a.shift, gpow(a.omega, __brev(t) >> (32 - a.lg_n)));
    const E2 den = {gsub(a.z[p][0], x), a.z[p][1]};
    const u64 ninv = ginv(gsub(gmul(den.a, den.a), gmul(EXT_W, gmul(den.b, den.b))));
    const E2 inv = {gmul(den.a, ninv), gmul(gsub(0, den.b), ninv)};
    const E2 w = escal(emul(inv, E2{a.c[p][0], a.c[p][1]}), x);
    u64* o = a.weights + ((size_t)p * n + t) * 2;
    o[0] = w.a;
    o[1] = w.b;
}

// One block per term: sum_t col[t] W_t, products summed in 160 bits per component per thread.
__global__ void __launch_bounds__(256) open_dot_kernel(const OpenArgs a) {
    __shared__ u64 sa[256], sb[256];
    const u32 n = 1u << a.lg_n, k = blockIdx.x;
    const u64* col = a.cols[k];
    const u64* w = a.weights + (size_t)a.point[k] * n * 2;
    Acc160 xa{0, 0, 0}, xb{0, 0, 0};
    for (u32 t = threadIdx.x; t < n; t += blockDim.x) {
        const u64 v = gcanon(col[t]);
        acc_mul(xa, v, w[2 * t]);
        acc_mul(xb, v, w[2 * t + 1]);
    }
    sa[threadIdx.x] = reduce160(xa);
    sb[threadIdx.x] = reduce160(xb);
    __syncthreads();
    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sa[threadIdx.x] = gadd(sa[threadIdx.x], sa[threadIdx.x + s]);
            sb[threadIdx.x] = gadd(sb[threadIdx.x], sb[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        a.out[2 * k] = sa[0];
        a.out[2 * k + 1] = sb[0];
    }
}

extern "C" int plonky2_openings(const OpenArgs* args, cudaStream_t stream) {
    const OpenArgs& a = *args;
    if (a.num_points < 1 || a.num_points > 2 || a.lg_n < 1 || a.num_terms == 0) return -1;
    const u32 n = 1u << a.lg_n;
    open_weights_kernel<<<dim3((n + 255) / 256, a.num_points), 256, 0, stream>>>(a);
    open_dot_kernel<<<a.num_terms, 256, 0, stream>>>(a);
    return (int)cudaGetLastError();
}
