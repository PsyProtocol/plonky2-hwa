// GPU evaluation of plonky2 quotient polynomials (lab).
//
// One thread per point of the quotient domain (the coset LDE of size N = n << rate_bits). Each
// thread evaluates exactly what `eval_vanishing_poly_base_batch` evaluates at its point: the
// `L_0(x) (Z(x) - 1)` terms, the permutation partial-product checks, and the filtered gate
// constraints, combined with powers of each alpha and divided by `Z_H(x)`. Field arithmetic is
// exact, so the result equals the CPU result element for element (all values are kept canonical).
//
// Inputs are the LDEs of the prover's polynomial batches, column-major in natural order:
// column `c` at point `i` is `lde[c * N + i]`, the value at `x_i = shift * omega^i`.

#include <cstdint>
#include <cuda_runtime.h>

typedef unsigned long long u64;
typedef unsigned int u32;

#define GL_P 0xFFFFFFFF00000001ull
#define GL_EPS 0xFFFFFFFFull
#define EXT_W 7ull  // GoldilocksField quadratic extension: X^2 = 7

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
    int debug_point;   // -1: off
};

// LDE values from the NTT are reduced, but canonicalise on load anyway: the field functions above
// assume canonical inputs.
__device__ __forceinline__ u64 gcanon(u64 x) { return x >= GL_P ? x - GL_P : x; }

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

// Receives one gate's constraints in order.
struct Sink {
    Acc& acc;
    u64 filter;
    u32 m;        // index of the gate's next constraint among all vanishing terms
    u64* dbg;     // non-null: also store the unfiltered value
    u32 c;
    __device__ __forceinline__ void one(u64 v) {
        if (dbg) dbg[c] = v;
        c++;
        acc.add(m++, gmul(v, filter));
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

__global__ void quotient_kernel(const QuotientArgs a) {
    const u32 N = 1u << a.lg_lde;
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Point p{a, N, i};
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
    for (u32 c = 0; c < NC; c++) acc.add(m++, gmul(l0, gsub(p.z(c, i), 1)));

    const u32 i_next = (i + a.next_step) & (N - 1);
    const u32 num_chunks = (a.num_routed + a.max_degree - 1) / a.max_degree;
    for (u32 c = 0; c < NC; c++) {
        const u64 beta = betas[c], gamma = gammas[c];
        u64 prev = p.z(c, i);
        for (u32 chunk = 0; chunk < num_chunks; chunk++) {
            u64 num = 1, den = 1;
            u32 hi = min((chunk + 1) * a.max_degree, a.num_routed);
            for (u32 j = chunk * a.max_degree; j < hi; j++) {
                u64 w = p.wire(j);
                num = gmul(num, gadd(gadd(w, gmul(beta, gmul(k_is[j], x))), gamma));
                den = gmul(den, gadd(gadd(w, gmul(beta, p.sigma(j))), gamma));
            }
            u64 next = chunk < a.num_prods ? p.z(NC + c * a.num_prods + chunk, i) : p.z(c, i_next);
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
        Sink sink{acc, filter, m, dbg ? a.debug + (size_t)gi * a.debug_stride : nullptr, 0};
        eval_gate(p, g, gate_data, sink);
    }

    const u64 zh_inv = a.zh_invs[r];
    for (u32 c = 0; c < NC; c++) a.out[(size_t)c * N + i] = gmul(acc.v[c], zh_inv);
}

extern "C" int plonky2_quotient_init(const u64* poseidon_constants, u32 len) {
    if (len != POS_LEN) return -1;
    return (int)cudaMemcpyToSymbol(c_pos, poseidon_constants, sizeof(u64) * POS_LEN);
}

extern "C" u32 plonky2_quotient_poseidon_constants_len() { return POS_LEN; }

extern "C" int plonky2_quotient_launch(const QuotientArgs* args) {
    const u32 N = 1u << args->lg_lde;
    const u32 threads = 128;
    quotient_kernel<<<(N + threads - 1) / threads, threads>>>(*args);
    return (int)cudaGetLastError();
}
