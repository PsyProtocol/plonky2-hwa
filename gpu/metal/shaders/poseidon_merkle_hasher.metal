#include <metal_stdlib>

#include "goldilocks.metal"

#include "poseidon_goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;


inline uint compute_digest_index(uint tree_length, uint num_layers, uint level_in, uint index_in) {
    uint remaining_layers = num_layers - level_in;
    uint tree_index = index_in >> remaining_layers;
    uint pair_index = index_in & ((1u << remaining_layers) - 1);
    uint parity = pair_index & 1;
    pair_index >>=1;

    uint d_index = (pair_index << (level_in + 2)) + (1u << (level_in + 1)) - 2 + parity;
    return d_index + tree_index * tree_length;

}

struct Uniforms {
    uint level;
    uint tree_length;
    uint num_layers;
    uint leaf_size;
    uint tree_height;
};

inline uint reverse_bits(uint value, uint num_bits) {
    value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
    value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
    value = ((value & 0x00ff00ffu) << 8) | ((value >> 8) & 0x00ff00ffu);
    value = (value << 16) | (value >> 16);
    return value >> (32 - num_bits);
}

inline Fp fp_sub(Fp lhs, Fp rhs) {
    ulong a = static_cast<ulong>(lhs);
    ulong b = static_cast<ulong>(rhs);
    if (a >= b) {
        return Fp(a - b);
    }
    return Fp(a - b - 0xfffffffful);
}

struct FftUniforms {
    uint poly_count;
    uint fft_len;
    uint log_fft_len;
    uint zero_factor;
    uint half_m;
};

struct FftPairUniforms {
    uint poly_count;
    uint fft_len;
    uint half_m;
};

struct FftTripleUniforms {
    uint poly_count;
    uint fft_len;
    uint half_m;
};

struct CosetLdeUniforms {
    uint poly_count;
    uint coeff_len;
    uint fft_len;
    uint log_fft_len;
    uint zero_factor;
};

struct LocalCosetLdeUniforms {
    uint poly_count;
    uint coeff_len;
    uint fft_len;
    uint log_fft_len;
    uint zero_factor;
    uint tile_log;
};

struct SingleCosetLdeUniforms {
    uint poly_count;
    uint coeff_len;
    uint fft_len;
    uint log_fft_len;
    uint zero_factor;
    uint thread_count;
};

struct QuotientPermUniforms {
    uint point_count;
    uint leaf_count;
    uint step;
    uint next_step;
    uint num_constants;
    uint num_routed_wires;
    uint num_challenges;
    uint num_partial_products;
    uint max_degree;
    uint alpha_count;
};

constant uint MAX_QUOTIENT_ALPHAS = 64;

inline void quotient_accum_term(
    Fp term,
    constant Fp * alphas,
    thread Fp * accum,
    thread Fp * powers,
    uint alpha_count
) {
    for (uint i = 0; i < alpha_count; i++) {
        Fp alpha = alphas[i];
        accum[i] = accum[i] + term * powers[i];
        powers[i] = powers[i] * alpha;
    }
}

kernel void quotient_perm_reduce(
    constant Fp * wires_lde[[buffer(0)]],
    constant Fp * constants_sigmas_lde[[buffer(1)]],
    constant Fp * zpp_lde[[buffer(2)]],
    constant Fp * shifted_xs[[buffer(3)]],
    constant Fp * l0_values[[buffer(4)]],
    constant Fp * k_is[[buffer(5)]],
    constant Fp * betas[[buffer(6)]],
    constant Fp * gammas[[buffer(7)]],
    constant Fp * alphas[[buffer(8)]],
    device Fp * output[[buffer(9)]],
    constant QuotientPermUniforms & uniforms[[buffer(10)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint point = gid[1] * 32768 + gid[0];
    if (point >= uniforms.point_count || uniforms.alpha_count > MAX_QUOTIENT_ALPHAS) {
        return;
    }

    uint lde_point = point * uniforms.step;
    uint next_lde_point = ((point + uniforms.next_step) % uniforms.point_count) * uniforms.step;
    Fp x = shifted_xs[point];
    Fp l0_x = l0_values[point];

    Fp accum[MAX_QUOTIENT_ALPHAS];
    Fp powers[MAX_QUOTIENT_ALPHAS];
    for (uint a = 0; a < uniforms.alpha_count; a++) {
        accum[a] = Fp(0);
        powers[a] = Fp(1);
    }

    for (uint challenge = 0; challenge < uniforms.num_challenges; challenge++) {
        Fp z_x = zpp_lde[challenge * uniforms.leaf_count + lde_point];
        quotient_accum_term(l0_x * fp_sub(z_x, Fp(1)), alphas, accum, powers, uniforms.alpha_count);
    }

    for (uint challenge = 0; challenge < uniforms.num_challenges; challenge++) {
        Fp beta = betas[challenge];
        Fp gamma = gammas[challenge];
        Fp z_x = zpp_lde[challenge * uniforms.leaf_count + lde_point];
        Fp z_gx = zpp_lde[challenge * uniforms.leaf_count + next_lde_point];

        for (uint chunk = 0; chunk <= uniforms.num_partial_products; chunk++) {
            uint start_wire = chunk * uniforms.max_degree;
            uint end_wire = min(start_wire + uniforms.max_degree, uniforms.num_routed_wires);
            Fp numerator_product = Fp(1);
            Fp denominator_product = Fp(1);

            for (uint wire = start_wire; wire < end_wire; wire++) {
                Fp wire_value = wires_lde[wire * uniforms.leaf_count + lde_point];
                Fp k_i = k_is[wire];
                Fp s_id = k_i * x;
                Fp s_sigma = constants_sigmas_lde[
                    (uniforms.num_constants + wire) * uniforms.leaf_count + lde_point
                ];
                numerator_product = numerator_product * (wire_value + beta * s_id + gamma);
                denominator_product = denominator_product * (wire_value + beta * s_sigma + gamma);
            }

            Fp prev_acc = chunk == 0
                ? z_x
                : zpp_lde[
                    (uniforms.num_challenges
                        + challenge * uniforms.num_partial_products
                        + chunk
                        - 1) * uniforms.leaf_count + lde_point
                ];
            Fp next_acc = chunk < uniforms.num_partial_products
                ? zpp_lde[
                    (uniforms.num_challenges
                        + challenge * uniforms.num_partial_products
                        + chunk) * uniforms.leaf_count + lde_point
                ]
                : z_gx;
            Fp term = fp_sub(prev_acc * numerator_product, next_acc * denominator_product);
            quotient_accum_term(term, alphas, accum, powers, uniforms.alpha_count);
        }
    }

    uint output_offset = point * uniforms.alpha_count;
    for (uint a = 0; a < uniforms.alpha_count; a++) {
        output[output_offset + a] = accum[a];
    }
}

kernel void fft_init_reverse_zero(
    constant Fp * input[[buffer(0)]],
    device Fp * values[[buffer(1)]],
    constant FftUniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];
    uint total = uniforms.poly_count * uniforms.fft_len;
    if (index >= total) {
        return;
    }

    uint point = index % uniforms.fft_len;
    uint poly = index / uniforms.fft_len;
    uint mask = uniforms.zero_factor >= 32 ? 0u : (0xffffffffu << uniforms.zero_factor);
    uint input_point = reverse_bits(point & mask, uniforms.log_fft_len);

    values[index] = input[poly * uniforms.fft_len + input_point];
}

kernel void fft_init_coset_lde(
    constant Fp * coeffs[[buffer(0)]],
    constant Fp * shift_powers[[buffer(1)]],
    device Fp * values[[buffer(2)]],
    constant CosetLdeUniforms & uniforms[[buffer(3)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];
    uint total = uniforms.poly_count * uniforms.fft_len;
    if (index >= total) {
        return;
    }

    uint point = index % uniforms.fft_len;
    uint poly = index / uniforms.fft_len;
    uint mask = uniforms.zero_factor >= 32 ? 0u : (0xffffffffu << uniforms.zero_factor);
    uint input_point = reverse_bits(point & mask, uniforms.log_fft_len);

    Fp coeff = coeffs[poly * uniforms.coeff_len + input_point];
    Fp shift = shift_powers[input_point];
    values[index] = coeff * shift;
}

kernel void fft_coset_lde_single(
    constant Fp * coeffs[[buffer(0)]],
    constant Fp * shift_powers[[buffer(1)]],
    device Fp * values[[buffer(2)]],
    constant Fp * flat_roots[[buffer(3)]],
    constant SingleCosetLdeUniforms & uniforms[[buffer(4)]],
    uint tid[[thread_index_in_threadgroup]],
    uint2 tgid[[threadgroup_position_in_grid]]
) {
    uint poly = tgid[1] * 32768 + tgid[0];
    if (poly >= uniforms.poly_count) {
        return;
    }

    uint mask = uniforms.zero_factor >= 32 ? 0u : (0xffffffffu << uniforms.zero_factor);
    uint poly_offset = poly * uniforms.fft_len;
    uint coeff_offset = poly * uniforms.coeff_len;
    for (uint point = tid; point < uniforms.fft_len; point += uniforms.thread_count) {
        uint input_point = reverse_bits(point & mask, uniforms.log_fft_len);
        Fp coeff = coeffs[coeff_offset + input_point];
        Fp shift = shift_powers[input_point];
        values[poly_offset + point] = coeff * shift;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint stage = uniforms.zero_factor; stage < uniforms.log_fft_len; stage++) {
        uint half_m = 1u << stage;
        uint m = half_m << 1;
        uint root_offset = stage == 0 ? 0 : (1u << stage);

        for (uint k = tid; k < (uniforms.fft_len >> 1); k += uniforms.thread_count) {
            uint block = k / half_m;
            uint j = k - block * half_m;
            uint left = poly_offset + block * m + j;
            uint right = left + half_m;

            Fp u = values[left];
            Fp omega = flat_roots[root_offset + j];
            Fp right_value = values[right];
            Fp t = omega * right_value;
            values[left] = u + t;
            values[right] = fp_sub(u, t);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void fft_init_coset_lde_local(
    constant Fp * coeffs[[buffer(0)]],
    constant Fp * shift_powers[[buffer(1)]],
    device Fp * values[[buffer(2)]],
    constant Fp * flat_roots[[buffer(3)]],
    constant LocalCosetLdeUniforms & uniforms[[buffer(4)]],
    threadgroup Fp * scratch[[threadgroup(0)]],
    uint tid[[thread_index_in_threadgroup]],
    uint2 tgid[[threadgroup_position_in_grid]]
) {
    uint group_index = tgid[1] * 32768 + tgid[0];
    uint tile_len = 1u << uniforms.tile_log;
    uint half_tile_len = tile_len >> 1;
    uint tiles_per_poly = uniforms.fft_len >> uniforms.tile_log;
    uint total_groups = uniforms.poly_count * tiles_per_poly;
    if (group_index >= total_groups) {
        return;
    }

    uint poly = group_index / tiles_per_poly;
    uint tile = group_index - poly * tiles_per_poly;
    uint tile_base = tile * tile_len;
    uint mask = uniforms.zero_factor >= 32 ? 0u : (0xffffffffu << uniforms.zero_factor);

    for (uint lane = tid; lane < tile_len; lane += half_tile_len) {
        uint point = tile_base + lane;
        uint input_point = reverse_bits(point & mask, uniforms.log_fft_len);
        Fp coeff = coeffs[poly * uniforms.coeff_len + input_point];
        Fp shift = shift_powers[input_point];
        scratch[lane] = coeff * shift;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stage = uniforms.zero_factor; stage < uniforms.tile_log; stage++) {
        uint half_m = 1u << stage;
        uint m = half_m << 1;
        uint block = tid / half_m;
        uint j = tid - block * half_m;
        uint left = block * m + j;
        uint right = left + half_m;
        uint root_offset = stage == 0 ? 0 : (1u << stage);

        Fp u = scratch[left];
        Fp omega = flat_roots[root_offset + j];
        Fp right_value = scratch[right];
        Fp t = omega * right_value;
        scratch[left] = u + t;
        scratch[right] = fp_sub(u, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint output_offset = poly * uniforms.fft_len + tile_base;
    for (uint lane = tid; lane < tile_len; lane += half_tile_len) {
        values[output_offset + lane] = scratch[lane];
    }
}

kernel void fft_stage(
    device Fp * values[[buffer(0)]],
    constant Fp * roots[[buffer(1)]],
    constant FftUniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];
    uint butterflies_per_poly = uniforms.fft_len >> 1;
    uint total = uniforms.poly_count * butterflies_per_poly;
    if (index >= total) {
        return;
    }

    uint poly = index / butterflies_per_poly;
    uint k = index - poly * butterflies_per_poly;
    uint half_m = uniforms.half_m;
    uint m = half_m << 1;
    uint block = k / half_m;
    uint j = k - block * half_m;
    uint left = poly * uniforms.fft_len + block * m + j;
    uint right = left + half_m;

    Fp u = values[left];
    Fp omega = roots[j];
    Fp right_value = values[right];
    Fp t = omega * right_value;
    values[left] = u + t;
    values[right] = fp_sub(u, t);
}

kernel void fft_stage_pair(
    device Fp * values[[buffer(0)]],
    constant Fp * roots0[[buffer(1)]],
    constant Fp * roots1[[buffer(2)]],
    constant FftPairUniforms & uniforms[[buffer(3)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];
    uint groups_per_poly = uniforms.fft_len >> 2;
    uint total = uniforms.poly_count * groups_per_poly;
    if (index >= total) {
        return;
    }

    uint poly = index / groups_per_poly;
    uint k = index - poly * groups_per_poly;
    uint half_m = uniforms.half_m;
    uint m = half_m << 1;
    uint block = k / half_m;
    uint j = k - block * half_m;
    uint base = poly * uniforms.fft_len + block * (m << 1) + j;
    uint i0 = base;
    uint i1 = base + half_m;
    uint i2 = base + m;
    uint i3 = base + m + half_m;

    Fp a0 = values[i0];
    Fp a1 = values[i1];
    Fp a2 = values[i2];
    Fp a3 = values[i3];

    Fp omega0 = roots0[j];
    Fp t0 = omega0 * a1;
    Fp t1 = omega0 * a3;
    Fp b0 = a0 + t0;
    Fp b1 = fp_sub(a0, t0);
    Fp b2 = a2 + t1;
    Fp b3 = fp_sub(a2, t1);

    Fp omega1_lo = roots1[j];
    Fp omega1_hi = roots1[j + half_m];
    Fp u0 = omega1_lo * b2;
    Fp u1 = omega1_hi * b3;

    values[i0] = b0 + u0;
    values[i2] = fp_sub(b0, u0);
    values[i1] = b1 + u1;
    values[i3] = fp_sub(b1, u1);
}

kernel void fft_stage_triple(
    device Fp * values[[buffer(0)]],
    constant Fp * roots0[[buffer(1)]],
    constant Fp * roots1[[buffer(2)]],
    constant Fp * roots2[[buffer(3)]],
    constant FftTripleUniforms & uniforms[[buffer(4)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];
    uint groups_per_poly = uniforms.fft_len >> 3;
    uint total = uniforms.poly_count * groups_per_poly;
    if (index >= total) {
        return;
    }

    uint poly = index / groups_per_poly;
    uint k = index - poly * groups_per_poly;
    uint half_m = uniforms.half_m;
    uint m = half_m << 1;
    uint block = k / half_m;
    uint j = k - block * half_m;
    uint base = poly * uniforms.fft_len + block * (m << 2) + j;

    uint i0 = base;
    uint i1 = base + half_m;
    uint i2 = base + m;
    uint i3 = base + m + half_m;
    uint i4 = base + (m << 1);
    uint i5 = i4 + half_m;
    uint i6 = i4 + m;
    uint i7 = i6 + half_m;

    Fp a0 = values[i0];
    Fp a1 = values[i1];
    Fp a2 = values[i2];
    Fp a3 = values[i3];
    Fp a4 = values[i4];
    Fp a5 = values[i5];
    Fp a6 = values[i6];
    Fp a7 = values[i7];

    Fp omega0 = roots0[j];
    Fp t0 = omega0 * a1;
    Fp t1 = omega0 * a3;
    Fp t2 = omega0 * a5;
    Fp t3 = omega0 * a7;
    Fp b0 = a0 + t0;
    Fp b1 = fp_sub(a0, t0);
    Fp b2 = a2 + t1;
    Fp b3 = fp_sub(a2, t1);
    Fp b4 = a4 + t2;
    Fp b5 = fp_sub(a4, t2);
    Fp b6 = a6 + t3;
    Fp b7 = fp_sub(a6, t3);

    Fp omega1_lo = roots1[j];
    Fp omega1_hi = roots1[j + half_m];
    Fp u0 = omega1_lo * b2;
    Fp u1 = omega1_hi * b3;
    Fp u2 = omega1_lo * b6;
    Fp u3 = omega1_hi * b7;
    Fp c0 = b0 + u0;
    Fp c2 = fp_sub(b0, u0);
    Fp c1 = b1 + u1;
    Fp c3 = fp_sub(b1, u1);
    Fp c4 = b4 + u2;
    Fp c6 = fp_sub(b4, u2);
    Fp c5 = b5 + u3;
    Fp c7 = fp_sub(b5, u3);

    Fp omega2_0 = roots2[j];
    Fp omega2_1 = roots2[j + half_m];
    Fp omega2_2 = roots2[j + (half_m << 1)];
    Fp omega2_3 = roots2[j + (half_m << 1) + half_m];
    Fp v0 = omega2_0 * c4;
    Fp v1 = omega2_1 * c5;
    Fp v2 = omega2_2 * c6;
    Fp v3 = omega2_3 * c7;

    values[i0] = c0 + v0;
    values[i4] = fp_sub(c0, v0);
    values[i1] = c1 + v1;
    values[i5] = fp_sub(c1, v1);
    values[i2] = c2 + v2;
    values[i6] = fp_sub(c2, v2);
    values[i3] = c3 + v3;
    values[i7] = fp_sub(c3, v3);
}

kernel void poseidon_hash_leaves(
    constant Fp * leaf_inputs[[buffer(0)]],
    device packed_ulong4 * output[[buffer(1)]],
    constant Uniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];

        
    Fp p2_state[12];
    
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;


    
    uint offset = index * uniforms.leaf_size;
    uint num_full_rounds = uniforms.leaf_size / 8;

    for (uint i = 0; i < num_full_rounds; i++) {
        p2_state[0] = leaf_inputs[offset];
        p2_state[1] = leaf_inputs[offset + 1];
        p2_state[2] = leaf_inputs[offset + 2];
        p2_state[3] = leaf_inputs[offset + 3];

        p2_state[4] = leaf_inputs[offset + 4];
        p2_state[5] = leaf_inputs[offset + 5];
        p2_state[6] = leaf_inputs[offset + 6];
        p2_state[7] = leaf_inputs[offset + 7];
        poseidon_permute(p2_state);
        offset+=8;
    }

    uint remaining = uniforms.leaf_size - num_full_rounds * 8;
    if (remaining != 0) {
        for(uint i=0;i<remaining;i++){
            p2_state[i] = leaf_inputs[offset + i];
        }
        poseidon_permute(p2_state);
    }

    offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, 0, index);
    output[offset] = packed_ulong4(
        static_cast<ulong>(p2_state[0]),
        static_cast<ulong>(p2_state[1]),
        static_cast<ulong>(p2_state[2]),
        static_cast<ulong>(p2_state[3])
    );
}

kernel void poseidon_hash_leaves_poly_major(
    constant Fp * poly_inputs[[buffer(0)]],
    device packed_ulong4 * output[[buffer(1)]],
    constant Uniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint point_index = gid[1] * 32768 + gid[0];
    uint leaf_index = reverse_bits(point_index, uniforms.tree_height);
    uint leaf_count = 1u << uniforms.tree_height;

    Fp p2_state[12];

    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    uint num_full_rounds = uniforms.leaf_size / 8;
    uint poly_index = 0;

    for (uint i = 0; i < num_full_rounds; i++) {
        p2_state[0] = poly_inputs[(poly_index + 0) * leaf_count + point_index];
        p2_state[1] = poly_inputs[(poly_index + 1) * leaf_count + point_index];
        p2_state[2] = poly_inputs[(poly_index + 2) * leaf_count + point_index];
        p2_state[3] = poly_inputs[(poly_index + 3) * leaf_count + point_index];

        p2_state[4] = poly_inputs[(poly_index + 4) * leaf_count + point_index];
        p2_state[5] = poly_inputs[(poly_index + 5) * leaf_count + point_index];
        p2_state[6] = poly_inputs[(poly_index + 6) * leaf_count + point_index];
        p2_state[7] = poly_inputs[(poly_index + 7) * leaf_count + point_index];
        poseidon_permute(p2_state);
        poly_index += 8;
    }

    uint remaining = uniforms.leaf_size - num_full_rounds * 8;
    if (remaining != 0) {
        for (uint i = 0; i < remaining; i++) {
            p2_state[i] = poly_inputs[(poly_index + i) * leaf_count + point_index];
        }
        poseidon_permute(p2_state);
    }

    uint offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, 0, leaf_index);
    output[offset] = packed_ulong4(
        static_cast<ulong>(p2_state[0]),
        static_cast<ulong>(p2_state[1]),
        static_cast<ulong>(p2_state[2]),
        static_cast<ulong>(p2_state[3])
    );
}

inline void poseidon_absorb8_poly_major(
    constant Fp *poly_inputs,
    uint leaf_count,
    uint point_index,
    uint poly_index,
    thread Fp *p2_state
) {
    p2_state[0] = poly_inputs[(poly_index + 0) * leaf_count + point_index];
    p2_state[1] = poly_inputs[(poly_index + 1) * leaf_count + point_index];
    p2_state[2] = poly_inputs[(poly_index + 2) * leaf_count + point_index];
    p2_state[3] = poly_inputs[(poly_index + 3) * leaf_count + point_index];
    p2_state[4] = poly_inputs[(poly_index + 4) * leaf_count + point_index];
    p2_state[5] = poly_inputs[(poly_index + 5) * leaf_count + point_index];
    p2_state[6] = poly_inputs[(poly_index + 6) * leaf_count + point_index];
    p2_state[7] = poly_inputs[(poly_index + 7) * leaf_count + point_index];
    poseidon_permute(p2_state);
}

inline void poseidon_hash_poly_major_init(
    thread Fp *p2_state
) {
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;
}

inline void poseidon_store_poly_major_digest(
    device packed_ulong4 *output,
    constant Uniforms &uniforms,
    uint leaf_index,
    thread Fp *p2_state
) {
    uint offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, 0, leaf_index);
    output[offset] = packed_ulong4(
        static_cast<ulong>(p2_state[0]),
        static_cast<ulong>(p2_state[1]),
        static_cast<ulong>(p2_state[2]),
        static_cast<ulong>(p2_state[3])
    );
}

kernel void poseidon_hash_leaves_poly_major_16(
    constant Fp * poly_inputs[[buffer(0)]],
    device packed_ulong4 * output[[buffer(1)]],
    constant Uniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint point_index = gid[1] * 32768 + gid[0];
    uint leaf_index = reverse_bits(point_index, uniforms.tree_height);
    uint leaf_count = 1u << uniforms.tree_height;

    Fp p2_state[12];
    poseidon_hash_poly_major_init(p2_state);
    poseidon_absorb8_poly_major(poly_inputs, leaf_count, point_index, 0, p2_state);
    poseidon_absorb8_poly_major(poly_inputs, leaf_count, point_index, 8, p2_state);
    poseidon_store_poly_major_digest(output, uniforms, leaf_index, p2_state);
}

kernel void poseidon_hash_tree_level(
    constant Uniforms & uniforms[[buffer(1)]],
    device packed_ulong4 * output[[buffer(0)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];

    Fp p2_state[12];
    uint offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level - 1, index * 2);
    packed_ulong4 left = output[offset];
    packed_ulong4 right = output[offset + 1];

    p2_state[0] = left.x;
    p2_state[1] = left.y;
    p2_state[2] = left.z;
    p2_state[3] = left.w;

    p2_state[4] = right.x;
    p2_state[5] = right.y;
    p2_state[6] = right.z;
    p2_state[7] = right.w;
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;
    poseidon_permute(p2_state);
    
    offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level, index);

    output[offset] = packed_ulong4(
        static_cast<ulong>(p2_state[0]),
        static_cast<ulong>(p2_state[1]),
        static_cast<ulong>(p2_state[2]),
        static_cast<ulong>(p2_state[3])
    );
}
