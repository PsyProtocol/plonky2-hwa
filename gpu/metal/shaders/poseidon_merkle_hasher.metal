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

kernel void poseidon_hash_leaves(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
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

    offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, 0, index) * 4;
    output[offset] = static_cast<ulong>(p2_state[0]);
    output[offset + 1] = static_cast<ulong>(p2_state[1]);
    output[offset + 2] = static_cast<ulong>(p2_state[2]);
    output[offset + 3] = static_cast<ulong>(p2_state[3]);
}

kernel void poseidon_hash_leaves_poly_major(
    constant Fp * poly_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
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

    uint offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, 0, leaf_index) * 4;
    output[offset] = static_cast<ulong>(p2_state[0]);
    output[offset + 1] = static_cast<ulong>(p2_state[1]);
    output[offset + 2] = static_cast<ulong>(p2_state[2]);
    output[offset + 3] = static_cast<ulong>(p2_state[3]);
}

kernel void poseidon_hash_tree_level(
    constant Uniforms & uniforms[[buffer(1)]],
    device ulong * output[[buffer(0)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint index = gid[1] * 32768 + gid[0];

    Fp p2_state[12];
    uint offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level - 1, index * 2) * 4;

    p2_state[0] = output[offset];
    p2_state[1] = output[offset + 1];
    p2_state[2] = output[offset + 2];
    p2_state[3] = output[offset + 3];

    p2_state[4] = output[offset + 4];
    p2_state[5] = output[offset + 5];
    p2_state[6] = output[offset + 6];
    p2_state[7] = output[offset + 7];
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;
    poseidon_permute(p2_state);
    
    offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level, index) * 4;
    
    output[offset] = static_cast<ulong>(p2_state[0]);
    output[offset + 1] = static_cast<ulong>(p2_state[1]);
    output[offset + 2] = static_cast<ulong>(p2_state[2]);
    output[offset + 3] = static_cast<ulong>(p2_state[3]);
}
