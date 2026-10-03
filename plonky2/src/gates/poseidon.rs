#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::poseidon_mds::PoseidonMdsGate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon;
use crate::hash::poseidon::{Poseidon, SPONGE_WIDTH};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Evaluates a full Poseidon permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently verifying Merkle proofs.
/// It has a flag which can be used to swap the first four inputs with the next four, for ordering
/// sibling digests.
#[derive(Debug, Default)]
pub struct PoseidonGate<F: RichField + Extendable<D>, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> PoseidonGate<F, D> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// The wire index for the `i`th input to the permutation.
    pub(crate) const fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub(crate) const fn wire_output(i: usize) -> usize {
        SPONGE_WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    pub(crate) const WIRE_SWAP: usize = 2 * SPONGE_WIDTH;

    const START_DELTA: usize = 2 * SPONGE_WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    const fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_FULL_0: usize = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the first set
    /// of full rounds.
    const fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_0 + SPONGE_WIDTH * (round - 1) + i
    }

    const START_PARTIAL: usize =
        Self::START_FULL_0 + SPONGE_WIDTH * (poseidon::HALF_N_FULL_ROUNDS - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    const fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < poseidon::N_PARTIAL_ROUNDS);
        Self::START_PARTIAL + round
    }

    const START_FULL_1: usize = Self::START_PARTIAL + poseidon::N_PARTIAL_ROUNDS;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the second set
    /// of full rounds.
    const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_1 + SPONGE_WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    const fn end() -> usize {
        Self::START_FULL_1 + SPONGE_WIDTH * poseidon::HALF_N_FULL_ROUNDS
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for PoseidonGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={SPONGE_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(PoseidonGate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::Extension::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon>::sbox_layer_field(&mut state);
            state = <F as Poseidon>::mds_layer_field(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        <F as Poseidon>::partial_first_constant_layer(&mut state);
        state = <F as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(state[0] - sbox_in);
            state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
            state[0] +=
                F::Extension::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <F as Poseidon>::mds_partial_layer_fast_field(&state, r);
        }
        let sbox_in = vars.local_wires[Self::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
        constraints.push(state[0] - sbox_in);
        state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
        state =
            <F as Poseidon>::mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_field(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon>::sbox_layer_field(&mut state);
            state = <F as Poseidon>::mds_layer_field(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon>::sbox_layer(&mut state);
            state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        <F as Poseidon>::partial_first_constant_layer(&mut state);
        state = <F as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
            state[0] += F::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <F as Poseidon>::mds_partial_layer_fast(&state, r);
        }
        let sbox_in = vars.local_wires[Self::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
        yield_constr.one(state[0] - sbox_in);
        state[0] = <F as Poseidon>::sbox_monomial(sbox_in);
        state = <F as Poseidon>::mds_partial_layer_fast(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon>::sbox_layer(&mut state);
            state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    /// Evaluates full `Packing::WIDTH`-row chunks with `eval_unfiltered_base_packed` and any
    /// remaining rows with `eval_unfiltered_base_one`. The output layout is the one of the default
    /// implementation: constraint `j` of row `i` is at index `j * batch_size + i`.
    ///
    /// When the packing width is 1 (no SIMD target features), every row takes the scalar path,
    /// whose MDS and partial-round routines are specialised for scalars and are faster than the
    /// generic packed ones instantiated at width 1.
    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        let batch_size = vars_base.len();
        let width = <F as Packable>::Packing::WIDTH;
        let mut res = vec![F::ZERO; batch_size * self.num_constraints()];
        let packed_rows = if width > 1 {
            let (vars_packed_iter, _) = vars_base.pack::<<F as Packable>::Packing>();
            let packed_rows = width * vars_packed_iter.len();
            for (i, vars_packed) in vars_packed_iter.enumerate() {
                self.eval_unfiltered_base_packed(
                    vars_packed,
                    StridedConstraintConsumer::new(&mut res, batch_size, width * i),
                );
            }
            packed_rows
        } else {
            0
        };
        for i in packed_rows..batch_size {
            self.eval_unfiltered_base_one(
                vars_base.view(i),
                StridedConstraintConsumer::new(&mut res, batch_size, i),
            );
        }
        res
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        // The naive method is more efficient if we have enough routed wires for PoseidonMdsGate.
        let use_mds_gate =
            builder.config.num_routed_wires >= PoseidonMdsGate::<F, D>::new().num_wires();

        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        // Compute the possibly-swapped input layer.
        let mut state = [builder.zero_extension(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_circuit(builder, &mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon>::sbox_layer_circuit(builder, &mut state);
            state = <F as Poseidon>::mds_layer_circuit(builder, &state);
            round_ctr += 1;
        }

        // Partial rounds.
        if use_mds_gate {
            for r in 0..poseidon::N_PARTIAL_ROUNDS {
                <F as Poseidon>::constant_layer_circuit(builder, &mut state, round_ctr);
                let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
                constraints.push(builder.sub_extension(state[0], sbox_in));
                state[0] = <F as Poseidon>::sbox_monomial_circuit(builder, sbox_in);
                state = <F as Poseidon>::mds_layer_circuit(builder, &state);
                round_ctr += 1;
            }
        } else {
            <F as Poseidon>::partial_first_constant_layer_circuit(builder, &mut state);
            state = <F as Poseidon>::mds_partial_layer_init_circuit(builder, &state);
            for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
                let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
                constraints.push(builder.sub_extension(state[0], sbox_in));
                state[0] = <F as Poseidon>::sbox_monomial_circuit(builder, sbox_in);
                let c = <F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r];
                let c = F::Extension::from_canonical_u64(c);
                let c = builder.constant_extension(c);
                state[0] = builder.add_extension(state[0], c);
                state = <F as Poseidon>::mds_partial_layer_fast_circuit(builder, &state, r);
            }
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = <F as Poseidon>::sbox_monomial_circuit(builder, sbox_in);
            state = <F as Poseidon>::mds_partial_layer_fast_circuit(
                builder,
                &state,
                poseidon::N_PARTIAL_ROUNDS - 1,
            );
            round_ctr += poseidon::N_PARTIAL_ROUNDS;
        }

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_circuit(builder, &mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }
            <F as Poseidon>::sbox_layer_circuit(builder, &mut state);
            state = <F as Poseidon>::mds_layer_circuit(builder, &state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints
                .push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = PoseidonGenerator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        SPONGE_WIDTH * (poseidon::N_FULL_ROUNDS_TOTAL - 1)
            + poseidon::N_PARTIAL_ROUNDS
            + SPONGE_WIDTH
            + 1
            + 4
    }
}

/// Applies the full MDS layer to every lane of a packed state.
///
/// The scalar `Poseidon::mds_layer` exploits the small MDS coefficients (FFT-based on Goldilocks)
/// and is much cheaper than twelve dense packed dot products, so each lane is transposed out,
/// multiplied with the scalar routine, and transposed back. The result is the same field element
/// per lane as the row-by-row evaluation computes.
#[inline(always)]
fn mds_layer_lanewise<F: RichField, P: PackedField<Scalar = F>>(
    state: &[P; SPONGE_WIDTH],
) -> [P; SPONGE_WIDTH] {
    let mut result = [P::ZEROS; SPONGE_WIDTH];
    for lane in 0..P::WIDTH {
        let mut lane_state = [F::ZERO; SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            lane_state[i] = state[i].as_slice()[lane];
        }
        let lane_result = <F as Poseidon>::mds_layer(&lane_state);
        for i in 0..SPONGE_WIDTH {
            result[i].as_slice_mut()[lane] = lane_result[i];
        }
    }
    result
}

/// Packed `x |--> x^7`, using the same multiplication chain as `Poseidon::sbox_monomial`.
#[inline(always)]
fn sbox_monomial_packed<P: PackedField>(x: P) -> P {
    let x2 = x.square();
    let x4 = x2.square();
    let x3 = x * x2;
    x3 * x4
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for PoseidonGate<F, D>
{
    /// Packed counterpart of `eval_unfiltered_base_one`. It must yield exactly the same
    /// constraints, in the same order, for every lane.
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * (swap - P::ONES));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [P::ZEROS; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_packed_field::<F, D, F, P, 1>(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            for s in state.iter_mut() {
                *s = sbox_monomial_packed(*s);
            }
            state = mds_layer_lanewise(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        <F as Poseidon>::partial_first_constant_layer_packed_field::<F, D, F, P, 1>(&mut state);
        state = <F as Poseidon>::mds_partial_layer_init_packed_field::<F, D, F, P, 1>(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = sbox_monomial_packed(sbox_in);
            state[0] += F::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state =
                <F as Poseidon>::mds_partial_layer_fast_packed_field::<F, D, F, P, 1>(&state, r);
        }
        let sbox_in = vars.local_wires[Self::wire_partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
        yield_constr.one(state[0] - sbox_in);
        state[0] = sbox_monomial_packed(sbox_in);
        state = <F as Poseidon>::mds_partial_layer_fast_packed_field::<F, D, F, P, 1>(
            &state,
            poseidon::N_PARTIAL_ROUNDS - 1,
        );
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer_packed_field::<F, D, F, P, 1>(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            for s in state.iter_mut() {
                *s = sbox_monomial_packed(*s);
            }
            state = mds_layer_lanewise(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }
}

#[derive(Debug, Default)]
pub struct PoseidonGenerator<F: RichField + Extendable<D> + Poseidon, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> SimpleGenerator<F, D>
    for PoseidonGenerator<F, D>
{
    fn id(&self) -> String {
        "PoseidonGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .map(|i| PoseidonGate::<F, D>::wire_input(i))
            .chain(Some(PoseidonGate::<F, D>::WIRE_SWAP))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state = (0..SPONGE_WIDTH)
            .map(|i| witness.get_wire(local_wire(PoseidonGate::<F, D>::wire_input(i))))
            .collect::<Vec<_>>();

        let swap_value = witness.get_wire(local_wire(PoseidonGate::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(PoseidonGate::<F, D>::wire_delta(i)), delta_i)?;
        }

        if swap_value == F::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();
        let mut round_ctr = 0;
        // The state is in the base field, so the base-field layers apply (the `_field` variants
        // are for extension elements and multiply in full width); the values are the same.

        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    out_buffer.set_wire(
                        local_wire(PoseidonGate::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    )?;
                }
            }
            <F as Poseidon>::sbox_layer(&mut state);
            state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }

        <F as Poseidon>::partial_first_constant_layer(&mut state);
        state = <F as Poseidon>::mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            out_buffer.set_wire(
                local_wire(PoseidonGate::<F, D>::wire_partial_sbox(r)),
                state[0],
            )?;
            state[0] = <F as Poseidon>::sbox_monomial(state[0]);
            state[0] += F::from_canonical_u64(<F as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = <F as Poseidon>::mds_partial_layer_fast(&state, r);
        }
        out_buffer.set_wire(
            local_wire(PoseidonGate::<F, D>::wire_partial_sbox(
                poseidon::N_PARTIAL_ROUNDS - 1,
            )),
            state[0],
        )?;
        state[0] = <F as Poseidon>::sbox_monomial(state[0]);
        state =
            <F as Poseidon>::mds_partial_layer_fast(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            <F as Poseidon>::constant_layer(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                out_buffer.set_wire(
                    local_wire(PoseidonGate::<F, D>::wire_full_sbox_1(r, i)),
                    state[i],
                )?;
            }
            <F as Poseidon>::sbox_layer(&mut state);
            state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(local_wire(PoseidonGate::<F, D>::wire_output(i)), state[i])?
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self {
            row,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::{Field64, PrimeField64, Sample};

    use super::*;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn wire_indices() {
        type F = GoldilocksField;
        type Gate = PoseidonGate<F, 4>;

        assert_eq!(Gate::wire_input(0), 0);
        assert_eq!(Gate::wire_input(11), 11);
        assert_eq!(Gate::wire_output(0), 12);
        assert_eq!(Gate::wire_output(11), 23);
        assert_eq!(Gate::WIRE_SWAP, 24);
        assert_eq!(Gate::wire_delta(0), 25);
        assert_eq!(Gate::wire_delta(3), 28);
        assert_eq!(Gate::wire_full_sbox_0(1, 0), 29);
        assert_eq!(Gate::wire_full_sbox_0(3, 0), 53);
        assert_eq!(Gate::wire_full_sbox_0(3, 11), 64);
        assert_eq!(Gate::wire_partial_sbox(0), 65);
        assert_eq!(Gate::wire_partial_sbox(21), 86);
        assert_eq!(Gate::wire_full_sbox_1(0, 0), 87);
        assert_eq!(Gate::wire_full_sbox_1(3, 0), 123);
        assert_eq!(Gate::wire_full_sbox_1(3, 11), 134);
    }

    #[test]
    fn generated_output() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 143,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = PoseidonGate<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..SPONGE_WIDTH)
            .map(F::from_canonical_usize)
            .collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs
            .set_wire(
                Wire {
                    row,
                    column: Gate::WIRE_SWAP,
                },
                F::ZERO,
            )
            .unwrap();
        for i in 0..SPONGE_WIDTH {
            inputs
                .set_wire(
                    Wire {
                        row,
                        column: Gate::wire_input(i),
                    },
                    permutation_inputs[i],
                )
                .unwrap();
        }

        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        let expected_outputs: [F; SPONGE_WIDTH] =
            F::poseidon(permutation_inputs.try_into().unwrap());
        for i in 0..SPONGE_WIDTH {
            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, expected_outputs[i]);
        }
    }

    #[test]
    fn low_degree() {
        type F = GoldilocksField;
        let gate = PoseidonGate::<F, 4>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonGate::<F, 2>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }

    /// Row-by-row reference: a verbatim copy of the default `Gate::eval_unfiltered_base_batch`,
    /// which calls `eval_unfiltered_base_one` once per row.
    fn eval_base_batch_row_by_row<F: RichField + Extendable<D>, G: Gate<F, D>, const D: usize>(
        gate: &G,
        vars_base: EvaluationVarsBaseBatch<F>,
    ) -> Vec<F> {
        let mut res = vec![F::ZERO; vars_base.len() * gate.num_constraints()];
        for (i, vars_base_one) in vars_base.iter().enumerate() {
            gate.eval_unfiltered_base_one(
                vars_base_one,
                StridedConstraintConsumer::new(&mut res, vars_base.len(), i),
            );
        }
        res
    }

    /// Random Goldilocks value that is an edge case (0, 1, p - 1, or a non-canonical
    /// representative >= p) with probability 1/2, and uniform otherwise.
    fn random_edge_value(rng: &mut impl rand::Rng) -> GoldilocksField {
        const P: u64 = GoldilocksField::ORDER;
        match rng.gen_range(0..10) {
            0 => GoldilocksField(0),
            1 => GoldilocksField(1),
            2 => GoldilocksField(P - 1),
            // Non-canonical representatives of 0, 1, ... and of p - 2.
            3 => GoldilocksField(P + rng.gen_range(0..4)),
            4 => GoldilocksField(u64::MAX - rng.gen_range(0..4)),
            _ => GoldilocksField(rng.gen_range(0..P)),
        }
    }

    fn assert_packed_batch_matches_row_by_row<const D: usize>()
    where
        GoldilocksField: Extendable<D>,
    {
        use rand::{Rng, SeedableRng};

        type F = GoldilocksField;
        let gate = PoseidonGate::<F, D>::new();
        let width = <F as Packable>::Packing::WIDTH;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x9053_1d0e + D as u64);

        let mut batch_sizes = vec![1, 2, 3, width, width + 1, 3 * width + 2, 32, 33, 64];
        batch_sizes.sort_unstable();
        batch_sizes.dedup();

        for &batch_size in &batch_sizes {
            for trial in 0..8 {
                // Trial 0 uses only edge values, trial 1 only uniform values, the rest a mix.
                let sample = |rng: &mut rand_chacha::ChaCha8Rng| match trial {
                    0 => [
                        GoldilocksField(0),
                        GoldilocksField(1),
                        GoldilocksField(F::ORDER - 1),
                        GoldilocksField(F::ORDER),
                        GoldilocksField(u64::MAX),
                    ][rng.gen_range(0..5)],
                    1 => GoldilocksField(rng.gen_range(0..F::ORDER)),
                    _ => random_edge_value(rng),
                };
                let wires: Vec<F> = (0..gate.num_wires() * batch_size)
                    .map(|_| sample(&mut rng))
                    .collect();
                // Extra constants model selector columns that the gate does not read.
                let constants: Vec<F> = (0..(gate.num_constants() + 2) * batch_size)
                    .map(|_| sample(&mut rng))
                    .collect();
                let public_inputs_hash = HashOut {
                    elements: core::array::from_fn(|_| sample(&mut rng)),
                };
                let vars = EvaluationVarsBaseBatch::new(
                    batch_size,
                    &constants,
                    &wires,
                    &public_inputs_hash,
                );

                let expected = eval_base_batch_row_by_row(&gate, vars);
                assert_eq!(expected.len(), gate.num_constraints() * batch_size);

                // The production entry point, and the generic packed driver, which also runs the
                // packed code at width 1 for the leftover rows.
                for (path, actual) in [
                    (
                        "eval_unfiltered_base_batch",
                        gate.eval_unfiltered_base_batch(vars),
                    ),
                    (
                        "eval_unfiltered_base_batch_packed",
                        gate.eval_unfiltered_base_batch_packed(vars),
                    ),
                ] {
                    assert_eq!(actual.len(), expected.len(), "{path}");
                    for (idx, (a, e)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_canonical_u64(),
                            e.to_canonical_u64(),
                            "{path} D={D} width={width} batch_size={batch_size} trial={trial}: \
                             constraint {} of row {} differs",
                            idx / batch_size,
                            idx % batch_size,
                        );
                    }
                    assert_eq!(actual, expected, "{path}");
                }
            }
        }
    }

    #[test]
    fn packed_batch_matches_row_by_row() {
        assert_packed_batch_matches_row_by_row::<2>();
        assert_packed_batch_matches_row_by_row::<4>();
        assert_packed_batch_matches_row_by_row::<5>();
    }

    /// Valid Poseidon witnesses (both swap settings) must still satisfy every packed constraint.
    #[test]
    fn packed_batch_is_zero_on_valid_witness() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type G = PoseidonGate<F, D>;

        let batch_size = 11;
        let gate = G::new();
        let mut rows = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let config = CircuitConfig {
                num_wires: 143,
                ..CircuitConfig::standard_recursion_config()
            };
            let mut builder = CircuitBuilder::new(config);
            let gate_row = builder.add_gate(G::new(), vec![]);
            let circuit = builder.build_prover::<C>();
            let mut inputs = PartialWitness::new();
            let swap = if row % 2 == 0 { F::ZERO } else { F::ONE };
            inputs
                .set_wire(
                    Wire {
                        row: gate_row,
                        column: G::WIRE_SWAP,
                    },
                    swap,
                )
                .unwrap();
            for i in 0..SPONGE_WIDTH {
                inputs
                    .set_wire(
                        Wire {
                            row: gate_row,
                            column: G::wire_input(i),
                        },
                        F::from_canonical_usize(row * 100 + i),
                    )
                    .unwrap();
            }
            let witness =
                generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();
            rows.push(
                (0..gate.num_wires())
                    .map(|column| {
                        witness.get_wire(Wire {
                            row: gate_row,
                            column,
                        })
                    })
                    .collect::<Vec<_>>(),
            );
        }
        // Point-major layout: wire `w` for every row, then wire `w + 1`, and so on.
        let wires: Vec<F> = (0..gate.num_wires())
            .flat_map(|w| rows.iter().map(move |r| r[w]))
            .collect();
        let public_inputs_hash = HashOut::rand();
        let vars = EvaluationVarsBaseBatch::new(batch_size, &[], &wires, &public_inputs_hash);
        let evals = gate.eval_unfiltered_base_batch(vars);
        assert_eq!(evals, eval_base_batch_row_by_row(&gate, vars));
        assert_eq!(evals, gate.eval_unfiltered_base_batch_packed(vars));
        assert!(evals.iter().all(|c| c.is_zero()));
    }
}
