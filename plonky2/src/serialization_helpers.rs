/// Speedy Serialization helpers
#[cfg(feature = "serialize_speedy")]
mod speedy_impls {
    use plonky2_field::extension::Extendable;
    use speedy::{Context, Readable, Reader, Writable, Writer};

    use crate::fri::proof::{FriProof, FriQueryRound, FriQueryStep};
    use crate::hash::hash_types::{BytesHash, RichField};
    use crate::plonk::circuit_data::VerifierOnlyCircuitData;
    use crate::plonk::config::{GenericConfig, Hasher};
    use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
    impl<'a, C: Context, F, H, const D: usize> Readable<'a, C> for FriQueryStep<F, H, D>
    where
        F: RichField + Extendable<D> + Readable<'a, C>,
        H: Hasher<F>,
        H::Hash: Readable<'a, C>,
        F::Extension: Readable<'a, C>, // Ensure the extension field is readable
    {
        #[inline]
        fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
            let evals = reader.read_value()?;
            let merkle_proof = reader.read_value()?;
            Ok(FriQueryStep {
                evals,
                merkle_proof,
            })
        }
    }

    impl<C: Context, F, H, const D: usize> Writable<C> for FriQueryStep<F, H, D>
    where
        F: RichField + Extendable<D>,
        H: Hasher<F>,
        H::Hash: Writable<C>,
        F::Extension: Writable<C>, // Ensure the extension field is writable
        F: Writable<C>,
    {
        #[inline]
        fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
            writer.write_value(&self.evals)?;
            writer.write_value(&self.merkle_proof)?;
            Ok(())
        }
    }
    // --- FriQueryRound ---
    impl<'a, C: Context, F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> Readable<'a, C>
        for FriQueryRound<F, H, D>
    where
        F: Readable<'a, C>,
        F::Extension: Readable<'a, C>,
        H::Hash: Readable<'a, C>,
    {
        #[inline]
        fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
            Ok(FriQueryRound {
                initial_trees_proof: reader.read_value()?,
                steps: reader.read_value()?,
            })
        }
    }

    impl<C: Context, F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> Writable<C>
        for FriQueryRound<F, H, D>
    where
        F::Extension: Writable<C>,
        F: Writable<C>,
        H::Hash: Writable<C>,
    {
        #[inline]
        fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
            writer.write_value(&self.initial_trees_proof)?;
            writer.write_value(&self.steps)?;
            Ok(())
        }
    }

    // --- FriProof ---
    impl<'a, C: Context, F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> Readable<'a, C>
        for FriProof<F, H, D>
    where
        F: Readable<'a, C>,
        F::Extension: Readable<'a, C>,
        H::Hash: Readable<'a, C>,
    {
        #[inline]
        fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
            Ok(FriProof {
                commit_phase_merkle_caps: reader.read_value()?,
                query_round_proofs: reader.read_value()?,
                final_poly: reader.read_value()?,
                pow_witness: reader.read_value()?,
            })
        }
    }

    impl<C: Context, F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> Writable<C>
        for FriProof<F, H, D>
    where
        H::Hash: Writable<C>,
        F::Extension: Writable<C>,
        F: Writable<C>,
    {
        #[inline]
        fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
            writer.write_value(&self.commit_phase_merkle_caps)?;
            writer.write_value(&self.query_round_proofs)?;
            writer.write_value(&self.final_poly)?;
            writer.write_value(&self.pow_witness)?;
            Ok(())
        }
    }

    impl<'a, C: Context, const N: usize> Readable<'a, C> for BytesHash<N> {
        #[inline]
        fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
            let mut bytes = [0u8; N];
            reader.read_bytes(&mut bytes)?;
            Ok(BytesHash(bytes))
        }

        #[inline]
        fn minimum_bytes_needed() -> usize {
            N
        }
    }
    impl<C: Context, const N: usize> Writable<C> for BytesHash<N> {
        #[inline]
        fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
            writer.write_bytes(&self.0)
        }
    }

    impl<'a, C: Context, F: RichField + Extendable<D>, const D: usize> Readable<'a, C>
        for OpeningSet<F, D>
    where
        F::Extension: Readable<'a, C>,
    {
        #[inline]
        fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
            Ok(OpeningSet {
                constants: reader.read_value()?,
                plonk_sigmas: reader.read_value()?,
                wires: reader.read_value()?,
                plonk_zs: reader.read_value()?,
                plonk_zs_next: reader.read_value()?,
                partial_products: reader.read_value()?,
                quotient_polys: reader.read_value()?,
                lookup_zs: reader.read_value()?,
                lookup_zs_next: reader.read_value()?,
            })
        }
    }

    impl<C: Context, F: RichField + Extendable<D>, const D: usize> Writable<C> for OpeningSet<F, D>
    where
        F::Extension: Writable<C>,
    {
        #[inline]
        fn write_to<T: ?Sized + Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
            writer.write_value(&self.constants)?;
            writer.write_value(&self.plonk_sigmas)?;
            writer.write_value(&self.wires)?;
            writer.write_value(&self.plonk_zs)?;
            writer.write_value(&self.plonk_zs_next)?;
            writer.write_value(&self.partial_products)?;
            writer.write_value(&self.quotient_polys)?;
            writer.write_value(&self.lookup_zs)?;
            writer.write_value(&self.lookup_zs_next)?;
            Ok(())
        }
    }

    impl<
            'a,
            Ctx: Context,
            F: RichField + Extendable<D>,
            C: GenericConfig<D, F = F>,
            const D: usize,
        > Readable<'a, Ctx> for Proof<F, C, D>
    where
        F: Readable<'a, Ctx>,
        F::Extension: Readable<'a, Ctx>,
        <<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash: Readable<'a, Ctx>,
    {
        #[inline]
        fn read_from<R: Reader<'a, Ctx>>(reader: &mut R) -> Result<Self, Ctx::Error> {
            Ok(Proof {
                wires_cap: reader.read_value()?,
                plonk_zs_partial_products_cap: reader.read_value()?,
                quotient_polys_cap: reader.read_value()?,
                openings: reader.read_value()?,
                opening_proof: reader.read_value()?,
            })
        }
    }

    impl<
            Ctx: Context,
            F: RichField + Extendable<D>,
            C: GenericConfig<D, F = F>,
            const D: usize,
        > Writable<Ctx> for Proof<F, C, D>
    where
        F: Writable<Ctx>,
        F::Extension: Writable<Ctx>,
        <<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash: Writable<Ctx>,
    {
        #[inline]
        fn write_to<T: ?Sized + Writer<Ctx>>(&self, writer: &mut T) -> Result<(), Ctx::Error> {
            writer.write_value(&self.wires_cap)?;
            writer.write_value(&self.plonk_zs_partial_products_cap)?;
            writer.write_value(&self.quotient_polys_cap)?;
            writer.write_value(&self.openings)?;
            writer.write_value(&self.opening_proof)?;
            Ok(())
        }
    }

    impl<
            'a,
            Ctx: Context,
            F: RichField + Extendable<D>,
            C: GenericConfig<D, F = F>,
            const D: usize,
        > Readable<'a, Ctx> for ProofWithPublicInputs<F, C, D>
    where
        F: Readable<'a, Ctx>,
        F::Extension: Readable<'a, Ctx>,
        <<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash: Readable<'a, Ctx>,
    {
        #[inline]
        fn read_from<R: Reader<'a, Ctx>>(reader: &mut R) -> Result<Self, Ctx::Error> {
            Ok(ProofWithPublicInputs {
                proof: reader.read_value()?,
                public_inputs: reader.read_value()?,
            })
        }
    }
    impl<
            Ctx: Context,
            F: RichField + Extendable<D>,
            C: GenericConfig<D, F = F>,
            const D: usize,
        > Writable<Ctx> for ProofWithPublicInputs<F, C, D>
    where
        F: Writable<Ctx>,
        F::Extension: Writable<Ctx>,
        <<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash: Writable<Ctx>,
    {
        #[inline]
        fn write_to<T: ?Sized + Writer<Ctx>>(&self, writer: &mut T) -> Result<(), Ctx::Error> {
            writer.write_value(&self.proof)?;
            writer.write_value(&self.public_inputs)?;
            Ok(())
        }
    }
    #[cfg(feature = "serialize_speedy")]
    impl<'a, Ctx: Context, C: GenericConfig<D>, const D: usize> Readable<'a, Ctx>
        for VerifierOnlyCircuitData<C, D>
    where
        <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash: Readable<'a, Ctx>,
    {
        #[inline]
        fn read_from<R: Reader<'a, Ctx>>(reader: &mut R) -> Result<Self, Ctx::Error> {
            Ok(VerifierOnlyCircuitData {
                constants_sigmas_cap: reader.read_value()?,
                circuit_digest: reader.read_value()?,
            })
        }
    }

    #[cfg(feature = "serialize_speedy")]
    impl<Ctx: Context, C: GenericConfig<D>, const D: usize> Writable<Ctx>
        for VerifierOnlyCircuitData<C, D>
    where
        <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash: Writable<Ctx>,
    {
        #[inline]
        fn write_to<T: ?Sized + Writer<Ctx>>(&self, writer: &mut T) -> Result<(), Ctx::Error> {
            writer.write_value(&self.constants_sigmas_cap)?;
            writer.write_value(&self.circuit_digest)?;
            Ok(())
        }
    }
}

/// Bytemuck Serialization helpers, only ones that are safe for all endianness (ie. straight up byte arrays)
#[cfg(all(feature = "serialize_bytemuck"))]
mod bytemuck_all_endian_impls {
    use crate::hash::hash_types::BytesHash;
    // ByteHash is Zeroable and Pod because it is literally a transparent array of bytes
    // This is the definition of Plain Old Data (aka. Pod).
    unsafe impl<const N: usize> bytemuck::Zeroable for BytesHash<N> {}
    unsafe impl<const N: usize> bytemuck::Pod for BytesHash<N> {}
}
