# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The negacyclic NTT over a `PseudoMersenneField`, on `PseudoMersenneVector`s.

One plan is one transform length over one field: the root it uses, and the
twiddle tables the C kernels read. The kernels run on the vector's limb
planes, so a butterfly stage processes a whole group of elements per
register; the C side states the transform's basis and output order at the
`PMFNTTPlan` declaration in ``arith.h``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.util.bindings import ffi, lib

if TYPE_CHECKING:
    from .pseudo_mersenne import PseudoMersenneElement, PseudoMersenneField
    from .vector import PseudoMersenneVector


class PseudoMersenneNTT:
    """The transform of F_p[X]/(X^n + 1) for one power-of-two ``n``.

    Same basis as arith's RNS transforms. With ``psi`` the primitive 2n-th
    root of unity in `root_of_unity`, `forward` of a vector holding the
    coefficients of P in natural order leaves position ``j`` holding
    ``P(psi ** (2 * brv(j) + 1))``, where ``brv`` reverses the ``log2(n)``
    index bits: natural in, bit-reversed out, so positions ``2i`` and
    ``2i + 1`` hold ``P(x)`` and ``P(-x)``. `inverse` takes that order back
    to coefficients, 1/n scaling included.

    The root is `PseudoMersenneField.root_of_unity` of order ``2n``. Every
    order's root comes from the same generator, so the plans of successive
    lengths satisfy ``plan(n // 2).root_of_unity == plan(n).root_of_unity ** 2``,
    the relation a fold across lengths relies on.

    Plans are cheap to keep and cost ``2 * n`` field elements of tables;
    `PseudoMersenneField.ntt_plan` memoizes one per length.
    """

    def __init__(self, field: PseudoMersenneField, n: int) -> None:
        """
        Build the plan for length ``n`` over ``field``.

        Raises ValueError unless ``n`` is a power of two no longer than the
        field allows: a 2n-th root of unity needs ``log2(n) + 1`` at most
        ``field.two_adicity``.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n & (n - 1):
            raise ValueError(f"transform length must be a power of two, got {n}")
        log_order = n.bit_length()  # log2(n) + 1, the order of psi as a power of 2
        if log_order > field.two_adicity:
            raise ValueError(
                f"no primitive {2 * n}-th root of unity: p - 1 has 2-adicity "
                f"{field.two_adicity}, so the longest transform is "
                f"{1 << (field.two_adicity - 1)}"
            )
        self.field = field
        self.n = n
        #: psi, the primitive 2n-th root of unity the transform evaluates at.
        self.root_of_unity: PseudoMersenneElement = field.root_of_unity(log_order)
        plan = lib.pmf_ntt_new_plan(n, self.root_of_unity._buf, field._params)
        if plan == ffi.NULL:
            raise RuntimeError(
                "C rejected the transform plan; see stderr for the reason"
            )
        self._plan = ffi.gc(plan, lib.pmf_ntt_free_plan)

    def forward(
        self,
        vector: PseudoMersenneVector,
        in_place: bool = False,
        blocks: int = 1,
    ) -> PseudoMersenneVector:
        """
        Coefficients in natural order to evaluations in bit-reversed order.

        With ``blocks`` above 1, ``vector`` holds that many transforms of
        length ``n`` end to end (block ``b`` at ``b * n``, the layout
        `batch_layout` names) and each is transformed on its own.

        Returns a fresh vector unless ``in_place``, in which case ``vector``
        is transformed and returned.
        """
        target = self._target(vector, in_place, blocks)
        if blocks == 1:
            lib.pmf_vec_ntt_forward(target._struct, self._plan)
        else:
            lib.pmf_vec_ntt_forward_blocks(target._struct, blocks, self._plan)
        return target

    def inverse(
        self,
        vector: PseudoMersenneVector,
        in_place: bool = False,
        blocks: int = 1,
    ) -> PseudoMersenneVector:
        """Evaluations in bit-reversed order back to coefficients; see `forward`."""
        target = self._target(vector, in_place, blocks)
        if blocks == 1:
            lib.pmf_vec_ntt_inverse(target._struct, self._plan)
        else:
            lib.pmf_vec_ntt_inverse_blocks(target._struct, blocks, self._plan)
        return target

    def encode_blocks(
        self, message: PseudoMersenneVector, blocks: int
    ) -> PseudoMersenneVector:
        """
        The forward transforms of ``blocks`` messages zero-padded to ``n``.

        ``message`` holds them interleaved -- coefficient ``i`` of message
        ``b`` at ``i * blocks + b`` -- each ``len(message) // blocks`` long, a
        power of two dividing ``n``. The result holds the transforms end to
        end, message ``b``'s at ``b * n``: what padding each message,
        `pack`-ing the batch and `forward` with ``blocks`` give, without the
        padding or the stages it leaves trivial.
        """
        from .vector import PseudoMersenneVector

        if not isinstance(message, PseudoMersenneVector):
            raise TypeError(
                f"expected a PseudoMersenneVector, not {type(message).__name__}"
            )
        if message.field.prime != self.field.prime:
            raise ValueError("vector belongs to a different field")
        if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
            raise ValueError(f"blocks must be a positive int, got {blocks}")
        k, rest = divmod(len(message), blocks)
        if rest or k < 1 or k & (k - 1) or self.n % k:
            raise ValueError(
                f"{len(message)} coefficients are not {blocks} messages of a "
                f"power of two dividing {self.n}"
            )
        result = message._like(self.n * blocks)
        lib.pmf_vec_ntt_encode_blocks(
            result._struct, message._struct, blocks, self._plan
        )
        return result

    #: How a batch of transforms is laid out: block ``b`` is the ``n``
    #: consecutive elements at ``b * n``. Named as `ExtensionFieldNTT` names
    #: its own, so a caller can lay out a batch for either plan the same way.
    batch_layout = "blocks"

    def pack(
        self,
        vector: PseudoMersenneVector,
        blocks: int = 1,
        out=None,
        layout: str = "blocks",
    ) -> PseudoMersenneVector:
        """`vector`, held in ``layout``, laid out the way a batched `forward`
        needs it.

        ``layout`` is ``'blocks'`` (the default) when each transform's
        elements are together, or ``'interleaved'`` when element ``i`` of
        block ``b`` sits at ``i * blocks + b``. From ``'blocks'`` there is
        nothing to do: the input is returned as it is unless `out` is given,
        in which case it is copied there. From ``'interleaved'`` it is a
        transpose, one pass.
        """
        return self._reshape(vector, blocks, out, layout, self.batch_layout)

    def unpack(
        self,
        vector: PseudoMersenneVector,
        blocks: int = 1,
        out=None,
        layout: str = "blocks",
    ) -> PseudoMersenneVector:
        """The inverse of `pack`: a transformed batch back to ``layout``."""
        return self._reshape(vector, blocks, out, self.batch_layout, layout)

    def _reshape(self, vector, blocks, out, source: str, target: str):
        for name in (source, target):
            if name not in ("blocks", "interleaved"):
                raise ValueError(
                    f"layout must be 'blocks' or 'interleaved', got {name!r}"
                )
        self._checked(vector, blocks)
        if source == target:
            if out is None:
                return vector
            if out is vector:
                return vector.copy()
            lib.pmf_vec_copy(vector._destination(out)._struct, vector._struct)
            return out
        result = vector._destination(out, len(vector))
        if result is vector:
            raise ValueError("a layout conversion cannot write into its input")
        kernel = (
            lib.pmf_ntt_to_interleaved
            if target == "interleaved"
            else lib.pmf_ntt_to_blocks
        )
        kernel(result._struct, vector._struct, blocks)
        return result

    def _checked(self, vector: PseudoMersenneVector, blocks: int) -> None:
        """Check the operand and the batch's shape against the plan."""
        # Imported here: the vector module is loaded through the field module,
        # which this one is imported from.
        from .vector import PseudoMersenneVector

        if not isinstance(vector, PseudoMersenneVector):
            raise TypeError(
                f"expected a PseudoMersenneVector, not {type(vector).__name__}"
            )
        if vector.field.prime != self.field.prime:
            raise ValueError("vector belongs to a different field")
        if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
            raise ValueError(f"blocks must be a positive int, got {blocks}")
        if len(vector) != self.n * blocks:
            raise ValueError(
                f"length mismatch: {blocks} transforms of length {self.n} are "
                f"{self.n * blocks} elements, got {len(vector)}"
            )

    def _target(
        self, vector: PseudoMersenneVector, in_place: bool, blocks: int
    ) -> PseudoMersenneVector:
        """Check the operand and pick the buffer the kernel writes."""
        self._checked(vector, blocks)
        return vector if in_place else vector.copy()

    def __repr__(self) -> str:
        return f"PseudoMersenneNTT(n={self.n}, 2^{self.field.bits} - {self.field.c})"
