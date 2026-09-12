# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The negacyclic NTT over an `ExtensionField`, with the root in F_(p^d).

The difference from running arith's F_p transform once per coefficient plane
is the evaluation domain: there the points are in the prime subfield, here
they are odd powers of a psi that lives in the extension. A code whose domain
must avoid F_p needs this one; a code that does not is better served by the
per-plane transform, which is cheaper.

The C side states the conventions at the `FieldNTTPlan` declaration in
``arith.h``, and the batched layout `forward` requires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.engine import ffi, lib

if TYPE_CHECKING:
    from .field import ExtensionField, ExtensionFieldElement
    from .vector import ExtensionFieldVector


class ExtensionFieldNTT:
    """The transform of F_(p^d)[X]/(X^n + 1) for one power-of-two ``n``.

    Same basis and order as arith's other negacyclic transforms: with ``psi``
    the primitive 2n-th root of unity in `root_of_unity`, `forward` of a
    vector holding the coefficients of P in natural order leaves position
    ``j`` holding ``P(psi ** (2 * brv(j) + 1))`` -- natural in, bit-reversed
    out, so positions ``2i`` and ``2i + 1`` hold ``P(x)`` and ``P(-x)``.

    **It transforms a batch, and that is the point.** A stage's butterfly
    works on runs of ``t`` elements and ``t`` halves every stage, so a single
    transform ends in runs of 1, 2 and 4 -- far too short to be worth a kernel
    call, and they dominate its cost. With ``blocks`` transforms laid out
    block-fastest (element ``i`` of block ``b`` at ``i * blocks + b``) every
    run is ``t * blocks`` elements instead, so every stage of every block is
    one call. `blocks=1` is the ordinary single transform and is exactly the
    case this layout exists to avoid.

    Plans cost ``2 * n`` field elements of tables;
    `ExtensionField.ntt_plan` memoizes one per length.
    """

    def __init__(self, field: ExtensionField, n: int) -> None:
        """
        Build the plan for length ``n`` over ``field``.

        Raises ValueError unless ``n`` is a power of two for which a primitive
        2n-th root of unity exists, i.e. unless ``2n`` divides ``p**d - 1``.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n & (n - 1):
            raise ValueError(f"transform length must be a power of two, got {n}")
        self.field = field
        self.n = n
        #: psi, the primitive 2n-th root of unity the transform evaluates at.
        self.root_of_unity: ExtensionFieldElement = field.root_of_unity(n)
        plan = lib.field_ntt_new_plan(
            n, self.root_of_unity.value, field.d, field.w, field.mod
        )
        if plan == ffi.NULL:
            raise RuntimeError(
                "C rejected the transform plan; see stderr for the reason"
            )
        self._plan = ffi.gc(plan, lib.field_ntt_free_plan)

    @property
    def subfield_degree(self) -> int:
        """The degree of the smallest field holding a 2n-th root of unity.

        1 means the evaluation domain lies in the prime subfield -- correct,
        but then the per-plane transform computes the same thing more cheaply.
        ``field.degree`` means the domain avoids every proper subfield. It is
        decided by ``p`` and ``n`` alone, not by which root is chosen; see
        `ExtensionField.root_subfield_degree`.
        """
        return self.field.root_subfield_degree(self.n)

    def _checked(self, vector: ExtensionFieldVector, blocks: int):
        """The plane pointers, with the batch's shape checked against the plan."""
        if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
            raise ValueError(f"blocks must be a positive int, got {blocks}")
        if vector.field is not self.field:
            raise ValueError("vector belongs to a different field")
        if len(vector) != self.n * blocks:
            raise ValueError(
                f"a batch of {blocks} transforms of length {self.n} is "
                f"{self.n * blocks} elements, but the vector holds {len(vector)}"
            )
        return vector._plane_ptrs

    def forward(self, vector: ExtensionFieldVector, blocks: int = 1) -> None:
        """Transform ``blocks`` batched transforms in place, natural to
        bit-reversed order. The vector's layout is block-fastest; see the class
        docstring."""
        lib.field_ntt_forward(self._checked(vector, blocks), blocks, self._plan)

    def inverse(self, vector: ExtensionFieldVector, blocks: int = 1) -> None:
        """The inverse of `forward`, 1/n scaling included."""
        lib.field_ntt_inverse(self._checked(vector, blocks), blocks, self._plan)

    def __repr__(self) -> str:
        return f"ExtensionFieldNTT(n={self.n}, field={self.field!r})"
