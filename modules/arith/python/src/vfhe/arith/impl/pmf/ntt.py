# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
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

from vfhe.engine import ffi, lib

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
        self, vector: PseudoMersenneVector, in_place: bool = False
    ) -> PseudoMersenneVector:
        """
        Coefficients in natural order to evaluations in bit-reversed order.

        Returns a fresh vector unless ``in_place``, in which case ``vector``
        is transformed and returned.
        """
        target = self._target(vector, in_place)
        lib.pmf_vec_ntt_forward(target._struct, self._plan)
        return target

    def inverse(
        self, vector: PseudoMersenneVector, in_place: bool = False
    ) -> PseudoMersenneVector:
        """Evaluations in bit-reversed order back to coefficients; see `forward`."""
        target = self._target(vector, in_place)
        lib.pmf_vec_ntt_inverse(target._struct, self._plan)
        return target

    def _target(
        self, vector: PseudoMersenneVector, in_place: bool
    ) -> PseudoMersenneVector:
        """Check the operand and pick the buffer the kernel writes."""
        # Imported here: the vector module is loaded through the field module,
        # which this one is imported from.
        from .vector import PseudoMersenneVector

        if not isinstance(vector, PseudoMersenneVector):
            raise TypeError(
                f"expected a PseudoMersenneVector, not {type(vector).__name__}"
            )
        if vector.field.prime != self.field.prime:
            raise ValueError("vector belongs to a different field")
        if len(vector) != self.n:
            raise ValueError(
                f"length mismatch: plan is for {self.n}, got {len(vector)}"
            )
        return vector if in_place else vector.copy()

    def __repr__(self) -> str:
        return f"PseudoMersenneNTT(n={self.n}, 2^{self.field.bits} - {self.field.c})"
