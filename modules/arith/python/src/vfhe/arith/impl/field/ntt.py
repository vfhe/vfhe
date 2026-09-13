# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The negacyclic NTT over an `ExtensionField`, with the root in F_(p^d).

Two transforms behind one class, differing in where the root of unity lives.
With it in F_p the map is F_p-linear and splits across the coefficient planes,
so arith's own kernels do it -- cheaper, and the default. With it outside F_p,
which is what a code whose evaluation domain must avoid the prime subfield
needs, the butterflies are extension multiplications and a batched layout is
what makes them vectorise.

The C side states the conventions at the `FieldNTTPlan` declaration in
``arith.h``, and the layout each path takes.
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

    **Two implementations, chosen by where the root lives** -- `domain`, and
    the reason `batch_layout` is not the same for both:

    - ``'base'``, when 2n divides ``p - 1``. The root is then in F_p, the
      transform is F_p-linear, and it splits into ``d`` independent transforms
      of the coefficient planes on arith's own kernels. About **3x** faster,
      because each block's transform then fits in cache and runs there.
    - ``'extension'``, otherwise. A stage's butterfly works on runs of ``t``
      elements and ``t`` halves every stage, so a single transform ends in runs
      of 1, 2 and 4 -- too short to vectorise, and they carry most of its cost.
      Batching sidesteps that by making every run ``t * blocks`` elements.

    The base domain is the default wherever it exists, being the faster one.
    **A caller whose evaluation domain must avoid F_p has to say
    ``domain='extension'``** -- and gets an error rather than the subfield if
    the prime cannot supply it.

    Plans cost ``2 * n`` field elements of tables (the base domain, arith's
    own); `ExtensionField.ntt_plan` memoizes one per length and domain.
    """

    def __init__(self, field: ExtensionField, n: int, domain: str = "auto") -> None:
        """
        Build the plan for length ``n`` over ``field``.

        ``domain`` says where the root of unity must live -- see
        `ExtensionField.ntt_plan`. Raises ValueError unless ``n`` is a power of
        two for which the chosen domain has a primitive 2n-th root.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n & (n - 1):
            raise ValueError(f"transform length must be a power of two, got {n}")
        self.field = field
        self.n = n
        #: Where the root lives: ``'base'`` (F_p) or ``'extension'``.
        self.domain: str = field._resolve_ntt_domain(n, domain)

        if self.domain == "base":
            plan = lib.ntt_new_plan(n, field.mod)
            if plan == ffi.NULL:
                raise RuntimeError(
                    "arith rejected the transform plan; see stderr for the reason"
                )
            self._base_plan = ffi.gc(plan, lib.ntt_free_plan)
            self._plan = None
            #: psi, the primitive 2n-th root the transform evaluates at. Read
            #: back from arith's plan rather than derived: `ntt_new_plan` picks
            #: its own and takes no root, so this is the only way the two agree.
            self.root_of_unity: ExtensionFieldElement = field(
                lib.ntt_plan_root(self._base_plan)
            )
            return

        self._base_plan = None
        self.root_of_unity = field.root_of_unity(n, domain="extension")
        plan = lib.field_ntt_new_plan(
            n, self.root_of_unity.value, field.d, field.w, field.mod
        )
        if plan == ffi.NULL:
            raise RuntimeError(
                "C rejected the transform plan; see stderr for the reason"
            )
        self._plan = ffi.gc(plan, lib.field_ntt_free_plan)

    @property
    def batch_layout(self) -> str:
        """How a batch of transforms must be laid out, which **differs by
        domain** and is not a detail a caller can ignore.

        - ``'blocks'`` (the base domain): block ``b`` is the ``n`` consecutive
          elements at ``b * n``. That is what lets one transform run inside the
          cache, which is where this path's speed comes from.
        - ``'interleaved'`` (the extension domain): element ``i`` of block ``b``
          sits at ``i * blocks + b``. That is what gives the extension butterfly
          runs long enough to vectorise at every stage -- a problem the base
          path does not have, since each of its butterflies is a scalar
          multiply that arith's kernels already handle.

        A caller picks a domain once, from what its evaluation domain has to
        avoid, and lays its data out to match.
        """
        return "blocks" if self.domain == "base" else "interleaved"

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

    def pack(
        self, vector: ExtensionFieldVector, blocks: int = 1, out=None
    ) -> ExtensionFieldVector:
        """`vector`, laid out the way this plan's `forward` needs it.

        A caller that holds each transform's elements together has its blocks
        contiguous, and the extension domain needs them interleaved. `pack`
        converts, `unpack` converts back, and neither the caller nor its data
        structure has to know which domain is in play.

        The base domain already takes the contiguous layout, so there is
        nothing to do: the input is returned as it is unless `out` is given,
        in which case it is copied there. A conversion is a tiled transpose,
        one pass.
        """
        return self._reshape(vector, blocks, out, to_plan=True)

    def unpack(
        self, vector: ExtensionFieldVector, blocks: int = 1, out=None
    ) -> ExtensionFieldVector:
        """The inverse of `pack`: a transformed batch back to blocks."""
        return self._reshape(vector, blocks, out, to_plan=False)

    def _reshape(self, vector, blocks, out, to_plan: bool):
        self._checked(vector, blocks)
        if self.domain == "base":
            if out is None:
                return vector
            return vector.copy() if out is vector else self._copy_into(vector, out)
        result = vector._destination(out, len(vector))
        if result is vector:
            raise ValueError("a layout conversion cannot write into its input")
        kernel = lib.field_ntt_to_interleaved if to_plan else lib.field_ntt_to_blocks
        kernel(result._struct, vector._struct, blocks)
        return result

    @staticmethod
    def _copy_into(vector, out):
        lib.field_vec_copy(out._struct, vector._struct)
        return out

    def forward(self, vector: ExtensionFieldVector, blocks: int = 1) -> None:
        """Transform ``blocks`` transforms in place, natural to bit-reversed
        order. `batch_layout` says how the batch must be laid out."""
        planes = self._checked(vector, blocks)
        if self.domain == "base":
            lib.field_ntt_forward_base(planes, blocks, self.field.d, self._base_plan)
        else:
            lib.field_ntt_forward(planes, blocks, self._plan)

    def inverse(self, vector: ExtensionFieldVector, blocks: int = 1) -> None:
        """The inverse of `forward`, 1/n scaling included."""
        planes = self._checked(vector, blocks)
        if self.domain == "base":
            lib.field_ntt_inverse_base(planes, blocks, self.field.d, self._base_plan)
        else:
            lib.field_ntt_inverse(planes, blocks, self._plan)

    def __repr__(self) -> str:
        return f"ExtensionFieldNTT(n={self.n}, domain={self.domain!r})"
