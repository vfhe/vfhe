# SPDX-FileCopyrightText: 2026 Robin Koestler
# SPDX-License-Identifier: Apache-2.0
"""Linear maps on CKKS slots, applied homomorphically.

A map is an ``n x n`` complex matrix ``A`` acting on every block of ``n``
slots, given by its diagonals ``diag_d[i] = A[i][(i + d) % n]`` (``n`` a power
of two dividing ``N/2``; ``n = N/2`` is the whole slot vector). Then
``A z = sum_d diag_d * rot(z, d)``, and the baby-step giant-step split
``d = j*b + i`` of [HS18] turns that into

    A z = sum_j rot( sum_i rot(z, i) * rot(diag_(j*b+i), -j*b), j*b )

so the plaintexts are rotated in the clear, the ``b`` baby rotations of ``z``
share one decomposition (`MLWE_Scheme.automorphisms`), and the map costs
``b + n/b`` rotations and one plaintext product per diagonal.

[HS18] S. Halevi, V. Shoup. *Faster Homomorphic Linear Transformations in
HElib.* CRYPTO 2018.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vfhe.arith import RNSPolynomial
    from vfhe.mlwe import MLWE_Set

    from .ckks import CKKS_Ciphertext, CKKS_Scheme


class CKKS_LinearTransform:
    """A slot matrix encoded for one level, ready to apply to ciphertexts there.

    ``diagonals`` maps ``d`` to ``diag_d`` (length ``n``), or is an iterable of
    such pairs; only the nonzero diagonals need be given. Each is rotated,
    tiled to ``N/2`` slots and encoded once, in ``scheme.rings[lvl]`` at
    ``scale`` (default: ``scaling_factor``). ``baby_steps`` is ``b`` (default:
    ``ceil(sqrt(n))``).
    """

    def __init__(
        self,
        scheme: CKKS_Scheme,
        diagonals: Mapping[int, Sequence[complex]]
        | Iterable[tuple[int, Sequence[complex]]],
        *,
        lvl: int = 0,
        scale: float | None = None,
        baby_steps: int | None = None,
    ) -> None:
        pairs = (
            cast("Mapping[int, Sequence[complex]]", diagonals).items()
            if isinstance(diagonals, Mapping)
            else cast("Iterable[tuple[int, Sequence[complex]]]", diagonals)
        )
        items: Iterator[tuple[int, Sequence[complex]]] = iter(pairs)
        first = next(items, None)
        if first is None:
            raise ValueError("expected at least one diagonal")
        n = len(first[1])
        slots = scheme.N // 2
        if n < 1 or n & (n - 1) or slots % n:
            raise ValueError(f"the matrix size {n} must be a power of two dividing N/2")
        b = baby_steps if baby_steps is not None else math.isqrt(n - 1) + 1
        if not 1 <= b <= n:
            raise ValueError(f"baby_steps must be between 1 and {n}")

        self.scheme = scheme
        self.n = n
        self.lvl = lvl
        self.baby_steps = b
        self.scale = scheme.scaling_factor if scale is None else scale
        ring = scheme.rings[lvl]
        # giant j -> baby i -> plaintext of rot(diag_(j*b+i), -j*b)
        self.plaintexts: dict[int, dict[int, RNSPolynomial]] = {}
        for d, diag in _chain(first, items):
            if not 0 <= d < n or len(diag) != n:
                raise ValueError(
                    f"diagonal {d} must have index below {n} and length {n}"
                )
            j, i = divmod(d, b)
            if i in self.plaintexts.get(j, {}):
                raise ValueError(f"diagonal {d} given twice")
            cut = n - j * b  # rotated[t] = diag[t - j*b]
            rotated = [*diag[cut:], *diag[:cut]]
            pt = scheme.encode(rotated * (slots // n), ring=ring, scale=self.scale)
            pt.to_NTT()
            self.plaintexts.setdefault(j, {})[i] = pt

    @property
    def rotations(self) -> list[int]:
        """The slot rotations :meth:`apply` needs a key for."""
        b = self.baby_steps
        babies = {i for row in self.plaintexts.values() for i in row if i}
        giants = {j * b for j in self.plaintexts if j}
        return sorted(babies | giants)

    def apply(
        self,
        ciphertext: CKKS_Ciphertext,
        ksks: Mapping[int, MLWE_Set | list[MLWE_Set]],
        n_threads: int = 0,
    ) -> CKKS_Ciphertext:
        """``A`` applied to the slots of ``ciphertext``, without rescaling.

        ``ksks[k]`` is the rotation key for ``k`` slots
        (`CKKS_Scheme.gen_rotation_key`), for every ``k`` in :attr:`rotations`. The
        result's ``delta`` is the input's times ``scale``. Runs on up to
        ``n_threads`` threads (0: the library limit, see
        `lib.vfhe_set_num_threads`).
        """
        scheme = self.scheme
        if ciphertext.lvl != self.lvl or ciphertext.ring != scheme.rings[self.lvl]:
            raise ValueError(
                f"the transform was encoded for level {self.lvl}, "
                f"the ciphertext is at level {ciphertext.lvl}"
            )
        missing = [k for k in self.rotations if k not in ksks]
        if missing:
            raise ValueError(f"missing rotation keys for {missing}")
        two_n = 2 * scheme.N
        b = self.baby_steps

        babies = sorted({i for row in self.plaintexts.values() for i in row})
        rotated = dict(
            zip(
                [i for i in babies if i],
                scheme.automorphisms(
                    ciphertext,
                    [pow(5, i, two_n) for i in babies if i],
                    [ksks[i] for i in babies if i],
                    n_threads,
                ),
                strict=True,
            )
        )
        rotated[0] = ciphertext
        giants = sorted(self.plaintexts)
        inner = scheme.linear_combinations(
            [rotated[i] for i in babies],
            [[self.plaintexts[j].get(i) for i in babies] for j in giants],
            self.scale,
            n_threads,
        )
        terms = scheme.automorphism_batch(
            inner,
            [pow(5, j * b, two_n) for j in giants],
            [ksks[j * b] if j else None for j in giants],
            n_threads,
        )
        out = terms[0]
        for term in terms[1:]:
            out += term
        return out

    @classmethod
    def slot_to_coeff(
        cls,
        scheme: CKKS_Scheme,
        *,
        slots: int | None = None,
        lvl: int = 0,
        scale: float | None = None,
        baby_steps: int | None = None,
    ) -> CKKS_LinearTransform:
        """SlotToCoeff: afterwards the plaintext's coefficients hold the slots.

        For ``n = slots`` (default ``N/2``) and ``R = N/(2n)``, the ``n`` slot values
        ``z`` become the plaintext whose coefficients at ``R*t`` and ``R*t + N/2``
        are ``delta * Re z_t`` and ``delta * Im z_t``, all others zero. The matrix is
        ``U_n[j][t] = eta^(5^j * t)`` with ``eta = exp(2 pi i / 4n)``.
        """
        n, roots, powers = _encoding_tables(scheme, slots)
        m = 4 * n

        def diagonal(d: int) -> list[complex]:
            return [roots[powers[i] * ((i + d) % n) % m] for i in range(n)]

        return cls(
            scheme,
            ((d, diagonal(d)) for d in range(n)),
            lvl=lvl,
            scale=scale,
            baby_steps=baby_steps,
        )

    @classmethod
    def coeff_to_slot(
        cls,
        scheme: CKKS_Scheme,
        *,
        slots: int | None = None,
        lvl: int = 0,
        scale: float | None = None,
        baby_steps: int | None = None,
    ) -> CKKS_LinearTransform:
        """CoeffToSlot, the inverse of :meth:`slot_to_coeff` (``U_n^H / n``).

        Afterwards slot ``t`` holds ``(m[R*t] + i*m[R*t + N/2]) / delta`` for the
        input plaintext ``m``, in every block of ``n`` slots. Exact when ``m`` has
        nonzero coefficients only at multiples of ``R``, always the case for
        ``n = N/2``.
        """
        n, roots, powers = _encoding_tables(scheme, slots)
        m = 4 * n

        def diagonal(d: int) -> list[complex]:
            return [roots[-powers[(i + d) % n] * i % m] / n for i in range(n)]

        return cls(
            scheme,
            ((d, diagonal(d)) for d in range(n)),
            lvl=lvl,
            scale=scale,
            baby_steps=baby_steps,
        )


def _chain(first, rest):
    yield first
    yield from rest


def _encoding_tables(
    scheme: CKKS_Scheme, slots: int | None
) -> tuple[int, list[complex], list[int]]:
    """``n``, the ``4n``-th roots of unity, and ``5^j mod 4n`` for ``j < n``."""
    n = scheme.N // 2 if slots is None else slots
    if n < 1 or n & (n - 1) or (scheme.N // 2) % n:
        raise ValueError(f"slots={n} must be a power of two dividing N/2")
    m = 4 * n
    roots = [cmath.exp(2j * cmath.pi * e / m) for e in range(m)]
    return n, roots, [pow(5, j, m) for j in range(n)]
