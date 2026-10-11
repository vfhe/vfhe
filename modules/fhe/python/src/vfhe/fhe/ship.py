# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The SHIP bootstrap [CHKS25]: a shallow, highly parallel CKKS bootstrap of
``N/2`` real slots.

The input's plaintext holds ``m_t ~ q0 * mu_t / gamma`` in its coefficients
``t < N/2``, at the lowest modulus ``q0`` and under a sparse key ``s``. With
``omega = exp(2 pi i / q0)``, the bootstrap evaluates

    omega^m = omega^b * prod_(j : s_j != 0) (omega^((s_j X^j a)_t))_t

(the paper's ``m = a s + b``; vfhe's linear decryption is ``b - a s``), and
``2 Re(gamma / (4 pi i) * omega^m)`` is ``mu`` up to
``(2 pi)^2 / (6 gamma^2)``. Each factor is a *blind rotation* of four public
vectors [CHKS25, Lemma 1]. The ``h`` factors are independent, and their
product tree, of depth ``ceil(log2(h + 1))``, is the only use of levels.

A blind rotation is the hybrid of [CHKS25, Section 4.3], in the order of their
Table 2: a *column* step for the low part of the index (plaintext products
over the special ring, so no level is used), then a *mux* step for the high
part, one base-``B`` digit at a time (`MLWE_Scheme.automorphism_mux`). The key's
non-zero coefficients lie in public windows [CHKS25, Section 5.1], so the
rotation to the window's start (its *alignment*) is an ordinary one.

`SHIP.bootstrap` starts from slots at the second-lowest level: SlotToCoeff
[CHK+18] moves them into the coefficients at ``q0``, and a key switch [BTH22]
moves to the sparse key.

[CHKS25] J. H. Cheon, G. Hanrot, J. Kim, D. Stehlé. *SHIP: A Shallow and
Highly Parallelizable CKKS Bootstrapping Algorithm.* EUROCRYPT 2025,
ePrint 2025/784.
[CHK+18] J. H. Cheon, K. Han, A. Kim, M. Kim, Y. Song. *Bootstrapping for
Approximate Homomorphic Encryption.* EUROCRYPT 2018.
[BTH22] J.-P. Bossuat, J. R. Troncoso-Pastoriza, J.-P. Hubaux. *Bootstrapping
for Approximate Homomorphic Encryption with Negligible Failure-Probability by
Using Sparse-Secret Encapsulation.* ACNS 2022.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from vfhe.arith import Polynomial, domain_of, repr
from vfhe.crypto import entropy
from vfhe.engine import ffi, lib
from vfhe.mlwe.mlwe import MLWE, GadgetParams, MLWE_Key, MLWE_Set, lib_rlwe

from .ckks import CKKS_Ciphertext, CKKS_Scheme
from .linear_transform import CKKS_LinearTransform

if TYPE_CHECKING:
    from vfhe.arith import RNSPolynomial


class SHIP_Key:
    """A SHIP bootstrapping key, from `SHIP.generate_bootstrap_key`. Every key
    is under the output key.

    Per window ``t`` of the input key:

    - ``columns[t]``: the ``4 * theta`` encrypted masks of the column step, over
      level 0's special ring, in the NTT domain;
    - ``muxes[t][p][d]``: the multiplexer key for value ``d`` of digit ``p``;
    - ``alignments[t]``: the rotation key to the window's start, or ``None``.

    Also ``sparse_switch_key`` (to the input key, at the lowest level),
    ``conjugation_key`` (at the output level) and ``slot_to_coeff_keys``
    (rotation keys by slot count, at the SlotToCoeff level).
    """

    def __init__(self) -> None:
        self.sparse_switch_key: MLWE_Set | None = None
        self.columns: list[list[MLWE]] = []
        self.muxes: list[list[list[MLWE_Set]]] = []
        self.alignments: list[MLWE_Set | None] = []
        self.conjugation_key: MLWE_Set | None = None
        self.slot_to_coeff_keys: dict[int, MLWE_Set] = {}


class SHIP:
    """The SHIP bootstrap [CHKS25] over ``scheme``, for input keys of Hamming
    weight ``h``.

    ``scheme`` is a rank-1 CKKS scheme with special primes (the paper's
    auxiliary modulus ``P``) and nested levels. From the top: the
    ``ceil(log2(h + 1))`` levels of the product tree, any levels left to the
    application, the SlotToCoeff level, and the lowest level ``q0``.
    `bootstrap` takes ciphertexts at the SlotToCoeff level, encoded at
    `input_scale` (``q0 / gamma``), and returns them at `output_lvl`.

    ``theta`` and ``base`` are the paper's ``theta`` and ``B``: the sizes of the
    column and mux steps. ``window`` is the paper's ``w`` (default
    ``max(175, N / (2 h))``): the input key has exactly one non-zero
    coefficient in each window ``[c_k - w, c_k + w)`` mod ``N``, with
    ``c_k = floor(k N / h)``. The windows are public, which narrows the key
    distribution; its security estimate must account for that
    [CHKS25, Section 5.1].

    ``gadget_params`` is the gadget of every key-switch key. ``slot_to_coeff``
    defaults to `CKKS_LinearTransform.slot_to_coeff` at the SlotToCoeff level,
    built on first use.
    """

    def __init__(
        self,
        scheme: CKKS_Scheme,
        *,
        h: int,
        theta: int,
        base: int = 4,
        gamma: float = 32.0,
        window: int | None = None,
        gadget_params: GadgetParams | None = None,
        slot_to_coeff: CKKS_LinearTransform | None = None,
    ) -> None:
        N = scheme.N
        M = N // 2
        if scheme.r != 1:
            raise ValueError("SHIP bootstraps samples of rank 1")
        if (
            scheme.special_primes < 1
            or int(scheme.special_rings[0].modulus_ratio(scheme.rings[0])) <= 1
        ):
            raise ValueError("SHIP needs special primes, for its auxiliary modulus")
        if not 1 <= h <= N:
            raise ValueError(f"h must be between 1 and N = {N}")
        if not 1 <= theta <= M:
            raise ValueError(f"theta must be between 1 and N/2 = {M}")
        if base < 2:
            raise ValueError("the mux step's base must be at least 2")
        self.scheme = scheme
        self.h = h
        self.theta = theta
        self.base = base
        self.gamma = gamma
        self.window = max(175, N // (2 * h)) if window is None else window
        if self.window < 1:
            raise ValueError("the window must be at least 1")
        self.gadget_params = gadget_params

        # Levels: the product tree from the top, SlotToCoeff second-lowest.
        self.depth = h.bit_length()
        self.lowest_lvl = len(scheme.rings) - 1
        self.slot_to_coeff_lvl = self.lowest_lvl - 1
        if self.slot_to_coeff_lvl < self.depth:
            raise ValueError(
                f"h = {h} needs {self.depth} levels for the product above the "
                "SlotToCoeff level and the lowest one"
            )
        if not all(
            scheme.rings[k + 1].is_quotient_ring(scheme.rings[k])
            for k in range(self.lowest_lvl)
        ):
            raise ValueError("SHIP needs nested levels")
        self.q0 = math.prod(scheme.rings[self.lowest_lvl].primes)
        self._slot_to_coeff = slot_to_coeff

        # A window covers an arc of slots starting at its alignment; the mux
        # step's index ranges over ceil(span / theta) values.
        span = min(2 * self.window, M)
        self.alignments = [
            0 if span == M else ((k * N) // h - self.window) % N % M for k in range(h)
        ]
        self.giant_steps = -(-span // theta)
        self.digits = 0
        while base**self.digits < self.giant_steps:
            self.digits += 1

    @property
    def output_lvl(self) -> int:
        """The level `bootstrap` returns at."""
        return self.depth

    @property
    def input_scale(self) -> float:
        """The scale `bootstrap`'s inputs are expected at, ``q0 / gamma``."""
        return self.q0 / self.gamma

    @property
    def slot_to_coeff(self) -> CKKS_LinearTransform:
        """The SlotToCoeff transform `bootstrap` applies first."""
        if self._slot_to_coeff is None:
            lvl = self.slot_to_coeff_lvl
            rings = self.scheme.rings
            dropped = math.prod(
                p for p in rings[lvl].primes if p not in rings[lvl + 1].primes
            )
            self._slot_to_coeff = CKKS_LinearTransform.slot_to_coeff(
                self.scheme, lvl=lvl, scale=float(dropped)
            )
        return self._slot_to_coeff

    # -- input keys ---------------------------------------------------------

    def _window_start(self, k: int) -> int:
        return ((k * self.scheme.N) // self.h - self.window) % self.scheme.N

    def _window_size(self) -> int:
        return min(2 * self.window, self.scheme.N)

    def sample_input_key(self, sigma_err: float, attempts: int = 1 << 10) -> MLWE_Key:
        """A key with one non-zero coefficient per window, uniform in
        ``{-1, 1}``, at a uniform free position of the window."""
        N = self.scheme.N
        size = self._window_size()
        poly = [0] * N
        for k in range(self.h):
            start = self._window_start(k)
            for _ in range(attempts):
                j = (start + entropy.below(size)) % N
                if poly[j] == 0:
                    poly[j] = 1 - 2 * entropy.below(2)
                    break
            else:
                raise RuntimeError("no free position in a window")
        return MLWE_Key([poly], sigma_err, self.scheme)

    def _assign_windows(self, key: MLWE_Key) -> list[int]:
        """The non-zero coefficient of ``key`` assigned to each window. Windows
        may overlap, so this is a matching; any one will do."""
        N = self.scheme.N
        size = self._window_size()
        positions = [j for j, c in enumerate(key.key[0]) if c != 0]
        if len(positions) != self.h or any(
            key.key[0][j] not in (-1, 1) for j in positions
        ):
            raise ValueError(
                f"the input key must have h = {self.h} non-zero coefficients, "
                "in {-1, 1}"
            )
        starts = [self._window_start(k) for k in range(self.h)]
        owner: dict[int, int] = {}

        def place(k: int, seen: set[int]) -> bool:
            for j in positions:
                if (j - starts[k]) % N < size and j not in seen:
                    seen.add(j)
                    if j not in owner or place(owner[j], seen):
                        owner[j] = k
                        return True
            return False

        for k in range(self.h):
            if not place(k, set()):
                raise ValueError(
                    "the input key does not have one non-zero coefficient per "
                    "window; draw it with SHIP.sample_input_key"
                )
        by_window = [0] * self.h
        for j, k in owner.items():
            by_window[k] = j
        return by_window

    # -- keys ---------------------------------------------------------------

    def _rotation_generator(self, k: int) -> int:
        """The Galois element of the paper's ``Rot_k`` (right by ``k`` slots),
        i.e. `CKKS_Scheme.rotate` by ``-k``."""
        N = self.scheme.N
        return pow(5, (-k) % (N // 2), 2 * N)

    def _mux_generators(self, p: int) -> list[int]:
        step = self.base**p * self.theta
        return [self._rotation_generator(d * step) for d in range(self.base)]

    def generate_bootstrap_key(
        self, output_key: MLWE_Key, input_key: MLWE_Key, n_threads: int = 0
    ) -> SHIP_Key:
        """The bootstrapping key for inputs and outputs under ``output_key``
        (dense), through ``input_key`` (sparse, from `sample_input_key`).

        The switch to ``input_key`` exists at the lowest level only, so the
        sparse key need only be secure there. Keys are drawn in batches on
        ``n_threads`` (0: the library limit).
        """
        scheme = self.scheme
        if output_key.scheme is not scheme or input_key.scheme is not scheme:
            raise ValueError("the keys must be keys of this SHIP's scheme")
        positions = self._assign_windows(input_key)
        signs = [input_key.key[0][j] for j in positions]
        N, M, theta = scheme.N, scheme.N // 2, self.theta
        special = scheme.special_rings[0]
        params = self.gadget_params

        key = SHIP_Key()
        key.sparse_switch_key = cast(
            "MLWE_Set",
            scheme.gen_ksk(
                input_key,
                output_key,
                self.lowest_lvl,
                gadget_params=params,
                n_threads=n_threads,
            ),
        )

        # Column step: with j = align + j0 + j1 theta (in slots), mask i0 is
        # 1[i0 = j0] Rot_-(j1 theta + align)(m_k), encoded at scale P.
        zeros = [0.0] * M
        values, shifts, giants = [], [], []
        for t, (j, sign) in enumerate(zip(positions, signs, strict=True)):
            align = self.alignments[t]
            j1, j0 = divmod((j % M - align) % M, theta)
            giants.append(j1)
            masks = _lemma1_masks(j, sign, M)
            for i0 in range(theta):
                for mask in masks:
                    values.append(mask if i0 == j0 else zeros)
                    shifts.append(j1 * theta + align)
        P = int(special.modulus_ratio(scheme.rings[0]))
        plaintexts = scheme._encode_rotated(  # noqa: SLF001 - same package
            values, shifts, special, float(P), n_threads
        )
        samples = scheme.sample_scaled(
            plaintexts,
            [[1] * special.ell],
            output_key.at_ring(special),
            n_threads=n_threads,
        )
        key.columns = [
            samples[t * 4 * theta : (t + 1) * 4 * theta] for t in range(self.h)
        ]

        # Mux step: digit p selects one of the rotations by d B^p theta, d < B.
        selections = [
            (gen, int((j1 // self.base**p) % self.base == d))
            for j1 in giants
            for p in range(self.digits)
            for d, gen in enumerate(self._mux_generators(p))
        ]
        muxes = iter(
            scheme.gen_automorphism_mux_keys(
                output_key, selections, 0, params, n_threads
            )
        )
        key.muxes = [
            [[next(muxes) for _ in range(self.base)] for _ in range(self.digits)]
            for _ in range(self.h)
        ]

        aligned = [t for t in range(self.h) if self.alignments[t]]
        alignment_keys = (
            cast(
                "list[MLWE_Set]",
                scheme.gen_ksk_automorphism_set(
                    output_key,
                    output_key.at_ring(special),
                    [self._rotation_generator(self.alignments[t]) for t in aligned],
                    0,
                    params,
                    n_threads,
                ),
            )
            if aligned
            else []
        )
        key.alignments = [None] * self.h
        for t, k in zip(aligned, alignment_keys, strict=True):
            key.alignments[t] = k

        out = self.output_lvl
        (key.conjugation_key,) = cast(
            "list[MLWE_Set]",
            scheme.gen_ksk_automorphism_set(
                output_key,
                output_key.at_ring(scheme.special_rings[out]),
                [2 * N - 1],
                out,
                params,
                n_threads,
            ),
        )

        lvl = self.slot_to_coeff_lvl
        rotations = self.slot_to_coeff.rotations
        rotation_keys = cast(
            "list[MLWE_Set]",
            scheme.gen_ksk_automorphism_set(
                output_key,
                output_key.at_ring(scheme.special_rings[lvl]),
                [pow(5, k, 2 * N) for k in rotations],
                lvl,
                params,
                n_threads,
            ),
        )
        key.slot_to_coeff_keys = dict(zip(rotations, rotation_keys, strict=True))
        return key

    # -- the bootstrap ------------------------------------------------------

    def bootstrap(
        self, ciphertext: CKKS_Ciphertext, key: SHIP_Key, n_threads: int = 0
    ) -> CKKS_Ciphertext:
        """Bootstraps the ``N/2`` real slots of ``ciphertext`` (at the
        SlotToCoeff level, scale about `input_scale`) to `output_lvl`.

        The slots' imaginary parts must be zero. The product tree uses the
        scheme's relinearization key (`MLWE_Scheme.rlk`).
        """
        if ciphertext.lvl != self.slot_to_coeff_lvl:
            raise ValueError(
                f"the input must be at the SlotToCoeff level {self.slot_to_coeff_lvl}"
            )
        coefficients = self.slot_to_coeff.apply(
            ciphertext, key.slot_to_coeff_keys, n_threads
        )
        return self.half_bootstrap(
            self.scheme.rescale(coefficients, n_threads), key, n_threads
        )

    def half_bootstrap(
        self, ciphertext: CKKS_Ciphertext, key: SHIP_Key, n_threads: int = 0
    ) -> CKKS_Ciphertext:
        """[CHKS25, Alg. 1]: from a ciphertext at the lowest level whose
        coefficients ``t < N/2`` hold ``delta * mu_t`` (``delta`` its scale) to
        one at `output_lvl` whose slots hold ``mu``. The other coefficients are
        ignored.
        """
        scheme = self.scheme
        if ciphertext.lvl != self.lowest_lvl:
            raise ValueError(f"the input must be at the lowest level {self.lowest_lvl}")
        if scheme.rlk is None:
            raise ValueError("the product tree needs the relinearization key (rlk)")
        sparse = scheme.keyswitch(
            ciphertext, cast("MLWE_Set", key.sparse_switch_key), n_threads
        )
        factors = self.blind_rotate(sparse, key, n_threads)

        # pt0 = gamma / (4 pi i) * omega^b, gamma from the input's actual scale.
        gamma = self.q0 / ciphertext.delta
        cr = scheme.complex_ring
        pt0 = Polynomial(scheme.rings[0])
        sparse.to_coeff()
        if not lib.ship_encode_pt0(
            pt0.obj,
            sparse.obj_b(),
            0.0,
            -gamma / (4 * math.pi),
            scheme.scaling_factor,
            cr.plan,
            cr.slot_positions,
        ):
            raise ValueError(
                f"the input's scale {ciphertext.delta} gives no finite gamma"
            )
        pt0.repr = repr.ntt
        _restrict(pt0, scheme.rings[0].mask)
        first = CKKS_Ciphertext(scheme, lvl=0)
        lib_rlwe.lib.mlwe_RNS_trivial_sample_of_zero(first.obj)
        first.repr = repr.ntt
        first += pt0

        z = scheme.product([first, *factors], n_threads)
        return z + scheme.conjugate(z, cast("MLWE_Set", key.conjugation_key), n_threads)

    def blind_rotate(
        self, sparse: MLWE, key: SHIP_Key, n_threads: int = 0
    ) -> list[CKKS_Ciphertext]:
        """The ``h`` factors of [CHKS25, Alg. 1], at level 0 and the scheme's
        scale. ``sparse`` is at the lowest level under the input key, and only
        its mask ``a`` is read. Factor ``t`` holds
        ``(omega^((-s_j X^j a)_i))_(i < N/2)``, ``s_j`` the coefficient of
        window ``t``.
        """
        scheme = self.scheme
        theta, count = self.theta, 4 * self.theta
        if sparse.lvl != self.lowest_lvl:
            raise ValueError(f"the input must be at the lowest level {self.lowest_lvl}")
        if len(key.columns) != self.h:
            raise ValueError(f"the key is not for h = {self.h}")

        # The plaintexts Rot_i(pi_k), shared by every window.
        special = scheme.special_rings[0]
        cr = scheme.complex_ring
        pis = [Polynomial(special) for _ in range(count)]
        sparse.to_coeff()
        if not lib.ship_encode_pi(
            ffi.new("void *[]", [p.obj for p in pis]),
            sparse.obj_a_i(0),
            theta,
            scheme.scaling_factor,
            cr.plan,
            cr.slot_positions,
            n_threads,
        ):
            raise ValueError(
                f"the scaling factor {scheme.scaling_factor} is not finite"
            )
        for p in pis:
            p.repr = repr.ntt
            _restrict(p, special.mask)

        # Column step, over the special ring, then divided by P.
        factors = [CKKS_Ciphertext(scheme, lvl=-1, ring=special) for _ in range(self.h)]
        masks = [c for column in key.columns for c in column]
        coefficients = ffi.new("ArithElement[]", self.h * len(masks))
        ntt = domain_of(repr.ntt)
        for t in range(self.h):
            for i, p in enumerate(pis):
                entry = coefficients[t * len(masks) + t * count + i]
                entry.handle = p.obj
                entry.domain = ntt
        for c in masks:
            c.to_NTT()
        lib_rlwe.lib.mlwe_RNS_linear_combinations(
            ffi.new("void *[]", [c.obj for c in factors]),
            ffi.new("void *[]", [c.obj for c in masks]),
            coefficients,
            self.h,
            len(masks),
            n_threads,
        )
        for c in factors:
            c.repr = repr.ntt
        scheme.round_division_batch(factors, 0, n_threads)
        for c in factors:
            c.delta = scheme.scaling_factor

        # Mux step, a digit of the giant index at a time, then the alignment.
        for p in range(self.digits):
            scheme.automorphism_mux_batch(
                factors,
                self._mux_generators(p),
                [mux[p] for mux in key.muxes],
                n_threads,
                in_place=True,
            )
        if any(k is not None for k in key.alignments):
            factors = scheme.automorphism_batch(
                factors,
                [
                    self._rotation_generator(align) if k is not None else 1
                    for align, k in zip(self.alignments, key.alignments, strict=True)
                ],
                key.alignments,
                n_threads,
            )
        return factors


def _restrict(poly: RNSPolynomial, mask: int) -> None:
    """Restricts an encoded polynomial's RNS mask to its ring's primes."""
    ffi.cast("RNS_Polynomial", poly.obj).rns_mask = mask


def _lemma1_masks(j: int, sign: int, M: int) -> list[list[float]]:
    """The masks of [CHKS25, Lemma 1] for the coefficient ``sign * X^j``, so
    that ``(omega^((sign X^j a)_t))_(t < M) = sum_k Rot_j(pi_k) * mask_k``.

    With ``s+ = (1 + sign)/2``, ``s- = (1 - sign)/2`` and ``m = (0^u, 1^(M-u))``,
    ``u = j mod M``: ``[s+ m, s- m, s- (1-m), s+ (1-m)]`` for ``j < M``, and
    ``[s- (1-m), s+ (1-m), s- m, s+ m]`` otherwise.
    """
    plus, minus = (1 + sign) / 2, (1 - sign) / 2
    u = j % M
    low = [0.0] * u + [1.0] * (M - u)
    high = [1.0 - x for x in low]
    if j < M:
        pairs = [(plus, low), (minus, low), (minus, high), (plus, high)]
    else:
        pairs = [(minus, high), (plus, high), (minus, low), (plus, low)]
    return [[c * x for x in mask] for c, mask in pairs]
