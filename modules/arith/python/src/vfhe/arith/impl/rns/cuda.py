# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""RNS polynomials held on a CUDA device: the ``cuda`` backend of ``rns``.

The backend registers only where `vfhe.util.gpu.available()`. Then
``Ring(..., backend="cuda")`` builds a `CUDARing`, the ring with the same
primes as the host one and elements on the device, `CUDAPolynomial`. Add,
subtract and negate run there; every other operation raises
`NotImplementedError` until its kernel exists. An element moves between the
host and the device through `Polynomial.to_backend`, never behind an
operator: combining a host element with a device one raises `TypeError`.

>>> ring = Ring(256, 300, split_degree=1)
>>> a, b = ring.random_element(), ring.random_element()
>>> (a.to_backend("cuda") + b.to_backend("cuda")).to_backend("ntt") == a + b
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vfhe.arith.base import Polynomial
from vfhe.arith.registry import common_spec, register, register_conversion
from vfhe.arith.spec import Capability, Constraints, Spec
from vfhe.util import gpu
from vfhe.util.bindings import ffi as host_ffi
from vfhe.util.bindings import lib as host_lib

from .polynomial import RNS_NTT, Representation, RNSPolynomial, RNSRing

if TYPE_CHECKING:
    from typing import Self

_DEVICE_RINGS: dict[RNSRing, CUDARing] = {}


class CUDARing(RNSRing):
    """An `RNSRing` whose elements live on the CUDA device.

    Same primes, shared base and mask as the host ring of the same parameters,
    and a different parent: the two compare unequal. `host` is that ring.
    """

    @classmethod
    def of(cls, host: RNSRing) -> CUDARing:
        """The device ring with `host`'s parameters, one per host ring."""
        ring = _DEVICE_RINGS.get(host)
        if ring is None:
            ring = cls(host.N, primes=host.primes, split_degree=host.split_degree)
            ring._host = host
            _DEVICE_RINGS[host] = ring
        return ring

    @property
    def host(self) -> RNSRing:
        """The host ring with the same parameters."""
        if not hasattr(self, "_host"):
            self._host = RNSRing(
                self.N, primes=self.primes, split_degree=self.split_degree
            )
        return self._host

    def __eq__(self, other: object) -> bool:
        return type(other) is CUDARing and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash((self.parameters, self.spec.backend))


class CUDAPolynomial(Polynomial):
    """An element of a `CUDARing`: `ring.ell` rows of `ring.N` residues on the device.

    `repr` is the representation the rows hold, kept from the host element
    that was uploaded; the device transforms nothing, so two operands must
    hold the same one.
    """

    if TYPE_CHECKING:
        # Declaration only, as on the fronts: the metaclass builds the class.
        def __new__(
            cls, ring: CUDARing, repr: Representation = ...
        ) -> CUDAPolynomial: ...

    ring: CUDARing
    #: The device element, released with this object.
    obj: Any

    def __init__(
        self, ring: CUDARing, repr: Representation = Representation.empty
    ) -> None:
        """Allocate an element of ``ring``, holding no value yet."""
        self.ring = ring
        ffi, lib = gpu.extension()
        handle = ffi.new("void *[1]")
        gpu.check(
            lib.arith_gpu_rns_new(
                handle, ring.ell, ring.N, ffi.new("uint64_t[]", _by_index(ring))
            )
        )
        self.obj = ffi.gc(handle[0], lib.arith_gpu_rns_free)
        self.repr = repr

    def copy(self) -> CUDAPolynomial:
        out = CUDAPolynomial(self.ring, self.repr)
        gpu.check(gpu.extension()[1].arith_gpu_rns_copy(out.obj, self.obj))
        return out

    def from_array(self, array) -> Self:
        _load(self, Polynomial(self.ring.host).from_array(list(array)))
        return self

    def get_polynomial(self, signed: bool = False) -> list[int]:
        return _download(self).get_polynomial(signed)

    def _operand(self, other) -> CUDAPolynomial:
        if not isinstance(other, CUDAPolynomial):
            common_spec(self.ring.spec, other.ring.spec)
        if other.ring != self.ring:
            raise ValueError("the elements belong to different rings")
        if other.repr != self.repr:
            raise ValueError("the elements hold different representations")
        return other

    def _run(self, name: str, out: CUDAPolynomial, *operands) -> CUDAPolynomial:
        lib = gpu.extension()[1]
        gpu.check(getattr(lib, name)(out.obj, *(p.obj for p in operands)))
        out.repr = self.repr
        return out

    def __add__(self, other) -> CUDAPolynomial:
        return self._run(
            "arith_gpu_rns_add", CUDAPolynomial(self.ring), self, self._operand(other)
        )

    def __sub__(self, other) -> CUDAPolynomial:
        return self._run(
            "arith_gpu_rns_sub", CUDAPolynomial(self.ring), self, self._operand(other)
        )

    def __neg__(self) -> CUDAPolynomial:
        return self._run("arith_gpu_rns_negate", CUDAPolynomial(self.ring), self)

    def __iadd__(self, other) -> Self:
        self._run("arith_gpu_rns_add", self, self, self._operand(other))
        return self

    def __isub__(self, other) -> Self:
        self._run("arith_gpu_rns_sub", self, self, self._operand(other))
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CUDAPolynomial):
            return _download(self) == _download(other)
        if isinstance(other, Polynomial):
            common_spec(self.ring.spec, other.ring.spec)
        return _download(self) == other

    __hash__ = Polynomial.__hash__


def _by_index(ring: RNSRing) -> list[int]:
    """The primes in ascending base index: the row order of the exported words."""
    return [p for _, p in sorted(zip(ring.prime_indices, ring.primes, strict=True))]


def _load(device: CUDAPolynomial, host: RNSPolynomial) -> None:
    """Copies `host`'s rows into `device`, representation included."""
    words = bytearray(8 * host.ring.ell * host.ring.N)
    host_lib.polynomial_RNS_export_u64(
        host_ffi.from_buffer("uint64_t[]", words), host.obj
    )
    ffi, lib = gpu.extension()
    gpu.check(
        lib.arith_gpu_rns_upload(device.obj, ffi.from_buffer("uint64_t[]", words))
    )
    device.repr = host.repr


def _upload(host: RNSPolynomial) -> CUDAPolynomial:
    device = CUDAPolynomial(CUDARing.of(host.ring))
    _load(device, host)
    return device


def _download(device: CUDAPolynomial) -> RNSPolynomial:
    host = Polynomial(device.ring.host, device.repr)
    words = bytearray(8 * host.ring.ell * host.ring.N)
    ffi, lib = gpu.extension()
    gpu.check(
        lib.arith_gpu_rns_download(ffi.from_buffer("uint64_t[]", words), device.obj)
    )
    host_lib.polynomial_RNS_import_u64(
        host.obj, host_ffi.from_buffer("uint64_t[]", words)
    )
    return host


#: RNS residues on a CUDA device; never the default, and absent without one.
RNS_CUDA = Spec(
    implementation="rns",
    backend="cuda",
    parent_cls=CUDARing,
    element_cls=CUDAPolynomial,
    capabilities=Capability.CORE | Capability.QUOTIENT_POLY_RING | Capability.EXACT,
    constraints=Constraints(max_prime_bits=62),
    rank=1,
)
CUDARing.spec = RNS_CUDA

if gpu.available():
    register(RNS_CUDA)
    register_conversion(RNS_NTT.key, RNS_CUDA.key, _upload)
    register_conversion(RNS_CUDA.key, RNS_NTT.key, _download)
