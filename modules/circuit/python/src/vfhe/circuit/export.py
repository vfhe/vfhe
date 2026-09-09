# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Export a GKR circuit to the polynomial objects the SNARK layers consume.

The GKR protocol reasons about a layered circuit through multilinear
extensions (MLEs) of three families of functions per layer ``l``:

* ``W_l(x)``     : the wire values of layer ``l`` (given a concrete input),
* ``add_l(z,x,y)``: 1 iff gate ``z`` of layer ``l`` is an ADD gate wired to
  outputs ``x`` and ``y`` of the previous layer,
* ``mul_l(z,x,y)``: likewise for MUL gates — and, for an XMULT (inner
  product) gate, 1 at every pair ``(x_i, y_i)`` of gate ``z``: the layer
  identity ``sum_{x,y} mul_l(z,x,y) W_l(x) W_l(y)`` is the inner product.

:func:`sparse_wiring` gives the same predicates in the generalized form of
Libra / zkCNN as sparse maps: ``lin_l(z,x)`` (the linear part — an ADD gate
contributes ``1`` at ``(z, left)`` and ``(z, right)``) and ``xmult_l(z,x,y)``
(the bilinear part — one entry per MUL gate, one per XMULT pair), so that
``W_{l+1}(z) = sum_x lin_l(z,x) W_l(x) + sum_{x,y} xmult_l(z,x,y) W_l(x) W_l(y)``
with as many nonzeros as wires, whatever the layer widths.

This module computes the *evaluation tables* of those functions over the
boolean hypercube (dense 0/1 or field-value vectors, padded to powers of
two), evaluates their MLEs at arbitrary points (:func:`mle_eval`, the
operation the sumcheck verifier needs), and packs the tables into
``vfhe.arith`` ring elements (:func:`wiring_polynomials`,
:func:`values_polynomial`) so the polynomial-commitment layers can commit to
them. Table entries are loaded as *coefficients* of the ring element.

Conventions
-----------
* ``circuit.layers`` are listed input to output: ``layers[0]`` reads the
  circuit inputs, ``layers[-1]`` produces the outputs.
* Every layer domain is padded to a power of two; positions past the real
  gate/input count read as 0.
* A wiring index packs its three coordinates as ``z || x || y`` with ``z``
  most significant: ``idx = ((z << s_in) | x) << s_in | y`` where ``s_in`` is
  the bit width of the previous layer's padded domain.
* ``circuit.field_modulus`` (decimal string) reduces wire values when set;
  otherwise evaluation is over the integers.

Note: a wiring table has ``2^(s_out + 2*s_in)`` entries; exponential in the
layer widths' bit sizes. That is inherent to the dense-MLE representation;
the sparse representations used by production GKR provers can be added
behind the same API later.
"""

from __future__ import annotations

from _vfhe_proto.vfhe.circuit.gkr.v1 import gkr_pb2 as gkr

__all__ = [
    "evaluate",
    "gate_pairs",
    "gate_wires",
    "layer_bit_sizes",
    "mle_eval",
    "sparse_wiring",
    "value_tables",
    "values_polynomial",
    "wiring_polynomials",
    "wiring_tables",
]


def _bits(n: int) -> int:
    """Bit width of the padded domain covering ``n`` positions (min 1)."""
    if n <= 1:
        return 1
    return (n - 1).bit_length()


def _modulus(circuit: gkr.Circuit) -> int | None:
    """The circuit's field modulus as an int, or None if unset."""
    return int(circuit.field_modulus) if circuit.field_modulus else None


def layer_bit_sizes(circuit: gkr.Circuit) -> list[int]:
    """Per-domain bit widths ``[s_input, s_layer0, s_layer1, ...]``.

    Entry 0 covers the circuit inputs; entry ``l + 1`` covers the gates of
    ``circuit.layers[l]``. Each is ``ceil(log2)`` of the domain size, so the
    padded domain has ``2^s`` positions.
    """
    sizes = [_bits(circuit.num_inputs)]
    sizes.extend(_bits(len(layer.gates)) for layer in circuit.layers)
    return sizes


def evaluate(circuit: gkr.Circuit, inputs: list[int]) -> list[list[int]]:
    """Run the circuit on ``inputs`` and return every layer's wire values.

    Args:
        circuit: the GKR circuit.
        inputs: ``circuit.num_inputs`` field elements.

    Returns:
        One list per domain, aligned with :func:`layer_bit_sizes`:
        ``[inputs, layer0 outputs, layer1 outputs, ...]`` (unpadded lengths,
        reduced mod ``field_modulus`` when set).

    Raises:
        ValueError: on an input-length mismatch, an out-of-range gate wire,
            or an unspecified gate type.
    """
    if len(inputs) != circuit.num_inputs:
        raise ValueError(f"expected {circuit.num_inputs} inputs, got {len(inputs)}")
    mod = _modulus(circuit)

    values = [list(inputs) if mod is None else [v % mod for v in inputs]]
    for layer_idx, layer in enumerate(circuit.layers):
        prev = values[-1]
        out: list[int] = []
        for g, gate in enumerate(layer.gates):
            if any(w >= len(prev) for w in gate_wires(gate)):
                raise ValueError(f"layer {layer_idx} gate {g}: wire index out of range")
            if gate.type == gkr.GATE_TYPE_ADD:
                v = prev[gate.left] + prev[gate.right]
            elif gate.type == gkr.GATE_TYPE_MUL:
                v = prev[gate.left] * prev[gate.right]
            elif gate.type == gkr.GATE_TYPE_XMULT:
                v = sum(prev[a] * prev[b] for a, b in gate_pairs(gate))
            else:
                raise ValueError(f"layer {layer_idx} gate {g}: unspecified gate type")
            out.append(v % mod if mod is not None else v)
        values.append(out)
    return values


def gate_pairs(gate: gkr.Gate) -> list[tuple[int, int]]:
    """The multiplied wire pairs of a MUL or XMULT gate (empty for ADD)."""
    if gate.type == gkr.GATE_TYPE_MUL:
        return [(gate.left, gate.right)]
    if gate.type == gkr.GATE_TYPE_XMULT:
        if len(gate.lefts) != len(gate.rights):
            raise ValueError("XMULT gate with unpaired wires")
        return list(zip(gate.lefts, gate.rights, strict=True))
    return []


def gate_wires(gate: gkr.Gate) -> list[int]:
    """Every previous-layer wire a gate reads."""
    if gate.type == gkr.GATE_TYPE_XMULT:
        return [*gate.lefts, *gate.rights]
    return [gate.left, gate.right]


def value_tables(circuit: gkr.Circuit, inputs: list[int]) -> list[list[int]]:
    """:func:`evaluate`, with every layer zero-padded to its ``2^s`` domain.

    These are the dense MLE tables of the ``W_l`` functions.
    """
    sizes = layer_bit_sizes(circuit)
    padded = []
    for s, vals in zip(sizes, evaluate(circuit, inputs), strict=True):
        padded.append(vals + [0] * ((1 << s) - len(vals)))
    return padded


def wiring_tables(circuit: gkr.Circuit, layer: int) -> tuple[list[int], list[int]]:
    """Dense MLE tables of ``add_l`` and ``mul_l`` for one layer.

    Args:
        circuit: the GKR circuit.
        layer: index into ``circuit.layers``.

    Returns:
        ``(add_table, mul_table)``: 0/1 vectors of length
        ``2^(s_out + 2*s_in)`` indexed by ``z || x || y`` (see module docs).

    Raises:
        ValueError: on a bad layer index or an unspecified gate type.
    """
    if not 0 <= layer < len(circuit.layers):
        raise ValueError(f"layer {layer} out of range (0..{len(circuit.layers) - 1})")
    sizes = layer_bit_sizes(circuit)
    s_in, s_out = sizes[layer], sizes[layer + 1]

    length = 1 << (s_out + 2 * s_in)
    add_table = [0] * length
    mul_table = [0] * length
    for z, gate in enumerate(circuit.layers[layer].gates):
        if gate.type == gkr.GATE_TYPE_ADD:
            add_table[(((z << s_in) | gate.left) << s_in) | gate.right] = 1
        elif gate.type in (gkr.GATE_TYPE_MUL, gkr.GATE_TYPE_XMULT):
            for a, b in gate_pairs(gate):
                mul_table[(((z << s_in) | a) << s_in) | b] += 1
        else:
            raise ValueError(f"layer {layer} gate {z}: unspecified gate type")
    return add_table, mul_table


def sparse_wiring(
    circuit: gkr.Circuit, layer: int
) -> tuple[dict[int, int], dict[int, int]]:
    """One layer's wiring in the generalized (Libra / zkCNN) form, as sparse
    maps from packed index to coefficient:

    * ``lin``: indexed ``z || x || y`` **with y = 0**, one entry per linear
      summand — an ADD gate ``z = in[a] + in[b]`` gives ``lin[(z, a, 0)] += 1``
      and ``lin[(z, b, 0)] += 1``;
    * ``xmult``: indexed ``z || x || y``, one entry per multiplied pair — a
      MUL gate gives ``xmult[(z, a, b)] += 1``, an XMULT gate one entry per
      pair.

    Both are indexed like :func:`wiring_tables` (``y`` in the low ``s_in``
    bits, then ``x``, then ``z``); ``lin`` keeps the ``y`` bits at zero so that
    summing ``lin(z,x,y) W(x)`` over the full ``(x, y)`` cube counts each
    summand once. The number of entries is the number of wires read, not
    ``2^(s_out + 2 s_in)``.

    Raises:
        ValueError: on a bad layer index or an unspecified gate type.
    """
    if not 0 <= layer < len(circuit.layers):
        raise ValueError(f"layer {layer} out of range (0..{len(circuit.layers) - 1})")
    sizes = layer_bit_sizes(circuit)
    s_in = sizes[layer]
    lin: dict[int, int] = {}
    xmult: dict[int, int] = {}

    def idx(z: int, x: int, y: int) -> int:
        return (((z << s_in) | x) << s_in) | y

    for z, gate in enumerate(circuit.layers[layer].gates):
        if gate.type == gkr.GATE_TYPE_ADD:
            for a in (gate.left, gate.right):
                lin[idx(z, a, 0)] = lin.get(idx(z, a, 0), 0) + 1
        elif gate.type in (gkr.GATE_TYPE_MUL, gkr.GATE_TYPE_XMULT):
            for a, b in gate_pairs(gate):
                xmult[idx(z, a, b)] = xmult.get(idx(z, a, b), 0) + 1
        else:
            raise ValueError(f"layer {layer} gate {z}: unspecified gate type")
    return lin, xmult


def wiring_polynomials(circuit: gkr.Circuit, layer: int, ring):
    """Pack a layer's wiring tables into two ``vfhe.arith`` ring elements.

    The tables become the *coefficients* of the returned polynomials
    (zero-padded to ``ring.N``), ready for the polynomial-commitment layer.

    Args:
        circuit: the GKR circuit.
        layer: index into ``circuit.layers``.
        ring: a ``vfhe.arith.Ring`` with ``ring.N`` >= the table length.

    Returns:
        ``(add_poly, mul_poly)`` as ``vfhe.arith.Polynomial``.

    Raises:
        ValueError: if the tables do not fit the ring degree.
    """
    from vfhe.arith import (
        Polynomial,
    )  # lazy: circuit stays usable without the native ext

    add_table, mul_table = wiring_tables(circuit, layer)
    if len(add_table) > ring.N:
        raise ValueError(
            f"wiring tables of layer {layer} have {len(add_table)} entries; ring.N={ring.N}"
        )
    return (
        Polynomial(ring).from_array(add_table),
        Polynomial(ring).from_array(mul_table),
    )


def values_polynomial(circuit: gkr.Circuit, inputs: list[int], layer: int, ring):
    """Pack one layer's padded wire values into a ``vfhe.arith`` ring element.

    ``layer`` indexes the domains of :func:`layer_bit_sizes`: 0 is the input
    layer, ``l + 1`` is ``circuit.layers[l]``. Values load as coefficients
    (zero-padded to ``ring.N``).

    Raises:
        ValueError: on a bad layer index or if the table does not fit.
    """
    from vfhe.arith import Polynomial

    tables = value_tables(circuit, inputs)
    if not 0 <= layer < len(tables):
        raise ValueError(f"layer {layer} out of range (0..{len(tables) - 1})")
    table = tables[layer]
    if len(table) > ring.N:
        raise ValueError(
            f"value table of layer {layer} has {len(table)} entries; ring.N={ring.N}"
        )
    return Polynomial(ring).from_array(table)


def mle_eval(table: list[int], point: list[int], modulus: int) -> int:
    """Evaluate the multilinear extension of a hypercube table at a point.

    The table lists ``f`` over ``{0,1}^k`` with the *first* coordinate as the
    most significant index bit (matching the ``z || x || y`` packing above).
    Folds one variable per pass, so the cost is ``O(len(table))`` field
    operations.

    Args:
        table: ``2^k`` values of ``f`` on the hypercube.
        point: ``k`` field elements ``(r_1, ..., r_k)``.
        modulus: the field modulus.

    Returns:
        ``MLE(f)(r_1, ..., r_k) mod modulus``.

    Raises:
        ValueError: if the table length is not ``2^len(point)``.
    """
    if len(table) != 1 << len(point):
        raise ValueError(
            f"table of length {len(table)} needs {_bits(len(table))} variables"
        )
    acc = [v % modulus for v in table]
    for r in point:
        half = len(acc) // 2
        # f(r, rest) = (1 - r) * f(0, rest) + r * f(1, rest)
        acc = [(acc[i] + r * (acc[half + i] - acc[i])) % modulus for i in range(half)]
    return acc[0]
