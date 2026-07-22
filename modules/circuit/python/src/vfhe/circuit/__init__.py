# SPDX-License-Identifier: Apache-2.0
"""vfhe.circuit: layered GKR arithmetic circuits.

* :mod:`~vfhe.circuit.builder`: construct and (de)serialize circuits over
  the protobuf wire format (re-exported here).
* :mod:`~vfhe.circuit.export`: evaluate circuits and export their MLE
  tables / wiring predicates as ``vfhe.arith`` polynomials for the SNARK
  layers (imported explicitly; it is the only part that touches arith).
"""

from _vfhe_proto.vfhe.circuit.gkr.v1 import gkr_pb2 as gkr

from .builder import add_gate, deserialize, mul_gate, serialize

__all__ = ["gkr", "add_gate", "mul_gate", "serialize", "deserialize"]
