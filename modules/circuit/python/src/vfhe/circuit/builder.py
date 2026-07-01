# SPDX-License-Identifier: Apache-2.0
from _vfhe_proto.vfhe.circuit.gkr.v1 import gkr_pb2 as gkr

__all__ = ["add_gate", "mul_gate", "serialize", "deserialize"]


def add_gate(left: int, right: int) -> "gkr.Gate":
    """A fan-in-2 ADD gate wiring two inputs from the previous layer."""
    return gkr.Gate(type=gkr.GATE_TYPE_ADD, left=left, right=right)


def mul_gate(left: int, right: int) -> "gkr.Gate":
    """A fan-in-2 MUL gate wiring two inputs from the previous layer."""
    return gkr.Gate(type=gkr.GATE_TYPE_MUL, left=left, right=right)


def serialize(circuit: "gkr.Circuit") -> bytes:
    """Serialize a GKR circuit to its protobuf wire format."""
    return circuit.SerializeToString()


def deserialize(data: bytes) -> "gkr.Circuit":
    """Parse a GKR circuit from its protobuf wire format."""
    c = gkr.Circuit()
    c.ParseFromString(data)
    return c
