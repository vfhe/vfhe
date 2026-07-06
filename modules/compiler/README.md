# vfhe.compiler

Circuit compiler / frontend that lowers programs to GKR circuits
(`vfhe.circuit` protobufs). The circuit -> polynomial export lives in
`vfhe.circuit.export`; this module's job is the level above it: turning
source programs into layered circuits.
