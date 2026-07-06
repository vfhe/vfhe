# base

Foundational C utilities shared across modules: safe (abort-on-failure)
allocation, power-of-two and bit-reversal helpers, modular switching, and
debug printing. Internal C only — no Python API, no internal dependencies.

Why this is its own module: both `rng` and `arith` need these helpers, and
they depend on each other in one direction only (`arith -> rng`). Folding
`base` into `arith` would force `rng -> arith` and create a cycle; folding it
into `rng` would put allocation policy in a randomness module. Keeping the
dependency-free bottom layer separate keeps the module graph a DAG:

```
base <- rng <- arith <- (mlwe, fhe, ...)
```
