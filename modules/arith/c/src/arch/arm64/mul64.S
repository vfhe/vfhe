/* SPDX-License-Identifier: Apache-2.0 */
/* asm_mul64(a, b) -> low 64 bits of a*b. AArch64 (arm64 / aarch64).
 * Preprocessed (.S): mach-o needs a leading underscore on symbols, ELF does not. */
#if defined(__APPLE__)
#define SYM(name) _##name
#else
#define SYM(name) name
#endif

.text
.globl SYM(asm_mul64)
.p2align 2
SYM(asm_mul64):
    mul x0, x0, x1          /* x0 = a * b (low 64 bits) */
    ret
