/* SPDX-License-Identifier: Apache-2.0 */
/* asm_mul64(a, b) -> low 64 bits of a*b. x86-64 (System V: a=rdi, b=rsi, ret=rax).
 * Preprocessed (.S): mach-o needs a leading underscore on symbols, ELF does not. */
#if defined(__APPLE__)
#define SYM(name) _##name
#else
#define SYM(name) name
#endif

.text
.globl SYM(asm_mul64)
SYM(asm_mul64):
    movq %rdi, %rax         /* rax = a */
    imulq %rsi, %rax        /* rax = a * b (low 64 bits) */
    ret
