.section .text.entry
.globl _start
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, _stack_top
    li t0, 0x6600
    csrs mstatus, t0
    csrw fcsr, zero
    call test_main
    li t0, 0x100000
    li t1, 0x5555
    beqz a0, 1f
    slli a0, a0, 16
    li t1, 0x3333
    or t1, t1, a0
1:  sw t1, 0(t0)
2:  wfi
    j 2b
