ENTRY(_start)
SECTIONS {
    /DISCARD/ : { *(.note*) *(.eh_frame*) }
    . = 0x80000000;
    .text : { *(.text.entry) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .sdata : { __global_pointer$ = . + 0x800; *(.sdata*) }
    .bss : { *(.sbss*) *(.bss*) *(COMMON) }
    . = ALIGN(16);
    . += 0x100000;
    _stack_top = .;
}
