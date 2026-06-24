# Projects
A compilation of projects that I've done. 

**Duck Machine Assembler — Phase 1**

What This Is

Writing code directly in machine language (raw numbers the computer understands) is painful and error-prone. Assembly language is a human-readable alternative: instead of calculating numeric instruction values by hand, you write things like ADD or STORE, and an assembler translates that into the binary the machine actually runs.

This project is Phase 1 of a two-phase assembler for the Duck Machine, a simulated CPU architecture used in CIS 211 at the University of Oregon. Phase 1's job is to take "shorthand" assembly code and resolve it into a fully explicit form that Phase 2 can then convert into machine code.

Specifically, Phase 1 handles two things that raw Duck Machine assembly doesn't support:


Symbolic labels — instead of calculating memory addresses by hand (e.g., "jump 3 instructions back"), you label a line and refer to it by name
JUMP pseudo-instructions — a cleaner way to write conditional and unconditional branches, which get translated into the actual ADD r15,... instructions the CPU understands


Example

Instead of writing:

     ADD    r1,r1,r0[2]
     STORE  r1,r0,r0[511]
     SUB    r0,r1,r0[10]
     ADD/P  r15,r0,r15[-3]

You can write:

again: ADD    r1,r1,r0[2]
       STORE  r1,r0,r0[511]
       SUB    r0,r1,r0[10]
       JUMP/P again

Phase 1 resolves again to its actual address and rewrites JUMP/P again as ADD/P r15,r0,r15[-3], which Phase 2 can then encode into machine code.


How It Works

Two-Pass Algorithm

Because a label might be used before it's defined (e.g., jumping forward to a label that appears later in the file), the assembler makes two passes through the source code:


Pass 1 (resolve) — reads every line and builds a dictionary mapping each label name to its memory address
Pass 2 (transform) — goes through the lines again and rewrites any instruction that references a label, replacing it with a PC-relative address


PC-relative means the offset is calculated as target_address - current_address, so the code works regardless of where in memory it's loaded.

Line Types

Each line of assembly is matched against one of four patterns:

KindExampleWhat happensCOMMENT# this is a comment or a blank linePassed through unchanged; does not count as a memory addressDATAx: DATA 42Passed through unchanged; counts as one memory wordFULLADD r1,r1,r0[2]Passed through unchanged; already fully resolvedJUMPJUMP/P againRewritten as ADD/P r15,r0,r15[offset] #again

Labels on any line type are recorded during the first pass.


Files


assembler_phase1.py — this file; Phase 1 assembler
assembler_phase2.py (provided) — takes fully resolved assembly and produces object code (machine instructions as integers)
run/asmgo.py (provided) — convenience script that chains Phase 1, Phase 2, and the Duck Machine simulator together in one command



How to Run

Basic usage

bashpython3 assembler_phase1.py input.asm output.asm


input.asm — your assembly source file (can use labels and JUMP)
output.asm — the resolved output, ready for Phase 2


If no files are specified, it reads from stdin and writes to stdout.

Full pipeline (Phase 1 + Phase 2 + run)

bashpython3 run/asmgo.py programs/asm/your_program.asm


Key Functions

resolve(lines) → dict[str, int]

First pass. Scans all lines and maps each label to its memory address. Comment lines don't count toward addresses; all other line types take up one memory word.

transform(lines) → list[str]

Second pass. Calls resolve first to get the label table, then rewrites each line. JUMP instructions become ADD r15,... instructions with a computed PC-relative offset. Lines that don't need changes are passed through as-is.

parse_line(line) → dict

Tries each regex pattern against a line and returns a dictionary of named fields (label, opcode, predicate, target, offset, comment, etc.) plus a kind field indicating which pattern matched. Raises SyntaxError if nothing matches.

resolve_labels(fields, labels, address)

Given parsed fields, the label table, and the current address, computes the PC-relative offset for a label reference.


Error Handling


Syntax errors, unknown labels, and unexpected exceptions are printed to stderr with the offending line number
The assembler stops after 5 errors to avoid flooding output



Dependencies


Python 3.10+
Standard library only (re, argparse, sys, logging)


