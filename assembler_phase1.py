import argparse
import sys
import re
import logging as log

ERROR_LIMIT = 5


class AsmSrcKind:
    FULL = "full"
    DATA = "data"
    COMMENT = "comment"
    MEMOP = "memop"
    JUMP = "jump"


PATTERNS = [
    (re.compile(
        r"\s*(?P<label>[a-zA-Z]\w*):?\s*(?P<comment>[\#;].*)?$"), AsmSrcKind.COMMENT),
    (re.compile(r"\s*(?P<label>[a-zA-Z]\w*):?\s*DATA\s+(?P<value>0x[a-fA-F0-9]+|[0-9]+)?\s*(?P<comment>[\#;].*)?$"),
     AsmSrcKind.DATA),
    (re.compile(
        r"\s*(?P<label>[a-zA-Z]\w*)?\s*(?P<opcode>[a-zA-Z]+)(/(?P<predicate>[a-zA-Z]+))?\s+(?P<target>r[0-9]+),"
        r"\s*(?P<src1>r[0-9]+),\s*(?P<src2>r[0-9]+)(\[(?P<offset>[-]?[0-9]+)\])?\s*(?P<comment>[\#;].*)?$"),
     AsmSrcKind.FULL),
    (re.compile(
        r"\s*(?P<label>[a-zA-Z]\w*)?\s*JUMP(/(?P<predicate>[a-zA-Z]+))?\s+(?P<labelref>[a-zA-Z]\w*)\s*(?P<comment>[\#;].*)?$"),
     AsmSrcKind.JUMP),
]


def parse_line(line: str) -> dict:
    """Parse one line of assembly code.
    Returns a dict containing the matched fields,
    some of which may be empty.  Raises SyntaxError
    if the line does not match assembly language
    syntax. Sets the 'kind' field to indicate
    which of the patterns was matched.
    """
    log.debug(f"\nParsing assembler line: '{line}'")
    for pattern, kind in PATTERNS:
        match = pattern.fullmatch(line)
        if match:
            fields = match.groupdict()
            fields["kind"] = kind
            log.debug(f"Extracted fields {fields}")
            return fields
    raise SyntaxError(f"Assembler syntax error in {line}")


def fill_defaults(fields: dict) -> None:
    """Fill in defaults for missing fields in an instruction.
    Should not modify fields that are already present.
    """
    if "predicate" not in fields:
        fields["predicate"] = "ALWAYS"
    if "offset" not in fields:
        fields["offset"] = "0"


def value_parse(value: str) -> int:
    """Return integer value corresponding to a DATA value field"""
    if value.startswith("0x"):
        return int(value, 16)
    else:
        return int(value)


def fix_optional_fields(fields: dict):
    """Fill in values of optional fields label,
    predicate, and comment, adding the punctuation
    they require.
    """
    if fields["label"] is None:
        fields["label"] = "    "
    else:
        fields["label"] = fields["label"] + ":"

    if fields["predicate"] is None:
        fields["predicate"] = ""

    if fields["comment"] is None:
        fields["comment"] = ""


def transform(lines: list[str]) -> list[str]:
    """
    Transform some assembly language lines, leaving others
    unchanged.
    Initial version:  No changes to any source line.

    Planned version:
       again:   STORE r1,r0,r15[4]   # x
                SUB   r1,r0,r0[1]
                ADD   r15,r0,r15[-2]
                HALT r0,r0,r0
       x:       DATA 0
    should become
       again:   STORE r1,r0,r15[4]   # x
                SUB   r1,r0,r0[1]
                ADD   r15,r0,r15[-2]
                HALT r0,r0,r0
       x:       DATA 0
     """
    labels = resolve(lines)
    transformed = []
    address = 0

    for lnum in range(len(lines)):
        line = lines[lnum].rstrip()
        log.debug(f"Processing line {lnum}: {line}")

        try:
            fields = parse_line(line)

            if fields["kind"] == AsmSrcKind.FULL:
                fill_defaults(fields)
                fix_optional_fields(fields)
                ref = fields["labelref"]
                mem_addr = labels[ref]
                pc_relative = mem_addr - address
                full = (f"{fields['label']}   {fields['opcode']}{fields['predicate']} "
                        f"{fields['target']},r0,r15[{pc_relative}] #{ref} {fields['comment']}")
                transformed.append(squish(full))
            elif fields["kind"] == AsmSrcKind.DATA:
                transformed.append(line)
            else:
                transformed.append(line)

            if fields["kind"] != AsmSrcKind.COMMENT:
                address += 1

        except SyntaxError as e:
            print(f"Syntax error in line {lnum}: {line}", file=sys.stderr)
        except KeyError as e:
            print(f"Unknown word in line {lnum}: {e}", file=sys.stderr)
        except Exception as e:
            print(
                f"Exception encountered in line {lnum}: {e}", file=sys.stderr)

        if lnum >= ERROR_LIMIT:
            print("Too many errors; abandoning", file=sys.stderr)
            sys.exit(1)

    return transformed


def resolve(lines: list[str]) -> dict[str, int]:
    """
    Build table associating labels in the source code
    with addresses.
    """
    labels = {}
    address = 0

    for lnum in range(len(lines)):
        line = lines[lnum].rstrip()
        log.debug(f"Processing line {lnum}: {line}")

        try:
            fields = parse_line(line)
            if fields["kind"] == AsmSrcKind.COMMENT and fields.get("label"):
                labels[fields["label"]] = address
            if fields["kind"] != AsmSrcKind.COMMENT:
                address += 1
        except Exception as e:
            log.debug(f"Exception encountered line {lnum}: {e}")

    return labels


def cli() -> object:
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(
        description="Duck Machine Assembler (phase 1)")
    parser.add_argument("sourcefile", type=argparse.FileType('r'), nargs="?", default=sys.stdin,
                        help="Duck Machine assembly code file")
    parser.add_argument("objfile", type=argparse.FileType('w'), nargs="?", default=sys.stdout,
                        help="Transformed assembly language file")
    args = parser.parse_args
