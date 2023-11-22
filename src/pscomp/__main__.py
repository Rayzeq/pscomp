from __future__ import annotations

import argparse
import subprocess
import sys

from . import lexer, parser, typer
from .compiler import compile
from .logger import Error
from .parser import ParserError
from .source import FileSection, SourceFile, UnsupportedFileTypeError

print("import", __name__)


def compile_section(section: FileSection) -> str:
    tokens = lexer.lexer(section)
    asts, constants = parser.parse(tokens)
    typed_asts, context = typer.parse(asts, constants)
    return compile(typed_asts, context)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        nargs="+",
        type=argparse.FileType("r"),
        help="The files you want to compile. Supported files are .md, .pseudo, .txt or stdin (using `-`)",
    )
    parser.add_argument("-r", "--run", action="store_true", help="Automatically run the compiled code")
    args = parser.parse_args()

    for file in args.file:
        try:
            sections = SourceFile(file).sections
        except UnsupportedFileTypeError as e:
            Error("Unsupported file format").detail(e.args[0]).log()
            continue

        for section in sections:
            try:
                print(f"Compiling {section.out_path}...")
                section.out_path.write_text(compile_section(section))
            except ParserError:  # noqa: PERF203
                Error("Some errors were unrecoverable, no file was written").log()
            else:
                print(f"Done, the result was written to {section.out_path}")

                if args.run:
                    print(f"Running {section.out_path}...")
                    subprocess.run([sys.executable, str(section.out_path)], check=False)  # noqa: S603 - false positive


if __name__ == "__main__":
    main()
