from __future__ import annotations

from typing import cast

from . import lexer
from .source import SourceFile
from .typer import Argument, BuiltinSignature
from .types import Any as AnyType
from .types import Float, Integer, Type


def _i(token: lexer.Token) -> lexer.Identifier:
    return cast(lexer.Identifier, token)


def _parse_builtin_functions(tokens: list[lexer.Token]) -> dict[str, BuiltinSignature]:
    result: dict[str, BuiltinSignature] = {}
    current_source = ""

    while tokens:
        if _i(tokens[0]).name == "source":
            tokens.pop(0)  # source
            tokens.pop(0)  # :
            current_source = cast(lexer.String, tokens.pop(0)).value
            continue

        tokens.pop(0)  # fonction
        name = _i(tokens.pop(0)).name
        tokens.pop(0)  # (

        args = []
        while not isinstance(tokens[0], lexer.RParen):
            arg_name = _i(tokens.pop(0)).name
            tokens.pop(0)  # :
            arg_typ = _i(tokens.pop(0))
            args.append(Argument(arg_name, _TYPES[arg_typ.name](arg_typ.span)))
            if isinstance(tokens[0], lexer.Comma):
                tokens.pop(0)

        tokens.pop(0)  # )
        tokens.pop(0)  # retourne

        ret_typ = _i(tokens.pop(0))

        source = ".".join(filter(bool, [current_source, name]))
        result[name] = BuiltinSignature(name, args, _TYPES[ret_typ.name](ret_typ.span), source=source)

    return result


BuiltinSource = SourceFile(
    """
Unknown

source: ""
fonction entier(valeur: reel) retourne entier
fonction reel(valeur: entier) retourne reel

source: "math"
fonction sqrt(number: reel) retourne reel
fonction pow(number: reel) retourne reel
fonction sin(number: reel) retourne reel
fonction cos(number: reel) retourne reel
fonction exp(number: reel) retourne reel
fonction log(number: reel) retourne reel

source: "random"
fonction random() retourne reel
""".strip(),
    path="<builtins>",
)

_TYPES: dict[str, type[Type]] = {
    "entier": Integer,
    "reel": Float,
}
_tokens = lexer.lexer(BuiltinSource.sections[0])

Unknown = AnyType(_tokens.pop(0).span)

builtins = _parse_builtin_functions(_tokens)
