from __future__ import annotations

from typing import Any, cast

from . import lexer
from .parser import Float, Integer, Value
from .source import SourceFile, Spanned
from .typer import Argument, BuiltinSignature


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
            args.append(Argument(arg_name, arg_typ.span.wrap(_TYPES[arg_typ.name])))
            if isinstance(tokens[0], lexer.Comma):
                tokens.pop(0)

        tokens.pop(0)  # )
        tokens.pop(0)  # retourne

        ret_typ = _i(tokens.pop(0))

        source = ".".join(filter(bool, [current_source, name]))
        result[name] = BuiltinSignature(name, args, ret_typ.span.wrap(_TYPES[ret_typ.name]), source=source)

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

_TYPES: dict[str, type[Value[Any]]] = {
    "entier": Integer,
    "reel": Float,
}
_tokens = lexer.lexer(BuiltinSource.sections[0])

Unknown = Spanned(Value, span=_tokens.pop(0).span)

builtins = _parse_builtin_functions(_tokens)
