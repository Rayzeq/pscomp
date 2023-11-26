from __future__ import annotations

from typing import cast

from . import lexer
from .source import SourceFile
from .typer import Argument, BuiltinSignature, Cast
from .types import Any as AnyType
from .types import Bool, Char, Float, Integer, String, Type


def _i(token: lexer.Token) -> lexer.Identifier:
    return cast(lexer.Identifier, token)


def _parse_builtin_functions(tokens: list[lexer.Token]) -> dict[str, BuiltinSignature | Cast]:
    result: dict[str, BuiltinSignature | Cast] = {}
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

        if sig := result.get(name):
            assert isinstance(sig, Cast)
            assert len(args) == 1
            sig.overload(args[0])
        elif current_source == "cast":
            assert len(args) == 1
            result[name] = Cast(name, args[0], _TYPES[ret_typ.name](ret_typ.span), _TYPE_NAMES[name])
        else:
            result[name] = BuiltinSignature(name, args, _TYPES[ret_typ.name](ret_typ.span), source=source)

    return result


BuiltinSource = SourceFile(
    """
Unknown

source: "cast"
fonction entier(valeur: booleen) retourne entier
fonction entier(valeur: reel) retourne entier
fonction entier(valeur: car) retourne entier
fonction entier(valeur: chaine) retourne entier

fonction reel(valeur: booleen) retourne reel
fonction reel(valeur: entier) retourne reel
fonction reel(valeur: chaine) retourne reel

fonction chaine(valeur: booleen) retourne chaine
fonction chaine(valeur: entier) retourne chaine
fonction chaine(valeur: reel) retourne chaine
fonction chaine(valeur: car) retourne chaine

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
    "booleen": Bool,
    "entier": Integer,
    "reel": Float,
    "car": Char,
    "chaine": String,
}
_TYPE_NAMES: dict[str, str] = {
    "booleen": "bool",
    "entier": "int",
    "reel": "float",
    "car": "str",
    "chaine": "str",
}
_tokens = lexer.lexer(BuiltinSource.sections[0])

Unknown = AnyType(_tokens.pop(0).span)

builtins = _parse_builtin_functions(_tokens)
