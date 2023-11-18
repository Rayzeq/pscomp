from typing import cast

import lexer
from parser_ import Float, Integer, Value
from source import SourceFile, Spanned
from typer import Argument, BuiltinSignature

BuiltinSource = SourceFile(
    """
Unknown
# Conversions
fonction entier(valeur: reel) retourne entier
fonction reel(valeur: entier) retourne reel

# Math
fonction sqrt(number: reel) retourne reel

# Random
fonction random() retourne reel
""".strip(),
    path="<builtins>",
)

_tokens = cast(list[lexer.Identifier], lexer.lexer(BuiltinSource.sections[0]))

Unknown = Spanned(Value, span=_tokens[0].span)

builtins = {
    "entier": BuiltinSignature(
        _tokens[2].name,
        [
            Argument(_tokens[4].name, _tokens[6].span.wrap(Float)),
        ],
        _tokens[9].span.wrap(Integer),
        source="int",
    ),
    "reel": BuiltinSignature(
        _tokens[11].name,
        [
            Argument(_tokens[13].name, _tokens[15].span.wrap(Integer)),
        ],
        _tokens[18].span.wrap(Float),
        source="float",
    ),
    "sqrt": BuiltinSignature(
        _tokens[20].name,
        [
            Argument(_tokens[22].name, _tokens[24].span.wrap(Float)),
        ],
        _tokens[27].span.wrap(Float),
        source="mat.sqrt",
    ),
    "random": BuiltinSignature(
        _tokens[29].name,
        [],
        _tokens[33].span.wrap(Float),
        source="random.random",
    ),
}
