from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from .errors import InternalCompilerError
from .logger import Error, Warn
from .source import FileSection, Position, SourceFile, Span, SpannedStr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Never, Self


class StringReader:
    __slots__ = ("file", "text", "span")

    file: SourceFile
    text: str
    span: Span

    def __init__(self: Self, section: FileSection) -> None:
        self.file = section.file
        self.text = self.file.content
        self.span = section.span

    @property
    def pos(self: Self) -> Position:
        return self.span.start

    def peek(self: Self) -> str:
        return self.text[self.span.start]

    def try_peek(self: Self) -> str | None:
        return self.peek() if self else None

    def pop(self: Self) -> str:
        self.span.start += 1
        return self.text[self.span.start - 1]

    def try_pop(self: Self) -> str | None:
        return self.pop() if self else None

    def __bool__(self: Self) -> bool:
        return self.span.start != self.span.end


T = TypeVar("T", bound="Token[Any]")


class Token(ABC, Generic[T]):
    __slots__ = ("span",)

    span: Span

    @classmethod
    def parse(cls: type[Self], reader: StringReader) -> T:
        start = reader.pos
        self = cls._parse(reader)
        self.span = Span.from_(start, reader.pos)

        return self

    @classmethod
    @abstractmethod
    def _parse(cls: type[Self], reader: StringReader) -> T:
        pass


class NoParse(Token["Never"]):
    @classmethod
    def _parse(cls: type[Self], _: StringReader) -> Never:
        msg = f"Can't call {cls.__name__}.parse directly"
        raise InternalCompilerError(msg)


class SingleCharToken(Token["SingleCharToken"]):
    __all__: ClassVar[dict[str, type[SingleCharToken]]] = {}

    def __init_subclass__(cls: type[Self], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__all__[cls.value] = cls

    __slots__ = ("value",)

    value: str

    @classmethod
    def _parse(cls: type[Self], reader: StringReader) -> Self:
        reader.pop()
        return cls()

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, type(self)):
            return True
        return NotImplemented

    def __hash__(self: Self) -> int:
        return hash(self.value)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__} {self.span}"

    def __str__(self: Self) -> str:
        return self.value


class Identifier(Token["Identifier | Operator"]):
    __slots__ = ("_name",)

    _name: str

    @classmethod
    def _parse(cls: type[Self], reader: StringReader) -> Self | Operator:
        name = ""
        while reader and (reader.peek().isalnum() or reader.peek() == "_"):
            name += reader.pop()

        if name in KEYWORDS.accented:
            name = KEYWORDS.accented[name]

        if name == "mod":
            return Operator(OperatorType.MOD)
        if name in ("et", "ou", "non"):
            return Operator(OperatorType(name))

        return cls(name)

    def __init__(self: Self, name: str) -> None:
        self._name = name

    @property
    def name(self: Self) -> SpannedStr:
        return SpannedStr(self._name, span=self.span)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Identifier):
            return self._name == other._name
        return NotImplemented

    def __hash__(self: Self) -> int:
        return hash(self._name)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self._name} {self.span if hasattr(self, 'span') else '[artificial]'})"

    def __str__(self: Self) -> str:
        return self._name


class KEYWORDS:
    programme = Identifier("programme")
    debut = Identifier("debut")
    fin = Identifier("fin")
    avec = Identifier("avec")
    booleen = Identifier("booleen")
    entier = Identifier("entier")
    reel = Identifier("reel")
    car = Identifier("car")
    chaine = Identifier("chaine")
    null = Identifier("null")
    vrai = Identifier("vrai")
    faux = Identifier("faux")
    saisir = Identifier("saisir")
    afficher = Identifier("afficher")
    si = Identifier("si")
    alors = Identifier("alors")
    sinon = Identifier("sinon")
    et = Identifier("et")
    ou = Identifier("ou")
    non = Identifier("non")
    selon = Identifier("selon")
    cas = Identifier("cas")
    defaut = Identifier("defaut")
    faire = Identifier("faire")
    pour = Identifier("pour")
    de = Identifier("de")
    a = Identifier("a")
    pas = Identifier("pas")
    tant = Identifier("tant")
    que = Identifier("que")
    repeter = Identifier("repeter")
    fonction = Identifier("fonction")
    retourne = Identifier("retourne")
    procedure = Identifier("procedure")
    detruire = Identifier("detruire")
    __python__ = Identifier("__python__")

    accented: Mapping[str, str] = {
        "à": "a",
        "début": "debut",
        "défaut": "defaut",
        "booléen": "booleen",
        "réel": "reel",
        "chaîne": "chaine",
        "répéter": "repeter",
        "procédure": "procedure",
    }


class Integer(Token["Integer | Float"]):
    __slots__ = ("value",)

    value: int

    @classmethod
    def _parse(cls: type[Self], reader: StringReader) -> Self | Float:
        value = ""
        while reader and reader.peek() in "0123456789_":
            value += reader.pop()

        if reader.try_peek() == ".":
            value += reader.pop()
            while reader and reader.peek() in "0123456789_":
                value += reader.pop()

            return Float(float(value))

        return cls(int(value))

    def __init__(self: Self, value: int) -> None:
        self.value = value

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Integer):
            return self.value == other.value
        return NotImplemented

    def __hash__(self: Self) -> int:
        return hash(self.value)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.value} {self.span})"

    def __str__(self: Self) -> str:
        return str(self.value)


class Float(NoParse):
    __slots__ = ("value",)

    value: float

    def __init__(self: Self, value: float) -> None:
        self.value = value

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Float):
            return self.value == other.value
        return NotImplemented

    def __hash__(self: Self) -> int:
        return hash(self.value)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.value} {self.span})"

    def __str__(self: Self) -> str:
        return str(self.value)


class Char(Token["Char"]):
    __slots__ = ("value",)

    value: str

    @classmethod
    def _parse(cls: type[Self], reader: StringReader) -> Self:
        start_pos = reader.pos
        reader.pop()

        if not reader:
            Error("Unterminated single quote").at(reader.pos).log()
            return cls("")
        value = reader.pop()

        if not reader:
            Error("Unterminated single quote").at(reader.pos).log()
            return cls(value)

        end_pos = reader.pos
        if (c := reader.pop()) != "'":
            value += c
            while reader and (c := reader.pop()) != "'":
                value += c
            if c != "'":
                Error("Unterminated single quote").at(reader.pos).log()
                return cls(value)

            Error("Unterminated single quote").at(end_pos).hint(
                "simple quotes (') can only used for the single-character type 'car'",
            ).replacement(
                (Span.from_(start_pos, start_pos + 1), '"'),
                (Span.from_(reader.pos - 1, reader.pos), '"'),
                msg="if you meant to write a 'chaine' replace single quotes with double quotes",
            ).replacement(
                (Span.from_(end_pos, reader.pos - 1), ""),
                msg="if you meant to write a 'car' keep only one character",
            ).log()

        return cls(value)

    def __init__(self: Self, value: str) -> None:
        self.value = value

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Char):
            return self.value == other.value
        return NotImplemented

    def __hash__(self: Self) -> int:
        return hash(self.value)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.value} {self.span})"

    def __str__(self: Self) -> str:
        return self.value


class String(Token["String"]):
    __slots__ = ("value",)

    value: str

    @classmethod
    def _parse(cls: type[Self], reader: StringReader) -> Self:
        start_pos = reader.pos
        reader.pop()

        value = ""
        escaped = False
        while reader and ((c := reader.pop()) != '"' or escaped):
            if escaped:
                if c in '"\\':
                    value += c
                elif c == "\n":
                    pass
                else:
                    Warn("Invalid escape character").at(reader.pos - 1).detail(c).hint(
                        "valid escapes are '\\\\', '\\\"' and '\\<line end>'",
                    ).replacement(
                        (Span.from_(reader.pos - 1, reader.pos - 1), "\\"),
                        msg="if you want to insert a literal backslash, use '\\\\'",
                    ).log()
                    value += c
                escaped = False
            elif c == "\\":
                escaped = True
            else:
                value += c

        if c != '"':
            Error("Unterminated double quote").at(reader.pos).comment(
                Span.from_(start_pos, start_pos + 1),
                msg="string started here",
            ).log()

        return cls(value)

    def __init__(self: Self, value: str) -> None:
        self.value = value

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, String):
            return self.value == other.value
        return NotImplemented

    def __hash__(self: Self) -> int:
        return hash(self.value)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.value!r} {self.span})"

    def __str__(self: Self) -> str:
        return self.value


class OperatorType(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    POW = "^"

    EQ = "="
    NEQ = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="

    OR = "ou"
    AND = "et"
    NOT = "non"


class Operator(Token["Operator | Assign"]):
    __slots__ = ("op",)

    op: OperatorType

    @classmethod
    def _parse(cls: type[Self], reader: StringReader) -> Self | Assign:
        operator = reader.pop()

        if operator == "=" and reader.try_peek() == "=":
            Warn("Pseudo-code uses `=` for equality check, not `==`").at(reader.pos).replacement(
                (Span.from_(reader.pos, reader.pos + 1), ""),
                msg="Replace `==` with `=`",
            ).log()
            reader.pop()
        elif operator in "!<>" and reader.try_peek() == "=":
            operator += reader.pop()
        elif operator == "!":
            Error("`!` can't be alone").at(reader.pos - 1).replacement(
                (Span.at(reader.pos), "="),
                msg="if you want the 'not equal to' operator, use `!=`",
            ).replacement(
                (Span.from_(reader.pos - 1, reader.pos), "non"),
                msg="if you want to invert a boolean expression, use `non`",
            ).log()

        if operator == "<" and reader.try_peek() == "-":
            reader.pop()
            return Assign()

        return cls(OperatorType(operator))

    def __init__(self: Self, op: OperatorType) -> None:
        self.op = op

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Operator):
            return self.op == other.op
        return NotImplemented

    def __hash__(self: Self) -> int:
        return hash(self.op)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.op.value} {self.span})"

    def __str__(self: Self) -> str:
        return self.op.value


class Comma(SingleCharToken):
    value = ","


class Colon(SingleCharToken):
    value = ":"


class Semicolon(SingleCharToken):
    value = ";"


class LParen(SingleCharToken):
    value = "("


class RParen(SingleCharToken):
    value = ")"


class LBracket(SingleCharToken):
    value = "["


class RBracket(SingleCharToken):
    value = "]"


class LBrace(SingleCharToken):
    value = "{"


class RBrace(SingleCharToken):
    value = "}"


class Assign(SingleCharToken):
    value = "<-"


class Dot(SingleCharToken):
    value = "."


def _lexer(reader: StringReader) -> list[Token[Any]]:
    tokens: list[Token[Any]] = []

    while reader:
        next_char = reader.peek()

        match next_char:
            case _ if next_char.isalpha() or next_char == "_":
                tokens.append(Identifier.parse(reader))
            case _ if next_char.isdigit():
                tokens.append(Integer.parse(reader))
            case "'":
                tokens.append(Char.parse(reader))
            case '"':
                tokens.append(String.parse(reader))
            case "=" | "<" | ">" | "!" | "+" | "-" | "*" | "/" | "%" | "^":
                tokens.append(Operator.parse(reader))
            case c if c in SingleCharToken.__all__:
                tokens.append(SingleCharToken.__all__[c].parse(reader))
            case "#":
                while reader.pop() != "\n":
                    pass
            case _ if next_char in " \t\n":
                reader.pop()
            case _:
                Error("Unknown token").at(reader.pos).detail(reader.pop()).log()

    return tokens


def lexer(section: FileSection) -> list[Token[Any]]:
    reader = StringReader(section)

    try:
        return _lexer(reader)
    except Exception:
        print(f"\033[31mError at {reader.pos}:\033[0m")
        raise
