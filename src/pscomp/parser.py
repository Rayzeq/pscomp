from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Mapping, Sequence, TypeAlias, TypeVar, cast, overload

from . import lexer
from .errors import InternalCompilerError
from .lexer import KEYWORDS, Token
from .logger import Error, Warn
from .source import Position, Span, SpannedStr
from .types import Any as AnyType
from .types import Bool, Char, Float, Integer, List, String, Type
from .types import Structure as StructType

if TYPE_CHECKING:
    from typing_extensions import Never, Self


class ParserError(Exception):
    """There was an unrecoverable syntax error while parsing."""


class TokenStream:
    tokens: list[Token[Any]]
    _last_pos: Position

    def __init__(self: Self, tokens: list[Token[Any]]) -> None:
        self.tokens = tokens

        if tokens:
            # this property should never be accessed when tokens is empty
            # so it's okay for it to not be set
            self._last_pos = tokens[0].span.start

    @property
    def next_pos(self: Self) -> Position:
        if self.tokens:
            return self.tokens[0].span.start
        else:
            return self._last_pos

    @property
    def last_pos(self: Self) -> Position:
        return self._last_pos

    def peek(self: Self, index: int = 0) -> Token[Any]:
        return self.tokens[index]

    def try_peek(self: Self, index: int = 0) -> Token[Any] | None:
        try:
            return self.tokens[index]
        except IndexError:
            return None

    def pop(self: Self) -> Token[Any]:
        token = self.tokens.pop(0)
        self._last_pos = token.span.end
        return token

    def try_pop(self: Self) -> Token[Any] | None:
        try:
            return self.pop()
        except IndexError:
            return None

    def __bool__(self: Self) -> bool:
        return bool(self.tokens)


N = TypeVar("N", bound="Node[Any]")


class Node(ABC, Generic[N]):
    span: Span

    @classmethod
    def parse(cls: type[Self], stream: TokenStream, **kwargs: Any) -> N:
        start = stream.next_pos
        self = cls._parse(stream, **kwargs)
        self.span = Span.from_(start, stream.last_pos)

        return self

    @classmethod
    @abstractmethod
    def _parse(cls: type[Self], stream: TokenStream) -> N:
        pass


class NoParse(Node["Never"]):
    @classmethod
    def _parse(cls: type[NoParse], _stream: TokenStream) -> Never:
        msg = f"Can't call {cls.__name__}.parse directly"
        raise InternalCompilerError(msg)


class Value(Node["Value"]):
    TOKENS: tuple[Token[Any] | type[Token[Any]], ...] = (
        KEYWORDS.vrai,
        KEYWORDS.faux,
        lexer.Integer,
        lexer.Float,
        lexer.Char,
        lexer.String,
    )

    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        next_token = assert_token(stream.try_pop(), cls.TOKENS)

        if next_token == KEYWORDS.vrai:
            return cls(Bool(), value=True)
        elif next_token == KEYWORDS.faux:
            return cls(Bool(), value=False)
        elif isinstance(next_token, lexer.Integer):
            return cls(Integer(), next_token.value)
        elif isinstance(next_token, lexer.Float):
            return cls(Float(), next_token.value)
        elif isinstance(next_token, lexer.Char):
            return cls(Char(), next_token.value)
        elif isinstance(next_token, lexer.String):
            return cls(String(), next_token.value)
        else:
            msg = f"Invalid literal: {next_token}"
            raise InternalCompilerError(msg)

    typ: Type
    value: bool | int | float | str

    def __init__(self: Self, typ: Type, value: bool | int | float | str) -> None:
        self.typ = typ
        self.value = value

    def __repr__(self: Self) -> str:
        return f"{self.typ.name}({self.value})"

    def __bool__(self: Self) -> Value:
        new_type = Bool().assign(self.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(Bool(), bool(self.value))

    def __pos__(self: Self) -> Value:
        new_type = self.typ.pos()
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, +self.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    def __neg__(self: Self) -> Value:
        new_type = self.typ.neg()
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, -self.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    def __pow__(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.pow(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value**other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    __rpow__ = __pow__

    def __mul__(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.mul(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value * other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    __rmul__ = __mul__

    def __div__(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.div(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()

        if isinstance(new_type, Integer):
            return Value(new_type, self.value // other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ
        else:
            return Value(new_type, self.value / other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    __rdiv__ = __div__

    def __mod__(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.mod(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value % other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    __rmod__ = __mod__

    def __add__(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.add(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value + other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    __radd__ = __add__

    def __sub__(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.sub(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value - other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    __rsub__ = __sub__

    def eq(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.eq(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value == other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    def ne(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.eq(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value != other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    def lt(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.order(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value < other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    def gt(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.order(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value > other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    def le(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.order(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value <= other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ

    def ge(self: Self, other: object) -> Value:
        if not isinstance(other, Value):
            return NotImplemented

        new_type = self.typ.order(other.typ)
        if not new_type or isinstance(new_type, AnyType):
            return Unknown()
        return Value(new_type, self.value >= other.value)  # type: ignore[operator] # the type is checked using self.typ and other.typ


class Unknown(Value):
    def __init__(self: Self) -> None:
        self.typ = AnyType()

    def __pos__(self: Self) -> Unknown:
        return self

    def __neg__(self: Self) -> Unknown:
        return self

    def __pow__(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    __rpow__ = __pow__

    def __mul__(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    __rmul__ = __mul__

    def __div__(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    __rdiv__ = __div__

    def __mod__(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    __rmod__ = __mod__

    def __add__(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    __radd__ = __add__

    def __sub__(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    __rsub__ = __sub__

    def eq(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    def ne(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    def lt(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    def gt(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    def le(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self

    def ge(self: Self, other: object) -> Unknown:
        if not isinstance(other, Value):
            return NotImplemented

        return self


TYPES: dict[lexer.Identifier, type[Type]] = {
    KEYWORDS.booleen: Bool,
    KEYWORDS.entier: Integer,
    KEYWORDS.reel: Float,
    KEYWORDS.car: Char,
    KEYWORDS.chaine: String,
}


def parse_type(token: Token[Any] | None) -> Type:
    token_: lexer.Identifier = assert_token(token, (*TYPES.keys(), lexer.Identifier))  # type: ignore[arg-type] # yeah mypy is really dumb
    typ = TYPES[token_](token_.span) if token_ in TYPES else StructType(token_.name, token_.span)

    return typ


class Program(Node["Program"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        name = assert_token(stream.try_pop(), lexer.Identifier).name

        typedefs, body = Block.parse(stream, name, KEYWORDS.debut, [KEYWORDS.fin], typedefs=True)
        return cls(name, typedefs, body)

    name: str
    types: list[TypeDef]
    body: list[Node[Any]]

    def __init__(self: Self, name: str, types: list[TypeDef], body: list[Node[Any]]) -> None:
        self.name = name
        self.types = types
        self.body = body

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.name}, ...)"


class Callable(Node["Callable[N]"], Generic[N]):
    @classmethod
    def _parse_arglist(
        cls: type[Self],
        stream: TokenStream,
        *,
        ignore_start: bool = False,
        ignore_end: bool = False,
    ) -> list[TypeDef]:
        if not ignore_start:
            assert_token(stream.try_pop(), lexer.LParen, crash=False)

        typedefs = []
        while stream and not isinstance(stream.peek(), (lexer.RParen, lexer.Semicolon) if ignore_end else lexer.RParen):
            binding = Binding.parse(stream, allow_lookups=False)

            assert_token(stream.try_pop(), lexer.Colon)
            typ = List.from_(binding, parse_type(stream.try_pop()))

            typedefs.append(TypeDef(binding.get_name(), typ, None))

            if isinstance(stream.try_peek(), lexer.Comma):
                stream.pop()

        if not ignore_end:
            assert_token(stream.try_pop(), lexer.RParen, crash=False)
        return typedefs


class Function(Callable["Function"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        introducer = stream.pop()
        name = assert_token(stream.try_pop(), lexer.Identifier).name

        args = cls._parse_arglist(stream)

        if stream.try_peek() == KEYWORDS.debut:
            Error("A function must return something").at(
                stream.peek().span,
                msg="expected `retourne`",
            ).replacement(
                (introducer.span, "procedure"),
                msg="if you don't want to return anything, use a `procedure`",
            ).log()
            ret: Type = AnyType(Span.at(stream.last_pos))
        else:
            assert_token(stream.try_pop(), KEYWORDS.retourne, crash=False)
            typ_ = parse_type(stream.try_peek())
            ret = List.from_(Binding.parse(stream, allow_lookups=False), typ_)

        typedefs, body = Block.parse(stream, name, KEYWORDS.debut, [KEYWORDS.fin], typedefs=True)
        return cls(name, args, ret, typedefs, body)

    name: SpannedStr
    args: list[TypeDef]
    ret: Type
    types: list[TypeDef]
    body: list[Node[Any]]

    def __init__(
        self: Self,
        name: SpannedStr,
        args: list[TypeDef],
        ret: Type,
        types: list[TypeDef],
        body: list[Node[Any]],
    ) -> None:
        self.name = name
        self.args = args
        self.ret = ret
        self.types = types
        self.body = body

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.name}, ...)"


class Procedure(Callable["Procedure"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        introducer = stream.pop()
        name = assert_token(stream.try_pop(), lexer.Identifier).name

        args = cls._parse_arglist(stream, ignore_end=True)

        ref_args = []
        if isinstance(stream.try_peek(), lexer.Semicolon):
            stream.pop()
            ref_args = cls._parse_arglist(stream, ignore_start=True)
        else:
            assert_token(stream.try_pop(), lexer.RParen, crash=False)

        if stream.try_peek() == KEYWORDS.retourne:
            retourne = stream.pop()
            typ = Binding.parse(stream, allow_lookups=False)

            if ref_args:
                new_span = Span.at(ref_args[-1].typ.span.end)
                link = ", "
            else:
                new_span = Span.at(args[-1].typ.span.end)
                link = "; "

            Error("A procedure can't return anything").at(
                retourne.span,
                msg="expected `debut`",
            ).replacement(
                (introducer.span, "fonction"),
                msg="if you want to return something use a `fonction`",
            ).replacement(
                (retourne.span + typ.span, ""),
                (new_span, f"{link}return_value: {typ.span.extract()}"),
                msg="if you want to edit the caller's variable, get it by reference",
            ).log()

        typedefs, body = Block.parse(stream, name, KEYWORDS.debut, [KEYWORDS.fin], typedefs=True)
        return cls(name, args, ref_args, typedefs, body)

    name: SpannedStr
    args: list[TypeDef]
    ref_args: list[TypeDef]
    types: list[TypeDef]
    body: list[Node[Any]]

    def __init__(
        self: Self,
        name: SpannedStr,
        args: list[TypeDef],
        ref_args: list[TypeDef],
        types: list[TypeDef],
        body: list[Node[Any]],
    ) -> None:
        self.name = name
        self.args = args
        self.ref_args = ref_args
        self.types = types
        self.body = body

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.name}, ...)"


class Condition(Node["Condition"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        condition = Expr.parse(stream)
        Block._parse_start(stream, KEYWORDS.alors)
        if_block = Block._parse_body(stream, breakers=[KEYWORDS.sinon, KEYWORDS.fin])

        else_block = None
        if stream.try_peek() == KEYWORDS.sinon:
            stream.pop()
            else_block = Block._parse_body(stream, breakers=[KEYWORDS.fin])

        Block._parse_end(stream, "si")
        return cls(condition, if_block, else_block)

    condition: Expr
    if_block: list[Node[Any]]
    else_block: list[Node[Any]] | None

    def __init__(self: Self, condition: Expr, if_block: list[Node[Any]], else_block: list[Node[Any]] | None) -> None:
        self.condition = condition
        self.if_block = if_block
        self.else_block = else_block

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.condition!r}, ..., ...)"


class Switch(Node["Switch"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        tested_value = Expr.parse(stream)
        Block._parse_start(stream, KEYWORDS.faire)

        blocks = []
        while stream.try_peek() == KEYWORDS.cas:
            stream.pop()
            value = Value.parse(stream)
            assert_token(stream.try_pop(), lexer.Colon)
            block = Block._parse_body(stream, breakers=[KEYWORDS.cas, KEYWORDS.defaut, KEYWORDS.fin])
            blocks.append((value, block))

        default = None
        if stream.try_peek() == KEYWORDS.defaut:
            stream.pop()
            assert_token(stream.try_pop(), lexer.Colon)
            default = Block._parse_body(stream, breakers=[KEYWORDS.fin])

        Block._parse_end(stream, "faire")
        return cls(tested_value, blocks, default)

    value: Expr
    blocks: list[tuple[Value, list[Node[Any]]]]
    default: list[Node[Any]] | None

    def __init__(
        self: Self,
        value: Expr,
        blocks: list[tuple[Value, list[Node[Any]]]],
        default: list[Node[Any]] | None,
    ) -> None:
        self.value = value
        self.blocks = blocks
        self.default = default

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.value!r}, ...)"


class ForLoop(Node["ForLoop"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        binding = Binding.parse(stream)

        assert_token(stream.try_pop(), KEYWORDS.de)
        start = Expr.parse(stream)

        assert_token(stream.try_pop(), KEYWORDS.a)
        end = Expr.parse(stream)

        step = None
        if stream.try_peek() == KEYWORDS.pas:
            stream.pop()
            step = Expr.parse(stream)

        block = Block.parse(stream, "faire", KEYWORDS.faire, [KEYWORDS.fin])
        return cls(binding, start, end, step, block)

    binding: Binding
    start: Expr
    end: Expr
    step: Expr | None
    block: list[Node[Any]]

    def __init__(
        self: Self,
        binding: Binding,
        start: Expr,
        end: Expr,
        step: Expr | None,
        block: list[Node[Any]],
    ) -> None:
        self.binding = binding
        self.start = start
        self.end = end
        self.step = step
        self.block = block

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}(for {self.binding!r} from {self.start!r} to {self.end!r}{f' step {self.step}' if self.step else ''}, ...)"


class WhileLoop(Node["WhileLoop"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        assert_token(stream.try_pop(), KEYWORDS.que, crash=False)
        condition = Expr.parse(stream)

        block = Block.parse(stream, "faire", KEYWORDS.faire, [KEYWORDS.fin])
        return cls(condition, block)

    condition: Expr
    block: list[Node[Any]]

    def __init__(self: Self, condition: Expr, block: list[Node[Any]]) -> None:
        self.condition = condition
        self.block = block

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.condition}, ...)"


class DoWhileLoop(Node["DoWhileLoop"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()

        block = Block._parse_body(stream, breakers=[KEYWORDS.tant])

        assert_token(stream.try_pop(), KEYWORDS.tant, crash=False)
        assert_token(stream.try_pop(), KEYWORDS.que, crash=False)
        condition = Expr.parse(stream)

        return cls(condition, block)

    condition: Expr
    block: list[Node[Any]]

    def __init__(self: Self, condition: Expr, block: list[Node[Any]]) -> None:
        self.condition = condition
        self.block = block

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.condition}, ...)"


class Print(Node["Print"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        elements = []

        while (e := Expr.parse(stream)).nodes:
            elements.append(e)
            if isinstance(stream.try_peek(), lexer.Comma):
                stream.pop()
            else:
                break

        return cls(elements)

    elements: list[Expr]

    def __init__(self: Self, elements: list[Expr]) -> None:
        self.elements = elements

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.elements))})"


class Input(Node["Input"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        bindings = []

        while isinstance(stream.try_peek(), lexer.Identifier):
            bindings.append(Binding.parse(stream))
            if isinstance(stream.try_peek(), lexer.Comma):
                stream.pop()
            else:
                break

        return cls(bindings)

    bindings: list[Binding]

    def __init__(self: Self, bindings: list[Binding]) -> None:
        if len(bindings) > 1:
            Warn("Inputting multiple variables in one instruction is discouraged").at(
                Span.from_(bindings[0].span.end, bindings[-1].span.end),
            ).log()
        self.bindings = bindings

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.bindings))})"


class Return(Node["Return"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        stream.pop()
        value = Expr.parse(stream)

        return cls(value)

    value: Expr

    def __init__(self: Self, value: Expr) -> None:
        self.value = value

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.value!r})"


class Assignement(Node["Assignement"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        binding = Binding.parse(stream)
        assert_token(stream.try_pop(), lexer.Assign)
        value = Expr.parse(stream)

        return cls(binding, value)

    binding: Binding
    value: Expr

    def __init__(self: Self, binding: Binding, value: Expr) -> None:
        self.binding = binding
        self.value = value

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.binding!r}, {self.value!r})"


class FuncCall(Node["FuncCall"]):
    @classmethod
    def _parse_expr_list(cls: type[Self], stream: TokenStream) -> list[Expr]:
        exprs = []
        while stream and not isinstance(stream.peek(), (lexer.RParen, lexer.Semicolon)):
            exprs.append(Expr.parse(stream))

            if isinstance(stream.try_peek(), lexer.Comma):
                stream.pop()

        return exprs

    @classmethod
    def _parse_binding_list(cls: type[Self], stream: TokenStream) -> list[lexer.Binding]:
        bindings = []
        while stream and not isinstance(stream.peek(), lexer.RParen):
            bindings.append(Binding.parse(stream))

            if isinstance(stream.try_peek(), lexer.Comma):
                stream.pop()

        return bindings

    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        name = assert_token(stream.pop(), lexer.Identifier).name

        assert_token(stream.try_pop(), lexer.LParen)

        arguments = cls._parse_expr_list(stream)
        references = []
        if isinstance(stream.try_peek(), lexer.Semicolon):
            stream.pop()
            references = cls._parse_binding_list(stream)

        assert_token(stream.try_pop(), lexer.RParen)

        return cls(name, arguments, references)

    name: SpannedStr
    arguments: list[Expr]
    references: list[Binding]

    def __init__(self: Self, name: SpannedStr, arguments: list[Expr], references: list[Binding]) -> None:
        self.name = name
        self.arguments = arguments
        self.references = references

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.name}, {', '.join(map(repr, self.arguments))}; {', '.join(map(repr, self.references))})"


class Expr(Node["Expr"]):
    UNARY_OPS_TOKS = (
        lexer.Identifier("+"),
        lexer.Identifier("-"),
        KEYWORDS.non,
    )
    UNARY_OPS = (
        lexer.OperatorType.ADD,
        lexer.OperatorType.SUB,
        lexer.OperatorType.NOT,
    )

    TOKENS: tuple[Token[Any] | type[Token[Any]], ...] = (
        *Value.TOKENS,
        *UNARY_OPS_TOKS,
        lexer.Identifier,
        lexer.LParen,
    )
    SubTypes: TypeAlias = "Value | FuncCall | Binding | Expr | Operator"

    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        nodes: list[cls.SubTypes] = []

        while stream:
            next_token = stream.peek()
            if (
                not nodes
                and not check_token(next_token, cls.TOKENS)
                and not (isinstance(next_token, lexer.Operator) and next_token.op in cls.UNARY_OPS)
            ):
                assert_token(stream.pop(), cls.TOKENS)

            if not nodes or isinstance(nodes[-1], Operator):
                if not (isinstance(next_token, lexer.Operator) and next_token.op in cls.UNARY_OPS):
                    assert_token(next_token, cls.TOKENS)

                if check_token(next_token, Value.TOKENS):
                    nodes.append(Value.parse(stream))
                elif isinstance(next_token, lexer.Identifier):
                    if isinstance(stream.try_peek(1), lexer.LParen):
                        nodes.append(FuncCall.parse(stream))
                    else:
                        nodes.append(Binding.parse(stream))
                elif isinstance(next_token, lexer.LParen):
                    stream.pop()
                    nodes.append(Expr.parse(stream))
                    assert_token(stream.try_pop(), lexer.RParen)
                elif isinstance(next_token, lexer.Operator):
                    stream.pop()
                    nodes.append(Operator(next_token))
                else:
                    msg = f"Invalid token in expression: {next_token}"
                    raise InternalCompilerError(msg)
            elif isinstance(next_token, lexer.Operator):
                stream.pop()
                nodes.append(Operator(next_token))
            else:
                break

        return cls(nodes)

    nodes: list[SubTypes]

    def __init__(self: Self, nodes: list[SubTypes]) -> None:
        self.nodes = nodes

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({' '.join(map(repr, self.nodes))})"


class Binding(Node["Binding"]):
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream, *, allow_lookups: bool = True) -> Binding:
        name = assert_token(stream.try_pop(), lexer.Identifier)
        self: Binding = Variable(name)
        self.span = name.span

        valid_tokens = (lexer.LBracket, lexer.Dot) if allow_lookups else (lexer.LBracket,)
        while isinstance(stream.try_peek(), valid_tokens):
            next_tok = stream.pop()
            if isinstance(next_tok, lexer.LBracket):
                index = Expr.parse(stream)
                rbrack = assert_token(stream.try_pop(), lexer.RBracket)
                self = Indexing(self, index)
                self.span = self.sub.span + rbrack.span
                self.bracket_span = next_tok.span + rbrack.span
            elif allow_lookups and isinstance(next_tok, lexer.Dot):
                name = assert_token(stream.try_pop(), lexer.Identifier)
                self = Lookup(self, next_tok, name.name)
            else:
                msg = f"Invalid token for binding continuation: {next_tok}"
                raise InternalCompilerError(msg)

        return self

    def get_name(self: Self) -> SpannedStr:
        if isinstance(self, Variable):
            return self.name
        elif isinstance(self, Indexing):
            return self.sub.get_name()
        else:
            msg = f"Cannot get binding name: {self}"
            raise InternalCompilerError(msg)


class Variable(Binding):
    name: SpannedStr

    def __init__(self: Self, name: lexer.Identifier) -> None:
        self.name = name.name
        self.span = name.span

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.name})"


class Indexing(Binding):
    sub: Binding
    index: Expr
    bracket_span: Span

    def __init__(self: Self, sub: Binding, index: Expr) -> None:
        self.sub = sub
        self.index = index

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.sub}[{self.index}])"


class Lookup(Binding):
    sub: Binding
    dot: lexer.Dot
    name: SpannedStr

    def __init__(self: Self, sub: Binding, dot: lexer.Dot, name: SpannedStr) -> None:
        self.sub = sub
        self.dot = dot
        self.name = name

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.sub}.{self.name})"


class Operator(NoParse):
    op: lexer.OperatorType

    def __init__(self: Self, operator: lexer.Operator) -> None:
        self.op = operator.op
        self.span = operator.span

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.op.value})"


class Structure:
    @classmethod
    def _parse(cls: type[Self], stream: TokenStream) -> Self:
        def parse_list(stream: TokenStream) -> list[Binding]:
            bindings = []

            while stream:
                bindings.append(Binding.parse(stream, allow_lookups=False))
                if isinstance(stream.try_peek(), lexer.Comma):
                    stream.pop()
                else:
                    break

            return bindings

        name = assert_token(stream.try_pop(), lexer.Identifier).name
        if isinstance(stream.try_peek(), lexer.LBrace):
            Error("Missing = in structure definition").at(stream.peek().span).replacement(
                (Span.at(stream.peek().span.start), "= "),
                msg="add the =",
            ).log()
        else:
            assert_token(stream.try_pop(), lexer.Operator(lexer.OperatorType.EQ))
        assert_token(stream.try_pop(), lexer.LBrace)

        fields = {}
        while not isinstance(stream.try_peek(), lexer.RBrace):
            names = parse_list(stream)
            assert_token(stream.try_pop(), lexer.Colon)

            typ_ = parse_type(stream.try_pop())

            for binding in names:
                typ = List.from_(binding, typ_)
                fields[binding.get_name()] = TypeDef(binding.get_name(), typ, None)

        assert_token(stream.try_pop(), lexer.RBrace)

        return cls(name, fields)  # type: ignore[arg-type] # apparently subclasses in dict keys are not ok

    name: SpannedStr
    fields: Mapping[SpannedStr, TypeDef]

    def __init__(self: Self, name: SpannedStr, fields: Mapping[SpannedStr, TypeDef]) -> None:
        self.name = name
        self.fields = fields


@dataclass
class TypeDef:
    name: SpannedStr
    typ: Type
    default: Expr | None


class Block:
    STARTERS: ClassVar[dict[lexer.Identifier, type[Node[Any]]]] = {
        KEYWORDS.programme: Program,
        KEYWORDS.fonction: Function,
        KEYWORDS.procedure: Procedure,
        KEYWORDS.si: Condition,
        KEYWORDS.selon: Switch,
        KEYWORDS.pour: ForLoop,
        KEYWORDS.tant: WhileLoop,
        KEYWORDS.repeter: DoWhileLoop,
        KEYWORDS.afficher: Print,
        KEYWORDS.saisir: Input,
        KEYWORDS.retourne: Return,
    }

    @classmethod
    @overload
    def parse(
        cls: type[Self],
        stream: TokenStream,
        name: str | None = None,
        starter: lexer.Identifier | None = None,
        enders: list[lexer.Identifier] | None = None,
    ) -> list[Node[Any]]:
        ...

    @classmethod
    @overload
    def parse(
        cls: type[Self],
        stream: TokenStream,
        name: str | None = None,
        starter: lexer.Identifier | None = None,
        enders: list[lexer.Identifier] | None = None,
        *,
        typedefs: Literal[False],
    ) -> list[Node[Any]]:
        ...

    @classmethod
    @overload
    def parse(
        cls: type[Self],
        stream: TokenStream,
        name: str | None = None,
        starter: lexer.Identifier | None = None,
        enders: list[lexer.Identifier] | None = None,
        *,
        typedefs: Literal[True],
    ) -> tuple[list[TypeDef], list[Node[Any]]]:
        ...

    @classmethod
    def parse(
        cls: type[Self],
        stream: TokenStream,
        name: str | None = None,
        starter: lexer.Identifier | None = None,
        enders: list[lexer.Identifier] | None = None,
        *,
        typedefs: bool = False,
    ) -> list[Node[Any]] | tuple[list[TypeDef], list[Node[Any]]]:
        if starter is not None:
            cls._parse_start(stream, starter)

        typedefs_list = []
        if typedefs and stream.try_peek() == KEYWORDS.avec:
            typedefs_list = cls._parse_typedefs(stream)

        nodes = cls._parse_body(stream, enders)

        if name is not None:
            cls._parse_end(stream, name)

        if typedefs:
            return typedefs_list, nodes
        return nodes

    @classmethod
    def _parse_start(cls: type[Self], stream: TokenStream, starter: lexer.Identifier) -> None:
        if stream.try_peek() == KEYWORDS.avec:
            Error(f"Missing `{starter}` to start a block").at(stream.last_pos).log()
        else:
            assert_token(stream.try_pop(), starter, crash=False)

    @overload
    @classmethod
    def _parse_typedefs(cls: type[Self], stream: TokenStream, *, allow_structs: Literal[False]) -> list[TypeDef]:
        ...

    @overload
    @classmethod
    def _parse_typedefs(
        cls: type[Self],
        stream: TokenStream,
        *,
        allow_structs: Literal[True],
    ) -> list[TypeDef | Structure]:
        ...

    @classmethod  # type: ignore[misc] # apparently my overloads are wrong
    def _parse_typedefs(
        cls: type[Self],
        stream: TokenStream,
        *,
        allow_structs: bool = False,
    ) -> list[TypeDef | Structure]:
        def check_line(stream: TokenStream) -> bool:
            if not stream:
                return False

            if allow_structs and stream.peek() == lexer.Identifier("type"):
                return True

            line = stream.peek().span.start.line
            i = 0
            while True:
                if (not stream) or stream.peek(i).span.end.line != line:
                    return False
                if stream and isinstance(stream.peek(i), lexer.Colon):
                    return True

                i += 1

        def parse_list(stream: TokenStream) -> list[Binding]:
            bindings = []

            while stream:
                bindings.append(Binding.parse(stream, allow_lookups=False))
                if isinstance(stream.try_peek(), lexer.Comma):
                    stream.pop()
                else:
                    break

            return bindings

        stream.pop()
        typedefs: list[TypeDef | Structure] = []

        while check_line(stream):
            if allow_structs and stream.peek() == lexer.Identifier("type"):
                stream.pop()
                typedefs.append(Structure._parse(stream))
                continue

            bindings = parse_list(stream)
            assert_token(stream.try_pop(), lexer.Colon)

            typ_ = parse_type(stream.try_pop())
            value = None
            if isinstance(stream.try_peek(), lexer.Assign):
                stream.pop()
                value = Expr.parse(stream)

            for binding in bindings:
                typ = List.from_(binding, typ_)
                typedefs.append(TypeDef(binding.get_name(), typ, value))

        return typedefs

    @classmethod
    def _parse_body(
        cls: type[Self],
        stream: TokenStream,
        breakers: list[lexer.Identifier] | None = None,
    ) -> list[Node[Any]]:
        nodes: list[Node[Any]] = []

        if breakers is None:
            breakers = []
        # we keep this out of the loop for optimisation purposes
        expected_tokens: tuple[Token[Any] | type[Token[Any]], ...] = (
            *breakers,
            *cls.STARTERS.keys(),
            lexer.Identifier,
        )

        while stream:
            next_token = assert_token(stream.peek(), expected_tokens, crash=False)

            if next_token in breakers:
                return nodes
            elif isinstance(next_token, lexer.Identifier) and next_token in cls.STARTERS:
                nodes.append(cls.STARTERS[next_token].parse(stream))
            elif (
                isinstance(next_token, lexer.Identifier)
                and next_token.name.startswith("fin")
                and KEYWORDS.fin in breakers
            ):
                Warn("Using a space for the end of a block is preferred").at(next_token.span).replacement(
                    (Span.at(next_token.span.start + 3), " "),
                    msg=f"add a space between 'fin' and '{next_token.name[3:]}'",
                ).log()
                return nodes
            elif isinstance(next_token, lexer.Identifier) and isinstance(stream.try_peek(1), lexer.LParen):
                nodes.append(FuncCall.parse(stream))
            elif isinstance(next_token, lexer.Identifier):
                nodes.append(Assignement.parse(stream))
            else:
                stream.pop()

        return nodes

    @classmethod
    def _parse_end(cls: type[Self], stream: TokenStream, name: str) -> None:
        if stream.try_peek() == lexer.Identifier("fin" + name):
            stream.pop()
            return

        fin = assert_token(stream.try_pop(), KEYWORDS.fin)
        if isinstance(stream.try_peek(), lexer.Identifier) and stream.peek().span.start.line == fin.span.start.line:
            ident = assert_token(stream.pop(), lexer.Identifier)
            if ident.name != name:
                Error("wrong name for end of block").at(ident.span).detail(ident.name).replacement(
                    (ident.span, name),
                    msg="replace it with the correct name",
                ).log()
        else:
            Error("missing name for end of block").at(fin.span.end).replacement(
                (Span.at(fin.span.end), f" {name}"),
                msg="add the name of the block to end it properly",
            ).log()


T = TypeVar("T", bound=Token[Any])


@overload
def assert_token(token: Token[Any] | None, expected: Sequence[T | type[T]], *, crash: bool = True) -> T:
    ...


@overload
def assert_token(token: Token[Any] | None, expected: type[T], *, crash: bool = True) -> T:
    ...


@overload
def assert_token(token: Token[Any] | None, expected: T, *, crash: bool = True) -> T:
    ...


def assert_token(token: Token[Any] | None, expected: Sequence[T | type[T]] | type[T] | T, *, crash: bool = True) -> T:
    if isinstance(expected, Token):
        # mypy is broken here
        expected_lst: Sequence[T | type[T]] = (cast(T, expected),)
    elif isinstance(expected, type):
        expected_lst = (expected,)
    else:
        expected_lst = expected

    expected_str = ((repr(str(x)) if isinstance(x, Token) else x.__name__) for x in expected_lst)

    if token is None:
        Error("No token found (end of file)").hint(f"expected one of: {', '.join(expected_str)}").log()
        if crash:
            raise ParserError
        return None

    if not check_token(token, expected_lst):
        Error("Unexpected token").at(token.span).detail(f"`{token}`").hint(
            f"expected one of: {', '.join(expected_str)}",
        ).log()
        if crash:
            raise ParserError
        return None

    return cast(T, token)


def check_token(token: Token[Any] | None, expected: Sequence[T | type[T]]) -> bool:
    for x in expected:
        if (isinstance(x, Token) and token == x) or (isinstance(x, type) and isinstance(token, x)):
            return True
    return False


def _parse(stream: TokenStream) -> tuple[list[Node[Any]], list[TypeDef], list[Structure]]:
    typdefs = Block._parse_typedefs(stream, allow_structs=True) if stream.try_peek() == KEYWORDS.avec else []

    constants = []
    structures = []
    for definition in typdefs:
        if isinstance(definition, Structure):
            if "_" in definition.name or definition.name[0].islower() or definition.name.isupper():
                Warn("Structures should be PascalCase").at(definition.name.span).replacement(
                    (definition.name.span, "".join(definition.name.title().split("_"))),
                    msg="Make the constant uppercase",
                ).log()

            structures.append(definition)
        else:  # it's a constant
            if not definition.name.isupper():
                Warn("Constant should be UPPER_CASE").at(definition.name.span).replacement(
                    (definition.name.span, definition.name.upper()),
                    msg="Make the constant uppercase",
                ).log()

            if definition.default is None:
                Error("Constants must have a value").at(definition.name.span).log()
                continue

            constants.append(definition)

    return Block.parse(stream), constants, structures


def parse(tokens: list[Token[Any]]) -> tuple[list[Node[Any]], list[TypeDef], list[Structure]]:
    stream = TokenStream(tokens)

    try:
        return _parse(stream)
    except ParserError:
        raise
    except Exception:
        print(f"\033[31mError at {stream.try_pop()!r}:\033[0m")
        raise
