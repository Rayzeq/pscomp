from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, cast

from . import parser
from .lexer import OperatorType
from .logger import Error, Warn
from .parser import Node, Value
from .source import Span, Spanned, SpannedStr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self


class Argument:
    name: SpannedStr
    typ: Spanned[type[Value[Any]]]

    def __init__(self: Self, name: SpannedStr, typ: Spanned[type[Value[Any]]]) -> None:
        self.name = name
        self.typ = typ

    @property
    def span(self: Self) -> Span:
        return self.name.span + self.typ.span

    def __str__(self: Self) -> str:
        return f"{self.name}: {self.typ.value.__name__}"


class Reference(Argument):
    def __str__(self: Self) -> str:
        return f"{self.name}: &{self.typ.value.__name__}"


class Signature:
    @classmethod
    def from_func(cls: type[Self], func: parser.Function) -> Self:
        return cls(
            func.name,
            [Argument(arg.name, arg.typ) for arg in func.args],
            func.ret,
        )

    @classmethod
    def from_proc(cls: type[Self], proc: parser.Procedure) -> Self:
        return cls(
            proc.name,
            [Argument(arg.name, arg.typ) for arg in proc.args]
            + [Reference(arg.name, arg.typ) for arg in proc.ref_args],
        )

    name: SpannedStr
    args: list[Argument]
    ret: Spanned[type[Value[Any]]] | None

    def __init__(
        self: Self,
        name: SpannedStr,
        args: list[Argument],
        ret: Spanned[type[Value[Any]]] | None = None,
    ) -> None:
        self.name = name
        self.args = args
        self.ret = ret

    @property
    def value_args(self: Self) -> list[Argument]:
        return [arg for arg in self.args if not isinstance(arg, Reference)]

    @property
    def ref_args(self: Self) -> list[Reference]:
        return [arg for arg in self.args if isinstance(arg, Reference)]

    def __str__(self: Self) -> str:
        return f"{self.name}({', '.join(str(arg) for arg in self.args)}){(' -> ' + self.ret.value.__name__) if self.ret else ''}"


class BuiltinSignature(Signature):
    source: str

    def __init__(
        self: Self,
        name: SpannedStr,
        args: list[Argument],
        ret: Spanned[type[Value[Any]]] | None = None,
        *,
        source: str,
    ) -> None:
        super().__init__(name, args, ret)
        self.source = source

    @property
    def python_name(self: Self) -> str:
        return self.source.rsplit(".", 1)[-1]


from .builtins import Unknown, builtins  # noqa: E402 - prevent circular import


class Context:
    python_imports: dict[str, set[str]]
    constants: dict[SpannedStr, tuple[Spanned[type[Value[Any]]], Expression]]
    functions: dict[str, Signature]
    variables: dict[SpannedStr, tuple[Spanned[type[Value[Any]]], Expression | None]]
    ret_type: Spanned[type[Value[Any]]] | None

    def __init__(self: Self) -> None:
        self.python_imports = {}
        self.constants = {}
        self.functions = {}
        self.variables = {}
        self.ret_type = None

    def add_constant(self: Self, name: SpannedStr, typ: Spanned[type[Value[Any]]], value: Expression) -> None:
        if name in self.constants:
            Warn(f"A constant named {name} is already defined, latest definition will be used").at(
                name.span,
            ).comment(next(n for n in self.constants if n == name).span, "previous definition here").log()

        if value is not None and not issubclass(value.typ, typ.value):
            Error(f"`{name}` has type {typ.value.name} but value has type {value.typ.name}").at(
                value.span,
                msg=f"this of type {value.typ.name}",
            ).comment(typ.span, "the variable's type is set here").log()
        self.constants[name] = (typ, value)
        self.add_variable(name, typ, value)

    def lookup_constant(self: Self, name: str) -> tuple[Spanned[type[Value[Any]]], Expression | None]:
        if name in self.constants:
            return self.constants[cast(SpannedStr, name)]

        return (Unknown, None)

    def add_function(self: Self, sig: Signature) -> None:
        if sig.name in self.functions:
            Warn(f"A function named {sig.name} is already defined, latest definition will be used").at(
                sig.name.span,
            ).comment(self.functions[sig.name].name.span, "previous definition here").log()
        self.functions[sig.name] = sig

    def lookup_function(self: Self, name: str) -> Signature | None:
        if name in self.functions:
            return self.functions[name]

        if name in builtins:
            *module, python_name = builtins[name].source.rsplit(".", maxsplit=1)
            if module:
                if module[0] not in self.python_imports:
                    self.python_imports[module[0]] = set()

                self.python_imports[module[0]].add(python_name)
            return builtins[name]

        return None

    def add_variable(self: Self, name: SpannedStr, typ: Spanned[type[Value[Any]]], value: Expression | None) -> None:
        if name in self.variables:
            Warn(f"A variable named {name} is already defined, latest definition will be used").at(
                name.span,
            ).comment(next(n for n in self.variables if n == name).span, "previous definition here").log()

        if value is not None and not issubclass(value.typ, typ.value):
            Error(f"`{name}` has type {typ.value.name} but value has type {value.typ.name}").at(
                value.span,
                msg=f"this of type {value.typ.name}",
            ).comment(typ.span, "the variable's type is set here").log()
        self.variables[name] = (typ, value)

    def lookup_variable(self: Self, name: str) -> tuple[Spanned[type[Value[Any]]], Expression | None]:
        if name in self.variables:
            return self.variables[cast(SpannedStr, name)]

        return (Unknown, None)

    def fork(self: Self) -> Self:
        new = type(self)()

        # global values
        new.python_imports = self.python_imports
        new.constants = self.constants

        new.functions = self.functions.copy()
        new.variables = self.variables.copy()
        return new


@dataclass
class TypeDef:
    name: SpannedStr
    typ: Spanned[type[Value[Any]]]
    default: Expression | None

    @classmethod
    def from_(cls: type[Self], typedef: parser.TypeDef, context: Context) -> Self:
        default = Expression.from_(typedef.default, context, constexpr=True) if typedef.default else None
        cls._check_indexes(typedef.typ.value, context)
        return cls(typedef.name, typedef.typ, default)

    @classmethod
    def _check_indexes(cls: type[Self], typ: type[Value[Any]], context: Context) -> None:
        if isinstance(typ, parser.ListMeta):
            length = Expression.parse(typ.length, context, constexpr=True)  # type: ignore[attr-defined] # mypy don't understand that type[List] == ListMeta
            # for the compiler
            typ.length = length  # type: ignore[attr-defined]
            cls._check_indexes(typ.subtype, context)  # type: ignore[attr-defined]


class Program:
    @classmethod
    def parse(cls: type[Self], program: parser.Program, context: Context) -> Self:
        context = context.fork()
        for typedef in program.types:
            context.add_variable(
                typedef.name,
                typedef.typ,
                Expression.from_(typedef.default, context) if typedef.default else None,
            )

        return cls(
            program.name,
            [TypeDef.from_(typedef, context) for typedef in program.types],
            parse_block(program.body, context),
        )

    name: str
    types: list[TypeDef]
    body: list[Statement]

    def __init__(self: Self, name: str, types: list[TypeDef], body: list[Statement]) -> None:
        self.name = name
        self.types = types
        self.body = body


class UnparsedFunction:
    @classmethod
    def from_func(cls: type[Self], func: parser.Function) -> Self:
        return cls(Signature.from_func(func), func.types, func.body)

    @classmethod
    def from_proc(cls: type[Self], proc: parser.Procedure) -> Self:
        return cls(Signature.from_proc(proc), proc.types, proc.body)

    signature: Signature
    types: list[parser.TypeDef]
    body: list[Node[Any]]

    def __init__(self: Self, signature: Signature, types: list[parser.TypeDef], body: list[Node[Any]]) -> None:
        self.signature = signature
        self.types = types
        self.body = body

    def parse(self: Self, context: Context) -> Function:
        context = context.fork()
        for arg in self.signature.args:
            context.add_variable(arg.name, arg.typ, None)
        for typedef in self.types:
            context.add_variable(
                typedef.name,
                typedef.typ,
                Expression.from_(typedef.default, context) if typedef.default else None,
            )
        context.ret_type = self.signature.ret

        return Function(
            self.signature,
            [TypeDef.from_(typedef, context) for typedef in self.types],
            parse_block(self.body, context),
        )


class Function:
    signature: Signature
    types: list[TypeDef]
    body: list[Statement]

    def __init__(self: Self, signature: Signature, types: list[TypeDef], body: list[Statement]) -> None:
        self.signature = signature
        self.types = types
        self.body = body


class Expression:
    PRECEDENCE: ClassVar[dict[OperatorType, int]] = {
        OperatorType.POW: 4,
        OperatorType.MUL: 3,
        OperatorType.DIV: 3,
        OperatorType.MOD: 3,
        OperatorType.ADD: 2,
        OperatorType.SUB: 2,
        OperatorType.EQ: 1,
        OperatorType.NEQ: 1,
        OperatorType.LT: 1,
        OperatorType.GT: 1,
        OperatorType.LTE: 1,
        OperatorType.GTE: 1,
        OperatorType.OR: 0,
        OperatorType.AND: 0,
        OperatorType.NOT: 0,
    }

    @classmethod
    def from_(
        cls: type[Self],
        node: parser.Expr.SubTypes | Expression,
        context: Context,
        *,
        constexpr: bool = False,
    ) -> Expression:
        if constexpr and isinstance(node, parser.FuncCall):
            Error("Expression is not allowed in this context").hint(
                "Only literals and other constants are allowed, in constants, array indexes, and default values",
            ).at(node.span).log()

        if isinstance(node, Expression):
            return node
        elif isinstance(node, Value):
            return Literal(node)
        elif isinstance(node, parser.Expr):
            return Expression.parse(node, context, constexpr=constexpr)
        elif isinstance(node, parser.Binding):
            return Binding.parse(node, context, constexpr=constexpr)
        elif isinstance(node, parser.FuncCall):
            return FuncCall.parse(node, context)
        else:
            msg = f"[COMPILER BUG] Missing support for {type(node).__name__}"
            raise TypeError(msg)

    @classmethod
    def parse(cls: type[Self], expr: parser.Expr, context: Context, *, constexpr: bool = False) -> Expression:
        # can't put this in the class body because subclasses aren't defined yet
        BINOPS = {
            OperatorType.POW: Power,
            OperatorType.MUL: Multiply,
            OperatorType.DIV: Divide,
            OperatorType.MOD: Modulo,
            OperatorType.ADD: Add,
            OperatorType.SUB: Substract,
            OperatorType.EQ: Equal,
            OperatorType.NEQ: NotEqual,
            OperatorType.LT: LessThan,
            OperatorType.GT: GreaterThan,
            OperatorType.LTE: LessThanOrEqual,
            OperatorType.GTE: GreaterThanOrEqual,
            OperatorType.OR: Or,
            OperatorType.AND: And,
        }

        nodes: list[parser.Expr.SubTypes | Expression] = expr.nodes.copy()  # type: ignore[assignment]

        while True:
            max_precedence = cls.get_max_precedence(nodes)
            if max_precedence == -1:
                break

            for i, node in enumerate(nodes):
                if isinstance(node, parser.Operator) and cls.PRECEDENCE[node.op] == max_precedence:
                    op_index = i
                    break
            else:
                msg = "[COMPILER BUG] Unreachable"
                raise ValueError(msg)

            operator = cast(parser.Operator, nodes[op_index])
            if (
                operator.op in (OperatorType.ADD, OperatorType.SUB)
                and (op_index == 0 or isinstance(nodes[op_index - 1], parser.Operator))
            ) or operator.op == OperatorType.NOT:
                cls.parse_unary(op_index, nodes, context, constexpr=constexpr)
            else:
                # if there is an unary operator at the right of this operator, this will take care of it
                cls.parse_unary(op_index + 1, nodes, context, constexpr=constexpr)
                new_expr = BINOPS[operator.op](
                    Expression.from_(nodes[op_index - 1], context, constexpr=constexpr),
                    operator,
                    Expression.from_(nodes[op_index + 1], context, constexpr=constexpr),
                )
                nodes[op_index - 1] = new_expr
                nodes.pop(op_index + 1)
                nodes.pop(op_index)

        if len(nodes) > 1:
            msg = "[COMPILER BUG] Unreachable"
            raise ValueError(msg)

        if isinstance(nodes[0], Expression):
            return nodes[0]
        else:
            return Expression.from_(nodes[0], context, constexpr=constexpr)

    @classmethod
    def parse_unary(
        cls: type[Self],
        index: int,
        nodes: list[parser.Expr.SubTypes | Expression],
        context: Context,
        *,
        constexpr: bool,
    ) -> None:
        operator = nodes[index]
        if not isinstance(operator, parser.Operator):
            return

        # if there is another unary operator after this one, this should parse it
        cls.parse_unary(index + 1, nodes, context, constexpr=constexpr)
        if operator.op in (OperatorType.ADD, OperatorType.SUB):
            new_expr: Expression = (
                Positive(operator, Expression.from_(nodes[index + 1], context, constexpr=constexpr))
                if operator.op == OperatorType.ADD
                else Negative(operator, Expression.from_(nodes[index + 1], context, constexpr=constexpr))
            )
        elif operator.op == OperatorType.NOT:
            new_expr = Not(operator, Expression.from_(nodes[index + 1], context, constexpr=constexpr))
        else:
            msg = f"[COMPILER BUG] Invalid unary operator: {operator}"
            raise TypeError(msg)

        nodes[index] = new_expr
        nodes.pop(index + 1)

    @classmethod
    def get_max_precedence(cls: type[Self], nodes: list[parser.Expr.SubTypes | Expression]) -> int:
        return max((cls.PRECEDENCE[node.op] for node in nodes if isinstance(node, parser.Operator)), default=-1)

    precedence: int = -1
    typ: type[Value[Any]]
    span: Span

    def __init__(self: Self, typ: type[Value[Any]], span: Span) -> None:
        self.typ = typ
        self.span = span


class Literal(Expression):
    precedence: int = 5
    value: Value[Any]

    def __init__(self: Self, value: Value[Any]) -> None:
        super().__init__(type(value), value.span)
        self.value = value


class Unary(Expression):
    right: Expression

    def __init__(self: Self, operator: parser.Operator, right: Expression) -> None:
        super().__init__(right.typ, operator.span + right.span)
        self.right = right


class Binary(Expression):
    @classmethod
    def _check_types(
        cls: type[Self],
        left: Expression,
        operator: parser.Operator,
        right: Expression,
    ) -> bool:
        if left.typ != right.typ:
            error = (
                Error(f"Incompatible types for operator {cls.__name__}: {left.typ.name} and {right.typ.name}")
                .at(
                    operator.span,
                )
                .comment(
                    left.span,
                    f"this is of type {left.typ.name}",
                )
                .comment(
                    right.span,
                    f"this is of type {right.typ.name}",
                )
            )

            not_float = None
            if (
                issubclass(left.typ, parser.Integer)
                and issubclass(right.typ, parser.Float)
                and isinstance(left, Literal)
            ):
                not_float = left
            if (
                issubclass(right.typ, parser.Integer)
                and issubclass(left.typ, parser.Float)
                and isinstance(right, Literal)
            ):
                not_float = right

            if not_float:
                error.replacement(
                    (Span.at(not_float.span.end), ".0"),
                    msg="replace this int literal with a float literal",
                )

            error.log()
            return False
        return True

    left: Expression
    right: Expression

    def __init__(self: Self, left: Expression, operator: parser.Operator, right: Expression) -> None:
        if Value in (left.typ, right.typ) or not self._check_types(left, operator, right):
            super().__init__(Unknown.value, left.span + right.span)
        else:
            super().__init__(left.typ, left.span + right.span)

        self.left = left
        self.right = right


class Check(Binary):
    def __init__(self: Self, left: Expression, operator: parser.Operator, right: Expression) -> None:
        super().__init__(left, operator, right)
        self.typ = parser.Bool


class Positive(Unary):
    precedence: int = 3


class Negative(Unary):
    precedence: int = 3


class Not(Unary):
    precedence: int = 0

    def __init__(self: Self, operator: parser.Operator, right: Expression) -> None:
        super().__init__(operator, right)
        self.typ = parser.Bool


class Power(Binary):
    precedence: int = 4


class Multiply(Binary):
    precedence: int = 3


class Divide(Binary):
    precedence: int = 3


class Modulo(Binary):
    precedence: int = 3


class Add(Binary):
    precedence: int = 2


class Substract(Binary):
    precedence: int = 2


class Equal(Check):
    precedence: int = 1


class NotEqual(Check):
    precedence: int = 1


class LessThan(Check):
    precedence: int = 1


class GreaterThan(Check):
    precedence: int = 1


class LessThanOrEqual(Check):
    precedence: int = 1


class GreaterThanOrEqual(Check):
    precedence: int = 1


class Or(Check):
    precedence: int = 0


class And(Check):
    precedence: int = 0


class Binding(Expression):
    precedence: int = 5

    @classmethod
    def parse(cls: type[Self], binding: parser.Binding, context: Context, *, constexpr: bool = False) -> Binding:  # type: ignore[override]
        if isinstance(binding, parser.Variable):
            return Variable(binding.name, context, constexpr=constexpr)
        elif isinstance(binding, parser.Indexing):
            return Indexing(
                Binding.parse(binding.sub, context, constexpr=constexpr),
                Expression.parse(binding.index, context),
                binding.bracket_span,
            )
        else:
            msg = f"[COMPILER BUG] Missing support for binding {type(binding).__name__}"
            raise TypeError(msg)


class Variable(Binding):
    name: SpannedStr

    def __init__(self: Self, name: SpannedStr, context: Context, *, constexpr: bool = False) -> None:
        if constexpr:
            typ = context.lookup_constant(name)[0].value
            if typ == Value:
                e = Error(f"Cannot find declaration for constant {name}").at(name.span)
                if context.lookup_variable(name)[0].value != Value:
                    e.hint("a variable with this name exist, but only constants are allowed in this context")
                e.log()
        else:
            typ = context.lookup_variable(name)[0].value
            if typ == Value:
                Error(f"Cannot find declaration for variable {name}").at(name.span).log()

        super().__init__(typ, name.span)
        self.name = name


class Indexing(Binding):
    base: Binding
    index: Expression
    brackets_span: Span

    def __init__(self: Self, base: Binding, index: Expression, brackets_span: Span) -> None:
        if base.typ == Value:
            typ = Unknown.value
        elif issubclass(base.typ, parser.List):
            typ = base.typ.subtype
        else:
            Error(f"Cannot index {base.typ.name}").at(brackets_span).comment(
                base.span,
                f"this is of type `{base.typ.name}`",
            ).log()
            typ = Unknown.value

        if not (issubclass(index.typ, parser.Integer) or index.typ == Value):
            Error(f"List index must be an {parser.Integer.name}").at(
                index.span,
                msg=f"this is of type {index.typ.name}",
            ).log()

        super().__init__(typ, base.span + brackets_span)
        self.base = base
        self.index = index
        self.brackets_span = brackets_span


class FuncCall(Expression):
    precedence: int = 5

    @classmethod
    def parse(cls: type[Self], fcall: parser.FuncCall, context: Context) -> Self:  # type: ignore[override]
        args = [Expression.parse(arg, context) for arg in fcall.arguments]
        ref_args = [Binding.parse(arg, context) for arg in fcall.references]

        signature = context.lookup_function(fcall.name)
        if signature is None:
            Error(f"Cannot find declaration for function {fcall.name}").at(fcall.name.span).log()
            return cls(fcall.name, args, ref_args, Unknown.value, fcall.span, None)

        def_args = signature.value_args
        if len(args) != len(def_args):
            e = Error(f"Function {fcall.name} takes {len(def_args)} arguments but {len(args)} were given").at(
                fcall.name.span,
            )

            if args:
                e.comment(reduce((lambda x, y: x + y), (x.span for x in args)), f"there are {len(args)} arguments here")

            if def_args:
                e.comment(
                    reduce((lambda x, y: x + y), (x.span for x in def_args)),
                    f"the function takes {len(def_args)} arguments",
                )
            else:
                e.comment(signature.name.span, "the function is defined here")

            e.log()

        def_ref_args = signature.ref_args
        if len(ref_args) != len(def_ref_args):
            e = Error(
                f"Function {fcall.name} takes {len(def_ref_args)} arguments by reference but {len(ref_args)} were given",
            ).at(fcall.name.span)

            if ref_args:
                e.comment(
                    reduce((lambda x, y: x + y), (x.span for x in ref_args)),
                    f"there are {len(ref_args)} arguments here",
                )

            if def_ref_args:
                e.comment(
                    reduce((lambda x, y: x + y), (x.span for x in def_ref_args)),
                    f"the function takes {len(def_ref_args)} arguments",
                )
            else:
                e.comment(signature.name.span, "the function is defined here")

            e.log()

        for arg, def_arg in zip(args, def_args):
            if arg.typ == Value:
                continue

            if arg.typ != def_arg.typ.value:
                Error(f"Expected {def_arg.typ.value.name}, found {arg.typ.name}").at(
                    arg.span,
                    msg=f"this is of type {arg.typ.name}",
                ).comment(def_arg.span, f"the argument is defined with a type of {def_arg.typ.value.name}").log()

        for arg, def_ref_arg in zip(ref_args, def_ref_args):
            if arg.typ == Value:
                continue

            if arg.typ != def_ref_arg.typ.value:
                Error(f"Expected {def_ref_arg.typ.value.name}, found {arg.typ.name}").at(
                    arg.span,
                    msg=f"this is of type {arg.typ.name}",
                ).comment(
                    def_ref_arg.span,
                    f"the argument is defined with a type of {def_ref_arg.typ.value.name}",
                ).log()

        if signature.ret:  # noqa: SIM108 (mypy is broken with ternary expressions)
            ret = signature.ret.value
        else:
            ret = parser.Void

        return cls(fcall.name, args, ref_args, ret, fcall.span, signature)

    name: SpannedStr
    args: list[Expression]
    ref_args: list[Binding]

    signature: Signature | None

    def __init__(
        self: Self,
        name: SpannedStr,
        args: list[Expression],
        ref_args: list[Binding],
        ret_type: type[Value[Any]],
        span: Span,
        signature: Signature | None,
    ) -> None:
        super().__init__(ret_type, span)
        self.name = name
        self.args = args
        self.ref_args = ref_args
        self.signature = signature


class Statement:
    pass


class Assignement(Statement):
    binding: Binding
    value: Expression

    def __init__(self: Self, binding: Binding, value: Expression) -> None:
        self.binding = binding
        self.value = value

        if Value in (self.binding.typ, self.value.typ):
            return

        if self.value.typ != self.binding.typ:
            Error(f"Cannot assign a {self.value.typ.name} to a {self.binding.typ.name}").at(
                self.value.span,
                msg=f"this is of type {self.value.typ.name}",
            ).comment(self.binding.span, f"the variable is of type {self.binding.typ.name}").log()


class ExpressionStmt(Statement):
    expr: Expression

    def __init__(self: Self, expr: Expression) -> None:
        self.expr = expr


class Print(Statement):
    elements: list[Expression]

    def __init__(self: Self, elements: list[Expression]) -> None:
        self.elements = elements

        for element in self.elements:
            if not (
                issubclass(element.typ, (parser.Bool, parser.Integer, parser.Float, parser.Char, parser.String))
                or element.typ == Value
            ):
                Error(f"{element.typ.name} is not printable").at(
                    element.span,
                    msg=f"this is of type {element.typ.name}",
                ).log()


class Input(Statement):
    bindings: list[Binding]

    def __init__(self: Self, bindings: list[Binding]) -> None:
        self.bindings = bindings

        for binding in self.bindings:
            if not (
                issubclass(binding.typ, (parser.Integer, parser.Float, parser.Char, parser.String))
                or binding.typ == Value
            ):
                Error(f"Can't input a variable of type {binding.typ.name}").at(
                    binding.span,
                    msg=f"this is of type {binding.typ.name}",
                ).log()


class Return(Statement):
    value: Expression

    def __init__(self: Self, value: Expression, span: Span, context: Context) -> None:
        if context.ret_type is None:
            Error("Cannot return here").hint("You can only return in functions").at(span).log()
        elif value.typ not in (context.ret_type.value, Value):
            Error(f"Expected {context.ret_type.value.name} but found {value.typ.name}").at(
                value.span,
                msg=f"this is of type {value.typ.name}",
            ).comment(context.ret_type.span, "the function's return type is declared here").log()

        self.value = value


class Condition(Statement):
    condition: Expression
    if_block: list[Statement]
    else_block: list[Statement] | None

    def __init__(
        self: Self,
        condition: Expression,
        if_block: list[Statement],
        else_block: list[Statement] | None,
    ) -> None:
        self.condition = condition
        self.if_block = if_block
        self.else_block = else_block

        if not (issubclass(condition.typ, parser.Bool) or condition.typ == Value):
            Error(f"A condition must be a {parser.Bool.name}").at(
                condition.span,
                msg=f"this is of type {condition.typ.name}",
            ).hint(f"There is no automatic conversion to {parser.Bool.name} in pseudo-code").log()


class Switch(Statement):
    binding: Binding
    cases: list[tuple[Literal, list[Statement]]]
    default: list[Statement] | None

    def __init__(
        self: Self,
        binding: Binding,
        cases: list[tuple[Literal, list[Statement]]],
        default: list[Statement] | None,
    ) -> None:
        self.binding = binding
        self.cases = cases
        self.default = default

        typ = self.binding.typ
        for lit, _ in self.cases:
            if not issubclass(lit.typ, typ):
                Error("Every case must be of the same type than the variable").at(
                    lit.span,
                    msg=f"this is of type {lit.typ.name}",
                ).comment(self.binding.span, f"the variable is of type {typ.name}").log()


class ForLoop(Statement):
    binding: Binding
    start: Expression
    end: Expression
    step: Expression | None
    body: list[Statement]

    def __init__(
        self: Self,
        binding: Binding,
        start: Expression,
        end: Expression,
        step: Expression | None,
        body: list[Statement],
    ) -> None:
        self.binding = binding
        self.start = start
        self.end = end
        self.step = step
        self.body = body

        if not (issubclass(binding.typ, parser.Integer) or binding.typ == Value):
            Error(f"A for loop variable must be an {parser.Integer.name}").at(
                binding.span,
                msg=f"this is of type {binding.typ.name}",
            ).log()

        if not (issubclass(start.typ, parser.Integer) or start.typ == Value):
            Error(f"The start value of a for-loop must be an {parser.Integer.name}").at(
                start.span,
                msg=f"this is of type {start.typ.name}",
            ).log()

        if not (issubclass(end.typ, parser.Integer) or end.typ == Value):
            Error(f"The end value of a for-loop must be an {parser.Integer.name}").at(
                end.span,
                msg=f"this is of type {end.typ.name}",
            ).log()

        if step and not (issubclass(step.typ, parser.Integer) or step.typ == Value):
            Error(f"The step value of a for-loop must be an {parser.Integer.name}").at(
                step.span,
                msg=f"this is of type {step.typ.name}",
            ).log()


class WhileLoop(Statement):
    condition: Expression
    body: list[Statement]

    def __init__(self: Self, condition: Expression, body: list[Statement]) -> None:
        self.condition = condition
        self.body = body

        if not (issubclass(condition.typ, parser.Bool) or condition.typ == Value):
            Error(f"A condition must be a {parser.Bool.name}").at(
                condition.span,
                msg=f"this is of type {condition.typ.name}",
            ).hint(f"There is no automatic conversion to {parser.Bool.name} in pseudo-code").log()


class DoWhileLoop(Statement):
    condition: Expression
    body: list[Statement]

    def __init__(self: Self, condition: Expression, body: list[Statement]) -> None:
        self.condition = condition
        self.body = body

        if not (issubclass(condition.typ, parser.Bool) or condition.typ == Value):
            Error(f"A condition must be a {parser.Bool.name}").at(
                condition.span,
                msg=f"this is of type {condition.typ.name}",
            ).hint(f"There is no automatic conversion to {parser.Bool.name} in pseudo-code").log()


def parse_block(nodes: list[Node[Any]], context: Context) -> list[Statement]:
    statements: list[Statement] = []

    for node in nodes:
        if isinstance(node, (parser.Program, parser.Function, parser.Procedure)):
            node_typename = type(node).__name__.lower()
            Error(
                f"You can't put a {node_typename} inside another block",
            ).at(
                node.span,
            ).hint(f"you can put this {node_typename} outside of the block").log()
        elif isinstance(node, parser.Assignement):
            lhs = Binding.parse(node.binding, context)
            rhs = Expression.parse(node.value, context)

            statements.append(Assignement(lhs, rhs))
        elif isinstance(node, parser.FuncCall):
            statements.append(ExpressionStmt(FuncCall.parse(node, context)))
        elif isinstance(node, parser.Print):
            elements = [Expression.parse(element, context) for element in node.elements]
            statements.append(Print(elements))
        elif isinstance(node, parser.Input):
            bindings = [Binding.parse(binding, context) for binding in node.bindings]
            statements.append(Input(bindings))
        elif isinstance(node, parser.Return):
            value = Expression.parse(node.value, context)
            statements.append(Return(value, node.span, context))
        elif isinstance(node, parser.Condition):
            condition = Expression.parse(node.condition, context)
            if_block = parse_block(node.if_block, context)
            else_block = parse_block(node.else_block, context) if node.else_block else None

            statements.append(Condition(condition, if_block, else_block))
        elif isinstance(node, parser.Switch):
            binding = Binding.parse(node.binding, context)
            cases = [(Literal(value), parse_block(block, context)) for value, block in node.blocks]
            default = parse_block(node.default, context) if node.default else None

            statements.append(Switch(binding, cases, default))
        elif isinstance(node, parser.ForLoop):
            binding = Binding.parse(node.binding, context)
            start = Expression.parse(node.start, context)
            end = Expression.parse(node.end, context)
            step = Expression.parse(node.step, context) if node.step else None
            body = parse_block(node.block, context)

            statements.append(ForLoop(binding, start, end, step, body))
        elif isinstance(node, parser.WhileLoop):
            condition = Expression.parse(node.condition, context)
            body = parse_block(node.block, context)

            statements.append(WhileLoop(condition, body))
        elif isinstance(node, parser.DoWhileLoop):
            condition = Expression.parse(node.condition, context)
            body = parse_block(node.block, context)

            statements.append(DoWhileLoop(condition, body))
        else:
            msg = f"[COMPILER BUG] missing support for statement {type(node).__name__}"
            raise TypeError(msg)

    return statements


def parse(
    nodes: Iterable[Node[Any]],
    constants: list[parser.TypeDef],
) -> tuple[list[Program | Function], Context]:
    context = Context()

    for constant in constants:
        # the parser already checked that this is not None
        value = Expression.parse(cast(parser.Expr, constant.default), context, constexpr=True)
        context.add_constant(constant.name, constant.typ, value)

    unparsed_program = []
    unparsed_function = []
    for node in nodes:
        if isinstance(node, parser.Function):
            function = UnparsedFunction.from_func(node)
            context.add_function(function.signature)
            unparsed_function.append(function)
        elif isinstance(node, parser.Procedure):
            function = UnparsedFunction.from_proc(node)
            context.add_function(function.signature)
            unparsed_function.append(function)
        elif isinstance(node, parser.Program):
            unparsed_program.append(node)
        else:
            Error("Invalid top level instruction").at(node.span).detail(type(node).__name__).hint(
                "you must put this instruction in a program, a function or a procedure",
            ).log()

    return [function.parse(context) for function in unparsed_function] + [
        Program.parse(program, context) for program in unparsed_program
    ], context
