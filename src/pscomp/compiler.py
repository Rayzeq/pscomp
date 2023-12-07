from __future__ import annotations

from itertools import tee, zip_longest
from typing import TYPE_CHECKING, Any, TypeVar, overload

from .errors import InternalCompilerError
from .parser import Unknown
from .typer import (
    Add,
    And,
    Argument,
    Assignement,
    Binding,
    BuiltinSignature,
    Cast,
    Comment,
    Condition,
    Destroy,
    Divide,
    DoWhileLoop,
    Equal,
    Expression,
    ExpressionStmt,
    ForLoop,
    FuncCall,
    Function,
    GreaterThan,
    GreaterThanOrEqual,
    Indexing,
    InlinePython,
    Input,
    LessThan,
    LessThanOrEqual,
    Literal,
    Lookup,
    Modulo,
    Multiply,
    Negative,
    Not,
    NotEqual,
    Or,
    Positive,
    Power,
    Print,
    Program,
    Return,
    Statement,
    Substract,
    Switch,
    TypeDef,
    VariableBinding,
    WhileLoop,
)
from .types import Bool, Char, Float, Integer, List, String, Type, Void
from .types import Structure as StructureType

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from . import typer
    from .parser import Structure, Value

CASTMAP = {
    Integer: int,
    Float: float,
}


T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T | None]]:
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b)


def compile_value(value: Value) -> str:
    if isinstance(value, Unknown):
        return "Uninit"
    return repr(value.value)


def precede(expr1: Expression, expr2: Expression) -> str:
    if expr1.precedence > expr2.precedence:
        return f"({compile_expr(expr2)})"
    else:
        return f"{compile_expr(expr2)}"


def compile_binding(binding: Binding) -> str:
    if isinstance(binding, VariableBinding):
        return binding.name
    elif isinstance(binding, Indexing):
        return f"{compile_binding(binding.base)}[{compile_expr(binding.index)}]"
    elif isinstance(binding, Lookup):
        return f"{compile_binding(binding.base)}.{binding.name}"
    else:
        msg = f"Cannot compile binding: {binding}"
        raise InternalCompilerError(msg)


def compile_funcall(funcall: FuncCall) -> str:
    if funcall.signature and isinstance(funcall.signature, (BuiltinSignature, Cast)):
        name = funcall.signature.python_name
    else:
        name = funcall.name
    ref_args = list(map(compile_binding, funcall.ref_args))
    args = ", ".join(list(map(compile_expr, funcall.args)) + ref_args)
    call = f"{name}({args})"

    if funcall.ref_args:
        call = ", ".join(ref_args) + " = " + call

    return call


def compile_expr(expr: Expression) -> str:
    if isinstance(expr, Literal):
        return compile_value(expr.value)
    elif isinstance(expr, Positive):
        return f"+{precede(expr, expr.right)}"
    elif isinstance(expr, Negative):
        return f"-{precede(expr, expr.right)}"
    elif isinstance(expr, Not):
        return f"not {precede(expr, expr.right)}"
    elif isinstance(expr, Power):
        return f"{precede(expr, expr.left)}**{precede(expr, expr.right)}"
    elif isinstance(expr, Multiply):
        return f"{precede(expr, expr.left)} * {precede(expr, expr.right)}"
    elif isinstance(expr, Divide):
        if isinstance(expr.typ, Integer):
            return f"{precede(expr, expr.left)} // {precede(expr, expr.right)}"
        else:
            return f"{precede(expr, expr.left)} / {precede(expr, expr.right)}"
    elif isinstance(expr, Modulo):
        return f"{precede(expr, expr.left)} % {precede(expr, expr.right)}"
    elif isinstance(expr, Add):
        return f"{precede(expr, expr.left)} + {precede(expr, expr.right)}"
    elif isinstance(expr, Substract):
        return f"{precede(expr, expr.left)} - {precede(expr, expr.right)}"
    elif isinstance(expr, Equal):
        return f"{precede(expr, expr.left)} == {precede(expr, expr.right)}"
    elif isinstance(expr, NotEqual):
        return f"{precede(expr, expr.left)} != {precede(expr, expr.right)}"
    elif isinstance(expr, LessThan):
        return f"{precede(expr, expr.left)} < {precede(expr, expr.right)}"
    elif isinstance(expr, GreaterThan):
        return f"{precede(expr, expr.left)} > {precede(expr, expr.right)}"
    elif isinstance(expr, LessThanOrEqual):
        return f"{precede(expr, expr.left)} <= {precede(expr, expr.right)}"
    elif isinstance(expr, GreaterThanOrEqual):
        return f"{precede(expr, expr.left)} >= {precede(expr, expr.right)}"
    elif isinstance(expr, Or):
        return f"{precede(expr, expr.left)} or {precede(expr, expr.right)}"
    elif isinstance(expr, And):
        return f"{precede(expr, expr.left)} and {precede(expr, expr.right)}"
    elif isinstance(expr, Binding):
        return compile_binding(expr)
    elif isinstance(expr, FuncCall):
        return compile_funcall(expr)
    else:
        msg = f"Cannot compile expression: {expr}"
        raise InternalCompilerError(msg)


def compile_input(binding: Binding) -> str:
    if (cast := CASTMAP.get(type(binding.typ))) is not None:
        return f"{compile_binding(binding)} = {cast.__name__}(input())"
    else:
        return f"{compile_binding(binding)} = input()"


def compile_statement(statement: Statement) -> str:
    if isinstance(statement, Assignement):
        return f"{compile_binding(statement.binding)} = {compile_expr(statement.value)}"
    elif isinstance(statement, ExpressionStmt):
        return f"{compile_expr(statement.expr)}"
    elif isinstance(statement, Print):
        return f"print({make_fstring(statement.elements)})"
    elif isinstance(statement, Input):
        return "\n".join(compile_input(b) for b in statement.bindings)
    elif isinstance(statement, Destroy):
        return f"{compile_binding(statement.binding)}.__destroy__()"
    elif isinstance(statement, InlinePython):
        return statement.code
    elif isinstance(statement, Comment):
        return f"#{statement.text}"
    elif isinstance(statement, Condition):
        base = f"if {compile_expr(statement.condition)}:\n    {indent(4, compile_block(statement.if_block))}"
        if statement.else_block:
            base += f"\nelse:\n    {indent(4, compile_block(statement.else_block))}"
        return base
    elif isinstance(statement, Switch):
        cases = "\n".join(
            f"case {compile_expr(case[0])}:\n    {indent(4, compile_block(case[1]))}" for case in statement.cases
        )
        if statement.default:
            cases += "\n" + f"case _:\n    {indent(4, compile_block(statement.default))}"

        return f"match {compile_expr(statement.value)}:\n    {indent(4, cases)}"
    elif isinstance(statement, ForLoop):
        binding = compile_binding(statement.binding)
        start = compile_expr(statement.start)
        end = compile_expr(statement.end)
        step = (", " + compile_expr(statement.step)) if statement.step else ""
        return f"for {binding} in range({start}, ({end}) + 1{step}):\n    {indent(4, compile_block(statement.body))}"
    elif isinstance(statement, WhileLoop):
        return f"while {compile_expr(statement.condition)}:\n    {indent(4, compile_block(statement.body))}"
    elif isinstance(statement, DoWhileLoop):
        return f"""
while True:
    {indent(4, compile_block(statement.body))}

    if not ({compile_expr(statement.condition)}):
        break""".strip()
    elif isinstance(statement, Return):
        return f"return {compile_expr(statement.value)}"
    else:
        msg = f"Cannot compile statement: {statement}"
        raise InternalCompilerError(msg)


def compile_block(statments: list[Statement], ret_values: Sequence[str] | None = None) -> str:
    if not statments:
        return "pass"

    result = []
    skip = False

    for statement, next_stmt in pairwise(statments):
        if skip:
            skip = False
            continue

        if isinstance(statement, Print) and isinstance(next_stmt, Input) and len(next_stmt.bindings) == 1:
            skip = True

            bd = compile_binding(next_stmt.bindings[0])
            cast = CASTMAP.get(type(next_stmt.bindings[0].typ))
            msg = make_fstring(statement.elements)
            if cast is None:
                result.append(f"{bd} = input({msg})")
            else:
                result.append(f"{bd} = {cast.__name__}(input({msg}))")
        else:
            result.append(compile_statement(statement))

    if ret_values:
        result.append(f"return {', '.join(ret_values)}")

    return "\n".join(result)


def compile_type(typ: Type) -> str:
    TYPEMAP = {
        Any: "Any",
        Void: "None",
        Bool: "bool",
        Integer: "int",
        Float: "float",
        Char: "str",
        String: "str",
    }

    if isinstance(typ, List):
        return f"list[{compile_type(typ.subtype)}]"
    elif isinstance(typ, StructureType):
        return typ.name
    else:
        return TYPEMAP[type(typ)]


def compile_argdef(arg: Argument) -> str:
    return f"{arg.name}: {compile_type(arg.typ)}"


def compile_typedef(typedef: TypeDef) -> str:
    def list_init(typ: Type) -> str:
        if isinstance(typ, List):
            return f"[{list_init(typ.subtype)} for _ in range({compile_expr(typ.length)})]"  # go check TypeDef._check_indexes
        else:
            return "Uninit"

    result = f"{typedef.name}: {compile_type(typedef.typ)}"
    if typedef.default is not None:
        result += f" = {compile_expr(typedef.default)}"
    elif isinstance(typedef.typ, List):
        result += f" = {list_init(typedef.typ)}"
    else:
        result += " = Uninit"
    return result


def compile_typedefs(typedefs: list[TypeDef]) -> list[str]:
    return list(map(compile_typedef, typedefs))


def compile_function(function: Function) -> str:
    if function.signature.ret is not None:
        ret = compile_type(function.signature.ret)
    elif refs := function.signature.ref_args:
        ret = f"tuple[{', '.join([compile_type(r.typ) for r in refs])}]"
    else:
        ret = "None"

    ret_values = [r.name for r in function.signature.ref_args]

    typedefs = indent(4, compile_typedefs(function.types)) + "\n\n    " if function.types else ""

    return f"""
def {function.signature.name}({", ".join(map(compile_argdef, function.signature.args))}) -> {ret}:
    {typedefs}{indent(4, compile_block(function.body, ret_values))}
""".strip()


def compile_program(program: Program) -> str:
    typedefs = indent(4, compile_typedefs(program.types)) + "\n\n    " if program.types else ""
    return f"""
def {program.name}() -> None:
    {typedefs}{indent(4, compile_block(program.body))}


if __name__ == "__main__":
    {program.name}()
""".strip()


def compile_toplevel(toplevel: Program | Function) -> str:
    if isinstance(toplevel, Program):
        return compile_program(toplevel)
    elif isinstance(toplevel, Function):
        return compile_function(toplevel)
    else:
        msg = f"Cannot compile toplevel: {toplevel}"
        raise InternalCompilerError(msg)


def compile_struct(struct: Structure) -> str:
    fields = "\n".join(f"{f.name}: {compile_type(f.typ)}" for f in struct.fields.values())
    field_inits = "\n".join(f"self.{f.name} = Uninit" for f in struct.fields.values())

    return f"""
class {struct.name}:
    __destroyed__: bool = False
    {indent(4, fields)}

    def __init__(self: {struct.name}) -> None:
        {indent(8, field_inits)}

    def __destroy__(self: {struct.name}) -> None:
        self.__destroyed__ = True

    def __getattribute__(self: {struct.name}, name: str) -> Any:
        if name ==  "__destroyed__":
            return super().__getattribute__(name)

        if self.__destroyed__:
            msg = "Cannot access destroyed structure"
            raise RuntimeError(msg)
        return super().__getattribute__(name)

    def __setattr__(self: {struct.name}, name: str, value: Any) -> None:
        if name ==  "__destroyed__":
            return super().__setattr__(name, value)

        if self.__destroyed__:
            msg = "Cannot access destroyed structure"
            raise RuntimeError(msg)
        return super().__setattr__(name, value)
""".strip()


@overload
def indent(level: int, lines: str) -> str:
    ...


@overload
def indent(level: int, lines: Iterator[str]) -> str:
    ...


@overload
def indent(level: int, lines: Sequence[str]) -> str:
    ...


def indent(level: int, lines: str | Iterator[str] | Sequence[str]) -> str:
    joint = "\n" + " " * level
    if isinstance(lines, str):
        return joint.join(lines.split("\n"))
    else:
        return joint.join(indent(level, x) for x in lines)


def make_fstring(elements: list[Expression]) -> str:
    result = []
    needs_f = False
    for element in elements:
        if isinstance(element, Literal) and isinstance(element.value.value, str):
            result.append(element.value.value.replace("{", "{{").replace("}", "}}"))
        else:
            needs_f = True
            result.append(f"{{{compile_expr(element)}}}")

    if len(result) == 1 and needs_f:
        return f"str({compile_expr(elements[0])})"

    return ("f" if needs_f else "") + repr(" ".join(result))


def compile(toplevels: list[Program | Function], context: typer.Context) -> str:
    linesep = "\n"
    joint = "\n\n\n"

    imports = "\n".join(
        f"from {module} import {', '.join(attributes)}" for module, attributes in context.python_imports.items()
    )

    constants = "\n".join(
        f"{name}: {compile_type(const.typ)} = {compile_expr(const.value)}" for name, const in context.constants.items()
    )

    structure = "\n\n\n".join(map(compile_struct, context.structures.values()))

    return f"""#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
{linesep * 2 + (linesep * 2).join(filter(bool, (imports, constants, structure))) + linesep * 2}
# ================ Compiler generated code, please don't edit ================
class UninitMeta(type):
    def __instancecheck__(cls: UninitMeta, instance: object) -> bool:
        return True


UninitClass = UninitMeta("UninitClass", (), {{}})
Uninit = UninitClass()
# ============================================================================


{joint.join(compile_toplevel(x) for x in toplevels)}
"""
