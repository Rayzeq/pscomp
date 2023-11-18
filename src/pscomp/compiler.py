from __future__ import annotations

from itertools import tee, zip_longest
from typing import TYPE_CHECKING, Any, TypeVar, overload

from . import parser, typer
from .parser import Bool, Char, Float, Integer, List, String, Value, Void
from .typer import (
    Add,
    And,
    Argument,
    Assignement,
    Binding,
    BuiltinSignature,
    Condition,
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
    Input,
    LessThan,
    LessThanOrEqual,
    Literal,
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
    Variable,
    WhileLoop,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


CASTMAP = {
    Integer: int,
    Float: float,
}


T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T | None]]:
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b)


def compile_value(value: Value[Any]) -> str:
    if isinstance(value, (Bool, Integer, Float, Char, String)):
        return repr(value.value)
    elif isinstance(value, List):
        msg = f"[COMPILER BUG] You're a liar, list literals don't exist ({value})"
        raise TypeError(msg)
    else:
        msg = f"[COMPILER BUG] Cannot compile value {value}"
        raise TypeError(msg)


def precede(expr1: Expression, expr2: Expression) -> str:
    if expr1.precedence > expr2.precedence:
        return f"({compile_expr(expr2)})"
    else:
        return f"{compile_expr(expr2)}"


def compile_binding(binding: Binding) -> str:
    if isinstance(binding, Variable):
        return binding.name
    elif isinstance(binding, Indexing):
        return f"{compile_binding(binding.base)}[{compile_expr(binding.index)}]"
    else:
        msg = f"[COMPILER BUG] Missing support for binding: {binding}"
        raise TypeError(msg)


def compile_funcall(funcall: FuncCall) -> str:
    if funcall.signature and isinstance(funcall.signature, BuiltinSignature):
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
        if issubclass(expr.typ, Integer):
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
        msg = f"[COMPILER BUG] Missing support for expression: {expr}"
        raise TypeError(msg)


def compile_input(binding: Binding) -> str:
    if (cast := CASTMAP.get(binding.typ)) is not None:
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

        return f"match {compile_expr(statement.binding)}:\n    {indent(4, cases)}"
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
        msg = f"[COMPILER BUG] Missing support for statement: {statement}"
        raise TypeError(msg)


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
            cast = CASTMAP.get(next_stmt.bindings[0].typ)
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


def compile_type(typ: type[Value[Any]]) -> str:
    TYPEMAP = {
        Value: "Any",
        Void: "None",
        Bool: "bool",
        Integer: "int",
        Float: "float",
        Char: "str",
        String: "str",
    }

    if issubclass(typ, List):
        return f"list[{compile_type(typ.subtype)}]"
    else:
        return TYPEMAP[typ]


def compile_argdef(arg: Argument) -> str:
    return f"{arg.name}: {compile_type(arg.typ.value)}"


def compile_typedef(typedef: parser.TypeDef) -> str:
    def list_init(typ: type[Value[Any]]) -> str:
        if issubclass(typ, parser.List):
            return f"[{list_init(typ.subtype)} for _ in range({typ.length})]"
        else:
            return "Uninit"

    result = f"{typedef.name}: {compile_type(typedef.typ.value)}"
    if typedef.default is not None:
        result += f" = {compile_value(typedef.default)}"
    elif issubclass(typedef.typ.value, parser.List):
        result += f" = {list_init(typedef.typ.value)}"
    else:
        result += " = Uninit"
    return result


def compile_typedefs(typedefs: list[parser.TypeDef]) -> list[str]:
    return list(map(compile_typedef, typedefs))


def compile_function(function: Function) -> str:
    if function.signature.ret is not None:
        ret = compile_type(function.signature.ret.value)
    elif refs := function.signature.ref_args:
        ret = f"tuple[{', '.join([compile_type(r.typ.value) for r in refs])}]"
    else:
        ret = "None"

    ret_values = [r.name for r in function.signature.ref_args]

    return f"""
def {function.signature.name}({", ".join(map(compile_argdef, function.signature.args))}) -> {ret}:
    {indent(4, compile_typedefs(function.types))}

    {indent(4, compile_block(function.body, ret_values))}
""".strip()


def compile_program(program: Program) -> str:
    return f"""
def {program.name}() -> None:
    {indent(4, compile_typedefs(program.types))}

    {indent(4, compile_block(program.body))}


if __name__ == "__name__":
    {program.name}()
""".strip()


def compile_toplevel(toplevel: Program | Function) -> str:
    if isinstance(toplevel, Program):
        return compile_program(toplevel)
    elif isinstance(toplevel, Function):
        return compile_function(toplevel)
    else:
        msg = f"[COMPILER BUG] Missing support for {toplevel}"
        raise TypeError(msg)


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
        if isinstance(element, Literal) and isinstance(element.value, (String, Char)):
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

    imports = linesep.join(
        f"from {module} import {', '.join(attributes)}" for module, attributes in context.python_imports.items()
    )

    return f"""#!/usr/bin/env python3
from __future__ import annotations
{linesep + imports + linesep if imports else ''}

# ================ Compiler generated code, please don't edit ================
class UninitMeta(type):
    def __instancecheck__(cls: UninitMeta, instance: object) -> bool:
        return True


UninitClass = UninitMeta("UninitClass", (), {{}})
Uninit = UninitClass()
# ============================================================================


{joint.join(compile_toplevel(x) for x in toplevels)}
"""