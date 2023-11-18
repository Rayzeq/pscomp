from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, SupportsIndex

from source import Position, SourceFile, Span

if TYPE_CHECKING:
    from typing_extensions import Self


class ANSI:
    @classmethod
    def csi(cls: type[Self], command: str) -> str:
        return f"\x1b[{command}"

    @classmethod
    def srg(cls: type[Self], attributes: list[int]) -> str:
        return cls.csi(";".join(map(str, attributes)) + "m")


class Color(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    BRIGHT_BLACK = 90
    BRIGHT_RED = 91
    BRIGHT_GREEN = 92
    BRIGHT_YELLOW = 93
    BRIGHT_BLUE = 94
    BRIGHT_MAGENTA = 95
    BRIGHT_CYAN = 96
    BRIGHT_WHITE = 97

    RESET = 39


@dataclass(frozen=True, slots=True)
class Format:
    foreground: Color | None = None
    bold: bool | None = None


@dataclass(frozen=True, slots=True)
class Reset:
    format: Format


class FormattedStr(str):
    __slots__ = ("regions",)

    regions: list[tuple[range, Format]]

    def __new__(cls: type[Self], value: str) -> Self:
        self = super().__new__(cls, value)
        self.regions = []
        return self

    def colored(self: Self, span: range, color: Color, *, bold: bool | None = None) -> None:
        self.regions.append((span, Format(foreground=color, bold=bold)))

    def black(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_BLACK if bright else Color.BLACK, **kwargs)

    def red(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_RED if bright else Color.RED, **kwargs)

    def green(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_GREEN if bright else Color.GREEN, **kwargs)

    def yellow(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_YELLOW if bright else Color.YELLOW, **kwargs)

    def blue(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_BLUE if bright else Color.BLUE, **kwargs)

    def magenta(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_MAGENTA if bright else Color.MAGENTA, **kwargs)

    def cyan(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_CYAN if bright else Color.CYAN, **kwargs)

    def white(self: Self, span: range, *, bright: bool = False, **kwargs: bool) -> None:
        self.colored(span, Color.BRIGHT_WHITE if bright else Color.WHITE, **kwargs)

    def make_ansi(self: Self) -> str:
        tags: list[tuple[int, Format | Reset]] = []

        for span, format in self.regions:
            tags.append((span.start, format))
            tags.append((span.stop, Reset(format)))

        result = ""
        last_pos = 0
        tag_stack: list[Format] = []

        for pos, tag in sorted(tags, key=lambda x: x[0]):
            result += self[last_pos:pos]
            last_pos = pos

            if isinstance(tag, Reset):
                tag_stack.remove(tag.format)

                attributes: list[int] = []
                if tag.format.foreground is not None:
                    # reset color to last used
                    for format in reversed(tag_stack):
                        if format.foreground is not None:
                            attributes.append(format.foreground.value)
                            break
                    else:
                        attributes.append(Color.RESET.value)
                if tag.format.bold is not None:
                    # reset bold to last used
                    for format in reversed(tag_stack):
                        if format.bold is not None:
                            attributes.append(1)
                            break
                    else:
                        attributes.append(22)
            else:
                tag_stack.append(tag)

                attributes = []
                if tag.foreground:
                    attributes.append(tag.foreground.value)
                if tag.bold is not None:
                    attributes.append(1)

            result += ANSI.srg(attributes)

        result += self[last_pos:]

        return result

    def __getitem__(self: Self, index: SupportsIndex | slice) -> str | SubStr:
        if not isinstance(index, slice) or index.step:
            return super().__getitem__(index)

        start, stop = index.start, index.stop
        if start is None:
            start = 0
        elif start < 0:
            start = len(self) + start

        if stop is None:
            stop = len(self)
        elif stop < 0:
            stop = len(self) + stop

        if start > stop:
            start, stop = stop, start

        return SubStr(super().__getitem__(index), self, range(start, stop))


class SubStr(str):
    __slots__ = ("parent", "span")

    parent: FormattedStr
    span: range

    def __new__(cls: type[Self], value: str, parent: FormattedStr, span: range) -> Self:
        self = super().__new__(cls, value)
        self.parent = parent
        self.span = span
        return self

    def colored(self: Self, color: Color, *, bold: bool | None = None) -> None:
        self.parent.colored(self.span, color, bold=bold)

    def black(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.black(self.span, bright=bright, **kwargs)

    def red(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.red(self.span, bright=bright, **kwargs)

    def green(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.green(self.span, bright=bright, **kwargs)

    def yellow(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.yellow(self.span, bright=bright, **kwargs)

    def blue(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.blue(self.span, bright=bright, **kwargs)

    def magenta(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.magenta(self.span, bright=bright, **kwargs)

    def cyan(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.cyan(self.span, bright=bright, **kwargs)

    def white(self: Self, *, bright: bool = False, **kwargs: bool) -> None:
        self.parent.white(self.span, bright=bright, **kwargs)


def srg(text: str, enable: list[int], reset: list[int]) -> str:
    return f"\x1b[{';'.join(map(str, enable))}m{text}\x1b[{';'.join(map(str, reset))}m"


def bold(text: str) -> str:
    return srg(text, [1], [22])


def colored(text: str, color: Color, *, bold: bool = False) -> str:
    enable, reset = [color.value], [39]
    if bold:
        enable.append(1)
        reset.append(22)
    return srg(text, enable, reset)


def red(text: str, **kwargs: bool) -> str:
    return colored(text, Color.RED, **kwargs)


def green(text: str, **kwargs: bool) -> str:
    return colored(text, Color.GREEN, **kwargs)


def blue(text: str, **kwargs: bool) -> str:
    return colored(text, Color.BLUE, **kwargs)


def bright_cyan(text: str, **kwargs: bool) -> str:
    return colored(text, Color.BRIGHT_CYAN, **kwargs)


@dataclass(frozen=True, slots=True)
class Annotation:
    span: Position | Span
    color: Color
    indicator: str = "^"
    comment: str | None = None


@dataclass(frozen=True, slots=True)
class Replacement:
    comment: str
    partials: list[PartialReplacement]


@dataclass(frozen=True, slots=True)
class PartialReplacement:
    span: Span
    text: str


class Logger:
    level: str
    color: Color

    __slots__ = ("msg", "pos", "_detail", "hints", "annotations", "replacements")

    msg: str
    pos: Position | Span | None
    _detail: str | None
    hints: list[str]
    annotations: dict[SourceFile, dict[int, list[Annotation]]]
    replacements: list[Replacement]

    def __init__(self: Self, msg: str) -> None:
        self.msg = msg
        self.pos = None
        self._detail = None
        self.hints = []
        self.annotations = {}
        self.replacements = []

    def at(self: Self, pos: Position | Span, *, msg: str | None = None) -> Self:
        self.pos = pos
        self._add_annotation(Annotation(pos, self.color, comment=msg))
        return self

    def detail(self: Self, detail: str) -> Self:
        self._detail = detail
        return self

    def hint(self: Self, msg: str) -> Self:
        self.hints.append(msg)
        return self

    def comment(self: Self, span: Span, msg: str) -> Self:
        self._add_annotation(Annotation(span, Color.BLUE, comment="info: " + msg))
        return self

    def replacement(self: Self, *args: tuple[Span, str], msg: str) -> Self:
        self.replacements.append(Replacement(msg, [PartialReplacement(pos, span) for pos, span in args]))
        return self

    def _add_annotation(self: Self, annotation: Annotation) -> None:
        if isinstance(annotation.span, Span) and annotation.span.start.line != annotation.span.end.line:
            pos = annotation.span.start
            while True:
                end = pos.line_end
                self._add_annotation(dataclasses.replace(annotation, span=Span.from_(pos, end), comment=None))
                pos = end + 1
                if pos.line == annotation.span.end.line:
                    break

            self._add_annotation(dataclasses.replace(annotation, span=Span.from_(pos, annotation.span.end)))
            return

        file = annotation.span.file
        line = annotation.span.line if isinstance(annotation.span, Position) else annotation.span.start.line
        if file not in self.annotations:
            self.annotations[file] = {}

        if line not in self.annotations[file]:
            self.annotations[file][line] = []

        self.annotations[file][line].append(annotation)

    def log(self: Self) -> None:
        result = f"{colored(self.level, self.color)}: {self.msg}"
        if self._detail:
            result += f": {self._detail}"
        result = bold(result)

        if self.pos is not None:
            start_pos = self.pos.start if isinstance(self.pos, Span) else self.pos
            result += f"\n   {blue('-->', bold=True)} {self.pos.file.path}:{start_pos}"
            result += f"\n    {blue('|', bold=True)}"
            result += self._get_source_display_for_file(self.annotations[self.pos.file])

        result += self._get_source_display()

        for hint in self.hints:
            result += f"\n{bright_cyan('hint', bold=True)}: {hint}"

        for replacement in self.replacements:
            result += self._get_replacement_display(replacement)

        result += "\n"

        print(result)

    def _get_source_display(self: Self) -> str:
        result = ""

        for file, annotations in self.annotations.items():
            if self.pos and file == self.pos.file:
                continue

            result += f"\n   {blue('-->', bold=True)} {file.path}"
            result += f"\n    {blue('|', bold=True)}"
            result += self._get_source_display_for_file(annotations)

        return result

    def _get_source_display_for_file(self: Self, annotations: dict[int, list[Annotation]]) -> str:
        start_line = min(annotations)
        end_line = max(annotations)

        result = ""
        lines_str = self._get_lines(annotations[start_line][0].span.file.content, (start_line, end_line))

        last_line = start_line
        for line in range(start_line, end_line + 1):
            if line not in annotations:
                continue

            if abs(last_line - line) > 1:
                result += f"\n    {blue('|', bold=True)}"
            last_line = line

            line_str, *sub_lines_str = self._annotate(lines_str[line - start_line], annotations[line])
            result += f"\n{blue(f'{line:<4}|', bold=True)} {line_str}"
            for sub_line_str in sub_lines_str:
                result += f"\n    {blue('|', bold=True)} {sub_line_str}"

        return result

    def _get_replacement_display(self: Self, replacement: Replacement) -> str:
        def shift_part(partials: list[PartialReplacement], index: int, amount: int) -> list[PartialReplacement]:
            return [
                dataclasses.replace(partial, span=self._shift_span(index, amount, partial.span)) for partial in partials
            ]

        result = f"\n{bright_cyan('hint', bold=True)}: {replacement.comment}"
        result += f"\n    {blue('|', bold=True)}"

        full_span = sum([partial.span for partial in replacement.partials], replacement.partials[0].span)
        full_span = Span.from_(full_span.start.line_start, full_span.end.line_end)
        partials = shift_part(replacement.partials, 0, -full_span.start + 1)

        lines = full_span.extract().split("\n")

        original_text_str = ""
        for line in range(full_span.start.line, full_span.end.line + 1):
            partials = shift_part(partials, len(original_text_str), 6)
            original_text_str += f"\n{line:<4}- {lines[line - full_span.start.line]}"

        index = 0
        original_text = FormattedStr(original_text_str)
        for line_str in original_text.split("\n"):
            original_text[index : index + 4].blue(bold=True)
            original_text[index + 4 : index + 5].red()
            index += len(line_str) + 1

        for partial in partials:
            original_text[partial.span.start : partial.span.end].red()

        patched_text_str = ""
        for line in range(full_span.start.line, full_span.end.line + 1):
            patched_text_str += f"\n{line:<4}+ {lines[line - full_span.start.line]}"

        for i in range(len(partials)):
            partial = partials[i]

            before = patched_text_str[: partial.span.start]
            after = patched_text_str[partial.span.end :]

            patched_text_str = before + partial.text + after

            del partials[i]
            displacement = len(partial.text) - len(partial.span)
            partials = shift_part(partials, partial.span.start, displacement)
            partials.insert(
                i,
                dataclasses.replace(partial, span=Span.from_(partial.span.start, partial.span.end + displacement)),
            )

        index = 0
        patched_text = FormattedStr(patched_text_str)
        for line_str in patched_text.split("\n"):
            patched_text[index : index + 4].blue(bold=True)
            patched_text[index + 4 : index + 5].green()
            index += len(line_str) + 1

        for partial in partials:
            patched_text[partial.span.start : partial.span.end].green()

        result += original_text.make_ansi()
        result += patched_text.make_ansi()
        result += f"\n    {blue('|', bold=True)}"

        return result

    @staticmethod
    def _shift_span(index: int, amount: int, span: Span) -> Span:
        start = span.start + amount if span.start >= index else span.start
        end = span.end + amount if span.end >= index else span.end
        return Span.from_(start, end)

    @staticmethod
    def _get_lines(source: str, lines: tuple[int, int]) -> list[str]:
        source_lines = source.splitlines()
        selected_lines = [line.replace("\t", " ") for line in source_lines[lines[0] - 1 : lines[1]]]
        selected_lines += ["<line not found>"] * (lines[1] - lines[0] + 1 - len(selected_lines))
        return selected_lines

    @staticmethod
    def _annotate(line: str, annotations: list[Annotation]) -> list[str]:
        annotations.sort(key=lambda x: x.span if isinstance(x.span, Position) else x.span.start)
        indicators = ""
        last_pos = 1
        for annotation in annotations:
            if isinstance(annotation.span, Position):
                start_column = annotation.span.column
                end_column = start_column + 1
            else:
                start_column = annotation.span.start.column
                end_column = annotation.span.end.column

            indicators += " " * (start_column - last_pos)
            indicators += colored(annotation.indicator, annotation.color, bold=True) * (end_column - start_column)
            last_pos = end_column

            if annotation is annotations[-1] and annotation.comment is not None:
                indicators += " " + colored(annotation.comment, annotation.color, bold=True)

        lines = [line, indicators]

        if len(annotations) > 1:
            line = ""
            last_pos = 1
            for ann in annotations[:-1]:
                if ann.comment is None:
                    continue

                column = ann.span.column if isinstance(ann.span, Position) else ann.span.start.column
                line += " " * (column - last_pos)
                line += colored("|", ann.color, bold=True)
                last_pos = column + 1
            lines.append(line)

            for i, annotation in reversed(list(enumerate(annotations[:-1]))):
                if annotation.comment is None:
                    continue

                line = ""
                last_pos = 1
                for ann in annotations[: i + 1]:
                    if ann.comment is None:
                        continue

                    column = ann.span.column if isinstance(ann.span, Position) else ann.span.start.column
                    line += " " * (column - last_pos)
                    line += colored("|", ann.color, bold=True)
                    last_pos = column + 1
                lines.append(line[:-16] + colored(annotation.comment, annotation.color, bold=True))

        return lines


class Error(Logger):
    level = "error"
    color = Color.RED


class Warn(Logger):
    level = "warning"
    color = Color.BRIGHT_YELLOW
