from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TextIO, TypeVar, overload

if TYPE_CHECKING:
    from typing_extensions import Self


class UnsupportedFileTypeError(Exception):
    pass


class SourceFile:
    """A file containing pseudo-code. It can contain several sections of code (in a Markdown file for example)."""

    __slots__ = ("path", "content", "sections")
    CODE_BLOCK_REGEX = re.compile(r"(#+ (?P<name>(\w| )+)\n+)?```(?P<language>\w+)?\n(?P<code>.+?)\n```", re.DOTALL)

    path: Path
    content: str
    sections: list[FileSection]

    @overload
    def __init__(self: Self, file: TextIO) -> None:
        ...

    @overload
    def __init__(self: Self, file: str, *, path: str) -> None:
        ...

    def __init__(self: Self, file: TextIO | str, *, path: str | None = None) -> None:
        if isinstance(file, str):
            self.path = Path(path or "<unknown>")
            self.content = file
        else:
            self.path = Path(file.name)
            with file:
                self.content = file.read()

        self._read_sections()

    def _read_sections(self: Self) -> None:
        sections = []
        match self.path.suffix:
            case ".pseudo" | ".txt":
                sections.append(FileSection(self, Span(self, 0, len(self.content)), self.content))
            case "" if self.path.name.startswith("<") and self.path.name.endswith(">"):
                sections.append(FileSection(self, Span(self, 0, len(self.content)), self.content, name="out"))
            case ".md":
                index = 0

                for match in self.CODE_BLOCK_REGEX.finditer(self.content):
                    if match.group("language") not in (None, "pseudo"):
                        continue

                    sections.append(
                        FileSection(
                            self,
                            Span(self, *match.span("code")),
                            match.group("code"),
                            name=match.group("name") or str(index),
                        ),
                    )
                    index += 1

            case suffix:
                raise UnsupportedFileTypeError(suffix)

        self.sections = sections


class FileSection:
    __slots__ = ("file", "name", "span", "code")
    SUFFIX = ".pseudo.py"

    file: SourceFile
    name: str | None
    span: Span
    code: str

    def __init__(self: Self, source: SourceFile, span: Span, code: str, *, name: str | None = None) -> None:
        self.file = source
        self.name = name
        self.span = span
        self.code = code

    @property
    def out_path(self: Self) -> Path:
        if self.file.path.name == "<stdin>":
            return self.file.path.with_name(f"{self.name}{self.SUFFIX}")

        if self.name:
            return self.file.path.with_name(f"{self.file.path.stem} - {self.name}{self.SUFFIX}")

        return self.file.path.with_suffix(self.SUFFIX)


class Position(int):
    file: SourceFile
    _line: int | None
    _column: int | None

    def __new__(cls: type[Self], file: SourceFile, value: int) -> Self:
        self = int.__new__(cls, value)
        self.file = file
        self._line = self._column = None

        return self

    @property
    def line(self: Self) -> int:
        if self._line is None:
            self._line = self.file.content.count("\n", 0, self) + 1
        return self._line

    @property
    def column(self: Self) -> int:
        if self._column is None:
            self._column = (int(self) - self.file.content.rfind("\n", 0, self)) if self.line != 1 else (int(self) + 1)
        return self._column

    @property
    def line_start(self: Self) -> Self:
        index = self.file.content.rfind("\n", 0, self)
        if index == -1:
            index = 0
        return type(self)(self.file, index + 1)

    @property
    def line_end(self: Self) -> Self:
        index = self.file.content.find("\n", self)
        if index == -1:
            index = len(self.file.content)
        return type(self)(self.file, index)

    def __add__(self: Self, other: object) -> Self:
        if isinstance(other, int):
            return type(self)(self.file, int(self) + other)
        return NotImplemented

    def __sub__(self: Self, other: object) -> Self:
        if isinstance(other, int):
            return type(self)(self.file, int(self) - other)
        return NotImplemented

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}(index={int(self)} line={self.line} column={self.column})"

    def __str__(self: Self) -> str:
        return f"{self.line}:{self.column}"


T = TypeVar("T")


class Span:
    __slots__ = ("file", "start", "end")

    file: SourceFile
    start: Position
    end: Position

    @classmethod
    @overload
    def at(cls: type[Self], position: Position, /) -> Self:
        ...

    @classmethod
    @overload
    def at(cls: type[Self], file: SourceFile, position: int, /) -> Self:
        ...

    @classmethod
    def at(cls: type[Self], file_or_pos: SourceFile | Position, position: int | None = None, /) -> Self:
        if isinstance(file_or_pos, SourceFile):
            if position is None:
                msg = "[COMPILER BUG] Position must not be null"
                raise ValueError(msg)
            return cls(file_or_pos, position, position)
        else:
            return cls(file_or_pos.file, file_or_pos, file_or_pos)

    @classmethod
    def from_(cls: type[Self], start: Position, end: Position) -> Self:
        if start.file is not end.file:
            msg = "[COMPILER BUG] File of start and end span must be equal"
            raise ValueError(msg)
        return cls(start.file, start, end)

    def __init__(self: Self, file: SourceFile, start: int, end: int) -> None:
        self.file = file
        self.start = Position(self.file, start)
        self.end = Position(self.file, end)

    @property
    def lines(self: Self) -> range:
        return range(self.start.line, self.end.line + 1)

    def extract(self: Self) -> str:
        return self.file.content[self.start : self.end]

    def wrap(self: Self, value: T) -> Spanned[T]:
        return Spanned(value, span=self)

    def __add__(self: Self, other: object) -> Self:
        if isinstance(other, Span):
            return type(self)(self.file, min(self.start, other.start), max(self.end, other.end))
        return NotImplemented

    def __len__(self: Self) -> int:
        return int(self.end - self.start)

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}(start={self.start!r} end={self.end!r})"

    def __str__(self: Self) -> str:
        return f"from {self.start} to {self.end}"


class Spanned(Generic[T]):
    __slots__ = ("value", "span")

    value: T
    span: Span

    def __init__(self: Self, value: T, *, span: Span) -> None:
        self.value = value
        self.span = span

    def __repr__(self: Self) -> str:
        return f"{type(self).__name__}({self.value!r}, span={self.span!r})"


class SpannedStr(str):
    __slots__ = ("span",)

    span: Span

    def __new__(cls: type[Self], value: str, *, span: Span) -> Self:
        self = super().__new__(cls, value)
        self.span = span
        return self
