from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable
from typing import Any as TAny

from .errors import InternalCompilerError

if TYPE_CHECKING:
    from typing_extensions import Self

    from .parser import Binding
    from .source import Span


class Type(ABC):
    span: Span | None = None

    @property
    @abstractmethod
    def name(self: Self) -> str:
        """The name of the type (displayed in error messages)."""
        ...

    def __init__(self: Self, span: Span | None = None) -> None:
        self.span = span

    def _always(self: Self, _other: Type, /) -> Self:
        return self

    @staticmethod
    def _never(name: str, /) -> Callable[[Type, Type], Type | None]:
        def wrapper(self: Self, other: Type, /) -> Type | None:
            return getattr(other, "r" + name)(self)  # type: ignore[no-any-return] # don't worry about this

        return wrapper

    def _rnever(self: Self, _other: Type, /) -> None:
        return None

    @staticmethod
    def _self(name: str, /) -> Callable[[Type, Type], Type | None]:
        def wrapper(self: Self, other: Type, /) -> Type | None:
            if self == other:
                return self
            return getattr(other, "r" + name)(self)  # type: ignore[no-any-return] # don't worry about this

        return wrapper

    def _rself(self: Self, other: Type, /) -> Type | None:
        if self == other:
            return self
        return None

    @staticmethod
    def _self_bool(name: str, /) -> Callable[[Type, Type], Type | None]:
        def wrapper(self: Self, other: Type, /) -> Type | None:
            if self == other:
                return Bool()
            return getattr(other, "r" + name)(self)  # type: ignore[no-any-return] # don't worry about this

        return wrapper

    def _rself_bool(self: Self, other: Type, /) -> Bool | None:
        if self == other:
            return Bool()
        return None

    @abstractmethod
    def pos(self: Self, /) -> Type | None:
        """The return type of `+self`."""
        ...

    @abstractmethod
    def neg(self: Self, /) -> Type | None:
        """The return type of `-self`."""
        ...

    @abstractmethod
    def add(self: Self, other: Type, /) -> Type | None:
        """The return type of `self + other`."""
        ...

    @abstractmethod
    def radd(self: Self, other: Type, /) -> Type | None:
        """The return type of `other + self`, used if `add` returned None."""
        ...

    @abstractmethod
    def sub(self: Self, other: Type, /) -> Type | None:
        """The return type of `self - other`."""
        ...

    @abstractmethod
    def rsub(self: Self, other: Type, /) -> Type | None:
        """The return type of `other - self`, used if `sub` returned None."""
        ...

    @abstractmethod
    def mul(self: Self, other: Type, /) -> Type | None:
        """The return type of `self * other`."""
        ...

    @abstractmethod
    def rmul(self: Self, other: Type, /) -> Type | None:
        """The return type of `other * self`, used if `mul` returned None."""
        ...

    @abstractmethod
    def div(self: Self, other: Type, /) -> Type | None:
        """The return type of `self / other`."""
        ...

    @abstractmethod
    def rdiv(self: Self, other: Type, /) -> Type | None:
        """The return type of `other / self`, used if `div` returned None."""
        ...

    @abstractmethod
    def mod(self: Self, other: Type, /) -> Type | None:
        """The return type of `self % other`."""
        ...

    @abstractmethod
    def rmod(self: Self, other: Type, /) -> Type | None:
        """The return type of `other % self`, used if `mod` returned None."""
        ...

    @abstractmethod
    def pow(self: Self, other: Type, /) -> Type | None:
        """The return type of `self ^ other`."""
        ...

    @abstractmethod
    def rpow(self: Self, other: Type, /) -> Type | None:
        """The return type of `other ^ self`, used if `pow` returned None."""
        ...

    @abstractmethod
    def eq(self: Self, other: Type, /) -> Type | None:
        """The return type of `self == other` and `self != other`."""
        ...

    @abstractmethod
    def req(self: Self, other: Type, /) -> Type | None:
        """The return type of `other == self` and `other != self`, used if `eq` returned None."""
        ...

    @abstractmethod
    def order(self: Self, other: Type, /) -> Type | None:
        """The return type of `self (> | < | <= | >=) other`."""
        ...

    @abstractmethod
    def rorder(self: Self, other: Type, /) -> Type | None:
        """The return type of `other (> | < | <= | >=) self`, used if `order` returned None."""
        ...

    @abstractmethod
    def assign(self: Self, other: Type, /) -> Type | None:
        """The return type of `self = other`."""
        ...

    @abstractmethod
    def rassign(self: Self, other: Type, /) -> Type | None:
        """The return type of `other = self`, used if `assign` returned None."""
        ...

    def __eq__(self: Self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return True


class Any(Type):
    name = "<inconnu>"

    def pos(self: Self, /) -> Self:
        return self

    def neg(self: Self, /) -> Self:
        return self

    add = Type._always
    radd = Type._always
    sub = Type._always
    rsub = Type._always
    mul = Type._always
    rmul = Type._always
    div = Type._always
    rdiv = Type._always
    mod = Type._always
    rmod = Type._always
    pow = Type._always
    rpow = Type._always
    eq = Type._always
    req = Type._always
    order = Type._always
    rorder = Type._always
    assign = Type._always
    rassign = Type._always


class Void(Type):
    name = "<rien>"

    def pos(self: Self, /) -> None:
        return None

    def neg(self: Self, /) -> None:
        return None

    add = Type._never("add")
    radd = Type._rnever
    sub = Type._never("sub")
    rsub = Type._rnever
    mul = Type._never("mul")
    rmul = Type._rnever
    div = Type._never("div")
    rdiv = Type._rnever
    mod = Type._never("mod")
    rmod = Type._rnever
    pow = Type._never("pow")
    rpow = Type._rnever
    eq = Type._never("eq")
    req = Type._rnever
    order = Type._never("order")
    rorder = Type._rnever
    assign = Type._never("assign")
    rassign = Type._rnever


class Bool(Type):
    name = "booleen"

    def pos(self: Self, /) -> None:
        return None

    def neg(self: Self, /) -> None:
        return None

    add = Type._never("add")
    radd = Type._rnever
    sub = Type._never("sub")
    rsub = Type._rnever
    mul = Type._never("mul")
    rmul = Type._rnever
    div = Type._never("div")
    rdiv = Type._rnever
    mod = Type._never("mod")
    rmod = Type._rnever
    pow = Type._never("pow")
    rpow = Type._rnever
    eq = Type._self("eq")
    req = Type._rself
    order = Type._never("order")
    rorder = Type._rnever
    assign = Type._self("assign")
    rassign = Type._rself


class Integer(Type):
    name = "entier"

    def pos(self: Self, /) -> Self:
        return self

    def neg(self: Self, /) -> Self:
        return self

    add = Type._self("add")
    radd = Type._rself
    sub = Type._self("sub")
    rsub = Type._rself
    mul = Type._self("mul")
    rmul = Type._rself
    div = Type._self("div")
    rdiv = Type._rself
    mod = Type._self("mod")
    rmod = Type._rself
    pow = Type._self("pow")
    rpow = Type._rself
    eq = Type._self_bool("eq")
    req = Type._rself_bool
    order = Type._self_bool("order")
    rorder = Type._rself_bool
    assign = Type._self("assign")
    rassign = Type._rself


class Float(Type):
    name = "reel"

    def pos(self: Self, /) -> Self:
        return self

    def neg(self: Self, /) -> Self:
        return self

    add = Type._self("add")
    radd = Type._rself
    sub = Type._self("sub")
    rsub = Type._rself
    mul = Type._self("mul")
    rmul = Type._rself
    div = Type._self("div")
    rdiv = Type._rself
    mod = Type._self("mod")
    rmod = Type._rself
    pow = Type._self("pow")
    rpow = Type._rself
    eq = Type._self_bool("eq")
    req = Type._rself_bool
    order = Type._self_bool("order")
    rorder = Type._rself_bool
    assign = Type._self("assign")
    rassign = Type._rself


class Char(Type):
    name = "car"

    def pos(self: Self, /) -> None:
        return None

    def neg(self: Self, /) -> None:
        return None

    def add(self: Self, other: Type, /) -> Type | None:
        if isinstance(other, (Char, Integer)):
            return self
        return other.radd(self)

    def radd(self: Self, other: Type, /) -> Type | None:
        if isinstance(other, (Char, Integer)):
            return self
        return None

    def sub(self: Self, other: Type, /) -> Type | None:
        if isinstance(other, (Char, Integer)):
            return self
        return other.rsub(self)

    def rsub(self: Self, other: Type, /) -> Type | None:
        if isinstance(other, (Char, Integer)):
            return self
        return None

    mul = Type._never("mul")
    rmul = Type._rnever
    div = Type._never("div")
    rdiv = Type._rnever
    mod = Type._never("mod")
    rmod = Type._rnever
    pow = Type._never("pow")
    rpow = Type._rnever
    eq = Type._self_bool("eq")
    req = Type._rself_bool
    order = Type._self_bool("order")
    rorder = Type._rself_bool
    assign = Type._self("assign")
    rassign = Type._rself


class String(Type):
    name = "chaine"

    def pos(self: Self, /) -> None:
        return None

    def neg(self: Self, /) -> None:
        return None

    add = Type._self("add")
    radd = Type._rself
    sub = Type._never("sub")
    rsub = Type._rnever

    def mul(self: Self, other: Type, /) -> Type | None:
        if isinstance(other, Integer):
            return self
        return other.rmul(self)

    def rmul(self: Self, other: Type, /) -> Type | None:
        if isinstance(other, Integer):
            return self
        return None

    div = Type._never("div")
    rdiv = Type._rnever
    mod = Type._never("mod")
    rmod = Type._rnever
    pow = Type._never("pow")
    rpow = Type._rnever
    eq = Type._self_bool("eq")
    req = Type._rself_bool
    order = Type._self_bool("order")
    rorder = Type._rself_bool
    assign = Type._self("assign")
    rassign = Type._rself


class List(Type):
    @classmethod
    def from_(cls: type[Self], binding: Binding, typ_: Type) -> Type:
        from .parser import Indexing, Variable

        if isinstance(binding, Variable):
            return typ_
        elif isinstance(binding, Indexing):
            return cls(cls.from_(binding.sub, typ_), binding.index, typ_.span)
        else:
            msg = f"Unsupported binding: {binding}"
            raise InternalCompilerError(msg)

    subtype: Type
    length: TAny

    @property
    def name(self: Self) -> str:
        lenght = self.length.compute()
        return f"liste[{self.subtype.name}, {lenght.value if isinstance(lenght.typ, Integer) else '???'}]"

    def __init__(self: Self, subtype: Type, length: TAny, span: Span | None = None) -> None:
        super().__init__(span)
        self.subtype = subtype
        self.length = length

    def pos(self: Self, /) -> Type | None:
        return None

    def neg(self: Self, /) -> Type | None:
        return None

    add = Type._never("add")
    radd = Type._rnever
    sub = Type._never("sub")
    rsub = Type._rnever
    mul = Type._never("mul")
    rmul = Type._rnever
    div = Type._never("div")
    rdiv = Type._rnever
    mod = Type._never("mod")
    rmod = Type._rnever
    pow = Type._never("pow")
    rpow = Type._rnever
    eq = Type._self_bool("eq")
    req = Type._rself_bool
    order = Type._never("order")
    rorder = Type._rnever
    assign = Type._self("assign")
    rassign = Type._rself

    def __eq__(self: Self, other: object) -> bool:
        if not isinstance(other, List):
            return False

        self_len = self.length.compute()
        if not isinstance(self_len.typ, Integer):
            return True

        other_len = other.length.compute()
        if not isinstance(other_len.typ, Integer):
            return True

        return self.subtype.assign(other.subtype) is not None and self_len.value == other_len.value


class Structure(Type):
    name: str = ""

    def __init__(self: Self, name: str, span: Span | None = None) -> None:
        super().__init__(span)
        self.name = name

    def pos(self: Self, /) -> Type | None:
        return None

    def neg(self: Self, /) -> Type | None:
        return None

    add = Type._never("add")
    radd = Type._rnever
    sub = Type._never("sub")
    rsub = Type._rnever
    mul = Type._never("mul")
    rmul = Type._rnever
    div = Type._never("div")
    rdiv = Type._rnever
    mod = Type._never("mod")
    rmod = Type._rnever
    pow = Type._never("pow")
    rpow = Type._rnever
    eq = Type._never("eq")
    req = Type._rnever
    order = Type._never("order")
    rorder = Type._rnever
    assign = Type._self("assign")
    rassign = Type._rself

    def __eq__(self: Self, other: object) -> bool:
        if not isinstance(other, Structure):
            return False

        return self.name == other.name
