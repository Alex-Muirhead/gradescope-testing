import inspect
from dataclasses import dataclass, field
from inspect import Signature, BoundArguments
from typing import TypeVar, Generic, Callable, Optional, cast
from typing import ParamSpec, ParamSpecArgs, ParamSpecKwargs


def foo(x: int, y: str) -> str:
    return x * y


def bar(x: int, y: str) -> str:
    return (x-1) * (y+"!")


T = TypeVar('T')
P = ParamSpec('P')


class TestGroup(Generic[P, T]):

    func: Callable[P, T]
    func_signature: Signature

    cases: list['TestCase[P, T]']

    def __init__(self, /, function: Callable[P, T]):
        self.func = function
        self.func_signature = inspect.signature(function)
        self.cases = []

    def add_case(self, *args: P.args, **kwargs: P.kwargs) -> 'TestCase[P, T]':
        test_case = TestCase[P, T](self.func_signature, *args, **kwargs)
        self.cases.append(test_case)
        return test_case

    def call(self):
        ...


@dataclass
class TestCase(Generic[P, T]):

    _signature: Signature = field(repr=False)
    params: BoundArguments

    def __init__(self, signature: Signature, *args: P.args, **kwargs: P.kwargs):
        self._signature = signature
        self.with_params(*args, **kwargs)

    def with_params(self, *args: P.args, **kwargs: P.kwargs) -> 'TestCase[P, T]':
        self.params = self._signature.bind(*args, **kwargs)
        return self

    def exec(self, func: Callable[P, T]) -> T:
        # FIXME: Safe by type system? Maybe should have a runtime check just in case...
        return func(*self.params.args, **self.params.kwargs)  # pyright: ignore


group = TestGroup(function=foo)
thing = group.add_case(5, "2")
print(thing)
print(group.cases)
print(thing.exec(bar))
