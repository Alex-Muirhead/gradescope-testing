import pytest


def foo(x: str, y: int) -> str:
    return x * y


def bar():
    name = input("What is your name?")
    print("This is some output!")
    return "Hello " + name


class TestDummy:

    @staticmethod
    @pytest.mark.parametrize("test_input", [(3, "aaa"), (0, "a"), (4, "aaaa")])
    def test_param(test_input):
        y, expected = test_input
        assert foo("a", y) == expected

    def test_foo(self):
        assert foo("a", 3) == "aaa"

    def test_fail(self):
        assert foo("b", 2) == "b"


def test_alone():
    assert foo("c", 3) == "cc"


def test_input(sysin):
    sysin.write("Test\n")
    sysin.seek(0)
    assert bar() == "Hello Test"
    assert 0
