from dummy import foo


class TestDummy:

    def test_foo(self):
        assert foo("a", 3) == "aaa"

    def test_fail(self):
        assert foo("b", 2) == "b"
