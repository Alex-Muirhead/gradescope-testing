"""
Tests for ENGG1001 Assignment 1 Sem2 2022
"""

from functools import partial  # , wraps
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar, ParamSpec, Generic

# from testrunner import AttributeGuesser, OrderedTestCase, RedirectStdIO, TestMaster, skipIfFailed
# from test_arrays import assert_allclose
import pytest
from pytest import approx

import a1_support
import a1


class A1:
    """ Just used for type hints """

    parameter_set = tuple[
        tuple[float, float, float],
        float, float, float, float, float, float,
        tuple[float, ...]
    ]

    @staticmethod
    def determine_hub_speed(
        v_meas: float,
        h_meas: float,
        h_hub: float,
        alpha: float
    ) -> float:
        ...

    @staticmethod
    def determine_windpower(v_hub: float, r: float, rho: float) -> float:
        ...

    @staticmethod
    def determine_mech_coeff(
        v_hub: float,
        radius: float,
        omega: float,
        coefs: tuple[float, ...]
    ) -> float:
        ...

    @staticmethod
    def determine_mech_power(
        v_hub: float,
        speeds: float,
        radius: float,
        omega: float,
        rho: float,
        coefs: tuple[float, ...]
    ) -> float:
        ...

    @staticmethod
    def determine_revenue(
        speeds: float,
        h_meas: float,
        h_hub: float,
        alpha: float,
        r: float,
        omega: float,
        rho: float,
        coeffs: tuple[float, ...]
    ) -> tuple[float, float]:
        ...

    @staticmethod
    def print_table(parameters: tuple[parameter_set, ...], separator: str) -> None:
        ...

    @staticmethod
    def print_table_general(
        parameters: tuple[parameter_set, ...],
        number_columns: int,
        separator: str
    ) -> None:
        ...

    @staticmethod
    def main() -> None:
        ...


T = TypeVar('T')
P = ParamSpec('P')


@dataclass
class TestCase(Generic[P, T]):
    __test__ = False
    args: tuple
    kwargs: dict
    output: None

    def __init__(self):
        self.args = ()
        self.kwargs = {}
        self.output = None

    def with_params(self, *args: P.args, **kwargs: P.kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def with_output(self, value: T):
        self.output = value
        return self

    def call(self, func: Callable[P, T]) -> T:
        return func(*self.args, **self.kwargs)

    @staticmethod
    def from_func(func: Callable[P, T]) -> "type[TestCase[P, T]]":
        return TestCase[P, T]


class TestA1(object):
    a1: A1 = a1
    a1_support: ...


# class TestDesign(TestA1):
#     """ Checks A1 design compliance """

#     def test_clean_import(self):
#         """ test no prints on import """
#         self.assertIsCleanImport(self.a1, msg="You should not be printing on import for a1.py")

#     def test_functions_defined(self):
#         """ test all functions are defined correctly """
#         a1 = AttributeGuesser.get_wrapped_object(self.a1)
#         for func_name, func in inspect.getmembers(A1, predicate=inspect.isfunction):
#             if func_name != "bonus":
#                 num_params = len(inspect.signature(func).parameters)
#                 self.aggregate(self.assertFunctionDefined, a1, func_name, num_params, tag=func_name)

#         self.aggregate_tests()

#     def test_doc_strings(self):
#         """ test all functions have documentation strings """
#         a1 = AttributeGuesser.get_wrapped_object(self.a1)
#         for attr_name, attr in inspect.getmembers(a1, predicate=inspect.isfunction):
#             self.aggregate(self.assertDocString, a1, attr_name)

#         self.aggregate_tests()


class TestFunctionality(TestA1):
    """ Base for all A1 functionality tests """

    TEST_DATA = (Path(__file__).parent / 'test_data').resolve()

    def load_test_data(self, filename: str):
        """ load test data from file """
        with open(self.TEST_DATA / filename, encoding='utf8') as file:
            return file.read()

    def write_test_data(self, filename: str, output: str):
        """ write test data to file """
        with open(self.TEST_DATA / filename, 'w', encoding='utf8') as file:
            file.write(output)


# skipIfNotDefined = partial(
#     skipIfFailed,
#     TestDesign,
#     TestDesign.test_functions_defined.__name__,
# )


# @skipIfNotDefined("determine_hub_speed")
class TestDetermineHubSpeed(TestFunctionality):
    """ test determine_hub_speed"""

    @classmethod
    def setup_class(cls):
        if not hasattr(cls.a1, "determine_hub_speed"):
            pytest.skip("Method not defined!")
        cls.parameters = a1_support.load_data("test_data", "task_1_parameters")
        cls.answers = a1_support.load_data("test_data", "task_1_answers")

    def test_water_pumped_0(self):
        """ test task sheet example """
        expected = self.answers[0]
        actual = self.a1.determine_hub_speed(*self.parameters[0])
        assert actual == pytest.approx(expected)

    def test_water_pumped_1(self):
        """ test alternative generator power """
        expected = self.answers[1]
        actual = self.a1.determine_hub_speed(*self.parameters[1])
        assert actual == pytest.approx(expected)

    def test_water_pumped_2(self):
        """ test alternative pumping time """
        expected = self.answers[2]
        actual = self.a1.determine_hub_speed(*self.parameters[2])
        assert actual == pytest.approx(expected)

    def test_water_pumped_3(self):
        """ test alternative efficiency """
        expected = self.answers[3]
        actual = self.a1.determine_hub_speed(*self.parameters[3])
        assert actual == pytest.approx(expected)


WindpowerTestCase = TestCase.from_func(A1.determine_windpower)

cases = (
    WindpowerTestCase().with_params(11.566, 40, 1.225).with_output(4763.49279),
    WindpowerTestCase().with_params(10, 1.0, 1.0).with_output(1.57079e+00),
    WindpowerTestCase().with_params(-10, 1.0, 1.0).with_output(-1.57079e+00),
    WindpowerTestCase().with_params(0, 1.0, 1.0).with_output(0.0e+0),
    WindpowerTestCase().with_params(10, 0.0, 1.0).with_output(0.0e+0),
    WindpowerTestCase().with_params(1_000_000.0, 1.0, 1.0).with_output(1.57079e+15),
    WindpowerTestCase().with_params(5, 4.0, 0.31831).with_output(1.0e+0),
)


@pytest.mark.parametrize("case", cases)
def test_determine_windpower(case):
    """ test task sheet example """
    actual = case.call(a1.determine_windpower)
    assert actual == approx(case.output, abs=1e-04, rel=1e-04)


def add_line_numbers(table):
    """
    Add line numbers to the table to help identify which line is which
    Parameters:
        table (string): The table to add line numbers to
    Returns:
        (string): the original string, but with "line x:" added to the start
            of each line, where x is the line number
    """
    table_with_line_numbers = ""
    lines = table.splitlines()
    for line_num, line in enumerate(lines):
        table_with_line_numbers += f"line {line_num+1}: {line}\n"
    return table_with_line_numbers


class TableComparison:
    """
    Class to assist with comparing if two tables are the same
    """

    def check_strings_almost_equal(self, reference_str, string):
        """
        Assert that two strings are equal to within a tolerance
        """
        # to make a start, just require an extact match
        return reference_str == string

    def assert_table_almost_equal(self, ref_table, table):
        """
        Assert that two tables are almost equal, in the sense
        that each row of the table is almost equal according to
        `assert_strings_almost_equal`.
        """
        __tracebackhide__ = True
        expected_lines = ref_table.splitlines()
        actual_lines = table.splitlines()

        # first check the table has the correct number of lines
        num_expected_lines = len(expected_lines)
        num_actual_lines = len(actual_lines)
        if num_expected_lines != num_actual_lines:
            raise AssertionError(
                f"Your table has {num_actual_lines} lines, "
                f"but {num_expected_lines} lines were expected.\n"
                f"Your table was:\n{add_line_numbers(table)}\n"
                f"The expected table was:\n{add_line_numbers(ref_table)}\n"
            )

        # now we'll check the table is the same
        for line_num, (expected, actual) in enumerate(zip(expected_lines, actual_lines)):
            equal = self.check_strings_almost_equal(expected, actual)
            if not equal:
                raise AssertionError(
                    f"Your table:\n{add_line_numbers(table)}\n"
                    "does not match the expected table:\n"
                    f"{add_line_numbers(ref_table)}\nat line {line_num+1}:\n"
                    f"expected: '{expected}'\n"
                    f" but got: '{actual}'"
                )


# @skipIfNotDefined("print_table")
class TestPrintTable(TestFunctionality):
    """ test print_table """

    @classmethod
    def setup_class(cls) -> None:
        cls.table_comparison = TableComparison()
        from test_data.task_6_parameters import test_1, test_2, test_3, test_4
        cls.params = (test_1, test_2, test_3, test_4)

    def test_task_sheet_example(self, capsys):
        """ Test the task sheet example """
        self.a1.print_table(*self.params[0])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table/test_2")
        # self.table_comparison.assert_table_almost_equal(expected, captured.out)
        assert captured.out == expected

    def test_alternative_parameters(self, capsys):
        """ Test with alternative attributes """
        self.a1.print_table(*self.params[1])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table/test_1")
        self.table_comparison.assert_table_almost_equal(expected, captured.out)

    def test_alternative_parameters_2(self, capsys):
        """ Test with alternative attributes 2 """
        self.a1.print_table(*self.params[2])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table/test_3")
        self.table_comparison.assert_table_almost_equal(expected, captured.out)

    def test_alternative_parameters_3(self, capsys):
        """ Test with alternative attributes 3 """
        self.a1.print_table(*self.params[3])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table/test_4")
        self.table_comparison.assert_table_almost_equal(expected, captured.out)


# @skipIfNotDefined("print_table_general")
class TestPrintTableGeneral(TestFunctionality):
    """ test print_table general """

    @classmethod
    def setup_class(cls) -> None:
        cls.table_comparison = TableComparison()
        from test_data.task_7_parameters import test_1, test_2, test_3, test_4
        cls.params = (test_1, test_2, test_3, test_4)

    def test_task_sheet_example(self, capsys):
        """ Test the task sheet example """
        self.a1.print_table_general(*self.params[0])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table_general/test_1")
        self.table_comparison.assert_table_almost_equal(expected, captured.out)

    def test_alternative_parameters(self, capsys):
        """ Test with alternative attributes """
        self.a1.print_table_general(*self.params[1])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table_general/test_2")
        self.table_comparison.assert_table_almost_equal(expected, captured.out)

    def test_alternative_parameters_2(self, capsys):
        """ Test with alternative attributes 2 """
        self.a1.print_table_general(*self.params[2])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table_general/test_3")
        self.table_comparison.assert_table_almost_equal(expected, captured.out)

    def test_alternative_parameters_3(self, capsys):
        """ Test with alternative attributes 3 """
        self.a1.print_table_general(*self.params[3])

        captured = capsys.readouterr()
        expected = self.load_test_data("print_table_general/test_4")
        self.table_comparison.assert_table_almost_equal(expected, captured.out)


# @skipIfNotDefined("main")
class TestMain(TestFunctionality):
    """ test main """

    @classmethod
    def setup_class(cls) -> None:
        cls.table_comparison = TableComparison()

    def _run_main(self, program_in: str, stop_early: bool):
        """ runs the main function and captures output """
        error = None
        result = None
        with RedirectStdIO(stdinout=True) as stdio:
            stdio.stdin = program_in
            try:
                result = self.a1.main()
            except EOFError as err:
                error = err

        # self.write_test_data(file_out, stdio.stdinout)
        if error is not None and not stop_early:
            last_output = "\n".join(stdio.stdinout.rsplit("\n")[-22:])
            raise AssertionError(
                f'Your program is asking for more input when it should have ended\n'
                f'EOFError: {error}\n\n{last_output}'
            ).with_traceback(error.__traceback__)

        return result, stdio

    @staticmethod
    def chunked_stdio(program_in: str, program_io: str):
        """Split the program IO by the inputs / interactions.

        The combined input-output of the program is broken down into
        blocks, with each line of the user input / interaction forming
        the start of each block. If an interaction cannot be found,
        the remaining IO is included in the block.

        Parameters
        ----------
        program_in: str
            The string containing user input / interactions given
            to the program via std_in
        program_io: str
            The combined input and output of the program through
            std_in and std_out. Produced by RedirectStdIO

        Generates
        ---------
        str
            The block of combined IO between two interactions.
        """
        io_by_line = iter(program_io.split('\n'))
        io_buffer = []
        io_line = next(io_by_line)

        for input_statement in program_in.split('\n'):
            while not io_line.endswith(input_statement):
                io_buffer.append(io_line)
                try:
                    io_line = next(io_by_line)
                except StopIteration:
                    # Program exited before completing input
                    break

            # Return buffer lines in original form
            if io_buffer:
                yield '\n'.join(io_buffer)
                io_buffer = []

        # Collect program IO after final command
        remaining_io = '\n'.join([io_line, *io_by_line])
        if remaining_io:
            yield remaining_io

    def assertMain(self, file_in: str, file_out: str, stop_early: bool = False):
        """ assert the main function ran correctly """
        expected_io = self.load_test_data(file_out)
        program_in = self.load_test_data(file_in)
        result, stdio = self._run_main(program_in, stop_early=stop_early)
        actual_io = stdio.stdinout

        for input_text, expected, actual in zip(
            program_in.split('\n'),
            self.chunked_stdio(program_in, expected_io),
            self.chunked_stdio(program_in, actual_io)
        ):
            # Check if we are printing a table
            if input_text.startswith('p'):
                expected_prompt, expected_table = expected.split('\n', maxsplit=1)
                actual_prompt, actual_table = actual.split('\n', maxsplit=1)

                self.assertEqual(expected_prompt, actual_prompt)
                self.table_comparison.assert_table_almost_equal(expected_table, actual_table)
            else:
                self.assertMultiLineEqual(expected, actual, strip=True)

        if stdio.stdin != '':
            self.fail(msg="Not all input was read")
        self.assertIsNone(result, msg="main function should not return a non None value")

    def test_main_task_sheet(self, capsys, sysin):
        """ test main task sheet example """
        with open("test_data/task_8_in_1") as filein:
            sysin.writelines(filein.readlines())
            sysin.seek(0)

        with open("test_data/main/task_8_out_1.final") as fileout:
            expected = fileout.read()

        self.a1.main()
        captured = capsys.readouterr()
        assert captured.out == expected

    def test_main_directory(self, capsys, sysin):
        """ test main task with data in different directory """
        with open("test_data/task_8_in_2") as filein:
            sysin.writelines(filein.readlines())
            sysin.seek(0)

        with open("test_data/main/task_8_out_2.final") as fileout:
            expected = fileout.read()

        self.a1.main()
        captured = capsys.readouterr()
        assert captured.out == expected


def main():
    """ run tests """
    test_cases = [
        # TestDesign,
        TestDetermineHubSpeed,
        TestDetermineWindpower,
        TestPrintTable,
        TestPrintTableGeneral,
        TestMain
    ]

    master = TestMaster(max_diff=None,
                        suppress_stdout=True,
                        # ignore_import_fails=True,
                        # timeout=560,
                        include_no_print=True,
                        # include_no_print=False,
                        scripts=[
                            ('a1_support', 'a1_support.py'),
                            ('a1', 'a1.py')
                        ])
    master.run(test_cases)


if __name__ == '__main__':
    main()
