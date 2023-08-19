import json
import pytest
from io import StringIO
from typing import Union
from _pytest.terminal import TerminalReporter


Report = Union[pytest.CollectReport, pytest.TestReport]
MIN_LINES_DIFF = 5


def pytest_assertrepr_compare(op, left, right):
    if not isinstance(left, str) or not isinstance(right, str) or op != "==":
        return None
    right_lines = right.splitlines()
    first_line = right_lines[0]
    # Check to ensure the first line is a constant character
    if any(char != first_line[0] for char in first_line[1:]):
        return None

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_pycollect_makeitem(
    collector: Union[pytest.Module, pytest.Class],
    name: str,
    obj: object
):
    outcome = (yield).get_result()
    if outcome is None:
        return
    # Turn into a list for iteration
    if not isinstance(outcome, (list, tuple)):
        outcome = [outcome]
    for item in outcome:
        # nodeid = collector.nodeid + "::" + item.name
        properties = {
            'max_score': getattr(obj, 'max_score', 0)
        }
        # test_group_stats[nodeid] = properties
        # item.stash[score_info_key] = properties
        item.user_properties = list(properties.items())


@pytest.fixture
def sysin(monkeypatch):
    input_stream = StringIO()
    monkeypatch.setattr("sys.stdin", input_stream)
    yield input_stream


def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus):
    json_results = {'tests': []}

    all_tests = []  # type: list[Report]
    if 'passed' in terminalreporter.stats:
        all_tests = all_tests + terminalreporter.stats['passed']
    if 'failed' in terminalreporter.stats:
        all_tests = all_tests + terminalreporter.stats['failed']

    for report in all_tests:
        output = report.capstdout + '\n' + report.capstderr
        group_stats = dict(report.user_properties)

        max_score = group_stats['max_score']
        score = group_stats.get('score', max_score if report.passed else 0)

        # Don't have access to config... or do we?
        from _pytest._io import TerminalWriter

        # NOTE: Recreating BaseReport.longreprtext but *with* markup
        file = StringIO()
        tw = TerminalWriter(file)
        # INFO: Can be properly controlled with the PY_COLOR env var
        # tw.hasmarkup = True
        report.toterminal(tw)
        exc = file.getvalue()

        output += exc.strip()

        json_results["tests"].append({
            'score': round(score, 4),
            'max_score': round(max_score, 4),
            'name': report.nodeid,
            'output': output,
            'visibility': 'visible',
        })

    with open('results.json', 'w') as results:
        results.write(json.dumps(json_results, indent=4))
