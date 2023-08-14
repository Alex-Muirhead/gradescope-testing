import json
import pytest
from typing import Union
from _pytest.terminal import TerminalReporter
from _pytest import capture
from _pytest import runner


Report = Union[pytest.CollectReport, pytest.TestReport]

metadata = {}
test_group_stats = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    print(call.excinfo)
    print(item)
    outcome = yield
    rep = outcome.get_result()
    print(rep.longrepr)


def pytest_generate_tests(metafunc):
    function = metafunc.function
    module   = function.__module__
    qualname = function.__qualname__.replace('.', '::')

    if hasattr(function, '_group_stats'):
        group_stats = function._group_stats

        for group_name, stats in group_stats.items():
            stats['max_score'] *= getattr(function, 'max_score', 0)
            stats['score'] *= getattr(function, 'max_score', 0)
            test_name = f'{module}.py::{qualname}[{group_name}]'
            test_group_stats[test_name] = stats

        metafunc.parametrize('group_name', group_stats.keys())
    else:
        test_name = f'{module}.py::{qualname}'
        test_group_stats[test_name] = {
            'max_score': getattr(metafunc.function, 'max_score', 0)
        }


def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus):
    json_results = {'tests': []}

    all_tests = []  # type: list[Report]
    if 'passed' in terminalreporter.stats:
        all_tests = all_tests + terminalreporter.stats['passed']
    if 'failed' in terminalreporter.stats:
        all_tests = all_tests + terminalreporter.stats['failed']

    for report in all_tests:
        output = report.capstdout + '\n' + report.capstderr
        group_stats = test_group_stats[report.nodeid]

        max_score = group_stats['max_score']
        score = group_stats.get('score', max_score if report.passed else 0)

        from io import StringIO
        from _pytest._io import TerminalWriter

        file = StringIO()
        tw = TerminalWriter(file)
        tw.hasmarkup = True
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
