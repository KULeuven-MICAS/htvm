# Pytest script to add option for running the code on the platform

def pytest_addoption(parser):
    parser.addoption(
        "--run",
        action='store_const',
        const=[True],
        default=[False],
        help="run on platform during testing",
    )


def pytest_generate_tests(metafunc):
    if "run" in metafunc.fixturenames:
        metafunc.parametrize("run", metafunc.config.getoption("run"))

