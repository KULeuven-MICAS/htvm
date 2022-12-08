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
        ids = "run" if metafunc.config.getoption("run")[0] else "no_run"
        metafunc.parametrize("run", 
                             metafunc.config.getoption("run"),
                             ids=[ids])

