import os

PULP_ENV = {"PULP_CONFIGS_PATH": "/pulp-sdk-diana/pkg/sdk/dev/install/ws/configs",
            "PULP_CURRENT_CONFIG": "pulpissimo@config_file=chips/pulpissimo/pulpissimo.json",
            "PULP_CURRENT_CONFIG_ARGS": "platform=rtl",
            "PULP_RISCV_GCC_TOOLCHAIN": "/pulp-riscv-gnu-toolchain",
            "PULP_RUNTIME_GCC_TOOLCHAIN": "/pulp-riscv-gnu-toolchain",
            "PULP_SDK_HOME": "/pulp-sdk-diana/pkg/sdk/dev",
            "PULP_SDK_INSTALL": "/pulp-sdk-diana/pkg/sdk/dev/install",
            "PULP_SDK_WS_INSTALL": "/pulp-sdk-diana/pkg/sdk/dev/install/ws",
            "PULP_TEMPLATE_ARGS": "platform(name(rtl))",
            "PYTHONPATH": "/pulp-sdk-diana/pkg/sdk/dev/install/ws/python: /tvm-fork/python",
            "RISCV": "/pulp-riscv-gnu-toolchain",
            "RULES_DIR": "/pulp-sdk-diana/pkg/sdk/dev/install/rules",
            "TARGET_INSTALL_DIR": "/pulp-sdk-diana/pkg/sdk/dev/install",
            "TILER_GENERATOR_PATH": "/pulp-sdk-diana/pkg/sdk/dev/install/ws/auto-tiler/generators",
            "TILER_PATH": "/pulp-sdk-diana/pkg/sdk/dev/install/ws/auto-tiler",
            "TVM_HOME": "/tvm-fork",
            "INSTALL_DIR": "/pulp-sdk-diana/pkg/sdk/dev/install/ws",
            "PATH": "/pulp-sdk-diana/pkg/sdk/dev/install/ws/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/pulp-sdk-diana/pkg/sdk/dev/install/ws/lib"}


from pygments import highlight
from pygments.lexers import CLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.shell import BashLexer
from pygments.lexers.make import MakefileLexer
from pygments.formatters import HtmlFormatter
from pygments.lexers.special import TextLexer

def get(path):
    if path.endswith(".c"):
        lexer = CLexer()
    elif path.endswith(".py"):
        lexer = PyLexer()
    elif path.endswith(".sh"):
        lexer = BashLexer()
    elif "Makefile" in path:
        lexer = MakefileLexer()
    else:
        lexer = TextLexer()
    with open(path) as f:
        code = f.read()
        return '<style type="text/css">{}</style>{}'.format(
            HtmlFormatter().get_style_defs('default'),
            highlight(code, lexer, HtmlFormatter()))

def set_pulp_env():
    """Utility function that sets the notebook from """
    os.environ.update(PULP_ENV)
