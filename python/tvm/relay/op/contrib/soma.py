"""
Operations to support the SOMA accelerator.
"""

import tvm

from tvm._ffi import register_func
from tvm.relay.expr import const
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.driver.tvmc import TVMCException

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr
from .register import register_pattern_table


def _register_external_op_helper(op_name, supported=True):
    """
    Easily register SOMA supported operations.

    Parameters
    ----------
    op_name : str
        Name of the operator.
    supported : bool
        Whether the operator is supported

    Returns
    -------
        The wrapped function to enable support for the operator.
    """
    @tvm.ir.register_op_attr(op_name, "target.soma")
    def _func_wrapper(attrs):#, args):
        return supported
    return _func_wrapper

_register_external_op_helper("add")
_register_external_op_helper("nn.bias_add")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("qnn.conv2d")
_register_external_op_helper("nn.relu")
_register_external_op_helper("qnn.relu")

def make_pattern(with_bias=True):
    """
    Helper function to register a pattern to recognize Conv + EWS + ReLU as a single operation.
    (Taken from DNNL example.)
    Parameters
    ----------
    with_bias : Whether or not to include bias

    Returns
    -------
        The created pattern from the pattern matching engine.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return is_op("nn.relu")(conv_out)


@register_pattern_table("soma")
def pattern_table():
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """
    # TODO: Register other operations
    conv2d_bias_relu_pat = ("soma.conv2d_bias_relu8", make_pattern(with_bias=True))
    conv2d_relu_pat = ("soma.conv2d_relu8", make_pattern(with_bias=False))
    soma_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]
    return soma_patterns


def partition_for_soma(mod, params=None, dpu=None, **opts):
    """
    The partitioning sequence for the soma byoc
    Parameters
    ----------
    mod The module to use

    Returns
    -------
    The partitioned module.

    """
    # Convert the layout of the graph where possible.
    seq = tvm.transform.Sequential(
        [
            transform.AnnotateTarget(["soma"]),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),

        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise TVMCException(
                "Error converting layout to {0}".format(str(err))
            )
