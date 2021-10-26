"""
Operations to support the SOMA accelerator.
"""

import tvm

from tvm._ffi import register_func
from tvm.relay.expr import const
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr

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
    def _func_wrapper(attrs, args):
        return supported
    return _func_wrapper


@tvm.ir.register_op_attr("qnn.add", "target.soma")
def qnn_add(expr):
    """
    Support only uint8 element wise sum.

    Parameters
    ----------
    expr : TODO: What type is Epr?
        Contains attribute and argument information.

    Returns
    -------
        Supported or not?
    """
    args = expr.args
    for typ in [args[0].checked_type, args[1].checked_type]:
        if typ.dtype != "uint8":
            return False

    return True


def make_pattern(with_bias=True):
    """
    Helper function to register a pattern to recognize Conv + EWS + ReLU as a single operation.
    (Taken from DNNL example.)
    Parameters
    ----------
    with_bias : Wheter or not to include bios

    Returns
    -------
        The created pattern from the patern matching engine.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("qnn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("qnn.add")(conv, bias)
    else:
        conv_out = conv
    return is_op("qnn.relu")(conv_out)


@register_pattern_table("soma")
def pattern_table():
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """
    # TODO: Register other operations
    return []

    conv2d_bias_relu_pat = ("soma.conv2d_bias_relu8", make_pattern(with_bias=True))
    conv2d_relu_pat = ("soma.conv2d_relu8", make_pattern(with_bias=False))
    soma_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]
    return soma_patterns
