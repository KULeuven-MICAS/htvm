"""
Operations to support the SOMA accelerator.
"""

import tvm

from tvm._ffi import register_func
from tvm.relay.expr import const
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
#from tvm.driver.tvmc import TVMCException

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr
from .register import register_pattern_table

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


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


@register_pattern_table("soma")
def pattern_table():
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """
    def qnn_conv2d_pattern():
        """Create pattern for qnn.conv2D with optional fused relu."""
        qnn_conv2d = is_op("qnn.conv2d")(
            wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        bias_add = is_op("nn.bias_add")(qnn_conv2d, wildcard())
        req = is_op("qnn.requantize")(
            qnn_conv2d | bias_add, is_constant(), is_constant(), is_constant(), is_constant()
        )
        clip_or_req = req.optional(is_op("clip"))
        return clip_or_req

    def check_qnn_conv2d(pattern):
        """Check if the Conv2D is supported by CMSIS-NN."""

        # just pretend that it is always supported for now
        return (True)

    return [
        ("soma.qnn_conv2d", qnn_conv2d_pattern(), check_qnn_conv2d),
    ]


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
    print('Before transform -----')
    print(mod)

    mod = transform.InferType()(mod)
    print('transform.InferType() -----')
    print(mod)

    mod = transform.MergeComposite(pattern_table())(mod)
    print('transform.MergeComposite(pattern_table()) -----')
    print(mod)

    mod = transform.AnnotateTarget(["soma"])(mod)
    print('transform.AnnotateTarget(["soma"]) -----')
    print(mod)

    #mod = transform.MergeCompilerRegions()(mod)
    #print('transform.MergeCompilerRegions() -----')
    #print(mod)

    mod = transform.PartitionGraph()(mod)
    print('transform.PartitionGraph() -----')
    print(mod)

    mod = transform.InferType()(mod)
    print('transform.InferType() -----')
    print(mod)

    with tvm.transform.PassContext(opt_level=3):
        try:
            return mod
        except Exception as err:
            raise Exception(
                "Error converting layout to {0}".format(str(err))
            )
