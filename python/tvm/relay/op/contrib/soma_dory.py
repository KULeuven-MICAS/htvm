"""
Operations to support the SOMA accelerator.
"""

import tvm
from tvm import relay

from ...expr_functor import ExprMutator

from tvm._ffi import register_func
from tvm.relay.expr import const
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
#from tvm.driver.tvmc import TVMCException

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr
from .register import register_pattern_table

# don't remove this import even if it does not seem to be used
# because this is the point where the soma_dory backend is registered
import tvm.relay.backend.contrib.soma_dory


@transform.function_pass(opt_level=0)
class SomaDoryGraphQuantizer(ExprMutator):
    """Convert fake-quantized relay graph (from Soma ONNX file) into a real quantized relay graph
    """

    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]

        if call.op.name == 'nn.conv2d':
            # replace nn.conv2d with qnn.conv2d and quantize weights (assume they are constant)
            w = relay.const(new_args[1].data.numpy().astype(self.dtype))
            new_call = relay.qnn.op.conv2d(new_args[0], w,
                                           relay.const(0),
                                           relay.const(0),
                                           relay.const(1.0),
                                           relay.const(1.0),
                                           w.data.shape[-2:],
                                           channels=w.data.shape[0],
                                           strides=call.attrs.strides,
                                           padding=call.attrs.padding,
                                           dilation=call.attrs.dilation,
                                           groups=call.attrs.groups)

        elif call.op.name == 'nn.bias_add':
            # quantize bias to int32
            new_args[1] = relay.const(new_args[1].data.numpy().astype('int32'))
            new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        elif call.op.name == 'divide':
            # We currently assume that a divide op represents a requant operations after bias_add or element-wise sum
            new_call = relay.qnn.op.requantize(new_args[0],
                                               relay.const(1.0),
                                               relay.const(0),
                                               new_args[1],
                                               relay.const(0),
                                               axis=1,
                                               out_dtype=self.dtype)
        else:
            new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        return new_call

    def visit_function(self, fn):
        """Rewrite function arguments
        """
        new_params = []
        binds = {}

        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # bias params are int32
            if param.name_hint.endswith('bias'):
                dtype = 'int32'
            else:
                dtype = self.dtype

            # Generate new variable.
            new_param = relay.var(param.name_hint, shape=var_type.shape, dtype=dtype)

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit(fn.body)
        # Rewrite the body to use new parameters.
        new_body = relay.bind(new_body, binds)

        # Construct the updated function and return.
        return relay.Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=fn.type_params,
            attrs=fn.attrs,
        )


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
        ("soma_dory.qnn_conv2d", qnn_conv2d_pattern(), check_qnn_conv2d),
    ]


def partition_for_soma_dory(mod, params=None, dpu=None, **opts):
    """
    The partitioning sequence for the soma_dory byoc
    Parameters
    ----------
    mod The module to use

    Returns
    -------
    The partitioned module.

    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            SomaDoryGraphQuantizer('int8'),
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget(["soma_dory"]),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise Exception(
                "Error converting layout to {0}".format(str(err))
            )
