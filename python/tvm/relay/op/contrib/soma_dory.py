"""
Operations to support the SOMA accelerator.
"""

import tvm
import logging
import numpy as np
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

logger = logging.getLogger("SomaDory")


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
            # Since the currently existing 'qnn.op.requantize' does not support floor-based rounding, we construct our
            # own requantization using a set of primitive relay ops. We expect that the division factor is power-of-two
            # and therefore our custom requantization is a sequence of these ops: right_shift, clip, cast.
            shift_factor = int(np.log2(new_args[1].data.numpy()))
            right_shift = relay.op.right_shift(new_args[0], relay.const(shift_factor))
            clip = relay.op.clip(right_shift, a_min=-128, a_max=127)
            new_call = relay.op.cast(clip, self.dtype)

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


def qnn_conv2d_pattern():
    """Create pattern for qnn.conv2D with optional fused relu."""
    qnn_conv2d = is_op("qnn.conv2d")(
        wildcard(), wildcard(), is_constant(), is_constant(),
        is_constant(), is_constant()
    )
    bias_add = is_op("nn.bias_add")(qnn_conv2d, wildcard())
    right_shift = is_op("right_shift")(bias_add,
                                       is_constant())
    # TODO: figure out how to match on attributes for clip?
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip).has_attr({"dtype": "int8"})
    # optionally have extra clip/ReLU
    act_or_cast = cast.optional(lambda x: is_op("clip")(x))
    return act_or_cast


def check_qnn_conv2d(pattern):
    """Check if the Conv2D is supported by the soma dory accelerator"""

    if str(pattern.op.name) == "clip":
        clip = pattern
        cast = clip.args[0]
    else:
        cast = pattern
    right_shift = cast.args[0].args[0]

    # Check range of shift factor
    shift_factor = right_shift.args[1].data.numpy()
    if shift_factor < 0 or shift_factor > 31:
        logger.warning("Conv2d shift factor must be in range [0, 31], but got {shift_factor}. Acceleration for this conv2d is not supported")
        return False

    right_shift_input = right_shift.args[0]

    # For now, we don't support convolutions without bias
    if str(right_shift_input.op.name) != "nn.bias_add":
        logger.warning("Found convolution without nn.bias_add. Acceleration for this conv2d is not supported")
        return False

    bias_add = right_shift_input

    # Check bias dtype
    bias_dtype = bias_add.args[1].checked_type.dtype
    if bias_dtype != 'int32':
        logger.warning(f"Expected nn.bias_add parameters to be of type int32, but got {bias_dtype}. Acceleration for this conv2d is not supported")
        return False

    conv2d = bias_add.args[0]

    def is_conv2d_attr_value_supported(attrs, name, supported_values):
        attr = attrs[name]

        if isinstance(attr, tvm.ir.container.Array):
            attr = list(attr)

        if attr not in supported_values:
            logger.warning(f"Expected qnn.conv2d {name} to be one of {supported_values}, but got {attr}. \
                            Acceleration for this conv2d is not supported")
            return False

        return True

    # check conv2d attributes
    if (not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_size', [[1, 1], [3, 3], [5, 5], [7, 7]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'padding', [4*[0], 4*[1], [1, 1, 0, 0], [0, 0, 1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', [[1, 1], [2, 2]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', [[1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', [1])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_layout', ['OIHW'])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'data_layout', ['NCHW'])):

       return False

    #conv2d_input = conv2d.args[0]
    #conv2d_weight = conv2d.args[1]

    return True


def pattern_table():
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """

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
