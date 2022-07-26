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

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


@transform.function_pass(opt_level=0)
class SomaGraphQuantizer(ExprMutator):
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
            # replace nn.conv2d with qnn.conv2d
            # TODO: add assertion on weights to be a variable
            w_shape = new_args[1].type_annotation.shape
            new_call = relay.qnn.op.conv2d(new_args[0], new_args[1],
                                           relay.const(0),
                                           relay.const(0),
                                           relay.const(1.0),
                                           relay.const(1.0),
                                           w_shape[-2:],
                                           channels=w_shape[0],
                                           strides=call.attrs.strides,
                                           padding=call.attrs.padding,
                                           dilation=call.attrs.dilation,
                                           groups=call.attrs.groups)

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


def soma_params_quantizer(params, dtype):
    """Convert a fake-quantized relay graph, read from soma onnx, into a real quantized relay graph
    """
    for k, v in params.items():
        if k.endswith('bias'):
            params[k] = tvm.nd.array(v.asnumpy().astype('int32'))
        else:
            params[k] = tvm.nd.array(v.asnumpy().astype(dtype))

    return params


#def _register_external_op_helper(op_name, supported=True):
#    """
#    Easily register SOMA supported operations.
#
#    Parameters
#    ----------
#    op_name : str
#        Name of the operator.
#    supported : bool
#        Whether the operator is supported
#
#    Returns
#    -------
#        The wrapped function to enable support for the operator.
#    """
#    @tvm.ir.register_op_attr(op_name, "target.soma")
#    def _func_wrapper(attrs):#, args):
#        return supported
#    return _func_wrapper
#
#_register_external_op_helper("add")
#_register_external_op_helper("nn.bias_add")
#_register_external_op_helper("nn.conv2d")
#_register_external_op_helper("qnn.conv2d")
#_register_external_op_helper("nn.relu")
#_register_external_op_helper("qnn.relu")


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
    if params:
        params = soma_params_quantizer(params, 'int8')

    print('Before transform -----')
    print(mod)

    mod = SomaGraphQuantizer('int8')(mod)
    print('SomaGraphMutator() -----')
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
