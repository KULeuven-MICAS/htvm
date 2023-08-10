# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Soma dory layout transform unit tests"""

import tvm
import pytest
from tvm import relay
from tvm.relay.dataflow_pattern import is_op, wildcard, FunctionPattern
from operator import attrgetter
import numpy as np

from tvm.relay.backend.contrib.soma_dory.layout_transform import SomaDoryLayoutTransform


def ir_module_from_expr(x):
    """Create IR module from relay expression
    """
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)

    # Infer data types
    return tvm.relay.transform.InferType()(mod)


def create_linear_relay_graph(ops_list):
    """ Create a linear relay graph based info from ops_list.
        For example:

        create_linear_relay_graph([('nn.relu', (), None),
                                   ('divide', (relay.const(1, dtype='int8'),), None),
                                   ('nn.dense', (relay.const(np.ones((1, 5), 'int8'))), "soma_dory.dense")])
    """
    x = relay.var('x', shape=(1,8))
    for op in ops_list:
        o = attrgetter(op[0])(relay)
        x = o(x, *op[1])

        # partition this op in a Function
        if len(op) == 3:
            pattern_args = [wildcard()] + len(op[1])*[wildcard()]
            x = is_op(op[0])(*pattern_args).partition(x, attrs={'Composite': op[2]})

    return ir_module_from_expr(x)


def create_layout_transform_pattern(x=wildcard()):
    """Return a search pattern for matching to a layout transform
    """
    x = is_op('reshape')(x)
    x = is_op('reverse')(x)
    x = is_op('reshape')(x)

    return x


def test_linear_single_composite_1():
    # Setup
    x = create_linear_relay_graph([('nn.relu', ()),
                                   ('divide', (relay.const(1.0), )),
                                   ('nn.dense', (relay.const(np.ones((1, 8))), ), "soma_dory.dense")])

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = create_layout_transform_pattern(expected)

    assert expected.match(actual)


def test_linear_single_composite_2():
    # Setup
    x = create_linear_relay_graph([('nn.relu', ()),
                                   ('divide', (relay.const(1.0), )),
                                   ('nn.dense', (relay.const(np.ones((1, 8))), ), "soma_dory.dense"),
                                   ('divide', (relay.const(1.0), ))])

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = create_layout_transform_pattern(expected)
    expected = is_op('divide')(expected, wildcard())    # a don't care op

    assert expected.match(actual)


def test_linear_multiple_composite_1():
    # Divide with division factor size 1 is not considered layout sensitive, so we only expect two layout transforms
    # Setup
    x = create_linear_relay_graph([('nn.relu', ()),
                                   ('nn.dense', (relay.const(np.ones((8, 8))), ), "soma_dory.dense"),
                                   ('divide', (relay.const(1.0), )),
                                   ('nn.dense', (relay.const(np.ones((1, 8))), ), "soma_dory.dense")])

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = is_op('divide')(expected, wildcard())
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = create_layout_transform_pattern(expected)

    assert expected.match(actual)


def test_linear_multiple_composite_2():
    # Add with rhs different from size 1 is considered layout sensitive, so we only expect four layout transforms
    # Setup
    x = create_linear_relay_graph([('nn.relu', ()),
                                   ('nn.dense', (relay.const(np.ones((8, 8))), ), "soma_dory.dense"),
                                   ('add', (relay.const(np.zeros((1, 8))), )),
                                   ('nn.dense', (relay.const(np.ones((1, 8))), ), "soma_dory.dense")])

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = create_layout_transform_pattern(expected)
    expected = is_op('add')(expected, wildcard())
    expected = create_layout_transform_pattern(expected)
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = create_layout_transform_pattern(expected)

    assert expected.match(actual)


def test_linear_multiple_composite_3():
    # Multiple consecutive ops on the accelerator should not contain intermediate transforms
    # Setup
    x = create_linear_relay_graph([('nn.relu', ()),
                                   ('nn.dense', (relay.const(np.ones((8, 8))), ), "soma_dory.dense"),
                                   ('nn.dense', (relay.const(np.ones((8, 8))), ), "soma_dory.dense"),
                                   ('nn.dense', (relay.const(np.ones((1, 8))), ), "soma_dory.dense")])

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = create_layout_transform_pattern(expected)

    assert expected.match(actual)


def test_residual_single_composite_1():
    # Residual block with only one composite op just after branch
    # Setup
    x = relay.var('x', shape=(1,8))
    x_ = relay.nn.relu(x)
    x = relay.nn.dense(x_, relay.const(np.ones((8, 8))))
    x = is_op('nn.dense')(wildcard(), wildcard()).partition(x, attrs={'Composite': 'soma_dory.dense'})
    x = relay.nn.dense(x, relay.const(np.ones((8, 8))))
    x = relay.add(x, x_)
    x = ir_module_from_expr(x)

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected = create_layout_transform_pattern()
    expected = is_op('nn.dense')(expected, wildcard())
    expected = is_op('add')(expected, wildcard())

    assert expected.match(actual)


def test_residual_multiple_composite_1():
    # Residual block with multiple composite ops
    # Setup
    x = relay.var('x', shape=(1,8))
    x_ = relay.nn.relu(x)
    x = relay.nn.dense(x_, relay.const(np.ones((8, 8))))
    x = relay.nn.dense(x, relay.const(np.ones((8, 8))))
    x_ = relay.nn.dense(x_, relay.const(np.ones((8, 8))))
    x = relay.add(x, x_)
    x = is_op('nn.dense')(wildcard(), wildcard()).partition(x, attrs={'Composite': 'soma_dory.dense'})
    x = ir_module_from_expr(x)

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected_ = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected_, wildcard())
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected_ = FunctionPattern(None, wildcard())(expected_, wildcard())
    expected = is_op('add')(expected, expected_)
    expected = create_layout_transform_pattern()

    assert expected.match(actual)


def test_residual_multiple_composite_2():
    # Residual block with multiple composite ops and composite add
    # Setup
    x = relay.var('x', shape=(1,8))
    x_ = relay.nn.relu(x)
    x = relay.nn.dense(x_, relay.const(np.ones((8, 8))))
    x = relay.nn.dense(x, relay.const(np.ones((8, 8))))
    x_ = relay.nn.dense(x_, relay.const(np.ones((8, 8))))
    x = relay.add(x, x_)
    x = is_op('add')(wildcard(), wildcard()).partition(x, attrs={'Composite': 'soma_dory.add'})
    x = is_op('nn.dense')(wildcard(), wildcard()).partition(x, attrs={'Composite': 'soma_dory.dense'})
    x = ir_module_from_expr(x)

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected_ = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected_, wildcard())
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected_ = FunctionPattern(None, wildcard())(expected_, wildcard())
    expected = FunctionPattern(None, wildcard())(expected, expected_)
    expected = create_layout_transform_pattern()

    assert expected.match(actual)


@pytest.mark.skip(reason="Concatenate is not yet supported in the layout transform.")
def test_concat_multiple_composite():
    # Residual block with multiple composite ops and concat
    # Setup
    x = relay.var('x', shape=(1,8))
    x_ = relay.nn.relu(x)
    x = relay.nn.dense(x_, relay.const(np.ones((8, 8))))
    x = relay.nn.dense(x, relay.const(np.ones((8, 8))))
    x_ = relay.nn.dense(x_, relay.const(np.ones((8, 8))))
    x = relay.concatenate((x, x_), 1)
    x = is_op('nn.dense')(wildcard(), wildcard()).partition(x, attrs={'Composite': 'soma_dory.dense'})
    x = ir_module_from_expr(x)

    # Action
    x = SomaDoryLayoutTransform()(x)
    actual = x["main"].body

    # Assert
    expected_ = create_layout_transform_pattern()
    expected = FunctionPattern(None, wildcard())(expected_, wildcard())
    expected = FunctionPattern(None, wildcard())(expected, wildcard())
    expected_ = FunctionPattern(None, wildcard())(expected_, wildcard())
    expected = is_op('concatenate')((expected, expected_))
    expected = create_layout_transform_pattern()

    assert expected.match(actual)


if __name__ == '__main__':
    test_concat_multiple_composite()
