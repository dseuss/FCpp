import functools as ft

import autograd.numpy as np
import pytest as pt
from autograd import grad
from numpy.testing import assert_almost_equal, assert_array_equal

from fcnn import FCNN


TEST_SHAPES = [(5, 1), (10, 5, 3, 10)]

def fcnn_predict(x_in, weights, activation):
    x_current = x_in
    for w in weights:
        x_current = activation(np.dot(w[:, 1:], x_current) + w[:, 0])
    return x_current


@pt.fixture(scope="module")
def rgen():
    return np.random.RandomState(seed=3476583865)


@pt.mark.parametrize('shape', TEST_SHAPES)
def test_init(shape):
    nn = FCNN(shape[0], shape[1:])
    assert nn.hlayers == len(shape) - 2
    assert nn.n_inputs == shape[0]
    assert nn.n_outputs == shape[-1]
    assert tuple(nn.neurons) == shape[1:]


def test_view_set_weights():
    shape = (5, 8, 1)
    nn = FCNN(shape[0], shape[1:])

    for i in range(len(shape) - 1):
        # +1 Due to implicit weights
        assert nn.get_weights(i).shape == (shape[i + 1], shape[i] + 1)

    new_weight = np.random.randn(shape[1], shape[0] + 1)
    nn.set_weights(0, new_weight)
    assert_array_equal(new_weight, nn.get_weights(0))


def test_view_set_weights_permissions():
    shape = (5, 1)
    nn = FCNN(shape[0], shape[1:])
    new_weight = np.random.randn(shape[1], shape[0] + 1)
    new_weight_copy = new_weight.copy()
    nn.set_weights(0, new_weight)

    new_weight[:] = 0
    nn_weight = nn.get_weights(0)
    assert_array_equal(new_weight_copy, nn_weight)
    assert nn_weight.flags['OWNDATA']
    assert nn_weight.flags['WRITEABLE']

    del nn
    nn_weight[:] = 0


def test_set_weights_exception():
    shape = (5, 1)
    nn = FCNN(shape[0], shape[1:])
    new_weight = (np.random.randn(shape[1], shape[0]))

    try:
        nn.set_weights(0, new_weight)
    except ValueError:
        pass
    else:
        raise AssertionError("Setting weight with wrong shape should raise error")


@pt.mark.parametrize('shape', TEST_SHAPES)
def test_predict(shape, rgen):
    nn = FCNN(shape[0], shape[1:])
    weights = [nn.get_weights(i) for i in range(len(shape) - 1)]

    x_in = rgen.randn(shape[0])
    sigmoid = lambda x: 1 / (1 + np.exp(x))
    x_ref = fcnn_predict(x_in, weights, sigmoid)
    x_out = nn.predict(x_in)
    assert_almost_equal(x_out, x_ref)
