import itertools as it

import autograd.numpy as np
import pytest as pt
from autograd import grad
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)

from fcclass import FcClassifier

TEST_HIDDEN_UNITS = [tuple(), (5,), (5, 10, 3, 4)]
TEST_INPUT_UNITS = [1, 10, 5]


###############################################################################
#                              Helper functions                               #
###############################################################################

def fcnn_predict(x_in, weights, biases, activations):
    x_current = x_in
    for w, b, f in zip(weights, biases, activations):
        x_current = f(w @ x_current + b[:, None])
    return x_current


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(p, q):
    return np.sum(- p * np.log(q) - (1 - p) * np.log(1 - q))


###############################################################################
#                              Testing functions                              #
###############################################################################

@pt.mark.parametrize('input_units', TEST_INPUT_UNITS)
@pt.mark.parametrize('hidden_units', TEST_HIDDEN_UNITS)
def test_init(input_units, hidden_units):
    nn = FcClassifier(input_units, hidden_units)
    assert nn.hidden_layers == len(hidden_units)
    assert nn.input_units == input_units
    assert tuple(nn.hidden_units) == hidden_units


def test_view_set_weights():
    shape = (5, 8, 1)
    nn = FcClassifier(shape[0], shape[1:-1])
    weights = nn.get_weights()

    for i in range(len(shape) - 1):
        # +1 Due to implicit weights
        w, b = weights[i]
        assert w.shape == (shape[i + 1], shape[i])
        assert b.shape == (shape[i + 1], )

    new_weight = np.random.randn(shape[1], shape[0])
    new_bias = np.random.randn(shape[1],)
    nn.set_weights(0, new_weight, new_bias)
    assert_array_equal(new_weight, nn.get_weights()[0][0])
    assert_array_equal(new_bias, nn.get_weights()[0][1])


def test_view_set_weights_permissions():
    shape = (5, 1)
    nn = FcClassifier(shape[0], tuple())
    new_weight = np.random.randn(shape[1], shape[0])
    new_bias = np.random.randn(shape[1], )

    new_weight_copy = new_weight.copy()
    new_bias_copy = new_bias.copy()
    nn.set_weights(0, new_weight, new_bias)

    new_weight[:] = 0
    new_bias[:] = 0
    nn_weight, nn_bias = nn.get_weights()[0]

    assert_array_equal(new_weight_copy, nn_weight)
    assert_array_equal(new_bias_copy, nn_bias)
    assert not nn_weight.flags['OWNDATA']
    assert not nn_bias.flags['OWNDATA']
    assert not nn_weight.flags['WRITEABLE']
    assert not nn_bias.flags['WRITEABLE']
    del nn
    assert_array_equal(new_weight_copy, nn_weight)
    assert_array_equal(new_bias_copy, nn_bias)


def test_set_weights_exception(rgen):
    shape = (5, 1)
    nn = FcClassifier(shape[0], tuple())

    try:
        nn.set_weights(0, rgen.randn(shape[1] + 3, shape[0]), rgen.randn(shape[1]))
    except ValueError:
        pass
    else:
        raise AssertionError("Setting weight with wrong shape should raise error")

    try:
        nn.set_weights(0, rgen.randn(shape[1], shape[0]), rgen.randn(shape[1] + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Setting bias with wrong shape should raise error")

@pt.mark.parametrize('input_units', TEST_INPUT_UNITS)
@pt.mark.parametrize('hidden_units', TEST_HIDDEN_UNITS)
def test_random_initialization(input_units, hidden_units):
    nn = FcClassifier(input_units, hidden_units)

    weights = nn.get_weights()
    for w, b in weights:
        assert_almost_equal(np.linalg.norm(w), 0)
        assert_almost_equal(np.linalg.norm(b), 0)

    nn.init_random()
    weights = nn.get_weights()
    for w, b in weights:
        assert np.linalg.norm(w) > .5
        assert np.linalg.norm(b) > .01


@pt.mark.parametrize('input_units', TEST_INPUT_UNITS)
@pt.mark.parametrize('hidden_units', TEST_HIDDEN_UNITS)
@pt.mark.parametrize('nr_samples', [1, 10])
def test_predict(input_units, hidden_units, nr_samples, rgen):
    nn = FcClassifier(input_units, hidden_units)
    nn.init_random()
    parameters = nn.get_weights()
    weights = list(w for w, _ in parameters)
    biases = list(b for _, b in parameters)

    x_in = rgen.randn(input_units, nr_samples)
    y_ref = fcnn_predict(x_in, weights, biases, it.repeat(sigmoid))[0, :]
    y_hat = nn.predict(x_in)
    assert_almost_equal(y_hat, y_ref)


@pt.mark.parametrize('input_units', TEST_INPUT_UNITS)
@pt.mark.parametrize('hidden_units', TEST_HIDDEN_UNITS)
@pt.mark.parametrize('nr_samples', [1, 10])
def test_evaluate(input_units, hidden_units, nr_samples, rgen):
    nn = FcClassifier(input_units, hidden_units)
    nn.init_random()
    parameters = nn.get_weights()
    weights = list(w for w, _ in parameters)
    biases = list(b for _, b in parameters)

    x_in = rgen.randn(input_units, nr_samples)
    y_in = rgen.randint(2, size=nr_samples)
    cost = nn.evaluate(x_in, y_in)

    y_ref = fcnn_predict(x_in, weights, biases, it.repeat(sigmoid))
    cost_ref = cross_entropy(y_in, y_ref)
    assert_almost_equal(cost, cost_ref)


@pt.mark.parametrize('input_units', TEST_INPUT_UNITS)
@pt.mark.parametrize('hidden_units', TEST_HIDDEN_UNITS)
def test_backprop(input_units, hidden_units, rgen):
    nn = FcClassifier(input_units, hidden_units)
    nn.init_random()
    parameters = nn.get_weights()
    weights = list(w for w, _ in parameters)
    biases = list(b for _, b in parameters)

    x_in = rgen.randn(input_units, 1)
    grads = nn.back_propagate(x_in, 1.0)
    grad_w = [w for w, _ in grads]
    grad_b = [b for _, b in grads]

    y_hat = lambda weights: fcnn_predict(x_in, weights, biases, it.repeat(sigmoid))
    costf = lambda weights: cross_entropy(1.0, y_hat(weights))
    grad_costf_ref = grad(costf)(weights)
    for w, w_ref in zip(grad_w, grad_costf_ref):
        assert_array_almost_equal(w, w_ref)

    y_hat = lambda biases: fcnn_predict(x_in, weights, biases, it.repeat(sigmoid))
    costf = lambda biases: cross_entropy(1, y_hat(biases))
    grad_costf_ref = grad(costf)(biases)
    for b, b_ref in zip(grad_b, grad_costf_ref):
        assert_array_almost_equal(b, b_ref)
