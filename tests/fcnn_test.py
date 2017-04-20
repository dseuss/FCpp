import autograd.numpy as np
import pytest as pt
from numpy.testing import assert_almost_equal, assert_array_equal

from fcclass import FcClassifier


TEST_HIDDEN_UNITS = [tuple(), (5,), (5, 10, 3, 4)]
TEST_INPUT_UNITS = [1, 10, 5]


def fcnn_predict(x_in, weights, activation):
    x_current = x_in
    for w in weights:
        x_current = activation(np.dot(w[:, 1:], x_current) + w[:, 0])
    return x_current


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

    for i in range(len(shape) - 1):
        # +1 Due to implicit weights
        assert nn.get_weights(i).shape == (shape[i + 1], shape[i] + 1)

    new_weight = np.random.randn(shape[1], shape[0] + 1)
    nn.set_weights(0, new_weight)
    assert_array_equal(new_weight, nn.get_weights(0))


def test_view_set_weights_permissions():
    shape = (5, 1)
    nn = FcClassifier(shape[0], tuple())
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


def test_set_weights_exception(rgen):
    shape = (5, 1)
    nn = FcClassifier(shape[0], tuple())
    new_weight = (rgen.randn(shape[1], shape[0]))

    try:
        nn.set_weights(0, new_weight)
    except ValueError:
        pass
    else:
        raise AssertionError("Setting weight with wrong shape should raise error")


@pt.mark.parametrize('input_units', TEST_INPUT_UNITS)
@pt.mark.parametrize('hidden_units', TEST_HIDDEN_UNITS)
def test_random_initialization(input_units, hidden_units):
    nn = FcClassifier(input_units, hidden_units)
    nr_layers = len(hidden_units) + 1

    for i in range(nr_layers):
        assert_almost_equal(np.linalg.norm(nn.get_weights(i)), 0)

    nn.init_random()
    for i in range(nr_layers):
        assert np.linalg.norm(nn.get_weights(i)) > .5


@pt.mark.parametrize('input_units', TEST_INPUT_UNITS)
@pt.mark.parametrize('hidden_units', TEST_HIDDEN_UNITS)
def test_predict(input_units, hidden_units, rgen):
    nn = FcClassifier(input_units, hidden_units)
    nn.init_random()
    weights = [nn.get_weights(i) for i in range(len(hidden_units) + 1)]

    x_in = rgen.randn(input_units)
    sigmoid = lambda x: 1 / (1 + np.exp(x))
    x_ref = fcnn_predict(x_in, weights, sigmoid)
    x_out = nn.predict(x_in)
    assert_almost_equal(x_out, x_ref)
    assert np.isscalar(x_out)
