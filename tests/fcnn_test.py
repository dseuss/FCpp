import numpy as np
import pytest as pt
from fcnn import FCNN


@pt.fixture(scope="module")
def rgen():
    return np.random.RandomState(seed=3476583865)


@pt.mark.parametrize('shape', [(5, 1), (10, 5, 3, 10)])
def test_init(shape):
    nn = FCNN(shape[0], shape[1:])
    assert nn.hlayers == len(shape) - 2
    assert nn.n_inputs == shape[0]
    assert nn.n_outputs == shape[-1]
    assert tuple(nn.neurons) == shape[1:]


def test_get_set_weights():
    shape = (5, 8, 1)
    nn = FCNN(shape[0], shape[1:])

    for i in range(len(shape) - 1):
        # +1 Due to implicit weights
        assert nn.get_weights(i).shape == (shape[i + 1], shape[i] + 1)

    new_weight = np.random.randn(shape[1], shape[0] + 1)
    nn.set_weights(0, new_weight)
    np.testing.assert_array_equal(new_weight, nn.get_weights(0))


def test_get_set_weights_permissions():
    shape = (5, 1)
    nn = FCNN(shape[0], shape[1:])
    new_weight = np.random.randn(shape[1], shape[0] + 1)
    new_weight_copy = new_weight.copy()
    nn.set_weights(0, new_weight)

    new_weight[:] = 0
    nn_weight = nn.get_weights(0)
    np.testing.assert_array_equal(new_weight_copy, nn_weight)
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
