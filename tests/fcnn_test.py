import fcnn
import pytest as pt


@pt.mark.parametrize('neurons', [(1, ), (10, 5, 3, 10)])
def test_init(neurons):
    a = fcnn.FCNN(5, neurons)
    assert a.hlayers == len(neurons) - 1
    assert a.n_inputs == 5
    assert a.n_outputs == neurons[-1]
    assert tuple(a.neurons) == neurons
