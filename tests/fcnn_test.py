import fcnn


def test_init():
    a = fcnn.FCNN(5, (10, 5, 3))
    assert a.hlayers == 2
    assert a.n_inputs == 5
    assert a.n_outputs == 3
    assert tuple(a.neurons) == (10, 5, 3)
