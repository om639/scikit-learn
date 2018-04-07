"""
Testing for ``Network`` and ``Variable`` functionality.
"""
from sklearn.bayesian_network import Network, Variable
from sklearn.utils.testing import assert_true, assert_false, assert_in, \
    assert_not_in, assert_raises, assert_equal


def test_add_remove_edge():
    # Test adding and removing an edge between two variables in a network
    a = Variable('test_a', ['false', 'true'])
    b = Variable('test_b', ['false', 'true'])
    n = Network([a, b])

    # Add edge and test
    n.add_edge(a.index, b.index)
    assert_in(a.index, n.parent_indices(b.index))
    assert_in(a, b.parents)

    # Remove edge and test
    n.remove_edge(a.index, b.index)
    assert_not_in(a.index, n.parent_indices(b.index))
    assert_not_in(a, b.parents)


def test_add_edge_cycle():
    # Test that adding a new edge raises an error if a cycle would be created
    n = Network(
        [Variable('test_{}'.format(i), ['false', 'true']) for i in range(3)])
    n.add_edge(0, 1)
    n.add_edge(1, 2)
    with assert_raises(ValueError):
        n.add_edge(2, 0)


def test_reverse_edge_cycle():
    # Test causes_cycle works when using reverse=True
    n = Network(
        [Variable('test_{}'.format(i), ['false', 'true']) for i in range(3)])
    n.add_edge(0, 1)
    n.add_edge(0, 2)
    n.add_edge(1, 2)
    assert_true(n.causes_cycle(2, 0, reversal=True))


def test_has_edge():
    # Test that Network.has_edge works as expected
    a = Variable('test_a', ['false', 'true'])
    b = Variable('test_b', ['false', 'true'])
    n = Network([a, b])
    n.add_edge(a.index, b.index)
    assert_true(n.has_edge(a.index, b.index))
    assert_false(n.has_edge(b.index, a.index))


def test_copy():
    # Test copying a network
    a = Variable('test_a', ['false', 'true'])
    b = Variable('test_b', ['false', 'true'])
    c = Variable('test_c', ['false', 'true'])
    n = Network([a, b, c])
    n.add_edge(a.index, b.index)
    n.add_edge(b.index, c.index)
    n_copy = n.copy()

    # Check networks are currently equal
    assert_equal(n, n_copy)

    # But that they are not the same object
    assert_false(n is n_copy)
    assert_false(a is n_copy.variables[n_copy.variable_index('test_a')])
    n_copy.remove_edge(a.index, b.index)

    # Check changes to one network do not affect other
    assert_true(n.has_edge(a.index, b.index))
    assert_false(n_copy.has_edge(a.index, b.index))
