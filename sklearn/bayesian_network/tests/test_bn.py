"""
Testing for ``BN`` and ``RV``.
"""
from sklearn.bayesian_network import BN, RV
from sklearn.utils.testing import assert_true, assert_false, assert_in, assert_not_in


def test_add_remove_rv():
    # Test adding and removing an RV to a BN
    bn = BN()
    rv = RV('test', ['false', 'true'])

    # Add RV and test
    bn.add_rv(rv)
    assert_in(rv.name, bn.rvs)

    # Remove RV and test
    bn.remove_rv(rv)
    assert_not_in(rv.name, bn.rvs)


def test_add_remove_rv_parent():
    # Test the parents dict automatically gets updated when a RV is removed
    bn = BN()
    rv_from = RV('test_from', ['false', 'true'])
    rv_to = RV('test_to', ['false', 'true'])
    bn.add_rv(rv_from)
    bn.add_rv(rv_to)
    bn.add_edge(rv_from, rv_to)
    bn.remove_rv(rv_from)
    assert_not_in(rv_from, bn.parents[rv_to.name])
    assert_not_in(rv_from, rv_to.parents)


def test_add_remove_edge():
    # Test adding and removing an edge between two RVs in a BN
    bn = BN()
    rv_from = RV('test_from', ['false', 'true'])
    rv_to = RV('test_to', ['false', 'true'])
    bn.add_rv(rv_from)
    bn.add_rv(rv_to)

    # Add edge and test
    bn.add_edge(rv_from, rv_to)
    assert_in(rv_from, bn.parents[rv_to.name])
    assert_in(rv_from, rv_to.parents)

    # Remove edge and test
    bn.remove_edge(rv_from, rv_to)
    assert_not_in(rv_from, bn.parents[rv_to.name])
    assert_not_in(rv_from, rv_to.parents)


def test_add_edge_cycle():
    # Test that adding a new edge fails if a cycle would be created
    bn = BN()
    rvs = [RV('test_{}'.format(i), ['false', 'true']) for i in range(3)]
    for rv in rvs:
        bn.add_rv(rv)
    bn.add_edge(rvs[0], rvs[1])
    bn.add_edge(rvs[1], rvs[2])
    assert_false(bn.add_edge(rvs[2], rvs[0]))


def test_has_edge():
    # Test that BN.has_edge works as expected
    bn = BN()
    rv_from = RV('test_from', ['false', 'true'])
    rv_to = RV('test_to', ['false', 'true'])
    bn.add_rv(rv_from)
    bn.add_rv(rv_to)
    bn.add_edge(rv_from, rv_to)
    assert_true(bn.has_edge(rv_from, rv_to))
    assert_false(bn.has_edge(rv_to, rv_from))
