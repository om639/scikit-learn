"""
Testing for Bayesian network scoring using data sampled from the Asia network.
"""
import numpy as np
import os

from sklearn.bayesian_network import Network, Variable, bic, bic_network, \
    load_discrete, max_add, max_remove, max_reverse, hc
from sklearn.utils.testing import assert_almost_equal, assert_equal, assert_greater, assert_true

ASIA_NETWORK = [('Smoker', ['no', 'yes'], []),
                ('LungCancer', ['no', 'yes'], ['Smoker']),
                ('VisitToAsia', ['no', 'yes'], []),
                ('Tuberculosis', ['no', 'yes'], ['VisitToAsia']),
                ('TuberculosisOrCancer', ['no', 'yes'], ['Tuberculosis', 'LungCancer']),
                ('X-ray', ['no', 'yes'], ['TuberculosisOrCancer']),
                ('Bronchitis', ['no', 'yes'], ['Smoker']),
                ('Dyspnea', ['no', 'yes'], ['TuberculosisOrCancer', 'Bronchitis'])]

ASIA_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'asia.csv')

ASIA_LEARNED = [[False, False, False, False, False, False, False, False],
                [True, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, True, False, False, False, False, False],
                [False, True, False, True, False, False, False, False],
                [False, False, False, False, True, False, False, False],
                [True, False, False, False, False, False, False, False],
                [False, False, False, False, True, False, True, False]]


def create_asia_network():
    """Create a ``Network`` modelling the Asia network."""
    n = Network(Variable(name, values) for name, values, _ in ASIA_NETWORK)
    # Add edges
    for name, _, parents in ASIA_NETWORK:
        for parent in parents:
            n.add_edge(n.variable_index(parent), n.variable_index(name))
    return n


def test_asia_load_discrete():
    # Test the load_discrete function for loading the Asia learning data
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    assert_true(isinstance(data, np.ndarray))
    assert_equal(data.shape, (10000, 8))


def test_asia_bic():
    # Test the BIC score for the Asia dataset against the correct network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Check calculated BIC is within given tolerance of correct score
    assert_almost_equal(bic_network(network, data), -22295.74566143257)


def test_asia_add():
    # Test that max_add returns the correct edge to add in a near-correct Asia network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Remove edge from LungCancer to TuberculosisOrCancer
    edge = (network.variable_index('LungCancer'), network.variable_index('TuberculosisOrCancer'))
    network.remove_edge(*edge)
    scores = np.array([bic(variable, data) for variable in network])

    # Check the new edge to add is the same as the one we just removed
    delta, edge_add = max_add(network, data, scores)
    assert_equal(edge_add, edge)
    assert_greater(delta, 0)


def test_asia_remove():
    # Test that max_remove returns the correct edge to remove in a near-correct Asia network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Add false edge from Bronchitis to TuberculosisOrCancer
    edge = (network.variable_index('Bronchitis'), network.variable_index('TuberculosisOrCancer'))
    network.add_edge(*edge)
    scores = np.array([bic(variable, data) for variable in network])

    # Check the edge to remove is the same as the one we just added
    delta, edge_add = max_remove(network, data, scores)
    assert_equal(edge_add, edge)
    assert_greater(delta, 0)


def test_asia_reverse():
    # Test that max_reverse returns the correct edge to reverse in a near-correct Asia network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Reverse edge from Tuberculosis to TuberculosisOrCancer
    edge = (network.variable_index('Tuberculosis'), network.variable_index('TuberculosisOrCancer'))
    network.remove_edge(*edge)
    network.add_edge(*edge[::-1])
    scores = np.array([bic(variable, data) for variable in network])

    # Check the new edge to add
    delta_total, delta, edge_add = max_reverse(network, data, scores)
    assert_equal(edge_add, edge[::-1])
    assert_greater(delta_total, 0)


def test_asia_hc():
    # Test that running hill-climbing on the Asia network produces the expected structure
    network = Network(Variable(name, values) for name, values, _ in ASIA_NETWORK)
    data = load_discrete(ASIA_DATA, network)

    # Run hill climbing on the network
    hc(network, data)
    assert_true(np.array_equal(network.parents, ASIA_LEARNED))
