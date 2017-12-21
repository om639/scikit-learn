"""
Testing for Bayesian network scoring using data sampled from the Asia network.
"""
import numpy as np
import os

from sklearn.bayesian_network import Network, Variable, bic, bic_network, \
    load_discrete, maximize_addition, maximize_removal, maximize_reversal
from sklearn.externals.six import iteritems
from sklearn.utils.testing import assert_almost_equal, assert_equal, assert_greater

ASIA_NETWORK = {'VisitToAsia': (['no', 'yes'],
                                []),
                'Smoker': (['no', 'yes'],
                           []),
                'Tuberculosis': (['no', 'yes'],
                                 ['VisitToAsia']),
                'LungCancer': (['no', 'yes'],
                               ['Smoker']),
                'Bronchitis': (['no', 'yes'],
                               ['Smoker']),
                'TuberculosisOrCancer': (['no', 'yes'],
                                         ['Tuberculosis', 'LungCancer']),
                'X-ray': (['no', 'yes'],
                          ['TuberculosisOrCancer']),
                'Dyspnea': (['no', 'yes'],
                            ['TuberculosisOrCancer', 'Bronchitis'])}

ASIA_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'asia.csv')


def create_asia_network():
    """Create a ``Network`` modelling the Asia network."""
    n = Network(Variable(name, values) for name, (values, _) in iteritems(ASIA_NETWORK))
    # Add edges
    for name, (_, parents) in iteritems(ASIA_NETWORK):
        for parent in parents:
            n.add_edge(n.variable_index(parent), n.variable_index(name))
    return n


def test_asia_bic():
    # Test the BIC score for the Asia dataset against the correct network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Check calculated BIC is within given tolerance of correct score
    assert_almost_equal(bic_network(network, data), -22295.74566143257)


def test_asia_addition():
    # Test that maximize_addition returns the correct edge to add in a near-correct Asia network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Remove edge from LungCancer to TuberculosisOrCancer
    edge = (network.variable_index('LungCancer'), network.variable_index('TuberculosisOrCancer'))
    network.remove_edge(*edge)
    scores = np.array([bic(variable, data) for variable in network])

    # Check the new edge to add is the same as the one we just removed
    delta, edge_add = maximize_addition(network, data, scores)
    assert_equal(edge_add, edge)
    assert_greater(delta, 0)


def test_asia_removal():
    # Test that maximize_removal returns the correct edge to remove in a near-correct Asia network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Add false edge from Bronchitis to TuberculosisOrCancer
    edge = (network.variable_index('Bronchitis'), network.variable_index('TuberculosisOrCancer'))
    network.add_edge(*edge)
    scores = np.array([bic(variable, data) for variable in network])

    # Check the edge to remove is the same as the one we just added
    delta, edge_add = maximize_removal(network, data, scores)
    assert_equal(edge_add, edge)
    assert_greater(delta, 0)


def test_asia_reversal():
    # Test that maximize_reversal returns the correct edge to reverse in a near-correct Asia network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Reverse edge from Tuberculosis to TuberculosisOrCancer
    edge = (network.variable_index('Tuberculosis'), network.variable_index('TuberculosisOrCancer'))
    network.remove_edge(*edge)
    network.add_edge(*edge[::-1])
    scores = np.array([bic(variable, data) for variable in network])

    # Check the new edge to add
    delta, edge_add = maximize_reversal(network, data, scores)
    assert_equal(edge_add, edge[::-1])
    assert_greater(sum(delta), 0)
