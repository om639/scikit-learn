"""
Testing for Bayesian network scoring using data sampled from the Asia network.
"""
import math
import os

from sklearn.bayesian_network import Network, Variable, bic, ll_variable, \
    load_discrete
from sklearn.externals.six import iteritems
from sklearn.utils.testing import assert_almost_equal

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
    assert_almost_equal(bic(network, data), -22295.74566143257)


def test_asia_decomposable():
    # Test that adding together the score for all variables results in the same score for the entire network
    network = create_asia_network()
    data = load_discrete(ASIA_DATA, network)

    # Compute log-likelihood for each variable
    ll = 0
    dimension = 0
    for variable in network.variables:
        ll += ll_variable(variable, data)
        dimension += variable.dimension

    # Compare with real score and score provided by bic function
    assert_almost_equal(ll - 0.5 * math.log(len(data)) * dimension, -22295.74566143257)
    assert_almost_equal(ll - 0.5 * math.log(len(data)) * dimension, bic(network, data))
