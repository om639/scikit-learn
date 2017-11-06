"""
Testing for Bayesian network scoring using data sampled from the Asia network.
"""
import os

from sklearn.bayesian_network import RV, BN, bic
from sklearn.bayesian_network.tests.common import load_recarray
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


def create_asia_bn():
    """Create a ``BN`` modelling the Asia network."""
    bn = BN()
    for name, (values, _) in iteritems(ASIA_NETWORK):
        bn.add_rv(RV(name, values))
    for name, (_, parents) in iteritems(ASIA_NETWORK):
        for parent in parents:
            bn.add_edge(bn.rvs[parent], bn.rvs[name])
    return bn


def test_asia_bic():
    # Test the BIC score for the Asia dataset against the correct BN
    bn = create_asia_bn()
    data = load_recarray(ASIA_DATA)

    # Check calculated BIC is within given tolerance of correct score
    assert_almost_equal(bic(bn, data), -22295.74566143257)
