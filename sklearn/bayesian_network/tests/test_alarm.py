"""
Testing for Bayesian network scoring using data sampled from the ALARM network.
"""
import numpy as np
import os

from sklearn.bayesian_network import Network, Variable, bic_network, \
    load_discrete, hc
from sklearn.utils.testing import assert_almost_equal, assert_equal, \
    assert_true

ALARM_NETWORK = [('HISTORY', ['FALSE', 'TRUE'], ['LVFAILURE']),
                 ('CVP', ['LOW', 'NORMAL', 'HIGH'], ['LVEDVOLUME']),
                 ('PCWP', ['LOW', 'NORMAL', 'HIGH'], ['LVEDVOLUME']),
                 ('HYPOVOLEMIA', ['FALSE', 'TRUE'], []),
                 ('LVEDVOLUME', ['LOW', 'NORMAL', 'HIGH'],
                  ['LVFAILURE', 'HYPOVOLEMIA']),
                 ('LVFAILURE', ['FALSE', 'TRUE'], []),
                 ('STROKEVOLUME', ['LOW', 'NORMAL', 'HIGH'],
                  ['LVFAILURE', 'HYPOVOLEMIA']),
                 ('ERRLOWOUTPUT', ['FALSE', 'TRUE'], []),
                 ('HRBP', ['LOW', 'NORMAL', 'HIGH'], ['ERRLOWOUTPUT', 'HR']),
                 ('HREKG', ['LOW', 'NORMAL', 'HIGH'], ['HR', 'ERRCAUTER']),
                 ('ERRCAUTER', ['FALSE', 'TRUE'], []),
                 ('HRSAT', ['LOW', 'NORMAL', 'HIGH'], ['HR', 'ERRCAUTER']),
                 ('INSUFFANESTH', ['FALSE', 'TRUE'], []),
                 ('ANAPHYLAXIS', ['FALSE', 'TRUE'], []),
                 ('TPR', ['LOW', 'NORMAL', 'HIGH'], ['ANAPHYLAXIS']),
                 ('EXPCO2', ['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                  ['VENTLUNG', 'ARTCO2']),
                 ('KINKEDTUBE', ['FALSE', 'TRUE'], []),
                 ('MINVOL', ['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                  ['INTUBATION', 'VENTLUNG']),
                 ('FIO2', ['LOW', 'NORMAL'], []),
                 ('PVSAT', ['LOW', 'NORMAL', 'HIGH'], ['FIO2', 'VENTALV']),
                 ('SAO2', ['LOW', 'NORMAL', 'HIGH'], ['PVSAT', 'SHUNT']),
                 ('PAP', ['LOW', 'NORMAL', 'HIGH'], ['PULMEMBOLUS']),
                 ('PULMEMBOLUS', ['FALSE', 'TRUE'], []),
                 ('SHUNT', ['NORMAL', 'HIGH'], ['PULMEMBOLUS', 'INTUBATION']),
                 ('INTUBATION', ['NORMAL', 'ONESIDED', 'ESOPHAGEAL'], []),
                 ('PRESS', ['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                  ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE']),
                 ('DISCONNECT', ['FALSE', 'TRUE'], []),
                 ('MINVOLSET', ['LOW', 'NORMAL', 'HIGH'], []),
                 (
                     'VENTMACH', ['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                     ['MINVOLSET']),
                 ('VENTTUBE', ['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                  ['DISCONNECT', 'VENTMACH']),
                 ('VENTLUNG', ['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                  ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE']),
                 ('VENTALV', ['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                  ['INTUBATION', 'VENTLUNG']),
                 ('ARTCO2', ['LOW', 'NORMAL', 'HIGH'], ['VENTALV']),
                 ('CATECHOL', ['NORMAL', 'HIGH'],
                  ['TPR', 'SAO2', 'ARTCO2', 'INSUFFANESTH']),
                 ('HR', ['LOW', 'NORMAL', 'HIGH'], ['CATECHOL']),
                 ('CO', ['LOW', 'NORMAL', 'HIGH'], ['STROKEVOLUME', 'HR']),
                 ('BP', ['LOW', 'NORMAL', 'HIGH'], ['TPR', 'CO'])]

ALARM_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                          'alarm.csv')

ALARM_LEARNED = [
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, True, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, True, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, True, False, True, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, True, False, False, True, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [True, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, True, True, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, True, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, True, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, True, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, True, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, True, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, True, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, True, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, True, False, True, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, True, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, True, False,
     False, False, False, False, False, False, False, False, False,
     False, False, True, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     True, False, False, False, False, False, False, False, False,
     False, False, False, False, True, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, True, False, False, False, True, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, True, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, True, False, True, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, True,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, True, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, True, False,
     False, False, False, False, False, False, True, False, False,
     False, False, True, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     True, False, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, True,
     False, True, False, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, True,
     False, False, False, False, False, False, True, False, False,
     False, False, False, False, True, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, True,
     False, False, False, False, False, False, False, False, False,
     False, False, True, False, False, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, True, False, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, True, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, True, False, False, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, True, False, False,
     False],
    [False, False, False, False, False, False, True, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, True, False,
     False],
    [False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, True, False, False, False,
     False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, False, False, False, True,
     False]]


def create_alarm_network():
    """Create a ``Network`` modelling the ALARM network."""
    n = Network(Variable(name, values) for name, values, _ in ALARM_NETWORK)
    # Add edges
    for name, _, parents in ALARM_NETWORK:
        for parent in parents:
            n.add_edge(n.variable_index(parent), n.variable_index(name))
    return n


def test_alarm_load_discrete():
    # Test the load_discrete function for loading the Asia learning data
    network = create_alarm_network()
    data = load_discrete(ALARM_DATA, network)

    assert_true(isinstance(data, np.ndarray))
    assert_equal(data.shape, (10000, 37))


def test_alarm_bic():
    # Test the BIC score for the ALARM dataset against the correct network
    network = create_alarm_network()
    data = load_discrete(ALARM_DATA, network)

    # Check calculated BIC is within given tolerance of correct score
    assert_almost_equal(bic_network(network, data), -105795.5734556365)


def test_alarm_hc():
    # Test that running hill-climbing on the ALARM network produces the
    # expected structure
    network = Network(
        Variable(name, values) for name, values, _ in ALARM_NETWORK)
    data = load_discrete(ALARM_DATA, network)

    # Run hill climbing on the network
    hc(network, data)
    assert_true(np.array_equal(network.parents, ALARM_LEARNED))
