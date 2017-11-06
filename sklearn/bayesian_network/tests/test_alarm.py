"""
Testing for Bayesian network scoring using data sampled from the ALARM network.
"""
import os

from sklearn.bayesian_network import RV, BN, bic
from sklearn.bayesian_network.tests.common import load_recarray
from sklearn.externals.six import iteritems
from sklearn.utils.testing import assert_almost_equal

ALARM_NETWORK = {'HISTORY': (['FALSE', 'TRUE'],
                             ['LVFAILURE']),
                 'CVP': (['LOW', 'NORMAL', 'HIGH'],
                         ['LVEDVOLUME']),
                 'PCWP': (['LOW', 'NORMAL', 'HIGH'],
                          ['LVEDVOLUME']),
                 'HYPOVOLEMIA': (['FALSE', 'TRUE'],
                                 []),
                 'LVEDVOLUME': (['LOW', 'NORMAL', 'HIGH'],
                                ['LVFAILURE', 'HYPOVOLEMIA']),
                 'LVFAILURE': (['FALSE', 'TRUE'],
                               []),
                 'STROKEVOLUME': (['LOW', 'NORMAL', 'HIGH'],
                                  ['LVFAILURE', 'HYPOVOLEMIA']),
                 'ERRLOWOUTPUT': (['FALSE', 'TRUE'],
                                  []),
                 'HRBP': (['LOW', 'NORMAL', 'HIGH'],
                          ['ERRLOWOUTPUT', 'HR']),
                 'HREKG': (['LOW', 'NORMAL', 'HIGH'],
                           ['HR', 'ERRCAUTER']),
                 'ERRCAUTER': (['FALSE', 'TRUE'],
                               []),
                 'HRSAT': (['LOW', 'NORMAL', 'HIGH'],
                           ['HR', 'ERRCAUTER']),
                 'INSUFFANESTH': (['FALSE', 'TRUE'],
                                  []),
                 'ANAPHYLAXIS': (['FALSE', 'TRUE'],
                                 []),
                 'TPR': (['LOW', 'NORMAL', 'HIGH'],
                         ['ANAPHYLAXIS']),
                 'EXPCO2': (['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                            ['VENTLUNG', 'ARTCO2']),
                 'KINKEDTUBE': (['FALSE', 'TRUE'],
                                []),
                 'MINVOL': (['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                            ['INTUBATION', 'VENTLUNG']),
                 'FIO2': (['LOW', 'NORMAL'],
                          []),
                 'PVSAT': (['LOW', 'NORMAL', 'HIGH'],
                           ['FIO2', 'VENTALV']),
                 'SAO2': (['LOW', 'NORMAL', 'HIGH'],
                          ['PVSAT', 'SHUNT']),
                 'PAP': (['LOW', 'NORMAL', 'HIGH'],
                         ['PULMEMBOLUS']),
                 'PULMEMBOLUS': (['FALSE', 'TRUE'],
                                 []),
                 'SHUNT': (['NORMAL', 'HIGH'],
                           ['PULMEMBOLUS', 'INTUBATION']),
                 'INTUBATION': (['NORMAL', 'ONESIDED', 'ESOPHAGEAL'],
                                []),
                 'PRESS': (['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                           ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE']),
                 'DISCONNECT': (['FALSE', 'TRUE'],
                                []),
                 'MINVOLSET': (['LOW', 'NORMAL', 'HIGH'],
                               []),
                 'VENTMACH': (['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                              ['MINVOLSET']),
                 'VENTTUBE': (['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                              ['DISCONNECT', 'VENTMACH']),
                 'VENTLUNG': (['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                              ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE']),
                 'VENTALV': (['ZERO', 'LOW', 'NORMAL', 'HIGH'],
                             ['INTUBATION', 'VENTLUNG']),
                 'ARTCO2': (['LOW', 'NORMAL', 'HIGH'],
                            ['VENTALV']),
                 'CATECHOL': (['NORMAL', 'HIGH'],
                              ['TPR', 'SAO2', 'ARTCO2', 'INSUFFANESTH']),
                 'HR': (['LOW', 'NORMAL', 'HIGH'],
                        ['CATECHOL']),
                 'CO': (['LOW', 'NORMAL', 'HIGH'],
                        ['STROKEVOLUME', 'HR']),
                 'BP': (['LOW', 'NORMAL', 'HIGH'],
                        ['TPR', 'CO'])}

ALARM_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'alarm.csv')


def create_alarm_bn():
    """Create a ``BN`` modelling the Asia network."""
    bn = BN()
    for name, (values, _) in iteritems(ALARM_NETWORK):
        bn.add_rv(RV(name, values))
    for name, (_, parents) in iteritems(ALARM_NETWORK):
        for parent in parents:
            bn.add_edge(bn.rvs[parent], bn.rvs[name])
    return bn


def test_alarm_bic():
    # Test the BIC score for the ALARM dataset against the correct BN
    bn = create_alarm_bn()
    data = load_recarray(ALARM_DATA)

    # Check calculated BIC is within given tolerance of correct score
    assert_almost_equal(bic(bn, data), -105795.5734556365)
