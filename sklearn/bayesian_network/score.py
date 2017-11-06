"""
Scoring functions for Bayesian networks.
"""
import math
import numpy as np

from collections import Counter, defaultdict
from sklearn.externals.six import iteritems, itervalues


def log_likelihood(bn, data):
    """Computer the log-likelihood for the given network with respect to the
    given data.

    Parameters
    ----------
    bn : ``BN``
        The network to compute the log-likelihood for.

    data : ``numpy.recarray``
        The data to use when computing the log-likelihood.

    Returns
    -------
    ll : float
        The log-likelihood of ``bn`` given ``data``.
    """
    return sum(_log_likelihood_rv(rv, data) for rv in itervalues(bn.rvs))


def _log_likelihood_rv(rv, data):
    """Return the log-likelihood for a single ``RV``."""
    ll = 0
    for count_key, counts in iteritems(_count_rv(rv, data)):
        parent_count = sum(itervalues(counts))
        for value, count in iteritems(counts):
            ll += count * math.log(count / parent_count)
    return ll


def _count_rv(rv, data):
    """Return the parent configuration counts for a single ``RV``."""
    counts = defaultdict(Counter)
    for row in data:
        count_key = tuple(row[parent.name] for parent in rv.parents)
        counts[count_key][row[rv.name]] += 1
    return counts


def bic(bn, data):
    """Compute the Bayesian Information Criterion (BIC) score for the given
    network with respect to the given data.

    Parameters
    ----------
    bn : ``BN``
        The network to compute the score for.

    data : ``numpy.recarray``
        The data to use when computing the score.

    Returns
    -------
    bic : float
        The BIC score for ``bn`` given ``data``.
    """
    return log_likelihood(bn, data) - 0.5 * math.log(len(data)) * _dim(bn)


def _dim(bn):
    """Return the dimension for the given ``BN``."""
    return sum(_dim_rv(rv) for rv in itervalues(bn.rvs))


def _dim_rv(rv):
    """Return the dimension for the given ``RV``."""
    return np.prod([len(parent.values) for parent in rv.parents]) * (len(rv.values) - 1)
