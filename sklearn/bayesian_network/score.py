"""
Scoring functions for Bayesian networks.
"""
from __future__ import division

import math
import numpy as np

from collections import Counter, defaultdict
from sklearn.externals.six import iteritems, itervalues


def log_likelihood(network, data):
    """Computer the log-likelihood for the given network with respect to the
    given data.

    Parameters
    ----------
    network : ``Network``
        The network to compute the log-likelihood for.

    data : ``numpy.array``
        The data to use when computing the log-likelihood.

    Returns
    -------
    log_likelihood : float
        The log-likelihood of ``network`` given ``data``.
    """
    return sum(_log_likelihood_variable(v, data) for v in network.variables)


def _log_likelihood_variable(variable, data):
    """Return the log-likelihood for a ``Variable`` for the specified data."""
    ll = 0
    for count_key, counts in iteritems(_counts(variable, data)):
        parent_count = sum(itervalues(counts))
        for value, count in iteritems(counts):
            ll += count * math.log(count / parent_count)
    return ll


def _counts(variable, data):
    """Return the parent configuration counts for a single ``Variable``."""
    # Select only the relevant columns
    ar = data[:, np.insert(variable.parent_indices, 0, variable.index)]

    # Convert each row into a single value and identify unique values
    ar = np.ascontiguousarray(ar)
    av = ar.view(np.dtype((np.void, ar.dtype.itemsize * ar.shape[1])))
    _, indices, counts = np.unique(av, return_index=True, return_counts=True)

    # Create dict from resulting unique counts
    counter = defaultdict(Counter)
    for index, count in zip(indices, counts):
        pk = tuple(ar[index, 1:])
        ck = ar[index, 0]
        counter[pk][ck] = count
    return counter


def bic(network, data):
    """Compute the Bayesian Information Criterion (BIC) score for the given
    network with respect to the given data.

    Parameters
    ----------
    network : ``Network``
        The network to compute the score for.

    data : ``numpy.array``
        The data to use when computing the score.

    Returns
    -------
    bic : float
        The BIC score for ``network`` given ``data``.
    """
    ll = log_likelihood(network, data)

    # Apply penalization
    return ll - 0.5 * math.log(len(data)) * _dim(network)


def _dim(n):
    """Return the dimension for the given ``Network``."""
    return sum(_dim_variable(v) for v in n.variables)


def _dim_variable(v):
    """Return the dimension for the given ``Variable``."""
    return np.prod([len(p.values) for p in v.parents]) * (len(v.values) - 1)
