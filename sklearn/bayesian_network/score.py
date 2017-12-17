"""
Scoring functions for Bayesian networks.
"""
from __future__ import division

import math
import numpy as np

from collections import Counter


def ll(network, data):
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
    ll : float
        The log-likelihood of ``network`` given ``data``.
    """
    return sum(ll_variable(v, data) for v in network.variables)


def ll_variable(variable, data):
    """Computer the log-likelihood for the given variable with respect to the
    given data. The variable must be attached to some ``Network``.

    Parameters
    ----------
    variable : ``Variable``
        The variable to compute the log-likelihood for.

    data : ``numpy.array``
        The data to use when computing the log-likelihood. The data columns must
        be in the same order as the variables in the network.

    Returns
    -------
    ll_variable : float
        The log-likelihood of ``variable`` given ``data``.
    """
    """Return the parent configuration counts for a single ``Variable``."""
    # Select only the relevant columns
    ar = data[:, np.insert(variable.parent_indices, 0, variable.index)]
    ar = np.ascontiguousarray(ar)

    # Convert each row into a single value and identify unique values
    av = ar.view(np.dtype((np.void, ar.dtype.itemsize * ar.shape[1])))
    _, indices, counts = np.unique(av, return_index=True, return_counts=True)

    # Count total occurrences of each parent configuration
    parent_counts = Counter()
    for index, count in zip(indices, counts):
        parent_counts[tuple(ar[index, 1:])] += count

    return sum(count * math.log(count / parent_counts[tuple(ar[index, 1:])])
               for index, count in zip(indices, counts))


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
    return ll(network, data) - 0.5 * math.log(len(data)) * network.dimension
