"""
Scoring functions for Bayesian networks.
"""
from __future__ import division

import math
import numpy as np

from collections import Counter


def ll(variable, data, parents=None):
    """Computer the log-likelihood for the given variable with respect to the
    given data. The variable must be attached to some ``Network``.

    Parameters
    ----------
    variable : ``Variable``
        The variable to compute the log-likelihood for.

    data : ``numpy.array``
        The data to use when computing the log-likelihood. The data columns must
        be in the same order as the variables in the network.

    parents : ``numpy.array``
        The indices of the parent variables to use for the variable. If None,
        the variable's parent indices from the network are used.

    Returns
    -------
    ll : float
        The log-likelihood of ``variable`` given ``data``.
    """
    if parents is None:
        parents = variable.parent_indices

    # Select only the relevant columns
    ar = data[:, np.insert(parents, 0, variable.index)]
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


def ll_network(network, data):
    """Computer the log-likelihood for the given network with respect to the
    given data.

    Parameters
    ----------
    network : ``Network``
        The network to compute the log-likelihood for.

    data : ``numpy.array``
        The data to use when computing the log-likelihood. The data columns must
        be in the same order as the variables in the network.

    Returns
    -------
    ll_network : float
        The log-likelihood of ``network`` given ``data``.
    """
    return sum(ll(v, data) for v in network.variables)


def bic(variable, data, parents=None):
    """Compute the Bayesian Information Criterion (BIC) score for the given
    variable with respect to the given data. The variable must be attached to
    some ``Network``.

    Parameters
    ----------
    variable : ``Variable``
        The variable to compute the score for.

    data : ``numpy.array``
        The data to use when computing the score. The data columns must be in
        the same order as the variables in the network.

    parents : ``numpy.array``
        The indices of the parent variables to use for the variable. If None,
        the variable's parent indices from the network are used.

    Returns
    -------
    bic : float
        The BIC score for ``variable`` given ``data``.
    """
    s = ll(variable, data, parents=parents)
    return s - 0.5 * math.log(len(data)) * variable.dimension


def bic_network(network, data):
    """Compute the Bayesian Information Criterion (BIC) score for the given
    network with respect to the given data.

    Parameters
    ----------
    network : ``Network``
        The network to compute the score for.

    data : ``numpy.array``
        The data to use when computing the score. The data columns must be in
        the same order as the variables in the network.

    Returns
    -------
    bic_network : float
        The BIC score for ``network`` given ``data``.
    """
    return sum(bic(variable, data) for variable in network.variables)
