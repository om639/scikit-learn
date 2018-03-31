"""
Hill-climbing structure learning for Bayesian networks.
"""
import numpy as np

from operator import itemgetter
from sklearn.bayesian_network.score import score

_OP_ADD = 0
_OP_REMOVE = 1
_OP_REVERSE = 2


# noinspection PyTypeChecker
def hc(network, data, use_cache=True):
    """Perform hill-climbing search on the specified network using the given
    data. The network is modified in-place.

    Parameters
    ----------
    network : ``Network``
        The network to learn the structure for.

    data : ``numpy.array``
        The data to use for learning.

    use_cache : ``bool``
        Whether or not to use a score cache. Using a score cache can result in a
        significant performance increase at the cost of increased memory usage.
        Defaults to ``True``.

    Returns
    -------
    hc : float
        The increase in network score resulting from performing hill-climbing
        search.
    """
    scores = np.array([score(variable, data) for variable in network])
    score_initial = np.sum(scores)

    # Use a cache to speed up computation
    cache = {} if use_cache else None
    while True:
        # Calculate maximum score increase for each operation
        ops = [(_OP_ADD,) + max_add(network, data, scores, cache=cache),
               (_OP_REMOVE,) + max_remove(network, data, scores, cache=cache),
               (_OP_REVERSE,) + max_reverse(network, data, scores, cache=cache)]

        # Get operation that results in the maximum score increase
        op = max(ops, key=itemgetter(1))

        # If no improvement, abort
        if op[1] <= 0:
            break

        # Apply op to network
        if op[0] in (_OP_ADD, _OP_REMOVE):
            edge = op[2]
            scores[edge[1]] += op[1]
            if op[0] == _OP_ADD:
                network.add_edge(*edge)
            else:
                network.remove_edge(*edge)
        elif op[0] == _OP_REVERSE:
            deltas, edge = op[2], op[3]
            scores[edge[0]] += deltas[0]
            scores[edge[1]] += deltas[1]
            network.remove_edge(*edge)
            network.add_edge(*reversed(edge))
    return np.sum(scores) - score_initial


def max_add(network, data, scores, cache=None):
    """Find the edge addition that will result in the largest score increase in
    the specified network.

    Parameters
    ----------
    network : ``Network``
        The network to try adding an edge to.

    data : ``numpy.array``
        The data to score the network against.

    scores : ``numpy.array``
        The current scores for each variable in the network.

    cache : dict of int to (dict of tuple to float)
        The score cache to use. If None, do not use a cache.

    Returns
    -------
    delta : float
        The increase in score resulting from adding the edge.

    (from, to) : tuple of (int, int), or None
        The indices of the variables from and to which adding an edge maximizes
        the score increase. If no edges result in a score increase, None.
    """
    max_delta = 0
    max_edge = None
    for b, variable in enumerate(network):
        for a in variable.not_parent_indices:
            # Make sure new edge would not cause cycle
            # Note this check will also prevent a loop
            if network.causes_cycle(a, b):
                continue

            # Calculate the score as if an edge has been added from a to b
            delta = score(variable, data, parent_include=a,
                          cache=cache) - scores[b]

            # Check if best solution
            if delta > max_delta:
                max_delta = delta
                max_edge = (a, b)
    return max_delta, max_edge


def max_remove(network, data, scores, cache=None):
    """Find the edge removal that will result in the largest score increase in
    the specified network.

    Parameters
    ----------
    network : ``Network``
        The network to try removing an edge from.

    data : ``numpy.array``
        The data to score the network against.

    scores : ``numpy.array``
        The current scores for each variable in the network.

    cache : dict of int to (dict of tuple to float)
        The score cache to use. If None, do not use a cache.

    Returns
    -------
    delta : float
        The increase in score resulting from removing the edge.

    (from, to) : tuple of (int, int), or None
        The indices of the variables from and to which removing an edge
        maximizes the score increase. If no edges result in a score increase,
        None.
    """
    max_delta = 0
    max_edge = None
    for b, variable in enumerate(network):
        for a in variable.parent_indices:
            # Calculate the score as if edge from a to b has been removed
            delta = score(variable, data, parent_exclude=a,
                          cache=cache) - scores[b]

            # Check if best solution
            if delta > max_delta:
                max_delta = delta
                max_edge = (a, b)
    return max_delta, max_edge


def max_reverse(network, data, scores, cache=None):
    """Find the edge reversal that will result in the largest score increase in
    the specified network.

    Parameters
    ----------
    network : ``Network``
        The network to try reversing an edge in.

    data : ``numpy.array``
        The data to score the network against.

    scores : ``numpy.array``
        The current scores for each variable in the network.

    cache : dict of int to (dict of tuple to float)
        The score cache to use. If None, do not use a cache.

    Returns
    -------
    delta : float
        The total increase in score resulting from reversing the edge.

    (delta_from, delta_to) : float
        The increase and decrease in respective scores resulting from reversing
        the edge.

    (from, to) : tuple of (int, int)
        The indices of the variables from and to which reversing an edge
        maximizes the score increase. If no edges result in a score increase,
        None.
    """
    max_delta = (0, 0)
    max_delta_sum = 0
    max_edge = None
    for b, variable in enumerate(network):
        for a in variable.parent_indices:
            if network.causes_cycle(b, a, reversal=True):
                continue
            delta = (score(network[a], data, parent_include=b,
                           cache=cache) - scores[a],
                     score(variable, data, parent_exclude=a,
                           cache=cache) - scores[b])
            delta_sum = sum(delta)

            # Check if best solution
            if delta_sum > max_delta_sum:
                max_delta = delta
                max_delta_sum = delta_sum
                max_edge = (a, b)
    return max_delta_sum, max_delta, max_edge
