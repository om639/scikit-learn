"""
Hill-climbing structure learning for Bayesian networks.
"""
from sklearn.bayesian_network.score import score


def maximize_addition(network, data, scores, cache=None):
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


def maximize_removal(network, data, scores, cache=None):
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


def maximize_reversal(network, data, scores, cache=None):
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
    (delta_from, delta_to) : float
        The increase in score resulting from reversing the edge.

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
    return max_delta, max_edge
