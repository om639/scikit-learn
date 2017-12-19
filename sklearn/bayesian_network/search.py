"""
Hill-climbing structure learning for Bayesian networks.
"""
import numpy as np

from sklearn.bayesian_network.score import bic


def maximize_addition(network, data, scores):
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
    for b, variable in enumerate(network.variables):
        for a in variable.not_parent_indices:
            # Make sure new edge would not cause cycle
            # Note this check will also prevent a loop
            if not network.causes_cycle(a, b):
                # Calculate the score as if an edge has been added from a to b
                parents = np.append(variable.parent_indices, a)
                delta = bic(variable, data, parents=parents) - scores[b]

                # Check if best solution
                if delta > max_delta:
                    max_edge = (a, b)
                    max_delta = delta
    return max_delta, max_edge


def maximize_removal(network, data, scores):
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

    Returns
    -------
    delta : float
        The increase in score resulting from removing the edge.

    (from, to) : tuple of (int, int)
        The indices of the variables from and to which removing an edge
        maximizes the score increase.
    """
    for b, variable in enumerate(network.variables):
        for a in variable.parent_indices:
            # Remove edge from a to b
            # TODO: implement me
            pass


def maximize_reversal(network, data, scores):
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

    Returns
    -------
    delta : float
        The increase in score resulting from reversing the edge.

    (from, to) : tuple of (int, int)
        The indices of the variables from and to which reversing an edge
        maximizes the score increase.
    """
    for b, variable in enumerate(network.variables):
        for a in variable.parent_indices:
            # Reverse edge from a to b
            # TODO: implement me
            pass
