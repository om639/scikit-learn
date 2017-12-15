"""
Hill-climbing structure learning for Bayesian networks.
"""


def maximize_addition(network, score):
    """Find the edge addition that will result in the largest score increase in
    the specified network.

    Parameters
    ----------
    network : ``Network``
        The network to try adding an edge to.

    score : float
        The current score for the network.

    Returns
    -------
    (from, to) : tuple of (int, int)
        The indices of the variables from and to which adding an edge maximizes
        the score increase.

    delta : float
        The increase in score resulting from adding the edge.
    """
    pass


def maximize_removal(network, score):
    """Find the edge removal that will result in the largest score increase in
    the specified network.

    Parameters
    ----------
    network : ``Network``
        The network to try removing an edge from.

    score : float
        The current score for the network.

    Returns
    -------
    (from, to) : tuple of (int, int)
        The indices of the variables from and to which removing an edge
        maximizes the score increase.

    delta : float
        The increase in score resulting from removing the edge.
    """
    pass


def maximize_reversal(network, score):
    """Find the edge reversal that will result in the largest score increase in
    the specified network.

    Parameters
    ----------
    network : ``Network``
        The network to try reversing an edge in.

    score : float
        The current score for the network.

    Returns
    -------
    (from, to) : tuple of (int, int)
        The indices of the variables from and to which reversing an edge
        maximizes the score increase.

    delta : float
        The increase in score resulting from reversing the edge.
    """
    pass
