"""
Representations of Bayesian networks and random variables.
"""
import numpy as np
from collections import deque
from scipy.sparse import lil_matrix


class Network(object):
    """Represents a Bayesian network.

    The constructor automatically calls the ``attach`` method of each variable
    passed in ``variables``.

    Parameters
    ----------
    variables : iterable of ``Variable``
        The variables that are part of the Bayesian network.
    """

    def __init__(self, variables):
        self._variables = tuple(variables)
        self._variable_indices = {variable.name: i for i, variable in
                                  enumerate(self.variables)}

        # Attach all variables to the network
        for variable in self.variables:
            variable.attach(self)

        n = len(self.variables)

        # Store edges as sparse adjacency matrix
        self._parents = lil_matrix((n, n), dtype=np.int32)

    def add_edge(self, a, b):
        """Add an edge from variable ``a`` to ``b``.

        If there is already an edge from ``a`` to ``b``, this method does
        nothing.

        If the new edge would cause a cycle, it is not added and this method
        raises a ValueError.

        Parameters
        ----------
        a : int
            The index of the variable from which the new edge should be added.

        b : int
            The index of the variable to which the new edge should be added.
        """
        if self._causes_cycle(a, b):
            raise ValueError('new edge would cause a cycle')

        self._parents[b, a] = 1

    def remove_edge(self, a, b):
        """Remove the edge from variable ``a`` to ``b``.

        If there is no edge from ``a`` to ``b``, this method does nothing.

        Parameters
        ----------
        a : int
            The index of the variable from which the edge to be removed
            originates.

        b : int
            The index of the variable to which the edge to be removed points.
        """
        self._parents[b, a] = 0

    def has_edge(self, a, b):
        """Return whether there exists an edge from variable ``a`` to ``b`` in
        the network.

        Parameters
        ----------
        a : int
            The index of the variable from which to check if an edge exists.

        b : int
            The index of the variable to which to check if an edge exists.

        Returns
        -------
        has_edge : bool
            True if there is an edge from ``a`` to ``b``, False otherwise.
        """
        return self._parents[b, a] != 0

    def _causes_cycle(self, a, b):
        """Return whether or not a new edge from variable ``a`` to ``b`` would
        cause a cycle in the network.

        Parameters
        ----------
        a : int
            The index of the variable to test from.

        b : int
            The index of the variable to test to.

        Returns
        -------
        _causes_cycle : bool
            True if the new edge would cause a cycle, False otherwise.
        """
        if a == b:
            return True

        current = deque([a])
        visited = {a}

        # Do BFS to check for path
        while current:
            for i in self.parents(current.popleft()):
                if i == b:
                    return True
                if i not in visited:
                    current.append(i)
                    visited.add(i)

        return False

    @property
    def variables(self):
        """Return the variables that are part of this network.

        Returns
        -------
        variables : tuple of ``Variable``
            The variables that are part of this network.
        """
        return self._variables

    def variable_index(self, name):
        """Return the index of the specified variable in this network.

        Parameters
        ----------
        name : str
            The name of the variable to return the index of.

        Returns
        -------
        variable_index : int
            The index of the specified variable in the network.
        """
        return self._variable_indices[name]

    def parents(self, i):
        """Return the indices of the parent variables of the specified variable
        index.

        Parameters
        ----------
        i : int
            The index of the variable to return the parent indices of.

        Returns
        -------
        parents : ``numpy.array``
            The indices of the parent variables of the specified variable.
        """
        _, parents = self._parents[i].nonzero()
        return parents


class Variable(object):
    """Represents a discrete random variable in a Bayesian network.

    Parameters
    ----------
    name : str
        The name of the random variable. Must be unique within a network.

    values : iterable of str
        The list of possible values that the random variable can take.
    """

    def __init__(self, name, values):
        self._name = name
        self._values = tuple(values)
        self._value_indices = {value: i for i, value in enumerate(self.values)}

        # Network that this variable is attached to
        self._network = None

    def attach(self, network):
        """Attach this variable to a ``Network``.

        This method must be called at least once before the ``index`` and
        ``parents`` properties can be used.

        Parameters
        ----------
        network : ``Network``
            The network to attach this variable to.
        """
        self._network = network

    @property
    def index(self):
        """Return the index of this variable in the network.

        If this variable is not attached to a ``Network``, this property raises
        an AttributeError.

        Returns
        -------
        index : int
            The index of the variable.
        """
        return self._network.variable_index(self.name)

    @property
    def name(self):
        """Return the name of this variable.

        Returns
        -------
        name : str
            The name of the variable.
        """
        return self._name

    @property
    def parents(self):
        """Return the parent variables of this variable in the network.

        If this variable is not attached to a ``Network``, this property raises
        an AttributeError.

        Returns
        -------
        parents : generator of ``Variable``
            The parent variables of the variable.
        """
        return (self._network.variables[i] for i in
                self._network.parents(self.index))

    @property
    def values(self):
        """Return the possible values that this variable can take.

        Returns
        -------
        values : tuple of str
            The possible values that the variable can take.
        """
        return self._values

    def value_index(self, value):
        """Return the index of the specified value of this variable.

        Parameters
        ----------
        value : str
            The value to return the index of.

        Returns
        -------
        value_index : int
            The index of the specified value.
        """
        return self._value_indices[value]

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self._name == other._name
                and self._values == other._values)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.values))
