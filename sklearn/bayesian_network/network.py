"""
Representations of Bayesian networks and random variables.
"""
import numpy as np
from collections import deque
from sklearn.externals.six import string_types


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
        for variable in self._variables:
            variable.attach(self)

        n = len(self._variables)

        # Store edges as adjacency matrix
        self._parents = np.zeros((n, n), dtype=np.bool)

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
        if self.causes_cycle(a, b):
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

    def causes_cycle(self, a, b, reversal=False):
        """Return whether or not a new edge from variable ``a`` to ``b`` would
        cause a cycle in the network.

        Parameters
        ----------
        a : int
            The index of the variable to test from.

        b : int
            The index of the variable to test to.

        reversal : bool
            True to ignore an existing arc from ``b`` to ``a`` when checking for
            cycles. Used to check whether an arc can be reversed.

        Returns
        -------
        causes_cycle : bool
            True if the new edge would cause a cycle, False otherwise.
        """
        if a == b:
            return True

        current = deque([a])
        visited = {a}

        # Do BFS to check for path
        while current:
            v = current.popleft()
            for p in self.parent_indices(v):
                if reversal and v == a and p == b:
                    # If reversing, pretend arc doesn't exist
                    continue
                if p == b:
                    return True
                if p not in visited:
                    current.append(p)
                    visited.add(p)

        return False

    def copy(self):
        """Return a copy of this network. All variables in the network are also
        copied and attached to the new network. Edge operations on the new
        network will have no effect on the original network.

        Returns
        -------
        copy : ``Network``
            A copy of this network.
        """
        n = Network([Variable(v.name, v.values) for v in self._variables])
        n._parents = self._parents.copy()
        return n

    @property
    def dimension(self):
        """Return the sum of the dimensions of all variables in this network.

        Returns
        -------
        dimension : int
            The sum of dimensions for all variables in the network.
        """
        return sum(variable.dimension for variable in self.variables)

    @property
    def parents(self):
        """Return a copy of the parent adjacency matrix of this network.


        Returns
        -------
        parents : ``numpy.array``
            A copy of the parent adjacency matrix of this network.
        """
        # Return a copy to prevent user from directly modifying parent matrix
        return self._parents.copy()

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

    def parent_indices(self, i):
        """Return the indices of the parent variables of the specified variable
        index.

        Parameters
        ----------
        i : int
            The index of the variable to return the parent indices of.

        Returns
        -------
        parent_indices : ``numpy.array``
            The indices of the parent variables of the specified variable.
        """
        parents, = np.nonzero(self._parents[i])
        return parents

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and len(self._variables) == len(other._variables)
                and all(v == w for v, w in zip(self._variables,
                                               other._variables))
                and np.array_equal(self._parents, other._parents))

    def __getitem__(self, item):
        if isinstance(item, string_types):
            return self._variables[self._variable_indices[item]]
        return self._variables[item]

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __str__(self):
        """Return a string representation of the structure of this network. The
        resulting string representation can then be used with bnlearn.

        The string returned by this method will be in the format of
        ``[A][B|A][C|A:B]``, where ``A``, ``B`` and ``C`` are the names of
        variables, ``A`` is a parent of ``B``, and both ``A`` and ``B`` are
        parents of ``C`` (``A`` itself does not have any parents).

        Returns
        -------
        __str__ : str
            A string representation of the structure of this network.
        """
        return ''.join(str(v) for v in self.variables)


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

    def dimension(self, parents=None):
        """Return the dimension of this variable.

        If this variable is not attached to a ``Network``, this property raises
        an AttributeError.

        Parameters
        ----------
        parents : ``numpy.array``
            The indices of the parent variables to use for the variable. If
            None, the variable's parent indices from the network are used.

        Returns
        -------
        dimension : int
            The dimension of the variable.
        """
        if parents is None:
            parents = self.parent_indices

        a = np.array([len(self._network[i].values) for i in parents])
        return np.prod(a) * (len(self.values) - 1)

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
        return (self._network[i] for i in self.parent_indices)

    @property
    def parent_indices(self):
        """Return the indices of the parent variables of this variable in the
        network.

        If this variable is not attached to a ``Network``, this property raises
        an AttributeError.

        Returns
        -------
        parent_indices : ``numpy.array``
            The indices of the parent variables of the variable.
        """
        return self._network.parent_indices(self.index)

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

    def __hash__(self):
        return hash((self.name, self.values))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        """Return a string representation of this variable, including the names
        of its parents.

        Returns
        -------
        __str__ : str
            A string representation of this variable.
        """
        if len(self.parent_indices) == 0:
            return '[{}]'.format(self._name)

        return '[{}|{}]'.format(self._name,
                                ':'.join(p.name for p in self.parents))
