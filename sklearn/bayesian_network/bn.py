"""
Representations of Bayesian networks and random variables.
"""


class BN(object):
    """Represents a Bayesian network.

    Attributes
    ----------
    rvs : dict of string to ``RV``
        Maps the name of a random variable names to the corresponding ``RV``.

    parents: dict of string to list of ``RV``
        Maps the name of a random variable to a list of its parent ``RV``s.
    """

    def __init__(self):
        self.rvs = dict()
        self.parents = dict()

    def add_rv(self, rv):
        """Add a random variable to the Bayesian network.

        Sets ``rv.bn`` to be equal to this ``BN``.

        Parameters
        ----------
        rv : ``RV``
            The random variable to be added to the network. The ``rv.name``
            attribute must be unique within the network, otherwise this method
            will replace the existing RV with the same name with the new one.
        """
        self.rvs[rv.name] = rv
        self.parents[rv.name] = []
        rv.bn = self

    def remove_rv(self, rv):
        """Remove a random variable from the network.

        If ``rv`` is a parent to any other RVs, it is automatically removed
        from their respective parent lists.

        Parameters
        ----------
        rv : ``RV``
            The random variable to be removed from the network. If ``rv`` is
            not part of the network, this method does nothing.
        """
        del self.rvs[rv.name]
        del self.parents[rv.name]

        # Remove from any child node's parent list
        for parent in self.parents.values():
            try:
                parent.remove(rv)
            except ValueError:
                pass

    def add_edge(self, a, b):
        """Add an edge from RV ``a`` to RV ``b``.

        If there is already an edge from ``a`` to ``b``, this method returns
        ``false``.

        If the new edge would cause a cycle, it is not added and this method
        returns ``false``.

        Parameters
        ----------
        a : ``RV``
            The random variable from which the new edge should be added.

        b : ``RV``
            The random variable to which the new edge should be added.

        Returns
        -------
        ``true`` if the new edge was successfully added, ``false`` otherwise.
        """
        # TODO: ensure new edge does not make cycle
        if not a == b and not self.has_edge(a, b):
            self.parents[b.name].append(a)
            return True
        return False

    def has_edge(self, a, b):
        """Return whether or not there exists an edge from RV ``a`` to RV ``b``
        in the network.

        Parameters
        ----------
        a : ``RV``
            The random variable from which to check if an edge exists.

        b : ``RV``
            The random variable to which to check if an edge exists.

        Returns
        -------
        ``true`` if there is an edge from ``a`` to ``b``, false otherwise.
        """
        return a in self.parents[b.name]

    def remove_edge(self, a, b):
        """Remove the edge from RV ``a`` to RV ``b``.

        If there is no edge from ``a`` to ``b``, this method does nothing.

        Parameters
        ----------
        a : ``RV``
            The random variable from which the edge to be removed originates.

        b : ``RV``
            The random variable to which the edge to be removed points.
        """
        self.parents[b.name].remove(a)


class RV(object):
    """Represents a random variable in a Bayesian network.

    Parameters
    ----------
    name : string
        The name of the random variable. Must be unique within a network.

    values : list of string
        The list of possible values that the random variable can take.

    Attributes
    ----------
    name : string
        The name of the random variable.

    values : list of string
        The list of possible values that the random variable can take.

    bn : ``BN``
        The ``BN`` that this ``RV`` is a part of.
    """

    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.bn = None

    @property
    def parents(self):
        """Return the parent RVs of this ``RV``.

        If this ``RV`` is not part of a ``BN``, a ``ValueError`` is raised.

        Returns
        -------
        parents : list of ``RV``
            The parent RVs of this ``RV``.
        """
        try:
            return self.bn.parents[self.name]
        except AttributeError:
            raise ValueError('RV not part of BN')

    def __repr__(self):
        """Return a string representation of this ``RV``.

        Returns
        -------
        repr : string
            A string representation of this ``RV``.
        """
        return "RV(name='{}', values={!r})".format(self.name, self.values)
