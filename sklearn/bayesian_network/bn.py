import itertools

from collections import Counter, defaultdict


class BN(object):
    """
    Represents a Bayesian network.
    """

    # TODO: make comments sklearn style

    def __init__(self):
        """
        Initialize a new BN.
        """
        self.rvs = dict()
        self.parents = dict()

    def add_rv(self, rv):
        """
        Add an RV to the network.
        """
        self.rvs[rv.name] = rv
        self.parents[rv.name] = []
        rv.bn = self

    def remove_rv(self, rv):
        """
        Remove an RV from the network.
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
        """
        Add an edge from ``a`` to ``b``.
        """
        if not a == b and not self.has_edge(a, b):
            self.parents[b.name].append(a)

    def has_edge(self, a, b):
        """
        Return ``True`` if there is an edge from ``a`` to ``b``, and ``False`` otherwise.
        """
        return a in self.parents[b.name]

    def remove_edge(self, a, b):
        """
        Remove an edge from ``a`` to ``b``.
        """
        self.parents[b.name].remove(a)

    def count(self, sample):
        """
        Update the marginal and conditional counts for all RVs in the network with the specified sample.
        """
        for rv in self.rvs:
            rv.count(sample)


class RV(object):
    """
    Represents a random variable in a BN.
    """

    def __init__(self, name, values):
        """
        Initialize a new RV with the specified name and possible values.
        """
        self.name = name
        self.values = values
        self.bn = None
        self.marginal = Counter()
        self.conditional = defaultdict(Counter)

        # Initialize counts to zero
        self.reset()

    def reset(self):
        """
        Reset the marginal and conditional counts for this RV to zero.
        """
        self.values.clear()
        self.conditional.clear()

        for value in self.values:
            self.marginal[value] = 0

        for parent_values in itertools.product(*(parent.values for parent in self.parents)):
            for value in self.values:
                self.conditional[parent_values][value] = 0

    def count(self, sample):
        """
        Update the marginal and conditional counts with the specified sample.
        """
        # Update marginal count
        self.marginal[sample[self.name]] += 1

        # Update conditional count
        self.conditional[tuple(sample[parent.name] for parent in self.parents)][sample[self.name]] += 1

    @property
    def parents(self):
        """
        Return the parent RVs of this RV.
        """
        return self.bn.parents[self.name]

    def __repr__(self):
        """
        Return a string representation of this RV.
        """
        return "RV(name='{}', values={!r})".format(self.name, self.values)
