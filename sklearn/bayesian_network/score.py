import math
import numpy as np

from collections import Counter, defaultdict


def bic(bn, data):
    return log_likelihood(bn, data) - 0.5 * math.log(len(data)) * _dim(bn)


def _dim(bn):
    return sum(_dim_rv(rv) for rv in bn.rvs.values())


def _dim_rv(rv):
    return np.prod([len(parent.values) for parent in rv.parents]) * (len(rv.values) - 1)


def log_likelihood(bn, data):
    return sum(_log_likelihood_rv(rv, data) for rv in bn.rvs.values())


def _log_likelihood_rv(rv, data):
    ll = 0
    for count_key, counts in _count_rv(rv, data).items():
        parent_count = sum(counts.values())
        for value, count in counts.items():
            ll += count * math.log(count / parent_count)
    return ll


def _count_rv(rv, data):
    counts = defaultdict(Counter)
    for row in data:
        count_key = tuple(row[parent.name] for parent in rv.parents)
        counts[count_key][row[rv.name]] += 1
    return counts
