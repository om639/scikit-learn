"""
Common utilities for testing Bayesian network structure learning.
"""
import csv
import numpy as np


def load_recarray(file):
    """Load a properly-formed CSV file ``file`` into a ``numpy.recarray``.

    Parameters
    ----------
    file : string
        The name of the file to load.

    Returns
    -------
    data : np.recarray
    """
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        dtype = [(name, 'O') for name in next(reader)]
        data = [tuple(row) for row in reader]
    return np.rec.array(data, dtype=dtype)
