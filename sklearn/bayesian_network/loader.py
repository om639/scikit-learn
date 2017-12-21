"""
Utilities for loading data used in Bayesian network learning.
"""
import csv
import numpy as np


def load_discrete(file, network):
    """Load a properly-formed CSV file ``file`` into a ``numpy.array`` using the
    variable order and value indices of ``network``.

    The returned ``numpy.array`` will contain N rows and M columns, where N is
    the number of data rows in the CSV file and M is the number of variables
    in ``network``.

    The data items in each row in the returned array will be ordered according
    to the ordering of the variables in ``network``. Additionally, each row in
    the CSV file will have its values in the array replaced by the corresponding
    value index as given by ``Variable.value_index`` for the corresponding
    variable.

    The input CSV file must contain a header row whose names correspond to the
    names of the variables in ``network`` (although they do not necessarily have
    to be in the same order).

    Parameters
    ----------
    file : str
        The name of the file to load.

    network : ``Network``
        The network defining the variables and their ordering, which in turn
        define the values to use in the array for each possible input.

    Returns
    -------
    load_discrete : ``numpy.array``
        The data as loaded from the CSV file.
    """

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [[variable.value_index(row[variable.name])
                 for variable in network] for row in reader]

    return np.array(rows, dtype=np.int32)
