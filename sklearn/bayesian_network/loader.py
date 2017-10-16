"""
Functions for loading discrete data sets.
"""
import csv


def load_discrete(name, variables):
    """Loads discrete data from a csv file.

    Parameters
    ----------
    name : string or unicode
        Name of the csv file to load data from.

    variables : dict
        A dictionary with each key representing the name of a variable,
        and values being iterables containing the possible outcomes for
        the respective variable.

    Returns
    -------


    """
    # Initialize empty data dictionary to return
    data = {key: [] for key in variables}

    with open(name, 'r') as f:
        for row in csv.DictReader(f):
            for key, outcomes in variables.items():
                if not row[key] in outcomes:
                    raise ValueError("invalid outcome '{}' for variable '{}'".format(row[key], key))

                data[key].append(row[key])

    return data
