"""
Data manipulation and i/o utils.
"""
import json


def load_json(filepath):
    """
    Load json into dictionary.

    Parameters:
    -----------
    filename : str
    directory : str

    Returns:
    --------
    dict
    """

    with open(filepath, 'r') as f:
        obj = json.load(f)
    return obj
