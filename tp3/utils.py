import numpy as np

def pad(array : np.array, dim : int):
    """
    Pad the array with zeros to the desired dimension.

    Parameters
    ----------
    array : np.array
        Array to pad.
    dim : int
        Desired dimension.

    Returns
    -------
    np.array
        Padded array.
    """
    if array.shape[0] < dim:
        return np.pad(array, (0, dim - array.shape[0]), 'constant')
    return array