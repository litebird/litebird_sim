import numpy as np
from astropy.io import ascii
from .detectors import DetectorInfo


"""
    Generate a random crosstalk matrix
    Input:

    dim := number of detectors

"""


def GenerateXtalk(dim: int):
    M = np.zeros((dim, dim))
    M = M + np.eye(dim)
    for i in range(dim):
        for j in range(dim):
            if j != i:
                rng = np.random.default_rng()
                a = rng.uniform(-10, 0)
                M[i, j] = 1.0e-5 * a
    return M


"""Implement the crosstalk over the TOD
    Input: 
    tods:= List of time ordered data
    X   :=  Crosstalk  matrix

"""


def Compute_Xtalk_TOD(tods: np.ndarray, X: np.ndarray):
    tods_x = X.dot(tods)
    return tods_x


"""
    Function that reads the full crosstalk matrix 
    written on an external file. 
    It returns the crosstalk matrix Xtalk
    and the dictionary d such that
      d[detector_name]=i
    where detector_name is a string specifying the name of the
    detetcor and i is the index of that detetctor in the 
    crosstalk matrix.
"""


def Get_Xtalk_from_file(path_matrix: str, path_dictionary: str):
    Xtalk = np.loadtxt(path_matrix, delimiter=",", unpack=True)
    d = ascii.read(path_dictionary)
    return (Xtalk, d)


"""
    This function takes as input a crosstalk matrix for a given squid,
    its related dictionary d[det.name]=idx, with idx the corresponding index 
    in the matrix file, and a list of detectors wichi can be a subset
    of the set of detectors related by the full crosstalk matrix.
    It returns a smaller matrix with the crosstalk of this subset of detectors.
    Input: 
        matrix       := the full crosstalk matrix
        d            := the associated dictionary
        detector_list:= list of detectors we want to consider
"""


def create_submatrix(
    matrix: np.ndarray, det_dict: dict[str, int], detector_list: list[DetectorInfo]
):
    N2 = np.size(detector_list)
    array = np.zeros(N2)
    dnames = [detector_list[i].name for i in range(N2)]
    for i, key in enumerate(dnames):
        array[i] = int(det_dict[key])
    newmatrix = np.zeros((N2, N2))
    for i in range(N2):
        for j in range(N2):
            newmatrix[i, j] = matrix[int(array[i]), int(array[j])]
    return newmatrix
