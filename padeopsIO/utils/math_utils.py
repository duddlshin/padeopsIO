"""
Math utility functions. 

Kirby Heck
2024 July 24
"""

import numpy as np
from ..gridslice import GridDataset


def assemble_tensor_1d(field_dict, keys, axis=-1):
    """
    Stacks keys in a 1d tensor in the order they are given.

    Parameters
    ----------
    field_dict : dict
        Dictionary of fields (e.g. {'ubar': <np.ndarray>, 'vbar': <np.ndarray>, 'wbar': <np.ndarray>})
    keys : list
        (Length-3) list of keys
    axis : int, optional
        Axis to stack. Default: -1

    Returns
    -------
    np.ndarray : (Nx, Ny, Nz, len(keys))
    """

    return np.stack([field_dict[key] for key in keys], axis=axis)


def assemble_tensor_nd(field_dict, keys):
    """
    Assembles an n-dimensional tensor from a dictionary of fields.

    Parameters
    ----------
    field_dict : dict
        Dictionary of fields
        (e.g. {'ubar': <np.ndarray>, 'vbar': <np.ndarray>, 'wbar': <np.ndarray>})
    keys : list of lists
        (Length-3) list of lists (of lists of lists...) of keys
    basedim : int, optional
        Number of base dimensions, default 3 (x, y, z).

    Returns
    -------
    np.ndarray
    """

    # keys is a list or a list of nested lists
    key_ls = list(field_dict.keys())
    try:
        ndim = field_dict[key_ls[0]].ndim
    except AttributeError:
        raise  # ???

    def _assemble_nd(keys):
        """Recursive helper function"""
        if isinstance(keys[0], str):  # TODO: what if the key is not a string?
            # base case, prepend stacks
            return assemble_tensor_1d(field_dict, keys, axis=-1)
        else:
            # recursive call: return stack [of stacks], also prepended
            return np.stack([_assemble_nd(key) for key in keys], axis=ndim)

    # use the recursive calls, which prepends each added index (e.g. [x, y, z, i, j, ...])
    tensors_rev = _assemble_nd(keys)

    return tensors_rev


def new_aggregation(ds, base_agg=0, in_place=False, **kwargs):
    """
    Aggregates all terms into dictionary by summing over extra axes

    Parameters
    ----------
    terms : dict
        Dictionary of np.ndarray
    base_agg : int, optional
        Base aggregation level. Default 0
    in_place : bool, optional
        If True, aggregates in-place. Default False.
    kwargs : dict, optional
        Additional levels of (dis)aggregation for keys `newkey`

    Returns
    -------
    xr.Dataset or None
        Dictionary of aggregated values, only if `ret` is None.
    """
    if base_agg < 0:
        raise ValueError("new_aggregation(): base_agg must be >= 0")

    ret = GridDataset(coords=ds.coords) if not in_place else ds

    for key in ds.data_vars.keys():
        if key in kwargs.keys() and not isinstance(kwargs[key], int):
            # assume these are dimensions to aggregate over
            dims = kwargs[key]
        else:
            if key in kwargs.keys():
                agg = kwargs[key]
            else:
                agg = base_agg

            # pick dimensions to sum over
            # e.g., for base_agg=0, sum over all free indices
            free_indices = [c for c in ds[key].coords if c not in ["x", "y", "z"]]
            free_indices.sort()
            dims = free_indices[agg:]

        ret[key] = ds[key].sum(dims)

    return ret


# ==================== index notation help ==========================


def e_ijk(i, j, k):
    """
    Permutation operator, takes i, j, k in [0, 1, 2]

    returns +1 if even permutation, -1 if odd permutation

    TODO: This is not elegant
    """

    if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        return 1
    elif (i, j, k) in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]:
        return -1
    else:
        return 0


# also E_ijk in tensor form:
E_ijk = np.array(
    [[[e_ijk(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)]
)


def d_ij(i, j):
    """
    Kronecker delta
    """
    return int(i == j)


def gradient(f, dx, axis=(0, 1, 2), stack=-1, edge_order=2, **kwargs):
    """
    Compute gradients of f using numpy.gradient

    Parameters
    ----------
    f : ndarray
    dx : float or tuple
        Float (dx) or vector (dx, dy, dz)
    axis : int or tuple
        Axes to compute gradients over. Default is (0, 1, 2) for x, y, z
    stack : int
        Which axis to stack gradients before returning a tensor.
        Default is -1
    edge_order : int
    """

    if hasattr(axis, "__iter__"):
        args = [dx[k] for k in axis]
    else:
        args = [dx[axis]]

    dfdxi = np.gradient(f, *args, axis=axis, edge_order=edge_order, **kwargs)

    if len(args) > 1:
        return np.stack(dfdxi, axis=stack)
    return dfdxi


def div(f, dxi, axis=-1, sum=False, **kwargs):
    """
    Computes the 3D divergence of vector or tensor field f: dfi/dxi

    Parameters
    ----------
    f : (Nx, Ny, Nz, 3) or (Nx, Ny, Nz, ...) array
        Vector or tensor field f
    dxi : tuple
        Vector (dx, dy, dz)
    axis : int, optional
        Axis to compute divergence, default -1
        (Requires that f.shape[axis] = 3)
    sum : bool, optional
        if True, performs implicit summation over repeated indices.
        Default False

    Returns
    -------
    dfi/dxi : f.shape array (if sum=True) OR drops the `axis`
        axis if sum=False
    """

    res = np.zeros(f.shape)

    def get_slice(ndim, axis, index):
        """
        Helper function to slice axis `axis` from ndarray
        """
        s = [slice(None) for i in range(ndim)]
        s[axis] = slice(index, index + 1)
        return tuple(s)

    # compute partial derivatives:
    for i in range(3):
        s = get_slice(f.ndim, axis, i)

        # TODO fix gradient functions everywhere
        res[s] = np.gradient(f[s], dxi[i], axis=i, **kwargs)

    if sum:
        return np.sum(res, axis=axis)
    else:
        return res


if __name__ == "__main__":
    # run basic tests:
    assert e_ijk(0, 0, 1) == 0
    assert e_ijk(0, 2, 1) == -1

    tmp = np.reshape(np.arange(24), (2, 3, 4))

    field = {str(k): np.ones((3, 4)) * k for k in range(8)}
    keys_test = [
        [
            [
                "0",
                "1",
            ],
            ["2", "3"],
        ],
        [["4", "5"], ["6", "7"]],
    ]
    ret = assemble_tensor_nd(field, keys_test)

    print(ret[..., 0, 1, 1])
