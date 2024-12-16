"""
Math utility functions. 

Kirby Heck
2024 July 24
"""


import numpy as np
from ..gridslice import Slice, SliceData


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
E_ijk = np.array([[[e_ijk(i, j, k)
                    for k in range(3)]
                   for j in range(3)]
                  for i in range(3)])


def d_ij(i, j): 
    """
    Kronecker delta
    """
    return int(i==j)


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
    
    if hasattr(axis, '__iter__'): 
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
            s[axis] = slice(index, index+1)
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


# =======================================
# === FLUIDS POSTPROCESSING FUNCTIONS ===
# =======================================

def compute_vort(field_dict, in_place=False):
    """
    Computes the vorticity vector w_i = [w_x, w_y, w_z]

    Parameters
    ----------
    sl_dict : Slice()
        Budget object or Slice() from BudgetIO.slice()
    in_place : bool, optional
        Returns result if False. Default False

    Returns
    -------
    w_i : (Nx, Ny, Nz, 3)
        4D Vorticity tensor, if in_place=False
    """

    u_i = assemble_tensor_1d(field_dict, ["ubar", "vbar", "wbar"])
    dxi = field_dict.grid.dxi

    dukdxj = gradient(
        u_i, dxi, axis=(0, 1, 2), stack=-2
    )  # stack along -2 axis so indexing is x, y, z, j, k

    w_i = np.zeros(field_dict.grid.shape + (3,))
    for i in range(3):  # can this be done w/o a loop?
        w_i[..., i] = np.sum(E_ijk[i, ...] * dukdxj, axis=(-2, -1))

    if in_place:
        field_dict["w_i"] = w_i
    else:
        return SliceData(w_i, grid=field_dict.grid, strict_shape=False, name='w_i')


def compute_vort_budget(
    field_dict,
    direction,
    Ro=None,
    lat=45.0,
    fplane=True,
    Fr=None,
    theta0=300.0,
):
    """
    Computes the offline vorticity budget in three component directions.

    All terms are nd arrays [x,y,z,i, ...] for the i-direction of vorticity.

    Parameters
    ----------
    field_dict : Slice()
        from BudgetIO.slice(), expects velocity, temperature, subgrid stresses, and reynolds stresses
    direction : int or list
        Direction (0, 1, 2), or a tuple/list of directions
        (e.g., i=(0, 1)) computes x, y vorticity
    Ro : float
        Rossby number as defined in LES
    lat : float
        Latitude in degrees, default is 45, NOT None.
    fplane : bool
        Use fplane approximation. Default True.
    Fr : float
        Froude number, defined Fr = G/sqrt(g*L_c)
    theta0 : float
        Reference potential temperature, Default 300 [K].

    Returns
    -------
    dict
        Vorticity budget terms
    """

    # if der is None:
    #     der = DerOps(dx=dx)
    dims = field_dict.grid.shape
    dxi = field_dict.grid.dxi
    dirs = np.unique(direction)  # np.unique returns an at least 1D array

    # allocate memory for all the tensors
    adv_ij = np.zeros(dims + (len(dirs), 3))
    str_ij = np.zeros(dims + (len(dirs), 3))
    buoy_ij = np.zeros(dims + (len(dirs), 3))
    sgs_ijkm = np.zeros(dims + (len(dirs), 3, 3, 3))  # 7D, [x, y, z, i, j, k, m]
    if fplane:
        cor_i = np.zeros(dims + (len(dirs),))
    else:
        cor_i = np.zeros(dims + (len(dirs), 3))
    rs_ijkm = np.zeros(dims + (len(dirs), 3, 3, 3))  # also 7D

    # check all required tensors exist:  (may throw KeyError)
    u_i = assemble_tensor_1d(field_dict, keys=["ubar", "vbar", "wbar"])
    w_i = compute_vort(field_dict, in_place=False)
    field_dict["w_i"] = w_i  # save vorticity keys to the budget object:
    uiuj = assemble_tensor_nd(field_dict, rs_keys)
    tau_ij = assemble_tensor_nd(field_dict, tau_keys)
    Tbar = field_dict["Tbar"]

    # compute tensor quantities:
    for ii in dirs:
        # compute coriolis
        if fplane:
            cor_i[:, :, :, ii] = (
                2 / Ro * np.sin(lat) * gradient(u_i[:, :, :, ii], dxi, axis=2)
            )
        else:
            raise NotImplementedError(
                "compute_vort_budget(): fplane = False not implemeneted"
            )

        for jj in range(3):
            # advection (on RHS, flipped sign)
            adv_ij[:, :, :, ii, jj] = -u_i[:, :, :, jj] * gradient(
                w_i[:, :, :, ii], dxi, axis=jj
            )

            # vortex stretching
            str_ij[:, :, :, ii, jj] = w_i[:, :, :, jj] * gradient(
                u_i[:, :, :, ii], dxi, axis=jj
            )

            # buoyancy torque
            if theta0 is not None:
                eijk = e_ijk(ii, jj, 2)  # buoyancy term has k=3
                if eijk == 0:
                    buoy_ij[:, :, :, ii, jj] = 0  # save compute time by skipping these
                else:
                    buoy_ij[:, :, :, ii, jj] = (
                        eijk * gradient(Tbar, axis=jj) / (Fr**2 * theta0),
                        dxi,
                    )

            for kk in range(3):
                # nothing is ijk at the moment, Coriolis w/o trad. approx. is, however

                for mm in range(3):
                    # compute permutation operator
                    eijk = e_ijk(ii, jj, kk)

                    if eijk == 0:
                        sgs_ijkm[:, :, :, ii, jj, kk, mm] = 0
                        rs_ijkm[:, :, :, ii, jj, kk, mm] = 0
                    else:
                        sgs_ijkm[:, :, :, ii, jj, kk, mm] = eijk * gradient(
                            gradient(
                                -tau_ij[:, :, :, kk, mm],
                                dxi,
                                axis=mm,
                            ),
                            dxi,
                            axis=jj,
                        )
                        rs_ijkm[:, :, :, ii, jj, kk, mm] = eijk * gradient(
                            gradient(-uiuj[:, :, :, kk, mm], dxi, axis=mm),
                            dxi,
                            axis=jj,
                        )

    # now sum over extra axes to collapse terms
    ret = {
        "adv_j": adv_ij,
        "str_j": str_ij,
        "buoy_j": buoy_ij,
        "sgs_jkm": sgs_ijkm,
        "cor": cor_i,
        "rs_jkm": rs_ijkm,
    }

    agg = base_aggregate(ret, ndim=len(dims), base_agg=1)  # aggregate along i
    ret["residual"] = sum(agg[key] for key in agg.keys())

    for key in ret.keys():  # collapse extra dims
        ret[key] = np.squeeze(ret[key])

    return Slice(ret, grid=field_dict.grid)


if __name__=='__main__': 
    # run basic tests: 
    assert e_ijk(0, 0, 1) == 0
    assert e_ijk(0, 2, 1) == -1
    
    tmp = np.reshape(np.arange(24), (2,3,4))

    field = {
        str(k): np.ones((3, 4)) * k for k in range(8)
    }
    keys_test = [[['0', '1',], ['2', '3']], [['4', '5'], ['6', '7']]]
    ret = assemble_tensor_nd(field, keys_test)

    print(ret[..., 0, 1, 1])
