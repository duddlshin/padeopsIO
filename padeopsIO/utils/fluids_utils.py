"""
Fluids-specific math operations, etc. 

Kirby Heck
2024 August 01
"""

import numpy as np
import xarray as xr

from . import math_utils as math
from ..gridslice import GridDataset, Slice, SliceData


# some helpful key pairings:
rs_keys = [["uu", "uv", "uw"], ["uv", "vv", "vw"], ["uw", "vw", "ww"]]
tau_keys = [
    ["tau11", "tau12", "tau13"],
    ["tau12", "tau22", "tau23"],
    ["tau13", "tau23", "tau33"],
]
AD_keys = ["xAD", "yAD", "zAD"]


# =======================================
# === FLUIDS POSTPROCESSING FUNCTIONS ===
# =======================================


def compute_vort(ds, in_place=False):
    """see compute_vorticity()"""
    return compute_vorticity(ds, in_place=in_place)


def compute_vorticity(ds, uvw_keys=None, in_place=False):
    """
    Computes the vorticity vector w_i = [w_x, w_y, w_z]

    Parameters
    ----------
    ds : xr.Dataset with keys ['ubar', 'vbar', 'wbar']
        Budget object or Slice() from BudgetIO.slice()
    uvw_keys : list, optional
        List of keys to use for u, v, w. Default ['ubar', 'vbar', 'wbar']
    in_place : bool, optional
        Returns result if False. Default False

    Returns
    -------
    w_i : (Nx, Ny, Nz, 3)
        4D Vorticity tensor, if in_place=False
    """
    u_i = math.assemble_tensor_1d(ds, uvw_keys or ["ubar", "vbar", "wbar"])
    dxi = ds.grid.dxi

    dukdxj = math.gradient(
        u_i, dxi, axis=(0, 1, 2), stack=-2
    )  # stack along -2 axis so indexing is x, y, z, j, k

    w_i = np.zeros(ds.grid.shape + (3,))
    for i in range(3):  # can this be done w/o a loop?
        w_i[..., i] = np.sum(math.E_ijk[i, ...] * dukdxj, axis=(-2, -1))

    if in_place:
        ds["w_i"] = xr.DataArray(w_i, dims=("x", "y", "z", "i"))
    else:
        return xr.DataArray(w_i, dims=("x", "y", "z", "i"))


# ================ BUDGET COMPUTATIONS ================


def compute_RANS(
    ds,
    i,
    Ro=None,
    lat=None,
    galpha=0,
    fplane=True,
    is_stratified=True,
    theta0=0,
    Fr=0.4,
):
    """
    Computes RANS momentum budgets in the i direction.

    NOTE: This code was written before the migration to xarray. It
    would be better to use xarray.Dataset and xarray.DataArray objects
    for all computations. #TODO

    Parameters
    ----------
    ds : xr.Dataset
    i : int
        Direction, either 0, 1, 2 (x, y, z, respectively)
    Ro : float
        Rossby number, defined Ro = G/(\Omega L)
    lat : float
        Latitude, in radians. Default None
    galpha : float, optional
        Geostrophic wind direction, in radians. Default 0
    fplane : bool
        Use f-plane approximation if true, default True
    is_stratified : bool
        Adds buoyancy term to z-momentum equations if True. Default True.
    theta0 : float
        Reference potential temperature in buoyancy equation. Default 0.
    Fr : float
        Froude number Fr = U/sqrt(gL). Default 0.4.
    """

    dims = ds.grid.shape
    dxi = ds.grid.dxi

    # assemble tensors
    u_j = math.assemble_tensor_1d(ds, ["ubar", "vbar", "wbar"])
    uu_ij = math.assemble_tensor_1d(ds, rs_keys[i])
    tau_ij = math.assemble_tensor_1d(ds, tau_keys[i])
    G = [np.cos(galpha), np.sin(galpha), 0]

    # compute gradients and divergence
    duidxj = math.gradient(u_j[..., i], dxi)
    duiujdxj = math.div(uu_ij, dxi)
    sgs_ij = math.div(tau_ij, dxi)
    dpdxi = math.gradient(ds["pbar"], dxi, axis=i)

    ret = dict()
    ret["adv"] = -u_j * duidxj
    ret["prss"] = -dpdxi
    ret["rs"] = -duiujdxj
    ret["sgs"] = -sgs_ij
    try:
        ret["adm"] = ds[AD_keys[i]]
    except KeyError:
        pass

    ret["geo"] = (
        -2 / Ro * np.sin(lat) * math.e_ijk(i, 1 - i, 2) * G[1 - i] * np.ones(dims)
    )
    if fplane:
        ret["cor"] = 2 / Ro * np.sin(lat) * math.e_ijk(i, 1 - i, 2) * (u_j[..., 1 - i])
    else:
        raise NotImplementedError("TODO: add full coriolis term")

    if i == 2 and is_stratified and theta0 is not None:  # stratification term:
        Tbar = ds["Tbar"]
        tref = np.mean(Tbar, (0, 1))  # how buoyancy is defined in PadeOps
        ret["buoy"] = (Tbar - tref) / (theta0 * Fr**2)

    # hotfix - cast to xarray 
    ret_ds = GridDataset(coords=ds.coords).expand_dims(j=(0,1,2))
    for key, val in ret.items(): 
        if val.ndim == 3: 
            ret_ds[key] = xr.DataArray(val, dims=("x", "y", "z"))
        elif val.ndim == 4: 
            ret_ds[key] = xr.DataArray(val, dims=("x", "y", "z", "j"))
    
    # compute residual
    tmp = math.new_aggregation(ret_ds, base_agg=0)
    ret_ds["residual"] = sum(ret_ds[key] for key in tmp)

    return ret_ds


def compute_vort_budget(
    ds,
    direction,
    Ro=None,
    lat=45.0,
    fplane=True,
    Fr=None,
    theta0=300.0,
    aggregate=1, 
):
    """
    Computes the offline vorticity budget in three component directions.

    All terms are nd arrays [x,y,z,i, ...] for the i-direction of vorticity.

    Parameters
    ----------
    field_dict : xr.Dataset
        Dataset expecting velocity, temperature, subgrid stresses, and reynolds stresses
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
    aggregate : int, optional
        Aggregation level for the output. Default 1.

    Returns
    -------
    dict
        Vorticity budget terms
    """

    # if der is None:
    #     der = DerOps(dx=dx)
    dims = ds.grid.shape
    dxi = ds.grid.dxi
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
    u_i = math.assemble_tensor_1d(ds, keys=["ubar", "vbar", "wbar"])
    w_i = compute_vorticity(ds, in_place=False)
    # save vorticity keys to the budget object:
    ds["w_i"] = w_i
    uiuj = math.assemble_tensor_nd(ds, rs_keys)
    tau_ij = math.assemble_tensor_nd(ds, tau_keys)
    Tbar = ds["Tbar"]

    # compute tensor quantities:
    for ii in dirs:
        # compute coriolis
        if fplane:
            cor_i[:, :, :, ii] = (
                2 / Ro * np.sin(lat) * math.gradient(u_i[:, :, :, ii], dxi, axis=2)
            )
        else:
            raise NotImplementedError(
                "compute_vort_budget(): fplane = False not implemeneted"
            )

        for jj in range(3):
            # advection (on RHS, flipped sign)
            adv_ij[:, :, :, ii, jj] = -u_i[:, :, :, jj] * math.gradient(
                w_i[:, :, :, ii], dxi, axis=jj
            )

            # vortex stretching
            str_ij[:, :, :, ii, jj] = w_i[:, :, :, jj] * math.gradient(
                u_i[:, :, :, ii], dxi, axis=jj
            )

            # buoyancy torque
            if theta0 is not None:
                eijk = math.e_ijk(ii, jj, 2)  # buoyancy term has k=3
                if eijk == 0:
                    buoy_ij[:, :, :, ii, jj] = 0  # save compute time by skipping these
                else:
                    buoy_ij[:, :, :, ii, jj] = (
                        eijk * math.gradient(Tbar, axis=jj) / (Fr**2 * theta0),
                        dxi,
                    )

            for kk in range(3):
                # nothing is ijk at the moment, Coriolis w/o trad. approx. is, however

                for mm in range(3):
                    # compute permutation operator
                    eijk = math.e_ijk(ii, jj, kk)

                    if eijk == 0:
                        sgs_ijkm[:, :, :, ii, jj, kk, mm] = 0
                        rs_ijkm[:, :, :, ii, jj, kk, mm] = 0
                    else:
                        sgs_ijkm[:, :, :, ii, jj, kk, mm] = eijk * math.gradient(
                            math.gradient(
                                -tau_ij[:, :, :, kk, mm],
                                dxi,
                                axis=mm,
                            ),
                            dxi,
                            axis=jj,
                        )
                        rs_ijkm[:, :, :, ii, jj, kk, mm] = eijk * math.gradient(
                            math.gradient(-uiuj[:, :, :, kk, mm], dxi, axis=mm),
                            dxi,
                            axis=jj,
                        )

    # hotfix - cast to xarray 
    ret = {
        "adv": adv_ij,
        "str": str_ij,
        "buoy": buoy_ij,
        "sgs": sgs_ijkm,
        "cor": cor_i,
        "rs": rs_ijkm,
    }
    ret_ds = GridDataset(coords=ds.coords).expand_dims(j=(0,1,2), k=(0,1,2), m=(0,1,2))
    for key, val in ret.items(): 
        dims = ["x", "y", "z", "i", "j", "k", "m"][:val.ndim]
        ret_ds[key] = xr.DataArray(val, dims=dims)

    # aggregate down
    print("DEBUG: Aggregating down vorticity budget")
    math.new_aggregation(ret_ds, base_agg=aggregate, in_place=True)
    if len(dirs) == 1:
        ret_ds = ret_ds.sel(i=dirs[0])
    print("Done with voricity tubdget")
    return ret_ds


def compute_residual(ds_budget, in_place=False): 
    """
    Computes the residual of a budget dataset. 

    Parameters
    ----------
    ds_budget : xr.Dataset
        Dataset with budget terms

    Returns
    -------
    xr.DataArray
        Residual of the budget
    """
    # compute residual
    tmp = math.new_aggregation(ds_budget, base_agg=0)
    residual = sum(ds_budget[key] for key in tmp)

    if in_place: 
        ds_budget["residual"] = residual
    else: 
        return residual
