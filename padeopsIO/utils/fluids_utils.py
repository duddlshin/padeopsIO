"""
Fluids-specific math operations, etc. 

Kirby Heck
2024 August 01
"""

from . import math_utils as math


# some helpful key pairings:
rs_keys = [["uu", "uv", "uw"], ["uv", "vv", "vw"], ["uw", "vw", "ww"]]
tau_keys = [
    ["tau11", "tau12", "tau13"],
    ["tau12", "tau22", "tau23"],
    ["tau13", "tau23", "tau33"],
]
AD_keys = ["xAD", "yAD", "zAD"]


def compute_RANS(
    field_dict,
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

    Parameters
    ----------
    field_dict : Slice
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

    dims = field_dict.grid.shape
    dxi = field_dict.grid.dxi
    u_j = math.assemble_tensor_1d(field_dict, ["ubar", "vbar", "wbar"])
    uu_ij = math.assemble_tensor_1d(field_dict, rs_keys[i])
    tau_ij = math.assemble_tensor_1d(field_dict, tau_keys[i])
    # der = DerOps(dx=field_dict.grid.dxi)

    duidxj = math.gradient(u_j[..., i], dxi)
    duiujdxj = math.div(uu_ij, dxi)
    sgs_ij = math.div(tau_ij, dxi)

    G = [np.cos(galpha), np.sin(galpha), 0]
    dpdxi = math.gradient(field_dict["pbar"], dxi, axis=i)

    ret = {}
    ret["adv_j"] = -u_j * duidxj
    ret["prss"] = -dpdxi
    ret["rs_j"] = -duiujdxj
    ret["sgs_ij"] = -sgs_ij
    try:
        ret["adm"] = field_dict[AD_keys[i]]
    except KeyError:
        pass
        # not including ADM

    ret["geo"] = (
        -2 / Ro * np.sin(lat) * math.e_ijk(i, 1 - i, 2) * G[1 - i] * np.ones(dims)
    )
    if fplane:
        ret["cor"] = 2 / Ro * np.sin(lat) * math.e_ijk(i, 1 - i, 2) * (u_j[..., 1 - i])
    else:
        raise NotImplementedError("TODO: add full coriolis term")

    if i == 2 and is_stratified and theta0 is not None:  # stratification term:
        Tbar = field_dict["Tbar"]
        tref = np.mean(Tbar, (0, 1))  # how buoyancy is defined in PadeOps
        ret["buoy"] = (Tbar - tref) / (theta0 * Fr**2)

    # aggregate to compute residual
    tmp = base_aggregate(ret, ndim=len(dims), base_agg=0)
    ret["residual"] = sum(tmp[key] for key in tmp.keys())

    return Slice(ret, grid=field_dict.grid, strict_shape=False)

