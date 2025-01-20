# Additional wake functions (some work-in-progress)
# this is crude but it works
# USAGE: from wake_utils import *

import numpy as np
import padeopsIO
from numpy.linalg import lstsq


def wake_centroid_2d(u_hub=None, u_wake_hub=None, y=None, thresh=0):
    """
    Computes the 2D wake centroid from the center of mass method. Assumes that u_wake
    is a positive quantity inside the wake such that u_wake_hub = u_inf - u_hub.

    Parameters
    ----------
    u_hub (nx * ny) : 2D slice at hub height (ASSUMES u_inf = 1; uniform inflow)
    u_wake_hub (nx * ny) : 2D slice of the wake field, u_inf-u_hub > 0
    y (ny * 1) : coordinate axis in y
    thresh : (optional) thresholding parameter on u_wake_hub, default 0.

    Returns
    -------
    yc (nx * 1) : wake centerline at hub plane
    """

    if u_wake_hub is None:
        del_u = 1 - u_hub  # assume freestream is normalized to 1
    else:
        del_u = u_wake_hub.copy()

    del_u[del_u < thresh] = 0

    numer = np.trapz(del_u * y, axis=1)
    denom = np.trapz(del_u, axis=1)

    return numer / denom


def wake_centroid_3d(u=None, u_wake=None, y=None, z=None, thresh=0):
    """
    Computes the 3D wake centroid from the center of mass method. Assumes that u_wake
    is a positive quantity inside the wake such that u_wake = u_inf - u.

    Parameters
    ----------
    u (nx * ny * nz) : 3D u-velocity field normalized to geostrophic wind. Assumes u_inf = 1
        (use for uniform inflow ONLY)
    u_wake (nx * ny * nz) : 3D wake field with background subtracted already. Supercedes
        `u` if included.
    y (ny * 1) : (optional) y-coordinate axis
    z (nz * 1) : (optional) z-coordinate axis
    thresh : (optional) thresholding parameter on u_wake_hub, default 0.

    Returns
    -------
    yc (nx * 1) : y-wake center (returned only if `y` is given)
    zc (nx * 1) : z-wake center (returned only if `z` is given)
    """

    if y is None and z is None:
        raise ValueError("Either y= or z= must be included. ")

    if u_wake is None:
        del_u = 1 - u  # assume freestream is normalized to 1
    else:
        del_u = u_wake.copy()

    # apply thresholding
    del_u[del_u < thresh] = 0

    ret = ()
    if y is not None:
        y = y[np.newaxis, :, np.newaxis]
        numer = np.trapz(np.trapz(del_u * y, axis=1), axis=1)
        denom = np.trapz(np.trapz(del_u, axis=1), axis=1)
        ret += (numer / denom,)

    if z is not None:
        z = z[np.newaxis, np.newaxis, :]
        numer = np.trapz(np.trapz(del_u * z, axis=1), axis=1)
        denom = np.trapz(np.trapz(del_u, axis=1), axis=1)
        ret += (numer / denom,)

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def usq_mean(case, diam=1, xlim=None):
    """
    Returns the mean squared velocity within the wake region

    Parameters
    ----------
    case : BudgetIO object
    diam : threshold for (square) averaging wake region; constant in x
    xlim :

    Returns
    -------
    ubar_sq : [nx] array of mean squared velocity values
    xlim (opt) : [nx] array xLine associated with the given xlim
    """

    sl = case.slice(
        budget_terms=["ubar"],
        xlim=xlim,
        ylim=(case.Ly / 2 - diam, case.Ly / 2 + diam),
        zlim=(case.Lz / 2 - diam, case.Lz / 2 + diam),
    )
    ubar_sq = np.mean(np.mean(sl["ubar"] ** 2, axis=1), axis=1)

    if xlim is not None:
        return ubar_sq, sl["x"]
    else:
        return ubar_sq


# ================= NUMERICS =================


# 2D partial derivatives for x, y
def partialx_2d(field, dx):
    """
    2nd order central differencing in x
    """
    dfdx = np.zeros(field.shape)
    dfdx[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)  # central differencing
    dfdx[0, :] = (-3 * field[0, :] + 4 * field[1, :] - field[2, :]) / (
        2 * dx
    )  # forward differencing
    dfdx[-1, :] = (3 * field[-1, :] - 4 * field[-2, :] + field[-3, :]) / (
        2 * dx
    )  # backward differencing

    return dfdx


def partialy_2d(field, dy):
    """
    2nd order central differencing in y
    """
    dfdy = np.zeros(field.shape)
    dfdy[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dy)  # central differencing
    dfdy[:, 0] = (-3 * field[:, 0] + 4 * field[:, 1] - field[:, 2]) / (
        2 * dy
    )  # forward differencing
    dfdy[:, -1] = (3 * field[:, -1] - 4 * field[:, -2] + field[:, -3]) / (
        2 * dy
    )  # backward differencing

    return dfdy


# 3D partial derivatives for x, y, z
def partialx(field, dx):
    """
    2nd order central differencing in x
    """
    dfdx = np.zeros(field.shape)
    dfdx[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (
        2 * dx
    )  # central differencing
    dfdx[0, :, :] = (-3 * field[0, :, :] + 4 * field[1, :, :] - field[2, :, :]) / (
        2 * dx
    )  # forward differencing
    dfdx[-1, :, :] = (3 * field[-1, :, :] - 4 * field[-2, :, :] + field[-3, :, :]) / (
        2 * dx
    )  # backward differencing

    return dfdx


def partialy(field, dy):
    """
    2nd order central differencing in y
    """
    dfdy = np.zeros(field.shape)
    dfdy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (
        2 * dy
    )  # central differencing
    dfdy[:, 0, :] = (-3 * field[:, 0, :] + 4 * field[:, 1, :] - field[:, 2, :]) / (
        2 * dy
    )  # forward differencing
    dfdy[:, -1, :] = (3 * field[:, -1, :] - 4 * field[:, -2, :] + field[:, -3, :]) / (
        2 * dy
    )  # backward differencing

    return dfdy


def partialz(field, dz):
    """
    2nd order central differencing in z
    """
    dfdz = np.zeros(field.shape)
    dfdz[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (
        2 * dz
    )  # central differencing
    dfdz[:, :, 0] = (-3 * field[:, :, 0] + 4 * field[:, :, 1] - field[:, :, 2]) / (
        2 * dz
    )  # forward differencing
    dfdz[:, :, -1] = (3 * field[:, :, -1] - 4 * field[:, :, -2] + field[:, :, -3]) / (
        2 * dz
    )  # backward differencing

    return dfdz


def partialr(fieldyz, dy, dz, theta):
    """
    Partial derivates in the radial direction in cylindrical coordinates.
    """
    dim = len(fieldyz.shape)
    if dim == 2:
        fieldyz = fieldyz[None, :, :]  # append x-axis

    ddy = partialy(fieldyz, dy)
    ddz = partialz(fieldyz, dz)

    ddr = ddy * np.cos(theta) + ddz * np.sin(theta)
    return np.squeeze(ddr)


def partialt(fieldyz, dy, dz, theta):
    """
    Partial derivates of a scalar field variable in the theta direction.

    This function actually computes 1/r (d/dtheta)
    """
    dim = len(fieldyz.shape)
    if dim == 2:
        fieldyz = fieldyz[None, :, :]  # append x-axis

    ddy = partialy(fieldyz, dy)
    ddz = partialz(fieldyz, dz)

    ddt = -ddy * np.sin(theta) + ddz * np.cos(theta)
    return np.squeeze(ddt)


# second derivatives in x, y, z
def partialx2(field, dx):
    """
    second derivative in x
    """
    dfdx = np.zeros(field.shape)
    dfdx[1:-1, :, :] = (field[2:, :, :] - 2 * field[1:-1, :, :] + field[:2, :, :]) / (
        dx**2
    )  # central differencing
    dfdx[0, :, :] = dfdx[1, :, :]  # for now, cheat at the boundaries...
    dfdx[-1, :, :] = dfdx[-2, :, :]  # for now, cheat at the boundaries...
    return dfdx


def partialy2(field, dy):
    """
    second derivative in y
    """
    dfdy = np.zeros(field.shape)
    dfdy[:, 1:-1, :] = (field[:, 2:, :] - 2 * field[:, 1:-1:, :] + field[:, :-2, :]) / (
        dy**2
    )  # central differencing
    dfdy[:, 0, :] = dfdy[:, 1, :]
    dfdy[:, -1, :] = dfdy[:, -2, :]
    return dfdy


def partialz2(field, dz):
    """
    second derivative in z
    """
    dfdz = np.zeros(field.shape)
    dfdz[:, :, 1:-1] = (field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]) / (
        dz**2
    )  # central differencing
    dfdz[:, :, 0] = dfdz[:, :, 1]
    dfdz[:, :, -1] = dfdz[:, :, -2]

    return dfdz


# Numpy partial derivative functions and index notation help:
def e_ijk(i, j, k):
    """
    Permutation operator, takes i, j, k in [1, 2, 3]

    returns +1 if even permutation, -1 if odd permutation

    TODO: This is not elegant
    """

    if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        return 1
    elif (i, j, k) in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]:
        return -1
    else:
        return 0


def d_ij(i, j):
    """
    Kronecker delta
    """
    return int(i == j)


def ddxi(f, i, dxi=1, edge_order=2):
    """
    Computes partials for a field on an evenly spaced grid in the xi
    direction along the i axis (expects 0, 1, 2).

    Parameters
    ----------
    f (array-like) : field with at least i dimensions
    i (int) : integer to differentiate along  (subtracts 1 for python)
    dxi (float) : normalization factor, default 1
    edge_order : optional, float
        Order of differentiation along edges

    Returns df/dxi
    """

    return np.gradient(f, axis=i, edge_order=edge_order) / dxi


def div(f, dx=(1, 1, 1), edge_order=2, axis=-1, sum=True):
    """
    Computes the 3D divergence of vector or tensor field f: dfi/dxi

    Parameters
    ----------
    f : (Nx, Ny, Nz, 3) or (Nx, Ny, Nz, ...) array
        Vector or tensor field f
    dx : (3, ) array-like
        discretization in 3 dimensions.
    edge_order : optional, int
        Order of differentiation, default 2.
    axis : optional, int
        Axis to compute divergence, default -1
        (Requires that f.shape[axis] = 3)
    sum : bool
        if True, performs implicit summation over repeated indices.
        Default True

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

        res[s] = ddxi(f[s], i, dx[i], edge_order=edge_order)

    if sum:
        return np.sum(res, axis=axis)
    else:
        return res


def fit_linear(x, y):
    """
    Fits a line to the data (x, y) using numpy.linalg.lstsq.

    Returns
    -------
    (a0, a1) : y-intercept (a0) and slope (a1) of the resulting curve
    """

    A = np.array([np.ones(x.shape), x]).T
    b = y

    ret = lstsq(A, b, rcond=None)

    return ret[0]


# ================= Fluid tensor functions ===================


def compute_duidxj(sl_dict, ui=None, save_ui=True, in_place=True):
    """
    Computes partial derivatives in x, y, and z for keys 'ubar', 'vbar', 'wbar'. Assumes the the
    slice is 3D.

    Parameters
    ----------
    sl_dict : dict
        dictionary from BudgetIO.slice()
    ui : (Nx, Ny, Nz, 3)
        4D velocity tensor (could be, e.g. velocity deficits)
    in_place : bool, optional
        If False, returns the computed quantities instead of appending them in the same dictionary.
        Default True

    Returns
    -------
    duidxj : array-like (if in_place is False)
        5D duidxj tensor, indexed [x, y, z, i, j]
    """

    if "duidxj" in sl_dict.keys():
        if in_place:
            return
        else:
            return sl_dict["duidxj"]

    if save_ui:
        if "ui" not in sl_dict.keys():
            assemble_u_tensor(sl_dict, in_place=True)
        ui = sl_dict["ui"]
    else:
        ui = assemble_u_tensor(sl_dict, in_place=False)

    x = sl_dict["x"]
    y = sl_dict["y"]
    z = sl_dict["z"]
    dx = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]

    ret = np.zeros((len(x), len(y), len(z), 3, 3))
    for i in range(3):
        for j in range(3):
            ret[:, :, :, i, j] = ddxi(ui[:, :, :, i], j, dxi=dx[j])

    if in_place:
        sl_dict["duidxj"] = ret
    else:
        return ret


def compute_duiujdxj(sl_dict, in_place=True, edge_order=2, sum=False):
    """
    Computes the Reynolds stress divergence tensor d<uiuj>/dxj

    Parameters
    ----------
    sl_dict : dict
        dictionary from BudgetIO.slice().
        Expects keys: ['uu', 'uv', 'uw', 'vv', 'vw', 'ww]
    in_place : optional, bool
        If False, returns the computed quantities instead of appending them in the same dictionary.
        Default True
    edge_order : optional, int
        Order of differentiation, default 2.
    sum : optional, bool
        If True, combines the terms within the implicit sum over j.
        Default False

    Returns
    -------
    d<uiuj>dxj : array-like (if in_place is False)
        5D d<uiuj>dxj tensor, indexed [x, y, z, i, j] if sum=False
        4D tensor summed over j, indexed [x, y, z, i] if sum=True
    """

    if "duiujdxj" in sl_dict.keys():
        if in_place:
            return
        else:
            return sl_dict["duiujdxj"]

    # assemble Reynolds stress tensor
    if not "uiuj" in sl_dict.keys():
        assemble_rs_tensor(sl_dict, in_place=True)

    x = sl_dict["x"]
    y = sl_dict["y"]
    z = sl_dict["z"]
    dx = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]

    duiujdxj = div(sl_dict["uiuj"], dx=dx, axis=-1, sum=sum, edge_order=edge_order)

    if in_place:
        sl_dict["duiujdxj"] = duiujdxj
    else:
        return duiujdxj


def assemble_sgs_tensor(sl_dict, in_place=True):
    """
    Reorganizes the SGS tensor terms 'tau11' etc. into a 5D tensor [x, y, z, i, j] from a 3D slice.

    Parameters
    ----------
    sl_dict : dict
        dictionary from BudgetIO.slice()
    in_place : optional, bool
        If False, returns the computed quantities instead of appending them in the same dictionary.
        Default True

    Returns
    -------
    tau_ij : array-like (if in_place is False)
        5D tau_ij tensor, indexed [x, y, z, i, j]
    """
    nx = len(sl_dict["x"])
    ny = len(sl_dict["y"])
    nz = len(sl_dict["z"])

    tau_ij = np.zeros((nx, ny, nz, 3, 3))  # subgrid stresses
    terms_sgs = [  # matching from budgetkey.py
        ["tau11", "tau12", "tau13"],
        ["tau12", "tau22", "tau23"],
        ["tau13", "tau23", "tau33"],
    ]

    # rearrange tensor
    for ii in range(3):
        for jj in range(3):
            tau_ij[:, :, :, ii, jj] = sl_dict[terms_sgs[ii][jj]]

    if in_place:
        sl_dict["tau_ij"] = tau_ij
    else:
        return tau_ij


def assemble_rs_tensor(sl_dict, in_place=True):
    """
    Reorganizes the reynolds stress tensor terms 'uu' etc. into a 5D tensor [x, y, z, i, j] from a 3D slice.

    Parameters
    ----------
    sl_dict : dict
        dictionary from BudgetIO.slice()
    in_place : optional, bool
        If False, returns the computed quantities instead of appending them in the same dictionary.
        Default True

    Returns
    -------
    uiuj : array-like (if in_place is False)
        5D reynolds stress (uiuj) tensor, indexed [x, y, z, i, j]
    """
    nx = len(sl_dict["x"])
    ny = len(sl_dict["y"])
    nz = len(sl_dict["z"])

    uiuj = np.zeros((nx, ny, nz, 3, 3))  # reynolds stresses
    terms_uiuj = [  # matching from budgetkey.py
        ["uu", "uv", "uw"],
        ["uv", "vv", "vw"],
        ["uw", "vw", "ww"],
    ]

    # rearrange tensor
    for ii in range(3):
        for jj in range(3):
            uiuj[:, :, :, ii, jj] = sl_dict[terms_uiuj[ii][jj]]

    if in_place:
        sl_dict["uiuj"] = uiuj
    else:
        return uiuj


def assemble_u_tensor(sl_dict, in_place=True):
    """
    Reorganizes the velocity vector into a 4D tensor [x, y, z, i] for component ui

    Parameters
    ----------
    sl_dict : dict
        dictionary from BudgetIO.slice()
    in_place : optional, bool
        If False, returns the computed quantities instead of appending them in the same dictionary.
        Default True

    Returns
    -------
    ui : array-like (if in_place is False)
        4D ui tensor, indexed [x, y, z, i]

    """
    nx = len(sl_dict["x"])
    ny = len(sl_dict["y"])
    nz = len(sl_dict["z"])

    ui = np.zeros((nx, ny, nz, 3))
    terms_ui = ["ubar", "vbar", "wbar"]  # matching from budgetkey.py

    # rearrange tensor
    for i in range(3):
        ui[:, :, :, i] = sl_dict[terms_ui[i]]

    if in_place:
        sl_dict["ui"] = ui
    else:
        return ui


def assemble_w_tensor(sl_dict, in_place=True):
    """
    Reorganizes the vorticity vector into a 4D tensor [x, y, z, i] for component ui

    Parameters
    ----------
    sl_dict : dict
        dictionary from BudgetIO.slice()
    in_place : optional, bool
        If False, returns the computed quantities instead of appending them in the same dictionary.
        Default True

    Returns
    -------
    wi : array-like (if in_place is False)
        4D vorticity tensor, indexed [x, y, z, i]
    """
    nx = len(sl_dict["x"])
    ny = len(sl_dict["y"])
    nz = len(sl_dict["z"])

    wi = np.zeros((nx, ny, nz, 3))
    terms_wi = [
        "wx",
        "wy",
        "wz",
    ]  # these terms might not exist and may need to be computed

    if not np.all([term in sl_dict.keys() for term in terms_wi]):
        compute_vort(sl_dict, save_ui=True, in_place=True)

    # rearrange tensor
    for i in range(3):
        wi[:, :, :, i] = sl_dict[terms_wi[i]]

    if in_place:
        sl_dict["wi"] = wi
    else:
        return wi


# ============= Vorticity computation functions ================


def compute_vort(sl_dict, save_ui=True, in_place=True):
    """
    Computes the vorticity vector and adds `wx`, `wy`, and `wz` the keys field.

    Parameters
    ----------
    sl_dict (dict) : slice from BudgetIO.slice()
    save_ui : bool
        saves intermediate key `ui` as a 4D tensor in the dictionary, but
        does not append to the keys field. Default True.
    in_place : bool
        returns result if False. Default True

    Returns
    -------
    wi : (Nx, Ny, Nz, 3)
        4D Vorticity tensor, if in_place=False
    """

    if not "duidxj" in sl_dict.keys():
        compute_duidxj(sl_dict, in_place=True, save_ui=save_ui)

    duidxj = sl_dict["duidxj"]

    wijk = np.zeros(duidxj.shape + (3,))  # vorticity component tensor
    for i in range(3):
        for j in range(3):
            for k in range(3):
                eijk = e_ijk(i, j, k)
                if eijk == 0:
                    continue
                wijk[:, :, :, i, j, k] = eijk * duidxj[:, :, :, k, j]

    wi = np.zeros(sl_dict["ubar"].shape + (3,))
    vort_keys = ["wx", "wy", "wz"]
    for i, wi_key in zip(range(3), vort_keys):
        wi[:, :, :, i] = np.sum(wijk[:, :, :, i, :, :], axis=(-2, -1))
        sl_dict[wi_key] = wi[:, :, :, i]
        sl_dict["keys"].append(wi_key)

    if in_place:
        sl_dict["wi"] = wi  # save this for further index notation computations
    else:
        return wi


# ============= offline budgets: vorticity ================


def compute_vort_budget(
    sl_dict, case=None, Ro=None, Ro_f=None, lat=45, Fr=None, theta0=300
):
    """
    Computes the offline vorticity budget in three component directions including the terms:
        advection '-vort_adv'
        stretching 'vort_str'
        buoyancy 'vort_buoy'
        coriolis 'vort_cor'
        subgrid stresses 'vort_sgs'
        Reynolds stresses 'vort_rs'

    All terms are 4D tensors [x,y,z,i] for the i-direction of vorticity.

    Parameters
    ----------
    sl_dict : dict
        from BudgetIO.slice(), expects velocity, temperature, subgrid stresses, and reynolds stresses
    case : BudgetIO object
        optional, if given, gleans Ro_f, Fr, and theta_0 from the following fields in
        the inputfile: 'ro', 'latitude', 'fr', 'tref'. Supersedes giving parameters individually; default None.
    Ro : float
        Rossby number as defined in LES
    Ro_f : float
        Rossby number as defined Ro = G/(f_c * L_c), where f_c = 2*Omega*sin(lat) is the Coriolis
        parmeter. Optional, supersedes Ro if both are given
    lat : float
        Latitude in degrees, default is 45, NOT None.
    Fr : float
        Froude number, defined Fr = G/sqrt(g*L_c)
    theta0 : float
        Reference potential temperature, Default 300 [K].

    Returns
    -------
    (None)
    """

    x = sl_dict["x"]
    y = sl_dict["y"]
    z = sl_dict["z"]
    nx = len(x)
    ny = len(y)
    nz = len(z)
    dxi = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]

    # allocate memory for all the tensors
    adv_ij = np.zeros((nx, ny, nz, 3, 3))
    str_ij = np.zeros((nx, ny, nz, 3, 3))
    buoy_ij = np.zeros((nx, ny, nz, 3, 3))
    sgs_ijkm = np.zeros((nx, ny, nz, 3, 3, 3, 3))  # 7D tensor, ugh
    cor_i = np.zeros((nx, ny, nz, 3))
    rs_ijkm = np.zeros((nx, ny, nz, 3, 3, 3, 3))  # also 7D

    if case is not None:
        Ro_LES = padeopsIO.key_search_r(case.input_nml, "ro")
        lat = padeopsIO.key_search_r(case.input_nml, "latitude") * np.pi / 180
        Ro = Ro_LES / (2 * np.sin(lat))  # we need this normalization
        Fr = padeopsIO.key_search_r(case.input_nml, "fr")
        theta0 = padeopsIO.key_search_r(case.input_nml, "tref")
        if theta0 is None:
            theta0 = 1  # hotfix for if flow is not stratified
    elif Ro_f is not None:
        Ro = Ro_f
    else:
        Ro = Ro / (
            2 * np.sin(lat * np.pi / 180)
        )  # convert Ro_LES -> Ro_f = G/(f_c*L_c)

    # check all required tensors exist:  (may throw KeyError)
    if not "ui" in sl_dict.keys():
        assemble_u_tensor(sl_dict, in_place=True)
    u_i = sl_dict["ui"]

    if not "wi" in sl_dict.keys():
        assemble_w_tensor(sl_dict, in_place=True)
    w_i = sl_dict["wi"]

    if not "uiuj" in sl_dict.keys():
        assemble_rs_tensor(sl_dict, in_place=True)
    uiuj = sl_dict["uiuj"]

    if not "tau_ij" in sl_dict.keys():
        assemble_sgs_tensor(sl_dict, in_place=True)
    tau_ij = sl_dict["tau_ij"]

    Tbar = sl_dict["Tbar"]

    # compute tensor quantities:
    for ii in range(3):
        # compute coriolis
        cor_i[:, :, :, ii] = 1 / Ro * ddxi(u_i[:, :, :, ii], 2, dxi=dxi[2])

        for jj in range(3):
            # advection (on RHS, flipped sign)
            adv_ij[:, :, :, ii, jj] = -u_i[:, :, :, jj] * ddxi(
                w_i[:, :, :, ii], jj, dxi=dxi[jj]
            )

            # vortex stretching
            str_ij[:, :, :, ii, jj] = w_i[:, :, :, jj] * ddxi(
                u_i[:, :, :, ii], jj, dxi=dxi[jj]
            )

            # buoyancy torque
            eijk = e_ijk(ii, jj, 2)  # buoyancy term has k=3
            if eijk == 0:
                buoy_ij[:, :, :, ii, jj] = (
                    0  # save a bit of compute time by skipping these
                )
            else:
                buoy_ij[:, :, :, ii, jj] = (
                    eijk * ddxi(Tbar, jj, dxi=dxi[jj]) / (Fr**2 * theta0)
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
                        sgs_ijkm[:, :, :, ii, jj, kk, mm] = eijk * ddxi(
                            ddxi(-tau_ij[:, :, :, kk, mm], mm, dxi=dxi[mm]),
                            jj,
                            dxi=dxi[jj],
                        )
                        rs_ijkm[:, :, :, ii, jj, kk, mm] = eijk * ddxi(
                            ddxi(-uiuj[:, :, :, kk, mm], mm, dxi=dxi[mm]),
                            jj,
                            dxi=dxi[jj],
                        )

    # now sum over extra axes to collapse terms
    vort_raw = {
        "vort_adv": adv_ij,
        "vort_str": str_ij,
        "vort_buoy": buoy_ij,
        "vort_sgs": sgs_ijkm,
        "vort_cor": cor_i,
        "vort_rs": rs_ijkm,
    }

    for key in vort_raw.keys():
        dims = len(vort_raw[key].shape)
        if dims == 4:
            sl_dict[key] = vort_raw[key]
        else:
            sum_axes = tuple(range(4, dims))
            sl_dict[key] = np.sum(vort_raw[key], axis=sum_axes)
    #         sl_dict['keys'].append(key)  # technically I suppose these should not be appended because they are 4D tensors

    # add the residual as well
    sl_dict["vort_res"] = np.sum([sl_dict[key] for key in vort_raw.keys()], axis=0)


#     sl_dict['keys'].append('vort_res')  # also a 4D tensor


# ============= polar coordinate functions ================


def compute_vort_xrt(sl_dict):
    """
    Transforms the vorticity vector x,y,z -> x,r,theta
    """

    if not np.all([keycheck in sl_dict.keys() for keycheck in ["wx", "wy", "wz"]]):
        compute_vort(sl_dict)

    yy, zz = np.meshgrid(sl_dict["y"], sl_dict["z"], indexing="ij")
    theta = np.arctan2(zz, yy)
    # append third axis for explicit broadcasting:
    theta = theta[None, :, :]

    sl_dict.update(
        {
            "wr": sl_dict["wy"] * np.cos(theta) + sl_dict["wz"] * np.sin(theta),
            "wt": -sl_dict["wy"] * np.sin(theta) + sl_dict["wz"] * np.cos(theta),
            "theta": theta,
        }
    )

    # also compute ur, utheta:
    sl_dict.update(
        {
            "ur": sl_dict["vbar"] * np.cos(theta) + sl_dict["wbar"] * np.sin(theta),
            "ut": -sl_dict["vbar"] * np.sin(theta) + sl_dict["wbar"] * np.cos(theta),
        }
    )
