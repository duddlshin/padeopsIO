"""
Budget term computations; return the most disaggregated form

Kirby Heck
2024 July 24
"""

import numpy as np
import copy

from .utils import math_utils as math
from .gridslice import Slice


# =============== NewBudget interface ================


def base_aggregate(terms, ret=None, ndim=3, base_agg=0, **kwargs):
    """
    Aggregates all terms into dictionary by summing over extra axes

    Parameters
    ----------
    terms : dict
        Dictionary of np.ndarray
    ret : dict, optional
        Dictionary that is updated and returned. If None, returns a
        clean dictionary. Default None
    ndim : int, optional
        Number of base dimensions (e.g. 3 for x, y, z). Default 3
    base_agg : int, optional
        Base aggregation level. Default 0
    kwargs : dict, optional
        Additional levels of (dis)aggregation for keys `newkey`

    Returns
    -------
    dict or None
        Dictionary of aggregated values, only if `ret` is None.
    """
    in_place = True
    if ret is None:
        ret = {}
        in_place = False

    for key in terms.keys():
        # for keys that are tensors, these are separated with '_' (e.g. 'adv_j')
        newkey = key.split("_")[0]

        # check aggregation levels:
        if newkey in kwargs.keys() and isinstance(kwargs[newkey], int):
            agg = kwargs[newkey]
        else:
            agg = base_agg

        # sum over all axes we are aggregating:
        sum_axes = tuple(range(ndim + agg, terms[key].ndim))
        summed = np.sum(terms[key], axis=sum_axes)  # OK even if sum_axes=()

        if summed.ndim == ndim:
            # fully collapsed, save into `newkey`
            ret[newkey] = summed
        else:
            # save each tensor by replacing e.g. 'tau_ij' -> 'tau_12' for i=1, j=2
            keys, flattened = flatten_tensor(summed, ndim, return_keys=True)
            for key, arr in zip(keys, flattened):
                ret[f"{newkey}_{key}"] = arr

    if not in_place:
        return ret  # return if `ret` is not given


def flatten_tensor(a, ndim_new, return_keys=True, index_st=1):
    """
    Flatten a tensor down to ndim+1 axes.
    reshape tensor by flattening the last `ndim` - `ndim_grid` axes
    for example, (8,8,8,4,3) -> (8,8,8,12)

    Parameters
    ----------
    a : ndarray
    ndim_new : int
        Number of dimensions to flatten to
    return_keys : bool (optional)
        Returns keys if True. Default True.
    index_st : int (optional)
        Value to start indexing from. Default is 1.
    """
    dims = a.shape

    if len(dims) > ndim_new + 1:
        new_dim = np.product(dims[ndim_new:])
        new_shape = dims[:ndim_new] + (new_dim,)
        tmp = np.reshape(a, new_shape)
    else:
        tmp = a

    # move this to the 0-axis so it can be iterated
    tmp = np.moveaxis(tmp, -1, 0)
    if return_keys:
        # returns a keys array [1, 2, ..., new_dim]
        keys = np.arange(tmp.shape[0]) + index_st
        return keys, tmp

    return tmp  # default: return reshaped array


def compute_delta_field(
    primary, precursor, budget_terms=None, avg_xy=True, in_place=True
):
    """
    Computes deficit fields between primary and precursor simulations, e.g.:
        \Delta u = u_primary - u_precursor

    This definition follows the double decomposition in Martinez-Tossas, et al. (2021).

    Parameters
    ----------
    primary : BudgetIO
        Primary simulation budget object
    precursor : BudgetIO
        Precursor simulation budget object
    budget_terms : list, optional
        Budget terms to compute deficits.
        Default is all shared keys between primary and precursor
    avg_xy : bool, optional
        Averages precursor terms horizontally if True. Default True

    Returns
    -------
    field : dict
        Deficit fields, either in-place if in_place=True, or

    """

    if budget_terms is None:
        budget_terms = set(primary.keys()) & set(precursor.keys())

    # background fields
    if avg_xy:
        tmp = {key: np.mean(precursor[key], (0, 1)) for key in budget_terms}
    else:
        tmp = precursor

    if in_place:
        field = primary
    else:
        # field = copy.deepcopy(primary)
        field = {}

    # for each key, compute primary - precursor:
    for key in budget_terms:
        dkey = key + "_deficit"
        if dkey in field.keys():
            continue  # skip repeat computations
        field[dkey] = primary[key] - tmp[key]

    return field


# ======================== Budget TEMPLATE ========================


class NewBudget(dict):
    """
    Informal interface for new budget classes to add (e.g. RANS).

    Aggregation is primarly defined based on how repeated indices are summed (or not)
    in the budget computation. For a term like the reynolds stress divergence,
        d<ui'uj'>/dxj,
    the base level of aggregation (level 0) is to sum over j. If we want one level lower
    of diaggregation (level 1), the j-indices are not summed.
    """

    req_keys = []  # required keys go here (e.g. 'ubar', etc...)
    opt_keys = []  # optional keys (e.g. 'xAD', etc... )

    # ====== the following functions should not need to change (much) =======

    def __init__(self, budget, base_agg=0):
        """Blueprint for budgets"""
        self.budget = budget
        self.base_terms = None  # these are computed in self.compute()
        self.base_agg = base_agg  # base level of aggregation, 0 is the most aggregated

    def compute(self, level=None, custom=False, **kwargs):
        """Compute disaggregated budget terms"""
        self._check_terms()  # make sure the terms exist to compute budgets
        self._compute_budget()  # carry out budget computation
        # aggregate terms and save them in the dictionary
        self.aggregate(level=level, custom=custom, **kwargs)

    def aggregate(self, level=None, custom=False, **kwargs):
        """Aggregate similar budget terms"""

        self.clear()  # clear previous aggregation
        if level is None:  # default aggregation
            level = self.base_agg

        if not custom:  # type(level) == int and level >= 0:
            base_aggregate(
                self.base_terms,
                ret=self,
                ndim=self.budget.grid.ndim,
                base_agg=level,
                **kwargs,
            )
        else:  # add a custom aggregation on top of base level
            self._aggregate_custom(level, **kwargs)

    def _check_terms(self, budget=None):
        """
        Ensure all the required terms exist before computing budgets

        Parameters
        ----------
        budget : budget.Budget object
            Checks terms in self.budget by default
        """

        if budget is None:
            budget = self.budget
        # first check required keys:
        missing_keys = [key for key in self.req_keys if key not in budget.keys()]

        if len(missing_keys) > 0:  # try to load missing keys
            try:
                budget._read_budgets(missing_keys)

            except AttributeError as e:
                print(
                    "NewBudget._check_terms(): missing",
                    missing_keys,
                    ", no source files associated.",
                )
                raise e

            except KeyError as e:
                raise e  # budgets in BudgetIO.read_budgets() do not exist

        # then check optional keys:
        missing_keys_opt = [key for key in self.opt_keys if key not in budget.keys()]

        if len(missing_keys_opt) > 0:  # try also to load optional keys:
            try:
                budget._read_budgets(missing_keys_opt)

            except AttributeError as e:
                print(
                    "NewBudget._check_terms(): missing optional keys",
                    missing_keys_opt,
                    ", no source files associated.",
                )
                # but do not raise these errors

            except KeyError as e:
                still_missing = [
                    key for key in self.opt_keys if key not in budget.keys()
                ]
                print(
                    "NewBudget._check_terms(): missing optional keys: ", still_missing
                )

    # the following functions will need to be overwritten

    def _aggregate_custom(self, level):
        """Custom aggregation for this budget"""
        raise Exception("_aggregate_custom(): No function definition. ")

    def _compute_budget(self):
        """Fill in budget computations here"""
        raise Exception("_compute_budget(): No function definition. ")


# ==================== LES Momentum budgets =========================


class LESMomentum(NewBudget):
    """
    Class for LES momentum budgets ("read directly from PadeOps")
    """

    def _compute_budget(self):
        """
        Assembles LES momentum budgets directly from PadeOps.
        """
        terms = {}
        for key in self.req_keys + self.opt_keys:
            try:
                terms[key] = self.budget[key]
            except KeyError as e:
                print("_compute_budget(): could not find term", key)

        # compute residual
        terms["residual"] = sum(terms[key] for key in terms.keys())

        self.base_terms = terms


class LESMomentum_x(LESMomentum):
    """
    Reads LES x-momentum budgets.
    """

    req_keys = ["DuDt", "dpdx", "xSGS"]
    opt_keys = ["xAD", "xCor", "xGeo"]

    def _aggregate_custom(self, aggregate):
        """
        Combines the coriolis and geostrophic terms. This is the "-1" level of aggregation
        (combining non-tensor terms).

        Parameters
        ----------
        aggregate : Any
            Dictionary {'coriolis': -1} or int (-1) to aggregate Coriolis terms
        """
        # may throw KeyError
        if aggregate == -1 or type(aggregate) == dict and aggregate["coriolis"] == -1:
            terms = self.base_terms
            for key in terms.keys():
                if key in ["xCor", "xGeo"]:
                    continue
                self[key] = terms[key]
            self["coriolis"] = terms["xCor"] + terms["xGeo"]

        else:
            raise ValueError


class LESMomentum_y(LESMomentum):
    """
    Reads LES y-momentum budgets.
    """

    req_keys = ["DvDt", "dpdy", "ySGS"]
    opt_keys = ["yAD", "yCor", "yGeo"]

    def _aggregate_custom(self, level, **kwargs):
        """
        Combines the coriolis and geostrophic terms. This is the "-1" level of aggregation
        (combining non-tensor terms).

        Parameters
        ----------
        level : int
            Base level of aggregation
        kwargs : dict
        """
        # may throw KeyError
        if level == -1:
            kwargs = {"coriolis": -1}
            level = 0

        self.aggregate(level)

        if "coriolis" in kwargs and kwargs["coriolis"] == -1:
            self["coriolis"] = self["yCor"] + self["yGeo"]
            del self["yCor"], self["yGeo"]

        self["residual"] = self.pop("residual")  # move the residual to the end


# =========================== RANS Budgets ===============================


class BudgetMomentum(NewBudget):
    """
    Base class for filtered RANS budgets.
    """

    req_keys = [
        ["ubar", "vbar", "wbar", "pbar", "tau11", "tau12", "tau13", "uu", "uv", "uw"],
        ["ubar", "vbar", "wbar", "pbar", "tau12", "tau22", "tau23", "uv", "vv", "vw"],
        ["ubar", "vbar", "wbar", "pbar", "tau13", "tau23", "tau33", "uw", "vw", "ww"],
    ]  # required keys in x, y, z
    opt_keys = [["xAD"], ["yAD"], ["Tbar"]]  # optional keys

    def __init__(
        self,
        budget,
        base_agg=0,
        Ro=None,
        Fr=None,
        lat=None,
        galpha=0,
        is_stratified=True,
        theta0=None,
        # TODO: what if we don't want to use Coriolis?
    ):
        """
        Initialize non-dimensional RANS budget terms.

        Parameters
        ----------
        budget : budget.Budget object
        base_agg : int
        Ro : float
            Rossby number, defined U/(\Omega L)
        Fr : float
            Froude number, defined U/\sqrt{g L}
        lat : float
            Latitude, in degrees
        galpha : float
            Geostrophic wind direction, in degrees
        is_stratified : bool
            Use stratification if True. Default True.
        theta0 : float
            Reference potential temperature (K)
        """
        super().__init__(budget, base_agg)
        self.Ro = Ro
        self.Fr = Fr
        self.lat = lat * np.pi / 180
        self.galpha = galpha * np.pi / 180
        self.is_stratified = is_stratified
        self.theta0 = theta0
        self.direction = None

    def _compute_budget(self):
        """
        Computes RANS momentum budgets in x.
        """
        self.base_terms = compute_RANS(
            self.budget,
            self.direction,
            Ro=self.Ro,
            lat=self.lat,
            galpha=self.galpha,
            is_stratified=self.is_stratified,
            Fr=self.Fr,
            theta0=self.theta0,
        )

    def _aggregate_custom(self, level, **kwargs):
        if level == -1:
            kwargs = {"totaladv": -1, "coriolis": -1}
            level = 0

        self.aggregate(level, **kwargs)
        if "totaladv" in kwargs.keys() and kwargs["totaladv"] == -1:
            self["totaladv"] = self["adv"] + self["rs"]
            del self["rs"], self["adv"]

        if "coriolis" in kwargs.keys() and kwargs["coriolis"] == -1:
            self["coriolis"] = self["cor"] + self["geo"]
            del self["geo"], self["cor"]

        self["residual"] = self.pop("residual")  # move the residual to the end


class BudgetMomentum_x(BudgetMomentum):
    """
    Computes the RANS budgets in the y-direction.
    """

    req_keys = BudgetMomentum.req_keys[0]
    opt_keys = BudgetMomentum.opt_keys[0]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.direction = 0  # y-direction


class BudgetMomentum_y(BudgetMomentum):
    """
    Computes the RANS budgets in the y-direction.
    """

    req_keys = BudgetMomentum.req_keys[1]
    opt_keys = BudgetMomentum.opt_keys[1]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.direction = 1  # y-direction


class BudgetMomentum_z(BudgetMomentum):
    """
    Computes the RANS budgets in the y-direction.
    """

    req_keys = BudgetMomentum.req_keys[2]
    opt_keys = BudgetMomentum.opt_keys[2]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.direction = 2  # z-direction


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


# =========================== RANS Deficit Budgets ===============================


class BudgetDeficit(NewBudget):
    """
    Base class for filtered RANS deficit budgets.
    """

    req_keys = BudgetMomentum.req_keys
    opt_keys = BudgetMomentum.opt_keys

    def __init__(
        self,
        budget,
        bkgd_budget,
        base_agg=0,
        Ro=None,
        Fr=None,
        lat=None,
    ):
        """
        Initialize non-dimensional RANS budget terms.

        Parameters
        ----------
        budget : budget.Budget object
        bkdg_budget : budget.Budget object
            Budget object for the background flow (precursor sim.)
        base_agg : int
        Ro : float
            Rossby number, defined U/(\Omega L)
        Fr : float
            Froude number, defined U/\sqrt{g L}
        lat : float
            Latitude, in degrees
        """
        super().__init__(budget, base_agg)
        self.bkgd = bkgd_budget
        self.Ro = Ro
        self.Fr = Fr
        self.lat = lat * np.pi / 180
        self.direction = None  # overwrite this in sub-classes

    def _compute_budget(self):
        """
        Computes RANS momentum budgets in x.
        """
        self.base_terms = deficit_budget(
            self.budget,
            self.bkgd,
            self.direction,
            dx=self.budget.dxi,
            Ro=self.Ro,
            lat=self.lat,
        )

    def _check_terms(self):
        """Check terms for background and primary budget objects"""
        super()._check_terms()
        super()._check_terms(budget=self.bkgd)

    def _aggregate_custom(self, level, **kwargs):
        """Custom aggregation for level -1 (combined base terms)"""
        if level == -1:
            kwargs = {"rs": -1}
            level = 0

        if "rs" in kwargs.keys():
            self.aggregate(level, rsfull=kwargs["rs"], rsbkgd=kwargs["rs"], **kwargs)
            if kwargs["rs"] <= 0:
                # combine advection with reynolds stresses
                self["rs"] = self["rsfull"] + self["rsbkgd"]
                del self["rsfull"], self["rsbkgd"]
            elif kwargs["rs"] == 1:
                for k in range(1, 4):
                    self[f"rs_{k}"] = self[f"rsfull_{k}"] + self[f"rsbkgd_{k}"]
                    del self[f"rsfull_{k}"], self[f"rsbkgd_{k}"]
        else:
            self.aggregate(level, **kwargs)

        self["residual"] = self.pop("residual")  # move the residual to the end


class BudgetDeficit_x(BudgetDeficit):
    """
    Computes the RANS deficit budgets in the x-direction.
    """

    req_keys = BudgetDeficit.req_keys[0]
    opt_keys = BudgetDeficit.opt_keys[0]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.direction = 0  # x-direction


class BudgetDeficit_y(BudgetDeficit):
    """
    Computes the RANS deficit budgets in the y-direction.
    """

    req_keys = BudgetDeficit.req_keys[1]
    opt_keys = BudgetDeficit.opt_keys[1]

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.direction = 1  # y-direction


def deficit_budget(full, bkgd, i, dxi, Ro=None, lat=None, fplane=True, avg_xy=True):
    """
    Computes the streamwise momentum deficit budget

    Parameters
    ----------
    full : dict
        Dictionary of full velocity fields
    bkgd : dict
        Dictionary of background fields
    i : int
        Direction to compute budgets (0 -> x, 1 -> y, 2 -> z)
    dxi : tuple
        Vector (dx, dy, dz)

    Returns
    -------
    dict
        Dictionary of deficit fields
    """

    if i == 2:
        raise NotImplementedError("TODO: Delta_w deficit budgets")

    ret = {}
    dims = full.grid.shape

    # compute deficit fields:
    compute_delta_field(full, bkgd, avg_xy=avg_xy, in_place=True)

    # construct tensors
    deltau_j = math.assemble_tensor_1d(
        full, ["ubar_deficit", "vbar_deficit", "wbar_deficit"]
    )
    u_j = math.assemble_tensor_1d(full, ["ubar", "vbar", "wbar"])  # full field
    U_j = math.assemble_tensor_1d(bkgd, ["ubar", "vbar", "wbar"])  # bkgd field
    uu_full_ij = math.assemble_tensor_1d(full, rs_keys[i])
    uu_bkgd_ij = math.assemble_tensor_1d(bkgd, rs_keys[i])
    tau_full_ij = math.assemble_tensor_1d(full, tau_keys[i])
    tau_bkgd_ij = math.assemble_tensor_1d(bkgd, tau_keys[i])

    # numerics:
    dduidxj = math.gradient(deltau_j[..., i], dxi)  # gradient of velocity deficit field
    dUidxj = math.gradient(U_j[..., i], dxi)  # gradient of background velocity field
    dpdxi = math.gradient(full["pbar_deficit"], dxi, axis=i)
    duiujdxj_full = math.div(uu_full_ij, dxi)
    duiujdxj_bkgd = math.div(uu_bkgd_ij, dxi)
    sgs_ij = math.div(
        tau_full_ij - tau_bkgd_ij, dxi
    )  # could split this into components as well

    # compute momentum deficit terms
    ret["adv_j"] = -u_j * dduidxj  # deficit advection
    ret["prss"] = -dpdxi
    ret["sgs_j"] = -sgs_ij
    ret["rsfull_j"] = -duiujdxj_full
    ret["rsbkgd_j"] = duiujdxj_bkgd

    try:
        ret["adm"] = full[AD_keys[i]]
    except KeyError:
        pass
        # not including ADM

    if fplane:
        ret["cor"] = (
            2 * math.e_ijk(i, 1 - i, 2) * np.sin(lat) / Ro * deltau_j[..., 1 - i]
        )
    else:
        raise NotImplementedError("TODO: Deficit budgets full Coriolis")
    ret["wakeadv_j"] = -deltau_j * dUidxj

    # aggregate to compute residual
    tmp = base_aggregate(ret, ndim=len(dims), base_agg=0)
    ret["residual"] = sum(tmp[key] for key in tmp.keys())

    return Slice(ret, grid=full.grid)


# ======================= Vorticity budgets ==========================


class BudgetVorticity(NewBudget):
    """
    Base class for filtered vorticity budgets.
    """

    req_keys = [
        "ubar",
        "vbar",
        "wbar",
        "Tbar",
        "uu",
        "uv",
        "uw",
        "vv",
        "vw",
        "ww",
        "tau11",
        "tau12",
        "tau13",
        "tau22",
        "tau23",
        "tau33",
    ]
    opt_keys = ["xAD", "yAD"]

    def __init__(
        self, budget, base_agg=0, Ro=None, lat=None, fplane=True, Fr=None, theta0=None
    ):
        """
        Initialize non-dimensional vorticity budget terms.

        Parameters
        ----------
        budget : budget.Budget object
        bkdg_budget : budget.Budget object
            Budget object for the background flow (precursor sim.)
        base_agg : int
        Ro : float
            Rossby number, defined U/(\Omega L)
        lat : float
            Latitude, in degrees
        fplane : bool
            Use f-plane approx. Default True
        Fr : float
            Froude number, defined U/\sqrt{g L}
        theta0 : float
            Reference potential temperature (K)
        """
        super().__init__(budget, base_agg)
        self.Ro = Ro
        self.Fr = Fr
        self.lat = lat * np.pi / 180
        self.fplane = fplane
        self.theta0 = theta0
        self.direction = None  # overwrite this in sub-classes

    def _compute_budget(self):
        """
        Computes RANS momentum budgets in x.
        """
        self.base_terms = compute_vort_budget(
            self.budget,
            self.direction,
            Ro=self.Ro,
            lat=self.lat,
            fplane=self.fplane,
            Fr=self.Fr,
            theta0=self.theta0,
        )

    def _aggregate_custom(self, level, **kwargs):
        """TODO"""
        self.aggregate(level, **kwargs)


class BudgetVorticity_x(BudgetVorticity):
    """
    Vorticity budgets in x
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize non-dimensional RANS budget terms.

        see BudgetMomentum()
        """
        super().__init__(*args, **kwargs)
        self.direction = 0  # x-direction


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

    u_i = math.assemble_tensor_1d(field_dict, ["ubar", "vbar", "wbar"])
    dxi = field_dict.grid.dxi

    dukdxj = math.gradient(
        u_i, dxi, axis=(0, 1, 2), stack=-2
    )  # stack along -2 axis so indexing is x, y, z, j, k

    w_i = np.zeros(field_dict.grid.shape + (3,))
    for i in range(3):  # can this be done w/o a loop?
        w_i[..., i] = np.sum(math.E_ijk[i, ...] * dukdxj, axis=(-2, -1))

    if in_place:
        field_dict["w_i"] = w_i
    else:
        return Slice(w_i, grid=field_dict.grid)


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
    u_i = math.assemble_tensor_1d(field_dict, keys=["ubar", "vbar", "wbar"])
    w_i = compute_vort(field_dict, in_place=False)
    # save vorticity keys to the budget object:
    field_dict["w_i"] = w_i
    uiuj = math.assemble_tensor_nd(field_dict, rs_keys)
    tau_ij = math.assemble_tensor_nd(field_dict, tau_keys)
    Tbar = field_dict["Tbar"]

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
