"""
Budget class for computing and aggregating budget terms. 

Kirby Heck
2024 July 24
"""

import numpy as np
from . import key_search_r
from .budgetIO import BudgetIO
from .budget_addons import *
from .gridslice import Grid3


class Budget(dict):
    """
    Computes "offline" budgets and links to a BudgetIO object.
    """

    def __init__(self, src):
        """
        Initialize a Budget object

        Parameters
        ----------
        src : Any
            If type(src) is a BudgetIO file, links the Budget class to the IO object
            If type(src) is a dictionary, only associates the current fields and axes
                For dictionary, expects keys ['x', 'y', 'z', <terms>, ...]
        """

        if isinstance(src, BudgetIO):
            self.src = src
            self.src_type = BudgetIO
            self.x = src.xLine
            self.y = src.yLine
            self.z = src.zLine
            self.Ro = src.Ro
            self.lat = src.lat
            self.galpha = src.galpha
            self.Fr = src.Fr
            self.is_stratified = key_search_r(src.input_nml, "isstratified")
            if self.is_stratified is None:
                self.is_stratified = False
            self.theta0 = key_search_r(src.input_nml, "tref")

        elif isinstance(src, dict):
            self.src = src
            self.src_type = dict
            self.x = src["x"]  # these may throw a KeyError
            self.y = src["y"]
            self.z = src["z"]
            self.Ro = None
            self.lat = None
            self.galpha = 0
            self.Fr = None
            self.is_stratified = None
            self.theta0 = None

        else:
            raise TypeError(f"`src` must be type BudgetIO or dict, not {type(src)}")

        # initialize grid
        self.grid = Grid3(x=self.x, y=self.y, z=self.z)

    def _read_budgets(self, budget_terms):
        """
        Reads budgets using BudgetIO, if linked.
        """
        if self.src_type == BudgetIO:
            self.src.read_budgets(budget_terms=budget_terms)

            for key in self.src.budget.keys():
                # equivalently:
                # self[key] = self.src.budget[key]
                super().__setitem__(key, self.src.budget[key])
        else:
            raise AttributeError(
                "_read_budgets(): Reading budgets requires a linked BudgetIO object."
            )

    # ================= Add budgets to interface with here ===================

    def init_momentumLES_x(self, **kwargs):
        self.momentumLES_x = LESMomentum_x(self, **kwargs)
        return self.momentumLES_x

    def init_momentumLES_y(self, **kwargs):
        self.momentumLES_y = LESMomentum_y(self, **kwargs)
        return self.momentumLES_y

    def init_momentum_x(
        self,
        base_agg=0,
        Ro=None,
        lat=None,
        galpha=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if lat is None:
            lat = self.lat
        if galpha is None:
            galpha = self.galpha
        self.momentum_x = BudgetMomentum_x(
            self,
            base_agg=base_agg,
            Ro=Ro,
            lat=lat,
            galpha=galpha,
        )
        return self.momentum_x

    def init_momentum_y(
        self,
        base_agg=0,
        Ro=None,
        lat=None,
        galpha=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if lat is None:
            lat = self.lat
        if galpha is None:
            galpha = self.galpha
        self.momentum_y = BudgetMomentum_y(
            self,
            base_agg=base_agg,
            Ro=Ro,
            lat=lat,
            galpha=galpha,
        )
        return self.momentum_y

    def init_momentum_z(
        self,
        base_agg=0,
        Ro=None,
        lat=None,
        galpha=None,
        Fr=None,
        theta0=None,
        is_stratified=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        if galpha is None:
            galpha = self.galpha
        if Fr is None:
            Fr = self.Fr
        if theta0 is None:
            theta0 = self.theta0
        if is_stratified is None:
            is_stratified = self.is_stratified
        self.momentum_z = BudgetMomentum_z(
            self,
            base_agg=base_agg,
            Ro=Ro,
            lat=lat,
            galpha=galpha,
            Fr=Fr,
            theta0=theta0,
            is_stratified=is_stratified,
        )
        return self.momentum_z

    def init_deficit_x(
        self,
        bkgd_budget,
        base_agg=0,
        Ro=None,
        Fr=None,
        lat=None,
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        self.deficit_x = BudgetDeficit_x(
            self,
            bkgd_budget,
            base_agg=base_agg,
            Ro=Ro,
            Fr=Fr,
            lat=lat,
        )
        return self.deficit_x

    def init_deficit_y(self, bkgd_budget, base_agg=0, Ro=None, Fr=None, lat=None):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        self.deficit_y = BudgetDeficit_y(
            self,
            bkgd_budget,
            base_agg=base_agg,
            Ro=Ro,
            Fr=Fr,
            lat=lat,
        )
        return self.deficit_y

    def init_vorticity_x(
        self, base_agg=0, Ro=None, lat=None, fplane=True, Fr=None, theta0=None
    ):
        if Ro is None:
            Ro = self.Ro
        if Fr is None:
            Fr = self.Fr
        if lat is None:
            lat = self.lat
        if theta0 is None and isinstance(self.src, BudgetIO):
            theta0 = key_search_r(self.src.input_nml, "tref")
        self.vorticity_x = BudgetVorticity_x(
            self, base_agg=base_agg, Ro=Ro, lat=lat, fplane=fplane, Fr=Fr, theta0=theta0
        )
        return self.vorticity_x


if __name__ == "__main__":
    pass  # TODO add unit tests
