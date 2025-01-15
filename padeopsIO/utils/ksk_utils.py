"""
Temporary extra budget functions for DeficitIO
#TODO: Merge with math_utils.py

Kerry Klemmer
2024 June
"""

# Additional budget functions 

import numpy as np
import padeopsIO

def advection(uj, arr, dx, dy, dz):
    """
    returns advection of arr by uj 
    i.e. u * d/dx(arr) + v * d/dy(arr) + w * d/dz(arr)
    """

    tmp_grad = np.gradient(arr, dx, dy, dz)

    adv = np.sum(np.multiply(uj, tmp_grad), axis=0, keepdims=False)

    return adv

def construct_delta_uiuj(defIO):
    """
    constructs and returns delta ui'uj'
    """

    delta_uiuj = np.zeros([defIO.nx, defIO.ny, defIO.nz, 3, 3])
    delta_uiuj[...,0,0] = defIO.budget['delta_uu']
    delta_uiuj[...,0,1] = defIO.budget['delta_uv']
    delta_uiuj[...,0,2] = defIO.budget['delta_uw']
    delta_uiuj[...,1,0] = defIO.budget['delta_uv']
    delta_uiuj[...,1,1] = defIO.budget['delta_vv']
    delta_uiuj[...,1,2] = defIO.budget['delta_vw']
    delta_uiuj[...,2,0] = defIO.budget['delta_uw']
    delta_uiuj[...,2,1] = defIO.budget['delta_vw']
    delta_uiuj[...,2,2] = defIO.budget['delta_ww']

    return delta_uiuj

def construct_delta_ui_base_uj(defIO):
    """
    constructs and return delta ui' base uj'
    """

    delta_ui_base_uj = np.zeros([defIO.nx, defIO.ny, defIO.nz, 3, 3])
    delta_ui_base_uj[...,0,0] = defIO.budget['delta_u_base_u']
    delta_ui_base_uj[...,0,1] = defIO.budget['delta_u_base_v']
    delta_ui_base_uj[...,0,2] = defIO.budget['delta_u_base_w']
    delta_ui_base_uj[...,1,0] = defIO.budget['base_u_delta_v']
    delta_ui_base_uj[...,1,1] = defIO.budget['delta_v_base_v']
    delta_ui_base_uj[...,1,2] = defIO.budget['delta_v_base_w']
    delta_ui_base_uj[...,2,0] = defIO.budget['base_u_delta_w']
    delta_ui_base_uj[...,2,1] = defIO.budget['base_v_delta_w']
    delta_ui_base_uj[...,2,2] = defIO.budget['delta_w_base_w']

    return delta_ui_base_uj

def construct_uiuj(budgetIO):
    """
    constructs and returns ui'uj' for BudgetIO object
    """    

    uiuj = np.zeros([budgetIO.nx, budgetIO.ny, budgetIO.nz, 3, 3])
    uiuj[...,0,0] = budgetIO.budget['uu']
    uiuj[...,0,1] = budgetIO.budget['uv']
    uiuj[...,0,2] = budgetIO.budget['uw']
    uiuj[...,1,0] = budgetIO.budget['uv']
    uiuj[...,1,1] = budgetIO.budget['vv']
    uiuj[...,1,2] = budgetIO.budget['vw']
    uiuj[...,2,0] = budgetIO.budget['uw']
    uiuj[...,2,1] = budgetIO.budget['vw']
    uiuj[...,2,2] = budgetIO.budget['ww']

    return uiuj

def construct_duidxj(io):
    """
    constructs and returns duidxj for BudgetIO or DeficitIO
    """

    duidxj = np.zeros([io.nx, io.ny, io.nz, 3, 3])
    duidxj[...,0,0] = io.budget['dUdx']
    duidxj[...,0,1] = io.budget['dUdy']
    duidxj[...,0,2] = io.budget['dUdz']
    duidxj[...,1,0] = io.budget['dVdx']
    duidxj[...,1,1] = io.budget['dVdy']
    duidxj[...,1,2] = io.budget['dVdz']
    duidxj[...,2,0] = io.budget['dWdx']
    duidxj[...,2,1] = io.budget['dWdy']
    duidxj[...,2,2] = io.budget['dWdz']

    return duidxj

def tke_calc(io):
    """
    Compute TKE for BudgetIO or DeficitIO
    """


    if isinstance(io, padeopsIO.DeficitIO):
        # check to make sure the BudgetIO object has uu, vv, and ww
        if 'delta_uu' not in io.budget:
            io.read_budgets(budget_terms=['delta_uu', 'delta_u_base_u'])
        if 'delta_vv' not in io.budget:
            io.read_budgets(budget_terms=['delta_vv', 'delta_v_base_v'])
        if 'delta_ww' not in io.budget:
            io.read_budgets(budget_terms=['delta_ww', 'delta_w_base_w'])

        io.budget['TKE'] = 0.5 * (io.budget['delta_uu'] + io.budget['delta_vv'] + io.budget['delta_ww']) \
                             + io.budget['delta_u_base_u'] + io.budget['delta_v_base_v'] + io.budget['delta_w_base_w']

    elif isinstance(io, padeopsIO.BudgetIO):
        # check to make sure the BudgetIO object has uu, vv, and ww
        if 'uu' not in io.budget:
            io.read_budgets(budget_terms=['uu'])
        if 'vv' not in io.budget:
            io.read_budgets(budget_terms=['vv'])
        if 'ww' not in io.budget:
            io.read_budgets(budget_terms=['ww'])

        io.budget['TKE'] = 0.5 * (io.budget['uu'] + io.budget['vv'] + io.budget['ww'])

def mke_calc(io):
    """
    Compute TKE for BudgetIO or DeficitIO
    """


    if isinstance(io, padeopsIO.DeficitIO):
        return

    elif isinstance(io, padeopsIO.BudgetIO):
        # check to make sure the BudgetIO object has uu, vv, and ww
        if 'ubar' not in io.budget:
            io.read_budgets(budget_terms='ubar')
        if 'vbar' not in io.budget:
            io.read_budgets(budget_terms='vbar')
        if 'wbar' not in io.budget:
            io.read_budgets(budget_terms='wbar')

        io.budget['MKE'] = 0.5 * (io.budget['ubar']**2 + io.budget['vbar']**2 + io.budget['wbar']**2)


def tke_wake_prod_calc(base, deficit):
    """
    Calculate wake tke production 
    """

    delta_uiuj = construct_delta_uiuj(deficit)
    delta_ui_base_uj = construct_delta_ui_base_uj(deficit)
    base_uiuj = construct_uiuj(base)

    delta_duidxj = construct_duidxj(deficit)
    base_duidxj = construct_duidxj(base)

    P_DDD = -np.sum(delta_uiuj * delta_duidxj, axis=(3,4))
    P_DDB = -np.sum(delta_uiuj * base_duidxj, axis=(3,4))
    P_DBB = -np.sum(delta_ui_base_uj * base_duidxj, axis=(3,4))
    P_BDD = -np.sum(delta_ui_base_uj * delta_duidxj, axis=(3,4))
    P_BBD = -np.sum(base_uiuj * delta_duidxj, axis=(3,4))
    P_BDB = -np.sum(delta_ui_base_uj * base_duidxj, axis=(3,4))
    P_DBD = -np.sum(delta_ui_base_uj * delta_duidxj, axis=(3,4))

    deficit.budget['P_DDD'] = P_DDD
    deficit.budget['P_DDB'] = P_DDB
    deficit.budget['P_DBB'] = P_DBB
    deficit.budget['P_BDD'] = P_BDD
    deficit.budget['P_BBD'] = P_BBD
    deficit.budget['P_BDB'] = P_BDB
    deficit.budget['P_DBD'] = P_DBD
    
    deficit.budget['P_grad_B'] = P_DDB + P_BDB + P_DBB

    deficit.budget['P_wake'] = P_DDD + P_DDB + P_DBB + P_BDD + P_BBD + P_DBD + P_BDB

def flux_calc(io, budget_terms, direction, coords=None, yc=None, overwrite=False, streamwise=False, suffix=''):
    """
    Calculate the flux of given quantity
    
    INPUTS:
    io - BudgetIO of DeficitIO object
    budget_terms - string or list of strings that correspond to budget terms
    direction - 'x', 'y', or 'z'. Indicates the normal direction of the surface
    coords (optional) - tuples of x, y, or z coordinates to take the flux over
    """

    if isinstance(budget_terms, str):
        budget_terms = [budget_terms]
    elif not isinstance(budget_terms, list):
        print("positional argument budget_terms must be either a string or a list.")
        return 

    if isinstance(io, padeopsIO.DeficitIO):
        velocities = ['delta_u', 'delta_v', 'delta_w']
    elif isinstance(io, padeopsIO.BudgetIO):
        velocities = ['ubar', 'vbar', 'wbar']
    else:
        print("keyword argument io must be either a BudgetIO or DeficitIO object.")
        return

    if not coords:
        xid = slice(0, len(io.xLine))
        yid = slice(0, len(io.yLine))
        zid = slice(0, len(io.zLine)) 
    else:
        xid, yid, zid = io.get_xids(x=coords[0], y=coords[1], z=coords[2], return_none=True, return_slice=True)

    if direction == 'x':
        vel = velocities[0]
        axis = (1,2)
        dx1 = io.dy
        dx2 = io.dz
    elif direction == 'y':
        vel = velocities[1]
        if streamwise:
            axis = 2
            dx1 = 1
            dx2 = io.dz
        else:
            axis = (0,2)
            dx1 = io.dx
            dx2 = io.dz
    elif direction == 'z':
        vel = velocities[2]
        if streamwise:
            axis = 1
            dx1 = 1
            dx2 = io.dy
        else:
            axis = (0,1)
            dx1 = io.dx
            dx2 = io.dy
    else:
        print("positional argument direction must be either 'x', 'y', or 'z'.") 
        return
        

    if yc is not None:
        # check to make sure that yc is the same length as xcoords
        if len(io.xLine[xid]) != len(yc) and len(io.xLine) != len(yc):
            print("yc must have the same length as the x coordinates or be the total length of the domain")
            return
        else:
            y_r =  (io.yLine[yid.stop-1] - io.yLine[yid.start])/2
            for term in budget_terms:
                if term + '_flux_' + direction + suffix in io.budget and not overwrite:
                    return
                else:
                    if vel not in io.budget:
                        io.read_budgets(budget_terms=[vel])
                    if term not in io.budget:
                        io.read_budgets(budget_terms=[term])
                #io.budget[term + '_flux_' + direction + suffix] = np.zeros(
                tmp = []
                for i in range(xid.start, xid.stop):
                    xid, yid, zid = io.get_xids(x=io.xLine[i], y=[yc[i] - y_r, yc[i] + y_r], z=coords[2], return_none=True, return_slice=True)
                    tmp.append(np.sum(io.budget[vel][xid, yid, zid] * io.budget[term][xid,yid,zid], axis = axis ) * dx1 * dx2)

                io.budget[term + '_flux_' + direction + suffix] = np.array(tmp).ravel()
                if isinstance(axis, tuple) and 0 in axis:
                    io.budget[term + '_flux_' + direction + suffix] = np.sum(io.budget[term + '_flux_' + direction + suffix], axis=0)
    else:
        for term in budget_terms:
            # check to see if flux term exists
            if term + '_flux_' + direction + suffix in io.budget and not overwrite:
                return
            else:
                if vel not in io.budget:
                    io.read_budgets(budget_terms=[vel])
                if term not in io.budget:
                    io.read_budgets(budget_terms=[term])
                io.budget[term + '_flux_' + direction + suffix] = np.sum(io.budget[vel][xid, yid, zid] * io.budget[term][xid,yid,zid], axis = axis ) * dx1 * dx2
                

    
def TI_calc_hh(prim, pre):
    """
    Calculate I0 and I_wake at hub height
    """

    zh = np.argmin(np.abs(prim.zLine))

    uh = np.mean(pre.budget['ubar'][...,zh], axis=(0,1))

    I0 = np.sqrt(pre.budget['uu'] + pre.budget['vv'])/uh

    _, yid, zid = prim.get_xids(y=[-1,1], z=[0,1], return_none=True, return_slice=True)

    print(yid, zid)

    I_wake = np.sqrt(prim.budget['uu'] + prim.budget['vv'])/uh
 
    return I0, I_wake


def TI_calc(io):
    
    I = np.sqrt(io.budget['uu'] + io.budget['vv'])

    return I

def vorticity_calc(io):

    if 'dUdx' not in io.budget:
        io.grad_calc()

    io.budget['omega_x'] = io.budget['dWdy'] - io.budget['dVdz']
    io.budget['omega_y'] = io.budget['dUdz'] - io.budget['dWdx']
    io.budget['omega_z'] = io.budget['dVdx'] - io.budget['dUdy']