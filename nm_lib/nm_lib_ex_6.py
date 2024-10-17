#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import external public "common" modules
import numpy as np


def ops_Lax_LL_Add(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list = [0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b
    a fix constant or array. Solve two advective terms separately
    with the Additive Operator Splitting scheme.  Both steps are
    with a Lax method.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    nt : `int`
        Number of time iterations.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    b : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y)
    bnd_type : `string`
        It allows one to select the type of boundaries
        By default, 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """


def ops_Lax_LL_Lie(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list = [0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b
    a fix constant or array. Solving two advective terms separately
    with the Lie-Trotter Operator Splitting scheme.  Both steps are
    with a Lax method.

    Requires:
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of time iterations.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    b : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y)
    bnd_type : `string`
        It allows one to select the type of boundaries.
        By default, 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """


def ops_Lax_LL_Strange(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list = [0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b
    a fix constant or array. Solving two advective terms separately
    with the Lie-Trotter Operator Splitting scheme. Both steps are
    with a Lax method.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger
    numpy.pad for boundaries.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of time iterations.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    b : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y)
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default, `wrap`
    bnd_limits : `list(int)`
        The number of pixels that will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """


def osp_Lax_LH_Strange(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list = [0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b
    a fix constant or array. Solving two advective terms separately
    with the Strange Operator Splitting scheme. One step is with a Lax method
    and the second is the Hyman predictor-corrector scheme.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of time iterations.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    b : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y)
    bnd_type : `string`
        It allows one to select the type of boundaries.
        By default, 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """


def hyman(
    xx: np.ndarray,
    f: np.ndarray,
    dth: float,
    a: np.ndarray,
    fold: np.ndarray = None,
    dtold: float = None,
    cfl_cut: float = 0.8,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list = [0, 1],
    **kwargs,
):
    """
    Hyman Corrector-predictor method

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    f : `array`
        A function that depends on xx.
    dth : `float`
        Time step interval
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    fold : `array`
        A function that depends on xx from the previous timestep.
    dtold : `array`
        Time step interval from previous step.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.45
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y)
    bnd_type : `string`
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default, [0,1]

    Returns
    -------
    f : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), where j represents
        all the domain elements.
    fold : `array`
        Spatial and time evolution of u^n_j_1/2 for n = (0,nt), where j represents
        all the domain elements, i.e., from the previous step.
    dt : `float`
        time interval
    """
    dt, u1_temp = step_adv_burgers(xx, f, a, ddx=ddx)

    if np.any(fold) is None:
        fold = np.copy(f)
        f = (np.roll(f, 1) + np.roll(f, -1)) / 2.0 + u1_temp * dth
        dtold = dth
    else:
        ratio = dth / dtold
        a1 = ratio**2
        b1 = dth * (1.0 + ratio)
        a2 = 2.0 * (1.0 + ratio) / (2.0 + 3.0 * ratio)
        b2 = dth * (1.0 + ratio**2) / (2.0 + 3.0 * ratio)
        c2 = dth * (1.0 + ratio) / (2.0 + 3.0 * ratio)

        f, fold, fsav = hyman_pred(f, fold, u1_temp, a1, b1, a2, b2)

        if bnd_limits[1] > 0:
            u1_c = f[bnd_limits[0] : -bnd_limits[1]]
        else:
            u1_c = f[bnd_limits[0] :]
        f = np.pad(u1_c, bnd_limits, bnd_type)

        dt, u1_temp = step_adv_burgers(xx, f, a, cfl_cut, ddx=ddx)

        f = hyman_corr(f, fsav, u1_temp, c2)

    if bnd_limits[1] > 0:
        u1_c = f[bnd_limits[0] : -bnd_limits[1]]
    else:
        u1_c = f[bnd_limits[0] :]
    f = np.pad(u1_c, bnd_limits, bnd_type)

    dtold = dth

    return f, fold, dtold


def hyman_corr(
    f: np.ndarray, fsav: np.ndarray, dfdt: np.ndarray, c2: float
) -> np.ndarray:
    """
    Hyman Corrector step

    Parameters
    ----------
    f : `array`
        A function that depends on xx.
    fsav : `array`
        A function that depends on xx from the interpolated step.
    dfdt : `array`
        A function that depends on xx. The right-hand side of the time derivative.
    c2: `float`
        Coefficient.

    Returns
    -------
    corrector : `array`
        A function of the Hyman corrector step
    """
    return fsav + c2 * dfdt


def hyman_pred(
    f: np.ndarray,
    fold: np.ndarray,
    dfdt: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
):
    """
    Hyman Predictor step

    Parameters
    ----------
    f : `array`
        A function that depends on xx.
    fold : `array`
        A function that depends on xx from the previous step.
    dfdt : `array`
        A function that depends on xx. The right-hand side of the time derivative.
    a1: `float`
        Coefficient.
    b1: `float`
        Coefficient.
    a2: `float`
        Coefficient.
    b2: `float`
        Coefficient.

    Returns
    -------
    f : `array`
        A function that depends on xx.
    fold : `array`
        A function that depends on xx from the previous step.
    fsav : `array`
        A function that depends on xx from the interpolated step.
    """
    fsav = np.copy(f)
    tempvar = f + a1 * (fold - f) + b1 * dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2 * (fsav - tempvar) + b2 * dfdt
    f = tempvar

    return f, fold, fsav
