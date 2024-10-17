#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import external public "common" modules
import numpy as np


def step_adv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    a: float,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    **kwargs,
) -> np.ndarray:
    r"""
    Right-hand side of Burger's eq. where a can be a constant or a function that
    depends on xx.

    Requires
    ----------
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default, clf_cut=0.98.
    ddx : `lambda function`
        Allows the selection of the type of spatial derivative.
        By default lambda x,y: deriv_fwd(x, y)

    Returns
    -------
    `array`
        Time interval.
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """


def cfl_adv_burger(a: float, x: np.ndarray) -> float:
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and
    Lewy's condition for the advective term in Burger's equation.

    Parameters
    ----------
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    x : `array`
        Spatial axis.

    Returns
    -------
    `float`
        min(dx/|a|)
    """


def evolv_adv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: float,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list = [0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    nt : `int`
        Number of time iterations.
    a : `float` or `array`
        Either constant or array, which multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y).
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels
        will need to be updated with the boundary information.
        By default [0,1].

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
