#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import external public "common" modules
import numpy as np
from nm_lib.nm_lib_ex_1 import deriv_fwd


def evolv_Lax_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list = [0, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax method.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    nt : `int`
        Number of time iterations.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `array`
        The lambda function allows to change of the space derivative function.
        By derault  lambda x,y: deriv_fwd(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries
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


def Rie_flux(
    hh: np.ndarray,
):
    """
     Flux

     Parameters
     ----------
     hh : `array`
         A function that depends on xx.

    Returns
     -------
     flux : `array`
         h^2
    """


def Rie_va(
    uL: np.ndarray,
    uR: np.ndarray,
):
    """
     absolute propagating speed (va), uses Rie_flux

     Parameters
     ----------
     uL: `array`
         Left and side variable
     uR: `array`
         Right and side variable

    Returns
     -------
     va : `array`
         absolute va speed
    """


def Rie_interface_flux(
    uL: np.ndarray,
    uR: np.ndarray,
):
    """
     absolute propagating speed (va). Uses Rie_va and Rie_flux

     Parameters
     ----------
     uL: `array`
         Left and side variable
     uR: `array`
         Right and side variable

    Returns
     -------
     F*: `array`
         Interface Rusanov flux
    """


def evolv_Rie_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    bnd_type: str = "wrap",
    bnd_limits: list = [1, 1],
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Riemann (Rusanov) method.

    Requires
    --------
    cfl_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `array`
        The lambda function allows to change of the space derivative function.
        By derault  lambda x,y: deriv_fwd(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries
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
