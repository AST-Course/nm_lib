"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

import numpy as np

from nm_lib.nm_lib_ex_1 import deriv_fwd


def deriv_bck(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    r"""
    Returns the backward derivative of hh with respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.

    Returns
    -------
    `array`
        The backward derivative of hh respect to xx. The first
        grid point is ill-calculated.
    """


def step_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    **kwargs,
):
    r"""
    Right-hand side of Burger's eq. where a is u, i.e., hh.

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_fwd(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """


def evolv_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of time iterations.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98.
    ddx : `lambda function`
        Allows to change the space derivative function.
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
