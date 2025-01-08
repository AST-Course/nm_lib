"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import external public "common" modules
import numpy as np


def deriv_fwd(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    """
    Returns the forward derivative of hh array with respect to xx array.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The forward derivative of hh respect to xx. The last
        grid point is ill (or missing) calculated.
    """


def order_conv(hh: np.ndarray, hh2: np.ndarray, hh4: np.ndarray, **kwargs) -> np.ndarray:
    """
    Computes the order of convergence of a derivative function

    Parameters
    ----------
    hh : `array`
        A function that depends on xx.
    hh2 : `array`
        A function that depends on xx but with twice the number of grid points than hh.
    hh4 : `array`
        A function that depends on xx but with twice the number of grid points than hh2.
    Returns
    -------
    `array`
        The order of convergence.
    """


def deriv_4tho(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    """
    Returns the 4th order derivative of hh with respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.

    Returns
    -------
    `array`
        The centered 4th order derivative of hh with respect to xx.
        The last and first two grid points are ill-calculated.
    """


def deriv_cent(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    r"""
    Returns the centered 2nd derivative of hh with respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh with respect to xx. The first
        and last grid points are ill-calculated.
    """
