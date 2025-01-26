"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

import numpy as np

from nm_lib.nm_ex.nm_lib_ex_1 import deriv_fwd


def cfl_diff_burger(a: float, x: np.ndarray) -> float:
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and
    Lewy condition for the diffusive term in the Burger's eq.

    Parameters
    ----------
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    x : `array`
        Spatial axis.

    Returns
    -------
    `float`
        min(0.5 dx**2 / nu)
    """


def step_diff_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    a: float,
    **kwargs,
) -> np.ndarray:
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that
    depends on xx. It will benefit from nm_lib_ex_1 functions

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.

    Returns
    -------
    `array`
        Diffusive part from right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., \frac{\partial^2 u}{\partial x^2}
    """


def evolv_diff_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: float,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv2_cent(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a fix constant or array.
    Requires
    ----------
    step_diff_burgers
    cfl_diff_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array, which multiplies the right-hand side of the DIFFUSIVE Burger's eq.
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


def step_diff_variable(
    xx: np.ndarray,
    hh: np.ndarray,
    mu=lambda x, y: y,
) -> np.ndarray:
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that
    depends on xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """


def evolv_diff_variable(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    mu=lambda x, y: y,
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a fix constant or array.
    Requires
    ----------
    step_diff_burgers
    cfl_diff_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array, which multiplies the right-hand side of the DIFFUSIVE Burger's eq.
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


def NR_f(
    xx: np.ndarray,
    un: np.ndarray,
    uo: np.ndarray,
    a: float,
    dt: float,
    **kwargs,
) -> np.ndarray:
    r"""
    NR F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        A function that depends on xx.
    uo : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    dt : `float`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """


def jacobian(xx: np.ndarray, un: np.ndarray, a: float, dt: float, **kwargs) -> np.ndarray:
    r"""
    Jacobian of the F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    dt : `float`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """


def Newton_Raphson(
    xx: np.ndarray,
    hh: np.ndarray,
    a: np.ndarray,
    dt: float,
    nt: int,
    toll: float = 1e-5,
    ncount: int = 2,
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
):
    r"""
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of time iterations.
    toll : `float`
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default, 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """
    if bnd_limits is None:
        bnd_limits = [0, 1]
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros(nt)
    countt = np.zeros(nt)
    unnt[:, 0] = hh
    t = np.zeros(nt)

    # Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):
            jac = jacobian(xx, ug, a, dt)  # Jacobian
            ff1 = NR_f(xx, ug, uo, a, dt)  # F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error:
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0] : -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0] :]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt


def NR_f_u(
    xx: np.ndarray,
    un: np.ndarray,
    uo: np.ndarray,
    dt: float,
    **kwargs,
) -> np.ndarray:
    r"""
    NR F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        A function that depends on xx.
    uo : `array`
        A function that depends on xx.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - u^{n}_{j} (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """


def jacobian_u(
    xx: np.ndarray,
    un: np.ndarray,
    dt: float,
    **kwargs,
) -> np.ndarray:
    """
    Jacobian of the F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        A function that depends on xx.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """


def Newton_Raphson_u(
    xx: np.ndarray,
    hh: np.ndarray,
    dt: float,
    nt: int,
    toll: float = 1e-5,
    ncount: int = 2,
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
):
    """
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    dt : `float`
        Time interval
    nt : `int`
        Number of time iterations.
    toll : `float`
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default, 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `array(int)`
        Number iterations for each timestep
    """
    if bnd_limits is None:
        bnd_limits = [0, 1]
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros(nt)
    countt = np.zeros(nt)
    unnt[:, 0] = hh
    t = np.zeros(nt)

    # Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):
            jac = jacobian_u(xx, ug, dt)  #  Jacobian
            ff1 = NR_f_u(xx, ug, uo, dt)  #  F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0] : -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0] :]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt


def taui_sts(nu: float, niter: int, iiter: int) -> float:
    """
    STS parabolic scheme. [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}

    Parameters
    ----------
    nu : `float`
        Coefficient, between (0,1).
    niter : `int`
        Number of time iterations.
    iiter : `int`
        Iterations number

    Returns
    -------
    `float`
        [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}
    """


def evol_sts(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    nu: float = 0.9,
    n_sts: float = 10,
):
    """
    Evolution of the STS method. It will benefit from nm_lib_ex_1 functions

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
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.45
    bnd_type : `string`
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default, [0,1]
    nu : `float`
        STS nu coefficient between (0,1).
        By default 0.9
    n_sts : `int`
        Number of STS sub iterations.
        By default 10

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), where j represents
        all the elements of the domain.
    """
