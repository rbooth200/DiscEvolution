################################################################################
# brent.py                                                                     #
# Author: R. Booth                                                             #
# Date: 18 - May - 2018                                                        #
#                                                                              #
# This file contains a vectorized implementation of Brents method. It is based #
# on implmentation in scipy, by Charles Harris charles.harris@sdl.usu.edu.     #
#                                                                              #
# The main change here is to allow the code to solve a vector of problems,     #
#   f(x_i) = 0,                                                                #
# in the case where each of the problems is indepedent.                        #
#                                                                              #
# Note:                                                                        #
#     Formally this code is less efficient than an a simple scalar             #
#     implementation as the number of iterations will be set by the maximum    #
#     number required for any of the sub-problems, and each interation also    #
#     requires more work.                                                      #
#                                                                              #
#     However, this can still be useful for solving problems that have been    #
#     vectorized with numpy as the time saved by vectorising the problem can   #
#     be ~2 orders of magnitude.                                               #
################################################################################
import numpy as np

def brentq(f, xa, xb, xtol=1e-7, rtol=1e-7, max_iter=100, 
           raise_failure=True):
    """Find's the roots of a system of independent problems via Brent's method.
    
    Formally, this code solves the equation,
        f_i(x_i) = 0,
    for a vector of points, {x_i}, under the assumption that the Jacobian matrix
        df_i / dx_j = 0,
    for all off-diagonal elements.

    args:
        f      : vectorized function to find the roots of.
        xa, xb : Bounds of the region. Requires f(xa)*f(xb) < 0.
        xtol   : Aboltute tolerence required for the root
        rtol   : Relative tolerence required.        
        max_iter : Maximum number of iterations, defaults to 100
        raise_failure : If true, raise an error if the iterations do not 
                        converge in max_iter steps. Otherwise return the best
                        guess.
    """
    xpre = xa;  xcur = xb
    xblk = 0.0; fblk = 0.0;  spre = 0.0; scur = 0.0

    fpre = f(xpre)
    fcur = f(xcur)

    if np.any(fpre*fcur > 0): raise ValueError("brentq: Region not bounded")

    # Check for any values which already solutions
    root = np.where(fpre == 0, xpre, np.where(fcur == 0, xcur, 0))
    done = (fpre == 0) | (fcur == 0)
    
    if np.all(done): return root

    # Protect the roots we already have
    xpre = np.where(done, root, xpre)
    xcur = np.where(done, root, xcur)
    xblk = np.where(done, root, xblk)
    
    fpre = np.where(done, 0, fpre)
    fcur = np.where(done, 0, fcur)
    fblk = np.where(done, 0, fblk)

    for _ in range(max_iter):
        args = (fpre*fcur < 0)
        xblk = np.where(args, xpre,        xblk)
        fblk = np.where(args, fpre,        fblk)
        spre = np.where(args, xcur - xpre, spre)
        scur = np.where(args, spre,        scur)
 
        args = (np.abs(fblk) < np.abs(fcur))
        xpre = np.where(args, xcur, xpre)
        xcur = np.where(args, xblk, xcur)
        xblk = np.where(args, xpre, xblk)

        fpre = np.where(args, fcur, fpre)
        fcur = np.where(args, fblk, fcur)
        fblk = np.where(args, fpre, fblk)

        tol  = xtol + rtol*np.abs(xcur)
        sbis = (xblk - xcur) / 2.

        args = (fcur == 0) | (np.abs(sbis) < tol) 
        root = np.where(args, xcur, root)
        done |= args

        if np.all(done): return root

        bisect = (np.abs(spre) <= tol) | (np.abs(fcur) >= np.abs(fpre))

        # Extrapolate / interpolate:
        with np.errstate(invalid='ignore'):
            dpre = (fpre - fcur + 1e-300)/(xpre - xcur + 1e-300)
            dblk = (fblk - fcur + 1e-300)/(xblk - xcur + 1e-300)
            stry = np.where(xpre == xblk, 
                            -fcur/dpre,                        # Interpolate
                            -fcur*(fblk*dblk - fpre*dpre) /    # Extrapolate
                            (dblk*dpre*(fblk - fpre+1e-300)))

            args = 2*np.abs(stry) >= np.minimum(np.abs(spre),
                                                3*np.abs(sbis) - tol)
        spre = np.where(args | bisect, sbis, scur)
        scur = np.where(args | bisect, sbis, stry)

        xpre = xcur ; fpre = fcur

        xcur = xcur + np.where((np.abs(scur) > tol) | done, scur, 
                               np.where(sbis > 0, tol, -tol))
        fcur = f(xcur)

    if raise_failure:
        raise RuntimeError("Iteration failed to converge")
    else:
        return np.where(done, root, xcur)
