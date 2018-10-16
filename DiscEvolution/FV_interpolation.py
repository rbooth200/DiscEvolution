# FV_interpolation.py
#
# Author: R. Booth
# Date: 12 - Jan - 2018
#
# Contains classes for constructing interpolation weights for finite-volume and
# finite-difference formulae.
################################################################################
from __future__ import print_function
import numpy as np
from scipy import sparse


def _construct_volume_factors(xi, m, order, dtype):
    '''Evaluates the left-most matrix in Mignone's equation 21 (the matrix B
    in Appendix A)

    args:
        xi    : Cell edge locations
        order : Order of the reconstruction
        dtype : numpy data-type to use
    '''
    beta = np.empty([len(xi) - 1, order], dtype=dtype)
    for n in range(order):
        beta[:, n] = np.diff(xi**(m+n+1)) / (m + n + 1.)

    beta.T[:] /= beta[:, 0]

    return beta

def _construct_difference_factors(xc, order, dtype):
    '''Evaluates the coefficient matrix for the finite-differnce interpolation
    formulae.

    args:
        xc    : Cell centre locations
        order : Order of the reconstruction
        dtype : numpy data-type to use
    '''
    beta = np.empty([len(xc) , order], dtype=dtype)
    for n in range(order):
        beta[:, n] = xc**n

    return beta


def _construct_poly_derivs(xk, order, dtype):
    '''Evaluates the RHS of Mignone's equation 21, along with its derivatives

    args:
        xi    : Cell edge locations
        order : Order of the reconstruction
        dtype : numpy data-type to use        
    '''
    rhs = np.zeros([order, order], dtype=dtype)
    eta = np.power(xk, np.arange(order))
    rhs[:, 0] = eta
    for n in range(1, order):
        rhs[n:,n] = rhs[n-1:order-1,n-1]*range(n, order)
    return rhs


def _solve_FV_matrix_weights(xi, iL, iR, beta, max_deriv, dtype):
    '''Solves Mignone's equation 21, along with its derivatives'''
    order = 1 + iL + iR
    
    w = np.zeros([len(xi), max_deriv+1, order], dtype=dtype)
    for i in range(0, len(xi)):
        
        start  = max(0,       i-iL)
        end    = min(len(xi), i+iR+1)
        N      = end - start
        N_term = min(N, max_deriv+1)
        
        beta_i = beta[start:end,:N]
       
        # Solve for the coefficients
        rhs = _construct_poly_derivs(xi[i], N_term, dtype)
        w[i, :N, start-i+iL:end-i+iL] = np.linalg.solve(beta_i.T, rhs).T
        
    return w

def _solve_FD_matrix_weights(beta, iL, iR, beta_FD, max_deriv, dtype):
    '''Solves for the weights used to reconstruct the volume averages from
    the centroid values.
    '''
    order = 1 + iL + iR
    
    w = np.zeros([len(beta), order], dtype=dtype)
    for i in range(0, len(beta)):
        
        start  = max(0,       i-iL)
        end    = min(len(beta), i+iR+1)
        N      = end - start
        N_term = min(N, max_deriv+1)
        
        # Solve for the coefficients
        w[i, start-i+iL:end-i+iL] = \
            np.linalg.solve(beta_FD[start:end,:N].T, beta[i,:N]).T
        
    return w

def compute_centroids(xe, m):
    '''First order upwind reconstruction

    args:
        xe : Cell edge locations
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
    '''
    return ((m + 1) * np.diff(xe**(m+2))) / ((m + 2) * np.diff(xe**(m+1)))
    
def compute_FV_weights(xe, xj, m, iL, iR, max_deriv=None, dtype='f8'):
    '''Solves for the finite-volume interpolation weights.

    This code follows the methods outlined in Mignone (2014, JCoPh 270 784) to
    compute the weights needed to reconstruct a function and its derivatives
    to the specified locations. The polynomial is reconstructed to
    reproduce the averages of the cell and its neighbours.

    Note that the polynomial computed near the domain edges will be lower 
    order, which also reduces the number of derivatives available

    args:
        xe : locations of cell edges
        xj : Reconstruction points (one for each cell).
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
        iL : Number of cells to the left of the target cell to use in the 
             interpolation.
        iR : Number of cells to the right of the target cell to use in the 
             interpolation.
        
        max_deriv : maximum derivative level to calculate. If not specified,
                    return all meaningful derivatives.
    
        dtype : data type used for calculation, default = 'f8'

    returns:
        w : The weights used for reconstructing the function and its 1st
            iL+iR derivatives to the specified points.
            The shape is [len(xj), max_deriv+1, 1+iL+iR]
    '''
    # Order of the polynomial
    order = 1 + iL + iR

    if max_deriv is None:
        max_deriv = order - 1
    elif max_deriv > order - 1:
        raise ValueError("Maximum derivative must be less than the order of the"
                         " polynomial fitted")
    
    # Setup the beta matrix of Mignone:
    beta =  _construct_volume_factors(xe, m, order, dtype)

    # Return the interpolated values
    return _solve_FV_matrix_weights(xj, iR, iL, beta, max_deriv, dtype)


def compute_FD_weights(xc, xj, iL, iR, max_deriv=None, dtype='f8'):
    '''Solves for the finite-difference interpolation weights.

    This code follows the methods outlined in Mignone (2014, JCoPh 270 784),
    adapted to point-wise values, to compute the weights needed to reconstruct
    a function and its derivatives to the specified locations. The polynomial
    is reconstructed to  reproduce the point-wise values of the cell and its
    neighbours.

    Note that the polynomial computed near the domain edges will be lower 
    order, which also reduces the number of derivatives available

    args:
        xc : Locations of cell centroids / point-wise data
        xj : Reconstruction points (one for each cell).
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
        iL : Number of cells to the left of the target cell to use in the 
             interpolation.
        iR : Number of cells to the right of the target cell to use in the 
             interpolation.
        
        max_deriv : maximum derivative level to calculate. If not specified,
                    return all meaningful derivatives.
    
        dtype : data type used for calculation, default = 'f8'

    returns:
        w : The weights used for reconstructing the function and its 1st
            iL+iR derivatives to the specified points.
            The shape is [len(xj), max_deriv+1, 1+iL+iR]
    '''
    # Order of the polynomial
    order = 1 + iL + iR

    if max_deriv is None:
        max_deriv = order - 1
    elif max_deriv > order - 1:
        raise ValueError("Maximum derivative must be less than the order of the"
                         " polynomial fitted")
    
    # Setup the beta matrix for finite-difference formulae
    beta =  _construct_difference_factors(xc, order, dtype)

    # Return the interpolated values
    return _solve_FV_matrix_weights(xj, iL, iR, beta, max_deriv, dtype)

        
def construct_FV_edge_weights(xi, m, iL, iR, max_deriv=None, dtype='f8'):
    '''Solves for the finite-volume interpolation weights.

    This code follows the methods outlined in Mignone (2014, JCoPh 270 784) to
    compute the weights needed to reconstruct a function and its derivatives
    to edges of computational cells. The polynomial is reconstructed to
    reproduce the averages of the cell and its neighbours.

    Note that the polynomial computed near the domain edges will be lower 
    order, which also reduces the number of derivatives available

    args:
        xi : locations of cell edges
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
        iL : Number of cells to the left of the target cell to use in the 
             interpolation.
        iR : Number of cells to the right of the target cell to use in the 
             interpolation.
        
        max_deriv : maximum derivative level to calculate. If not specified,
                    return all meaningful derivatives.
    
        dtype : data type used for calculation, default = 'f8'

    returns:
        wp, wm : The weights used for reconstructing the function and its 1st
                 iL+iR derivatives to the left and right of the cell edges. 
                 The shape is [len(xi)-1,max_deriv+1, 1+iL+iR]
    '''
    # Order of the polynomial
    order = 1 + iL + iR

    if max_deriv is None:
        max_deriv = order - 1
    elif max_deriv > order - 1:
        raise ValueError("Maximum derivative must be less than the order of the"
                         " polynomial fitted")
    
    # Setup the beta matrix of Mignone:
    beta =  _construct_volume_factors(xi, m, order, dtype)

    # The matrix of extrapolations to the RHS of cell
    wp = _solve_FV_matrix_weights(xi[1:], iL, iR, beta, max_deriv, dtype)
        
    # The matrix of extrapolations to the LHS of cell
    wm = _solve_FV_matrix_weights(xi[:-1], iR, iL, beta, max_deriv, dtype)

    return wp, wm


def construct_FV_centroid_weights(xi, m, iL, iR, max_deriv=None, dtype='f8'):
    '''Solves for the finite-volume interpolation weights.

    This code follows the methods outlined in Mignone (2014, JCoPh 270 784) to
    compute the weights needed to reconstruct a function and its derivatives
    to centroid of the computational cells. The polynomial is reconstructed to
    reproduce the averages of the cell and its neighbours.

    Note that the polynomial computed near the domain edges will be lower 
    order, which also reduces the number of derivatives available

    args:
        xi : locations of cell edges
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
        iL : Number of cells to the left of the target cell to use in the 
             interpolation.
        iR : Number of cells to the right of the target cell to use in the 
             interpolation.
        
        max_deriv : maximum derivative level to calculate. If not specified,
                    return all meaningful derivatives.
    
        dtype : data type used for calculation, default = 'f8'

    returns:
        xc : Cell centroids
        wc : The weights used for reconstructing the function and its 1st
             iL+iR derivatives to the cell centroid
             The shape is [len(xi)-1,max_deriv+1, 1+iL+iR]
        wv : The wieghts for reconstructing the volume averaged quantities from
             the cell centroid values. The shape is [len(xi)-1, 1+iL+iR]
    '''
    # Order of the polynomial
    order = 1 + iL + iR

    if max_deriv is None:
        max_deriv = order - 1
    elif max_deriv > order - 1:
        raise ValueError("Maximum derivative must be less than the order of the"
                         " polynomial fitted")
    
    # Setup the beta matrix of Mignone:
    beta = _construct_volume_factors(xi, m, order, dtype)

    # The matrix of extrapolations to the centroid of cell
    #    Note: xc is equivalent to beta[:,1]
    xc = compute_centroids(xi, m)
    wc = _solve_FV_matrix_weights(xc, iL, iR, beta, max_deriv, dtype)

    # Also build the inverse transform: cell centroid to volume average
    #     To do this we compute the volume integral of finite-difference
    #     interpolation formula
    beta_FD = _construct_difference_factors(xc, order, dtype)
    wv = _solve_FD_matrix_weights(beta, iL, iR, beta_FD, max_deriv, dtype)
    
    return xc, wc, wv

def join_symmetric_stencil(wp, wm):
    '''Join together the weights in the case of a symmetric stencil.

    In this case both the wp and wm weights for the same edge are equal.
    '''
    if wp.shape != wm.shape:
        raise AttributeError("Error:Left/Right weights must have equal shapes")
    if wp.shape[1] % 2:
        raise AttributeError("Error: Weights must have an even stencil")

    w = np.concatenate([wm[:1], wp], axis=0)

    return w

def _build_sparse_matrix(w,centroid=False):
    '''Builds a spare matrix from the weights for easy evaulation of the
    reconstruction.
    '''
    # Compute the shape of the stencil for centroid / edge values
    s = w.shape[1] // 2
    M = w.shape[0]
    if centroid:
        N = M
        stencil = np.arange(-s, s+1)
    else:
        N = M - 1
        stencil = np.arange(-s, s)

    # Seperate out the diagonals
    diags = []
    for j in range(w.shape[1]):
        ji = max(s-j, 0)
        je = min(N+s-j,M)
        diags.append(w[ji:je,j])

    # Create and return the sparse matrix
    return sparse.diags(diags, stencil, shape=(M,N))
    
class FV_Centred_Interpolator(object):
    '''Finite-volume interpolator using centered slopes.

    This class is designed for interpolating quantities volume-averaged over a
    cell to the cell edges. It uses an even stencil, i.e. the same number of 
    points on each side of the edge. At the edge of the domain a lower-order
    interpolation is used instead.

    args:
        xe     : Locations of the cell edges
        m      : Index of the radial scaling. The volume element is given by:
                 dV_i = (R_i+1^m+1 - R_i^m+1) / m + 1
       stencil : Number of points to use on each side of the edge

    -----
    Notes:
        Transforming the volume-averaged values to centroid values and back
        using the centroid and volume average functions will result in changes
        to values because these transformations are not unitary. This means that
        the centroid to volume average conversion should not be used if the 
        volume average is otherwise available.
    -----
    '''
    def __init__(self, xe, m, stencil):

        # Compute the finite-volume weights:
        wp, wm = construct_FV_edge_weights(xe, m, stencil-1, stencil)
        w = join_symmetric_stencil(wp, wm)

        # Create sparse matrices for each set of weights
        wgts = []
        N = len(xe)
        
        for i in range(stencil+1):
            wgts.append(_build_sparse_matrix(w[:,i], False))

        self._stencil = stencil
        self._wgts = wgts

        # Also setup functions for converting between volume averaged and cell
        # centroid values. A smaller symmetric stencil is sufficient
        s = max(stencil - 1, 0)
        xc, wc, wv = construct_FV_centroid_weights(xe, m, s, s)

        wgts_c = []
        for i in range(s+1):
            wgts_c.append(_build_sparse_matrix(wc[:,i], True))            
        wv = _build_sparse_matrix(wv,True)

        self._xc = xc
        self._wgts_c = wgts_c
        self._wgts_vol = wv

        # Now do the finite-difference weights for the centroid values
        #  Edge Values:
        wp = compute_FD_weights(xc, xe[1:], stencil-1, stencil)
        wm = compute_FD_weights(xc, xe[:-1], stencil-1, stencil)
        w = join_symmetric_stencil(wp, wm)

        wgts_fde = []
        for i in range(stencil+1):
            wgts_fde.append(_build_sparse_matrix(w[:,i], False))

        #  Centroid Values:
        w = compute_FD_weights(xc, xc, s, s)

        # Directly use the identity matrix for the interpolation to centroids
        # to avoid problems with roundoff
        wgts_fdc = [sparse.identity(w.shape[0],format='dia')]
        for i in range(1,s+1):
            wgts_fdc.append(_build_sparse_matrix(w[:,i], True))
        
        self._wgts_fd   = wgts_fde
        self._wgts_c_fd = wgts_fdc

    def edge(self, fc, deriv=0, FV=True):
        '''Interpolate the volume-averaged data or its derivatives to the cell
        edges.

        args:
            fc    : Volume averaged data
            deriv : Order of the derivative in the range [0, stencil]
            FV    : Are we interpolating volume-averaged data (True) or
                    centroid data (False). Default = True
        '''
        if FV:
            return self._wgts[deriv].dot(fc)
        else:
            return self._wgts_fd[deriv].dot(fc)

    def centroid(self, fc, deriv=0, FV=True):
        '''Interpolate the volume-averaged data or its derivatives to the cell
        centroids.

        args:
            fc    : Volume averaged data
            deriv : Order of the derivative in the range [0, stencil]
            FV    : Are we interpolating volume-averaged data (True) or
                    centroid data (False). Default = True
        '''
        if FV:
            return self._wgts_c[deriv].dot(fc)
        else:
            return self._wgts_c_fd[deriv].dot(fc)

    def volume_average(self, fc):
        '''Compute the volume averaged data from the centroid data.

        args:
            fc    : Centroid data
        '''
        return self._wgts_vol.dot(fc)
        
    @property
    def stencil(self):
        return self._stencil

    @property
    def centroids(self):
        return self._xc

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    xi = np.arange(7)
    wp, wm = construct_FV_edge_weights(xi, 0, 1, 2)

    print('Edge values')
    print(wp[0])
    print(wm[0])
    print()
    print(wp[3])
    print(wm[3])
    print()
    print(wp[-1])
    print(wm[-1])

    xc, wc, wv = construct_FV_centroid_weights(xi, 1, 2, 2)

    print('Centroid values')
    print(wc[0,0])
    print(wv[0])
    print()
    print(wc[3,0])
    print(wv[3])
    print()
    print(wc[-1,0])
    print(wv[-1])

    # Check that it runs reasonably quickly and that we can join symmetric
    # stencils
    xi = np.logspace(-1, 2, 3*10**2+1)
    wp, wm = construct_FV_edge_weights(xi, 1, 1, 2)
    w =  join_symmetric_stencil(wp, wm)
    
    assert (all((w[0] == wm[0]).flat))
    assert (all((w[1] == wp[0]).flat))

    ### Run a test interpolation / derivative

    # Volume average of 1/x^2
    Vx = 0.5 * (xi[1:]**2 -  xi[:-1]**2)                               
    fx = np.log(xi[1:]/xi[:-1]) / Vx
    
    # Interpolators with 1, 2 and 3 point stencils
    FV1 = FV_Centred_Interpolator(xi, 1, 1)
    FV2 = FV_Centred_Interpolator(xi, 1, 2)
    FV3 = FV_Centred_Interpolator(xi, 1, 3)

    xc = FV1.centroids
    plt.subplot(211)
    plt.loglog(xi, abs(1 - FV1.edge(fx,deriv=0)*xi**2))
    plt.loglog(xi, abs(1 - FV2.edge(fx,deriv=0)*xi**2))
    plt.loglog(xi, abs(1 - FV3.edge(fx,deriv=0)*xi**2))
    plt.ylabel('Relative error on f(R_edge)')
    plt.subplot(212)
    plt.loglog(xi, abs(1 + FV1.edge(fx,deriv=1)*xi**3/2))
    plt.loglog(xi, abs(1 + FV2.edge(fx,deriv=1)*xi**3/2))
    plt.loglog(xi, abs(1 + FV3.edge(fx,deriv=1)*xi**3/2))
    plt.ylabel('Relative error on f\'(R_edge)')
    plt.xlabel('R')

    plt.figure()
    plt.subplot(211)
    plt.loglog(xc, abs(1 - FV1.centroid(fx,deriv=0)*xc**2))
    plt.loglog(xc, abs(1 - FV2.centroid(fx,deriv=0)*xc**2))
    plt.loglog(xc, abs(1 - FV3.centroid(fx,deriv=0)*xc**2))
    plt.ylabel('Relative error on f(R_c)')
    plt.subplot(212)
    plt.loglog(xc, abs(1 - FV1.volume_average(FV1.centroid(fx))/fx))
    plt.loglog(xc, abs(1 - FV2.volume_average(FV2.centroid(fx))/fx))
    plt.loglog(xc, abs(1 - FV3.volume_average(FV3.centroid(fx))/fx))
    plt.ylabel('Relative error on Unitarity')
    plt.xlabel('R')
    plt.show()
