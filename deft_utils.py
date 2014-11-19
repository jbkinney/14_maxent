import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.misc import comb
import matplotlib as mpl
from scipy.interpolate import interp1d

# Empty container class for storing results
class Results: pass;

blue = [0.,0.,1.]
orange = [1.,0.5,0.]

def get_colormap():
    xs = sp.linspace(0,100,4)
    xgrid = sp.linspace(0,100,1000)
    ys = sp.array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 1.0, 1.0],
         #[1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0]])
    f = interp1d(xs,ys.T)
    C = f(xgrid).T
    #print C.shape()

    cm = mpl.colors.ListedColormap(C)
    return cm
#
# Computes derivative matrix
#
def derivative_matrix(G, grid_spacing=1.0):
    '''
    Returns a (G-1)xG dimensional derivative matrix. 
    '''
    
    # Make sure G is a positive integer
    assert(G == int(G) and G >= 2)
    
    # Create matrix
    tmp_mat = sp.diag(sp.ones(G),0) + sp.diag(-1.0*sp.ones(G-1),-1)
    right_partial = tmp_mat[1:,:]/grid_spacing
    
    return sp.mat(right_partial)

#
# Computes the bilateral laplacian
#
def bilateral_laplacian(G, alpha, grid_spacing=1.0):
    '''
    Returns a GxG bilateral laplacian matrix
    '''

    # Make sure G is a positive integer, alpha is a positive integer, and
    # G is larger than alpha
    assert(G==int(G) and alpha==int(alpha) and alpha>0 and G>alpha)
    
    # Initialize to GxG identity matrix
    right_side = sp.diag(sp.ones(G),0)
    
    # Multiply alpha derivative matrices of together. 
    # Reduce dimension going left
    for a in range(alpha):
        right_side = derivative_matrix(G-a, grid_spacing)*right_side
    
    # Return final bilateral_laplacian
    return right_side.T*right_side

#
# Compute the moments and cumulants of a distribution
#

def get_moments(x, Q, num=10):
    assert x.shape == Q.shape
    assert all(Q >= 0)
    
    h = x[1]-x[0]
    mus = sp.array([sum(Q*h*(x**n)) for n in range(num)])
    return mus

# Compute cumulants
def get_cumulants(x, Q, num=10):
    assert x.shape == Q.shape
    assert all(Q >= 0)
    
    # First compute moments
    mus = get_moments(x,Q,num=num)
    
    # Then compute cumulants using recursive relation
    ks = sp.ones(mus.shape)
    for n in range(1,num):
        ks[n] = mus[n] - sum([comb(n-1,m-1)*ks[m]*mus[n-m] for m in range(1,n)])
    
    return ks

#
# Compute maxent density
# 
def maxent_1d(counts, L, alpha):
    '''
    maxent_1d
    
    Estimates a maximum entropy in 1D
    
    Args:
        counts: histogram of the raw data. It is assumed that the histogram 
            evenly title the length L
            
        L: length of the interval on which the data histogram lives
        
        alpha: number of moments to constrain. Specifically, function will
            return a probability density whos x^0, x^1, ..., x^(alpha-1) moments
            match those of the data. 
            
    Returns:
        Q: value of maxent density at centers of histogram bins
        
        results: structure containing more detailed info; see below
        
        results.Q: same as Q above
        
        results.phi: the maxent field
        
        results.coeffs: polynomial coefficients of maxent field
        
        results.convergence: details on convergence of minimization procedure
        
        results.success: True if minimization procedure converged, False
            otherwise
    '''
    
    # I assume the xs are evenly spaced
    G = 1.*len(counts)
    h = L/G

    # Use only zs in interval [-Z, Z]; provides numerical stability.
    # Z = 1 seems to work best. 
    Z = 1.
    dz = 2./G
    zs = sp.linspace(-Z+dz/2., Z-dz/2.,G)

    # Compute normalized density
    R = counts/sp.sum(dz*counts)

    # Compute moments of data distribution
    mu = sp.zeros(alpha)
    for i in range(alpha):
        mu[i] = dz*sp.sum(R*zs**(alpha-1-i))

    # Define action
    def action(a):
        quasiQ = sp.exp(-sp.polyval(a,zs))/(2.*Z)
        return h*sp.sum(R*sp.polyval(a,zs) + quasiQ)

    # Define gradient function
    def grad(a):
        grad = sp.zeros(alpha)
        quasiQ = sp.exp(-sp.polyval(a,zs))/(2.*Z)
        for i in range(alpha):
            grad[i] = mu[i] - dz*sp.sum((zs**(alpha-1-i))*quasiQ)
        return grad
    
    # Define Hessian function
    def Hess(a):
        Hess = sp.zeros([alpha,alpha])
        quasiQ = sp.exp(-sp.polyval(a,zs))/(2.*Z)
        entries = [h*sp.sum((zs**(2*alpha-2-k))*quasiQ) for k in range(2*alpha-1)]
        for i in range(alpha):
            for j in range(alpha):
                Hess[i,j] = entries[i+j]
        return Hess

    # Initialize coefficients
    a0 = sp.zeros(alpha)
    
    # Compute the minimum
    opt_result = opt.minimize(action, a0, method='Newton-CG', jac=grad, hess=Hess)

    # Abort of optimization failed
    if not opt_result.success:
        print 'Falure!'
        assert(False)
    
    # Compute Qs in x-coords
    a = opt_result.x
    Q = sp.exp(-sp.polyval(a,zs))
    Q = Q/sp.sum(h*Q)
    phi = -sp.log(L*Q)
    
    # Gather other results
    results = Results()
    results.Q = Q
    results.phi = phi
    results.coeffs = a
    results.convergence = opt_result
    results.success = True
    
    return [Q, results]