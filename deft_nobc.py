import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from deft_utils import maxent_1d, bilateral_laplacian
from scipy.linalg import det, eigh
from scipy.interpolate import interp1d

class Args: pass;
class Results: pass;

#
# Mesures geodesic distance
#

phi_max = 50
phi_min = -50

# Convert field to non-normalized density
def field_to_quasidensity(phi,L):
    Q = sp.zeros(len(phi))
    indices = (phi < phi_max)*(phi > phi_min)
    Q[indices] = sp.exp(-phi[indices])/L
    Q[phi <= phi_min] = sp.exp(-phi_min)/L
    return Q

# Evaluate geodesic distance (independent of L or h)
def geo_dist(P,Q):
    assert all(P >= 0)
    assert all(Q >= 0)
    P = P/sp.sum(P)
    Q = Q/sp.sum(Q)
    return sp.real(2*sp.arccos(sp.sum(sp.sqrt(P*Q))))

# Evaluate action
def action(phi,args):
    N = args.N
    ell_factor = args.N*sp.exp(-args.t)
    quasiQ = field_to_quasidensity(phi,args.L)
    phi_col = sp.mat(phi).T
    return 0.5*ell_factor*float(phi_col.T*args.Delta*phi_col) + N*sp.sum(args.R*phi + quasiQ)

# Evaluate gradient of action

def grad(phi,args):
    N = args.N
    ell_factor = args.N*sp.exp(-args.t)
    quasiQ = field_to_quasidensity(phi,args.L)
    phi_col = sp.mat(phi).T
    return ell_factor*sp.ravel(args.Delta*phi_col) + N*(args.R - quasiQ)

# Evaluate Hessain of action

def hessian(phi,args):
    N = args.N
    ell_factor = args.N*sp.exp(-args.t)
    quasiQ = field_to_quasidensity(phi,args.L)
    return ell_factor*args.Delta + N*diags(quasiQ,0)

#
# Performs corrector step
#
    
def corrector_step(phi,args):
    convergence = False
    stats = Results()
    stats.num_corrector_steps = 0
    stats.num_action_evaluations = 0
    stats.num_spsolves = 0
    stats.num_gradient_evaluations=0
    stats.num_backtracks=0
    
    # Evaluate action
    S = action(phi,args)
    stats.num_action_evaluations+=1
    
    while not convergence:
        stats.num_corrector_steps += 1
        
        # Compute the gradient
        v = grad(phi,args)
        stats.num_gradient_evaluations+=1
        
        # Solve linear system
        Lambda = hessian(phi, args)
        dphi = -sp.real(spsolve(Lambda,v))
        stats.num_spsolves+=1
        
        dS = sp.sum(dphi*v)
        beta = 1.0
        S_new = action(phi + beta*dphi,args)
        stats.num_action_evaluations+=1
        
        # Reduce step size until in linear regime
        while S_new > S + 0.5*beta*dS:
            beta *= 0.5
            S_new = action(phi + beta*dphi,args)
            stats.num_action_evaluations+=1
            stats.num_backtracks+=1
            
        # Set new phi, S, etc
        S_change = S_new - S
        
        # Convergence reached if S_change is above threshold
        convergence = -S_change < 1E-3
        
        # If S_change is positive, don't change phi. Just get outta there
        if S_change > 0:
            beta = 0
            print 'Warning: S_change > 0'
        
        # Set new phi and correpsonding action
        phi = phi + beta*dphi
        S = S_new
    
    return [phi, stats]
#
# Performs predictor step
#
def predictor_step(phi,args,direction):
    assert abs(direction)==1
    
    Q = field_to_quasidensity(phi,args.L)
    Lambda = sp.exp(-args.t)*args.Delta + diags(Q,0)
    rho = sp.real(spsolve(Lambda, args.R-Q))
    
    delta_t = direction*args.epsilon/sp.real(sp.sqrt(sp.sum(rho*Q*rho)))
    delta_phi = phi + delta_t*rho
    
    return [delta_phi, delta_t]

#
# Computes MAP curve
#
def map_curve(xis_raw, G, bbox, alpha, epsilon=1E-2):

    # Check vailidity of arguments    
    assert len(xis_raw > 1)
    assert len(bbox)==2 and bbox[0] < bbox[1]
    assert G==int(G) and G > alpha
    assert alpha==int(alpha) and alpha >= 1
     	 	
    # Make sure xis is a numpy array
    xis_raw = sp.array(xis_raw)
            
    # Get upper and lower bounds on x and length of interval
    xlb = bbox[0]
    xub = bbox[1]
    L = xub-xlb
    
    # Only keep data points that are within the bounding box
    ok_data_indices = (xis_raw >= xlb) & (xis_raw <= xub)
    xis = xis_raw[ok_data_indices]
    
    # Converte to x -> z. z has grid spacing 1
    zis = (xis-xlb)*G/L
    zedges = sp.linspace(0, G, G+1)     # Edges of histogram grid. G+1 entries
                        
    # Determine number of valid data points
    N = len(zis)
    assert(N > 0) # Make sure there actually is data to work with
    
    # Create bilateral laplacian, the Delta matrix
    Delta = csr_matrix(bilateral_laplacian(G, alpha, grid_spacing=1.0))     
    
    #                            
    # Compute histogram
    #
    [R, xxx] = sp.histogram(zis, zedges, normed=1)                                                        
                                                                                                                    
    #
    # Compute maxent density               
    #
    [xxx, result] = maxent_1d(R, G, alpha)
    phi0 = sp.array(result.phi)
    M = field_to_quasidensity(phi0,G)
    
    #
    # Compute starting phi at ell0
    #
    args = Args()
    args.R = R
    args.G = G
    args.L = G
    args.N = N
    args.epsilon = epsilon
    args.Delta = Delta   
    
    #print 'Compting phi0...'
    #ell0 = sp.sqrt(G)
    #t0 = sp.log(N/ell0**(2*alpha-1.))
    t0 = 0
    args.t = t0
    [phi0,stats0] = corrector_step(M,args)
    Q0 = field_to_quasidensity(phi0,G)
    
    #
    # Algorithm along decreasing length scales
    #
    
    Qs_dir1 = []
    ts_dir1 = []   
    phis_dir1 = []
    stats_dir1 = []
    
    Q = Q0
    t = t0
    phi = phi0
    direction = +1.0
    step_num = 0
    args.t = t0
    while geo_dist(Q,R) > epsilon:    
        
        # Predictor step
        [dphi, dt] = predictor_step(phi,args,direction)
        
        beta = 1.0/0.8
        step_ok = False
        while not step_ok:
            # Corrector step
            beta = 0.8*beta
            t_new = t + beta*dt
            args.t = t_new
            [phi_new,stats] = corrector_step(phi + beta*dphi,args)
            Q_new = field_to_quasidensity(phi_new,G)
            step_ok = True if geo_dist(Q,Q_new) < 2.0*epsilon else False
        
        # Record step
        t = t_new
        phi = phi_new
        Q = Q_new
        Qs_dir1.append(Q)
        ts_dir1.append(t)
        phis_dir1.append(phi)
        stats_dir1.append(stats)
        Q = Q_new
        
        step_num += 1
        #if step_num%20 == 0:
        #    print 'direction = %d, step_num = %d, t = %f, geo_dist = %f'%(direction,step_num,t, geo_dist(Q,R))
    
    #
    # Algorithm along increasing length scales
    #
    
    Qs_dir2 = []
    ts_dir2 = []   
    phis_dir2 = []
    stats_dir2 = []
    
    
    Q = Q0
    t = t0
    phi = phi0
    direction = -1.0
    step_num = 0
    args.t = t0
    while geo_dist(Q,M) > epsilon:    
        
        # Predictor step
        [dphi, dt] = predictor_step(phi,args,direction)
        
        beta = 1.0/0.8
        step_ok = False
        while not step_ok:
            # Corrector step
            beta = 0.8*beta
            t_new = t + beta*dt
            args.t = t_new
            [phi_new, stats] = corrector_step(phi + beta*dphi,args)
            Q_new = field_to_quasidensity(phi_new,G)
            step_ok = True if geo_dist(Q,Q_new) < 2.0*epsilon else False
        
        # Record step
        t = t_new
        phi = phi_new
        Q = Q_new
        Qs_dir2.append(Q)
        ts_dir2.append(t)
        phis_dir2.append(phi)
        stats_dir2.append(stats)
        Q = Q_new
        
        step_num += 1
        #if step_num%20 == 0:
        #    print 'direction = %d, step_num = %d, t = %f, geo_dist = %f'%(direction,step_num,t, geo_dist(Q,M))
            
    # Package and return results
    results = Results()
    results.Qs = Qs_dir2[::-1] + [Q0] + Qs_dir1
    results.phis = phis_dir2[::-1] + [phi0] + phis_dir1
    results.ts = ts_dir2[::-1] + [t0] + ts_dir1
    results.stats = stats_dir2[::-1] + [stats0] + stats_dir1
    results.index0 = len(ts_dir2) 
    results.R = R
    results.M = M
    results.t0 = t0
    results.Q0 = Q0
    results.N = N
    results.G = G
    results.Delta = Delta
    
    return results

    
#
# Returns density estimate in function form
#    
def interpolated_density_estimate(xgrid, Q):
    xgrid = sp.array(xgrid)
    Q = sp.array(Q)
    
    G = len(xgrid)
    assert len(Q) == xgrid
    
    
    diffs = sp.diff(xgrid)
    h = diffs[0]
    assert all(diffs == h)
    
    xlb = xgrid[0]-h/2.
    xub = xgrid[-1]+h/2.
    L = xub - xlb
    
    assert all(Q > 0)
    phi = -sp.log(Q*L)

    extended_xgrid = sp.zeros(G+2)
    extended_xgrid[1:-1] = xgrid
    extended_xgrid[0] = xlb
    extended_xgrid[-1] = xub
    
    extended_phi = sp.zeros(G+2)
    extended_phi[1:-1] = phi
    extended_phi[0] = phi[0]-0.5*(phi[1]-phi[0])
    extended_phi[-1] = phi[-1]+0.5*(phi[-1]-phi[-2])
    phi_func = interp1d(extended_xgrid, extended_phi, kind='cubic')
    Z = sp.sum(h*sp.exp(-phi))
    
    def Q_func(x):
        x_array = sp.array(x)
        if len(x_array.shape) == 0:
            x_array = sp.array([x_array])
        
        in_indices = (x_array >= xlb) & (x_array <= xub)
        out_indices = True - in_indices
        
        values = sp.array(x_array.shape())
        values[in_indices] = sp.exp(-phi_func(x))/Z
        values[out_indices] = 0.0
        
        return values
    
    return Q_func
    
#
# Performs density estimation
#
def deft_nobc_1d(xis_raw, G, bbox, alpha, epsilon=3.14159E-2, details=False):
    
    # Comput MAP curve
    results = map_curve(xis_raw, G, bbox, alpha, epsilon=epsilon)
    
    xub = bbox[1]
    xlb = bbox[0]
    L = xub - xlb
    h = L/G
    xgrid = sp.linspace(xlb,xub,G+1)[:-1]+.5*h    
    
    num_ts = len(results.ts)
    ts = sp.array(results.ts)
    phis = results.phis
    args = Args()
    args.N = results.N
    args.G = results.G
    args.L = results.G
    args.R = results.R
    args.Delta = results.Delta
    
    
    # Get spectrum
    Delta = results.Delta.todense()
    lambdas, psis = eigh(Delta)
    indices = lambdas.argsort()
    lambdas = lambdas[indices]
    psis = psis[:,indices]

    #
    # Compute Occam factor at ell = infty
    #
    M = results.M
    N = results.N
    R = results.R
    phi_M = -sp.log(G*M)
    
    M_on_kernel = sp.zeros([alpha, alpha])
    for a in range(alpha):
	for b in range(alpha):
            psi_a = sp.ravel(psis[:,a])
            psi_b = sp.ravel(psis[:,b])
            M_on_kernel[a,b] = sp.sum(psi_a*psi_b*M)

    # Compute log occam factor
    log_Occam_at_infty = -0.5*sp.log(det(M_on_kernel)) - 0.5*sp.sum(sp.log(lambdas[alpha:]))
    log_likelihood_at_infty = -N*sp.sum(phi_M*R) - N
    log_ptgd_at_infty = log_likelihood_at_infty + log_Occam_at_infty
    
    #
    # Compute p(t|data)
    #
    
    log_ptgd = sp.zeros(num_ts)
    log_likelihood = sp.zeros(num_ts)
    log_Occam = sp.zeros(num_ts)
    for k in range(num_ts):
        phi = phis[k]
        t = ts[k]
        args.t = t
        
        S = action(phi,args)
        
        ell_factor = args.N*sp.exp(-args.t)  
        Lambda = hessian(phi,args).todense()/ell_factor
        
        log_likelihood[k] = -S
        log_Occam[k] = 0.5*alpha*t - 0.5*sp.log(det(Lambda)) 
        log_ptgd[k] = log_likelihood[k] + log_Occam[k]

    Qs = sp.array([results.M] + results.Qs + [results.R])
    num_Qs = Qs.shape[0]
    nn_dists = [geo_dist(Qs[k,:], Qs[k+1,:]) for k in range(num_Qs-1)]
    R = results.R*G/L
    M = results.M*G/L
    istar = sp.argmax(log_ptgd)
    Q_star = results.Qs[istar]*G/L
    phi_star = results.phis[istar]
    
    results.istar = istar
    results.Q_star = Q_star
    results.Qs = Qs*G/L
    results.phi_star = phi_star
    results.nn_dists = nn_dists
    results.L = L
    results.h = h
    results.R = R
    results.M = M
    results.Q_infty = M
    results.xgrid = xgrid
    results.log_ptgd = log_ptgd
    results.log_Occam = log_Occam
    results.log_likelihood = log_likelihood
    results.log_Occam_at_infty = log_Occam_at_infty
    results.log_likelihood_at_infty = log_likelihood_at_infty
    results.log_ptgd_at_infty = log_ptgd_at_infty
    results.ells = (args.N*sp.exp(-ts))**(1.0/(2.*alpha-1))
    
    if not details:
        return (Q_star, xgrid)
    else: 
        return (Q_star, xgrid, results)

#
# Comptues the K coefficient (Eq. 12)
#
def compute_K_coeff(res):
    
    # Compute the spectrum of Delta
    Delta = res.Delta.todense()
    alpha = int(-Delta[0,1])
    lambdas, psis = eigh(Delta) # Columns of psi are eigenvectors
    original_psis = sp.array(psis)
                
    R = res.R
    M = res.M
    N = res.N
    G = len(R)    

    # Get normalized M and R, with unit grid spacing
    M = sp.array(M/sp.sum(M)).T
    R = sp.array(R/sp.sum(R)).T

    # Diagonalize first alpha psis with respect to diag_M
    # This does the trick
    diag_M_mat = sp.mat(sp.diag(M))
    psis_ker_mat = sp.mat(original_psis[:,:alpha])
    diag_M_ker = psis_ker_mat.T*diag_M_mat*psis_ker_mat
    omegas, psis_ker_coeffs = eigh(diag_M_ker)
    
    psis = original_psis.copy()
    psis[:,:alpha] = psis_ker_mat*psis_ker_coeffs

    # Now compute relevant coefficients
    # i: range(G)
    # j,k: range(alpha)
    v_is = sp.array([sp.sum((M - R)*psis[:,i]) for i in range(G)])
    z_iis = sp.array([sp.sum(M*psis[:,i]*psis[:,i]) for i in range(G)])
    z_ijs = sp.array([[sp.sum(M*psis[:,i]*psis[:,j]) for j in range(alpha)] for i in range(G)] )
    z_ijks = sp.array([[[sp.sum(M*psis[:,i]*psis[:,j]*psis[:,k]) for j in range(alpha)] for k in range(alpha)] for i in range(G)] )

    K_pos_terms = sp.array([(N*v_is[i]**2)/(2*lambdas[i]) for i in range(alpha,G)])
    K_neg_terms = sp.array([(-z_iis[i])/(2*lambdas[i]) for i in range(alpha,G)])
    K_ker1_terms = sp.array([sum([z_ijs[i,j]**2 / (2*lambdas[i]*omegas[j])for j in range(alpha)]) for i in range(alpha,G)] )
    K_ker2_terms = sp.array([sum([v_is[i]*z_ijks[i,j,j] / (2*lambdas[i]*omegas[j]) for j in range(alpha)]) for i in range(alpha,G)] )
    K_ker3_terms = sp.array([sum([sum([-v_is[i]*z_ijs[i,j]*z_ijks[j,k,k] / (2*lambdas[i]*omegas[k]*omegas[j])for j in range(alpha)]) for k in range(alpha)])for i in range(alpha,G)] )
    
    # I THINK THIS IS RIGHT!!!
    K_coeff = K_pos_terms.sum() + K_neg_terms.sum() + K_ker1_terms.sum() + K_ker2_terms.sum() + K_ker3_terms.sum()

    # Return the K coefficient
    return K_coeff