import scipy as sp
import matplotlib.pyplot as plt
import time
from deft import deft
from deft_utils import get_cumulants, get_colormap
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib.patches import Rectangle
from numpy.random import choice
from numpy.linalg import eigh, det
import pickle

class Results: pass;

plt.close('all')

blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]
min_log_evidence = -30
subplot_labels = 'abcdefghi'

fontsize = 8
subplot_fontsize = 10
figsize = [7.0,2.25]
label_xpad = 0.26
label_ypad = -0.05

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=8)
cmap = cm.afmhot #get_colormap()


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def label_subplot(ax,label,label_xpad=label_xpad,label_ypad=label_ypad):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    Lx = xlim[1]-xlim[0]
    Ly = ylim[1]-ylim[0]
    x0 = xlim[0]
    y0 = ylim[1]
    
    x = x0 - label_xpad*Lx
    y = y0 + label_ypad*Ly
    ax.text(x,y,label,fontsize=subplot_fontsize)

def label_semilogx_subplot(ax,label,label_xpad=label_xpad,label_ypad=label_ypad):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    Lx = sp.log(xlim[1]) - sp.log(xlim[0])
    Ly = ylim[1]-ylim[0]
    x0 = sp.log(xlim[0])
    y0 = ylim[1]
    
    x = x0 - label_xpad*Lx
    y = y0 + label_ypad*Ly
    ax.text(sp.exp(x),y,label,fontsize=subplot_fontsize)

# Load info
all_examples = pickle.load(open('fig_2.pickle','rb'))

# Plot results
plt.figure(figsize=figsize)

num_examples = len(all_examples)

for n, example in enumerate(all_examples):

    Q_true = example.Q_true
    d = example.d
    num_trials = len(example.results)

    res = example.results[0]
    
    # Compute the spectrum of Delta
    Delta = res.Delta.todense()
    alpha = int(-Delta[0,1])
    lambdas, psis = eigh(Delta) # Columns of psi are eigenvectors
    original_psis = sp.array(psis)
                
    R = res.R
    M = res.M
    G = len(R)
    Qs = res.Qs[1:-1,:]
    ells = res.ells
    ell_star = res.ells[res.istar]
    xgrid = res.xgrid
    h = example.h
    bbox = example.bbox
    
    max_ell = min([max(res.ells) for res in example.results])
    min_ell = max([min(res.ells) for res in example.results])

    # Plot Q_true, R, and Q_Star for last trial
    subplot_num = n
    ax = plt.subplot(2,3,subplot_num+1)
    plt.bar(xgrid-h/2.0, res.R, width=h, edgecolor='none', color=gray)
    plt.plot(xgrid, Q_true, 'k', linewidth=2)
    #plt.plot(xgrid, Q_star,color=lightblue, linewidth=2)
    plt.xlim(bbox)
    plt.ylim([0, 1.5*max(Q_true)])
    plt.yticks([])
    plt.xlabel('$x$',labelpad=0,fontsize=fontsize)
    label_subplot(ax,'$(%s)$'%subplot_labels[subplot_num])

    # Plot evidence vs. ell
    subplot_num = n+3
    ax = plt.subplot(2,3,subplot_num+1)
    yl = [-40,40]
    xl = [0.3, max_ell]
    plt.semilogx(xl, [0, 0], '--', color='k')
    yrange = yl[1]-yl[0]
    num_max_ent = 0
    num_called_max_ent = 0
    num_called_max_ent_and_right = 0
    
    #
    # Compute B statistics
    #
    B_stats = sp.zeros(len(example.results))
    flags = sp.zeros(len(example.results))
    sign_matches = sp.zeros(len(example.results))
    max_ent = sp.array([False]*len(example.results))
    for res_num, res in enumerate(example.results):
        R = res.R
        M = res.M
        N = res.N
        shifted_log_ptgd = res.log_ptgd - res.log_ptgd_at_infty
    
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
        V_is = sp.array([sp.sum((M - R)*psis[:,i]) for i in range(G)])
        W_iis = sp.array([sp.sum(M*psis[:,i]*psis[:,i]) for i in range(G)])
        W_ijs = sp.array([[sp.sum(M*psis[:,i]*psis[:,j]) for j in range(alpha)] for i in range(G)] )
        W_ijks = sp.array([[[sp.sum(M*psis[:,i]*psis[:,j]*psis[:,k]) for j in range(alpha)] for k in range(alpha)] for i in range(G)] )
    
        B_pos_terms = sp.array([(N*V_is[i]**2)/(2*lambdas[i]) for i in range(alpha,G)])
        B_neg_terms = sp.array([(-W_iis[i])/(2*lambdas[i]) for i in range(alpha,G)])
        B_ker1_terms = sp.array([sum([W_ijs[i,k]**2 / (2*lambdas[i]*omegas[k])for k in range(alpha)]) for i in range(alpha,G)] )
        B_ker2_terms = sp.array([sum([V_is[i]*W_ijks[i,k,k]**2 / (2*lambdas[i]*omegas[k]) for k in range(alpha)]) for i in range(alpha,G)] )
        B_ker3_terms = sp.array([sum([sum([-V_is[i]*W_ijs[i,j]*W_ijks[j,k,k] / (2*lambdas[i]*omegas[k]*omegas[j])for j in range(alpha)]) for k in range(alpha)])for i in range(alpha,G)] )
        
        # I THINK THIS IS RIGHT!!!
        B_stat = B_pos_terms.sum() + B_neg_terms.sum() + B_ker1_terms.sum() + B_ker2_terms.sum() + B_ker3_terms.sum()

        # Compute B_empirical
        B_empirical = shifted_log_ptgd[0]

        if B_stat > 0:
            ell_star = res.ells[res.istar]
            this_yl = [yl[0]-sp.rand()*yrange, yl[1]+sp.rand()*yrange]
            #plt.semilogx([ell_star, ell_star], this_yl, ':k', linewidth=0.5, alpha=0.2)
            #color = orange
        else:
            color = lightblue
            num_max_ent += 1
                    
        indices = sp.isfinite(shifted_log_ptgd)
        max_ptgd =  max(shifted_log_ptgd[indices])
        
        max_ent[res_num] = (max_ptgd < 0.0)
        
        if (B_stat > 0.0) and (max_ptgd > 0.0):  # True positives
            flag = 1 
        elif (B_stat < 0.0) and (max_ptgd < 0.0):   # True negatives
            flag = -1
        else:
            flag = 0
        
        if (B_stat < 0.0) and (max_ptgd > 0.0):
            makes_sense = shifted_log_ptgd[0] < 0

        
        # Compute the quantity in the paper
        if (B_stat < 0.0):
            num_called_max_ent += 1.0
            if (max_ptgd < 0.0):
                num_called_max_ent_and_right += 1.0
                    
        if sp.sign(B_stat) == sp.sign(B_empirical):
            sign_match = True
        else:
            sign_match = False
            
        
        #plt.semilogx(res.ells, res.log_ptgd - res.log_ptgd_at_infty, color=color, linewidth=0.5, alpha=0.2)
        B_stats[res_num] = B_stat
        flags[res_num] = flag
        sign_matches[res_num] = sign_match
        
    #print flags
    num_trials = len(example.results)
    pct_max_ent = num_max_ent*100./num_trials
    pct_true_pos = sum(flags == 1)*100./num_trials + 1E-10
    pct_true_neg = sum(flags == -1)*100./num_trials + 1E-10
    
    pct_pos = 100.-pct_max_ent
    pct_neg = pct_max_ent        
                                              
    print 'Positive (not MaxEnt) rate: %d%%'%pct_pos
    if pct_pos > 0:
        print 'TP rate: %d%%'%(100.*pct_true_pos/pct_pos)
    print 'Negative (MaxEnt) rate: %d%%'%pct_neg
    if pct_neg > 0:
        print 'TN rate: %d%%'%int(100.*pct_true_neg/pct_neg)
    
    # This is the quantity that's in the paper: (# B < 0 & maxent) / # B < 0  
    #print 'Fidelity of B < 0 call:%d'%(100.*num_called_max_ent_and_right / num_called_max_ent)                         
                                                                                                                      
    plt.ylim(yl)
    plt.yticks([-40, -20, 0, 20, 40])
    plt.xlim(xl)
    #plt.xlim([1,10])
    #plt.xlim([min_ell, max_ell])
    plt.xlabel('$\ell$',labelpad=-1,fontsize=fontsize)
    plt.ylabel('$\ln~E$',labelpad=0,fontsize=fontsize)
    plt.title('MaxEnt: %d\%%'%pct_max_ent, fontsize=fontsize)
    #plt.title('MaxEnt: %d\%%,  TN: %d\%%,  TP: %d\%%'%(pct_max_ent, pct_true_neg, pct_true_pos), fontsize=fontsize)
    label_semilogx_subplot(ax,'$(%s)$'%subplot_labels[subplot_num], label_ypad=0.1)

    for res_num, res in enumerate(example.results):   
        ell_star = 2*xl[1] if max_ent[res_num] else res.ells[res.istar]
        this_yl = [yl[0]-sp.rand()*yrange, yl[1]+sp.rand()*yrange]  
        #plt.semilogx([ell_star, ell_star], this_yl, '-k', linewidth=0.5, alpha=0.2)
    
    for res_num, res in enumerate(example.results): 
        color = lightblue if max_ent[res_num] else orange
        plt.semilogx(res.ells, res.log_ptgd - res.log_ptgd_at_infty, color=color, linewidth=0.5, alpha=0.2)
        
    
plt.subplots_adjust(hspace=0.8, wspace=0.4, left=0.07, right=0.98, top=0.95, bottom=0.15)
plt.show()
plt.savefig('fig_2.pdf')