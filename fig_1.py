import scipy as sp
import matplotlib.pyplot as plt
import time
from deft import deft
from deft_utils import get_cumulants 
from scipy.interpolate import interp1d
from matplotlib import cm
from numpy.linalg import eigh, det


plt.close('all')

blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]
min_log_evidence = -30

fontsize = 8
subplot_fontsize = 10
figsize = [3.4,4.5]
label_xpad = 0.2
label_ypad = -0.05

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=8)
cmap = cm.afmhot #get_colormap()
class Args: pass;

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

#
# Set parameters
#

N = 100

alpha = 3
G = 100
#epsilon = 1E-1
epsilon = sp.pi*1E-2
#epsilon = 1E-2

#
# Generate data
#
x_min = -10.
x_max = 5.
L = x_max-x_min
h = L/G
xs = sp.linspace(x_min,x_max,G+1)[:-1]+h/2

mu1 = -3
mu2 = 2
sigma1 = 1
sigma2 = 2
f1 = 1./2.
f2 = 1./2.

data1 = mu1+sigma1*sp.randn(sp.floor(N*f1))
data2 = mu2+sigma2*sp.randn(sp.ceil(N*f2))
data = sp.array(list(data1) + list(data2))
[counts, xxx] = sp.histogram(data, G)
bbox = [x_min, x_max]

Q_true = (f1/sigma1)*sp.exp(-0.5*((xs-mu1)/sigma1)**2) + (f2/sigma2)*sp.exp(-0.5*((xs-mu2)/sigma2)**2)
Q_true = Q_true/sp.sum(h*Q_true)

#
# Do DEFT density estimation
#

start_time = time.clock()
Q_star, xgrid, res = deft(data, G, bbox, alpha=alpha, epsilon=epsilon, details=True)
stop_time = time.clock()
print 'Execution time: %f sec'%(stop_time-start_time)

R = res.R
M = res.M
Delta = res.Delta.todense()
Qs = res.Qs[1:-1,:]
ells = res.ells
ell_star = res.ells[res.istar]
num_Qs = Qs.shape[0]

# Plot results
plt.figure(figsize=figsize)

# Plot Q_true, R, M, Q_Star
ax = plt.subplot(411)
plt.bar(xgrid-h/2.0, res.R, width=h, edgecolor='none', color='gray')
plt.plot(xgrid, Q_true, 'k', linewidth=2)
plt.plot(xgrid, M, color=lightblue, linewidth=2)
plt.plot(xgrid, Q_star,color=orange, linewidth=2)
plt.xlim([x_min, x_max])
plt.ylim([0, 1.5*max(Q_true)])
plt.yticks([])
plt.xlabel('$x$',labelpad=0,fontsize=fontsize)
label_subplot(ax,'$(a)$')

# Compute interpolation of Qs so that length scale axis makes sense

K = 1000
min_log10_ell = sp.log10(min(res.ells))
max_log10_ell = sp.log10(max(res.ells))
edge= 0.05*(max_log10_ell - min_log10_ell)
shift = 0.075*(max_log10_ell - min_log10_ell)
log10_ells_grid = sp.linspace(min_log10_ell - edge, max_log10_ell + edge, K)
top_indices = log10_ells_grid < min_log10_ell
mid_indices = (log10_ells_grid >= min_log10_ell) & (log10_ells_grid <= max_log10_ell)
bot_indices = log10_ells_grid > max_log10_ell

phis = sp.array(res.phis)
log10_ells = sp.log10(res.ells)
phis_interp_func = interp1d(log10_ells, phis,axis=0)
phis_interp = phis_interp_func(log10_ells_grid[mid_indices])
Qs_interp = sp.exp(-phis_interp)/L

Qs_disp = sp.zeros([len(log10_ells_grid),G])
Qs_disp[mid_indices,:] = Qs_interp
Qs_disp[top_indices,:] = R
Qs_disp[bot_indices,:] = M

# Show heatmap of Qs
ax = plt.subplot(412)
plt.imshow(Qs_disp[::-1,:],interpolation='nearest', aspect='auto',extent=[x_min,x_max,min_log10_ell-edge,max_log10_ell+edge], cmap=cmap)
plt.clim([0, max(res.Q_star)])
plt.xlabel('$x$',labelpad=0,fontsize=fontsize)
plt.ylabel('$\ell$',labelpad=0,fontsize=fontsize)
plt.yticks([min_log10_ell-edge/2.0,-1.0,0,1.0,max_log10_ell+edge/2.0],['$0$','$10^{-1}$','$10^{0}$','$10^{1}$','$\infty$'])
xl = [x_min,x_max]
Lx = x_max-x_min

# Show where optimal ell is
log10_ell_star = sp.log10(res.ells[res.istar])
plt.plot(xl,[log10_ell_star,log10_ell_star], ':w', linewidth=1)
plt.plot(xl,[min_log10_ell,min_log10_ell], '-w', linewidth=0.5)
plt.plot(xl,[max_log10_ell,max_log10_ell], '-w', linewidth=0.5)
plt.xlim(xl)
label_subplot(ax,'$(b)$')

# Plot evidence vs. ell
ax = plt.subplot(413)
shifted_log_ptgd = res.log_ptgd - res.log_ptgd_at_infty
yl = [max(shifted_log_ptgd)-30, max(shifted_log_ptgd)+5]
xl = [min(res.ells), max(res.ells)]

plt.semilogx([ell_star, ell_star], yl, ':k')
plt.semilogx(xl, [0, 0], '--', color='k')
plt.semilogx(res.ells, shifted_log_ptgd, linewidth=2, color='k')

plt.ylim(yl)
plt.xlim([min(res.ells), max(res.ells)])
plt.xlabel('$\ell$',labelpad=-1,fontsize=fontsize)
plt.ylabel('$\ln~E$',labelpad=0,fontsize=fontsize)
label_semilogx_subplot(ax,'$(c)$',label_ypad=0.1)

# Compute and plot entropy
phis_list = res.phis
Qs_list = [Qs[n,:] for n in range(num_Qs)]
h = res.h
entropy_list = [-sp.sum(h*Qs_list[n]*sp.log2(Qs_list[n]+1E-20)) for n in range(num_Qs)]
max_entropy = -sp.sum(h*M*sp.log2(M+1E-20))
min_entropy = -sp.sum(h*R*sp.log2(R+1E-20))
true_entropy = -sp.sum(h*Q_true*sp.log2(Q_true+1E-20))

ax = plt.subplot(414)
yl = [min_entropy-.2,max_entropy+.2]
xl = [min(res.ells), max(res.ells)]
plt.semilogx([ell_star, ell_star], yl, ':k')
plt.semilogx(xl, [max_entropy, max_entropy], '--k')
plt.semilogx(res.ells, entropy_list, linewidth=2, color='k')
plt.ylim(yl)
plt.xlim(xl)
plt.xlabel('$\ell$',labelpad=-1,fontsize=fontsize)
plt.ylabel('$H$ (bits)',labelpad=0,fontsize=fontsize)
label_semilogx_subplot(ax,'$(d)$',label_ypad=0.1)



plt.subplots_adjust(hspace=0.5, wspace=0, left=0.17, right=0.99, top=0.98, bottom=0.07)
plt.show()
#plt.savefig('fig_1.pdf')