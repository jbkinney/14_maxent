import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from deft_utils import bilateral_laplacian
from numpy.linalg import eigh
linewidth = 2
fontsize = 8
subplot_fontsize = 10
figsize = [3.4,3.4]

blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
black = [0,0,0]

plt.close('all')

G = 100
alpha = 3
L = 1.
h = L/G
xs = - L/2. + (0.5+sp.arange(G))*h

# Get bilateral laplacian matrix
Delta = bilateral_laplacian(G=G,alpha=alpha,grid_spacing=h)

# Compute first few terms in the spectrum
w, v = eigh(Delta)
indices = w.argsort()

# Sort spectrum
plt.figure(figsize=figsize)
psi_0 = lambda(x): sp.sqrt(1./L)*sp.ones(len(x))
psi_1 = lambda(x): sp.sqrt(3./L)*(2.*x/L)
psi_2 = lambda(x): sp.sqrt(5./L)*0.5*(3.*(2.*x/L)**2 - 1.)

ax = plt.subplot(221)
plt.plot(xs+.5,psi_0(xs), label='$\psi_0$', color=black, linewidth=1, alpha=1)
plt.plot(xs+.5,psi_1(xs), label='$\psi_1$', color=black, linewidth=1, alpha=0.5)
plt.plot(xs+.5,psi_2(xs), label='$\psi_2$', color=black, linewidth=1, alpha=0.25)
plt.xticks([0,1], fontsize=fontsize)
plt.yticks([-2,-1,0,1,2], fontsize=fontsize)
#plt.legend(loc=2)
plt.xlim([0,1])
plt.ylim([-2.5,2.5])
ax.set_aspect(1/5.)
plt.xlabel('$(x-x_\mathrm{min})/L$', fontsize=fontsize, labelpad=-2)
plt.ylabel('$\psi_k$', fontsize=fontsize, labelpad=0)
plt.title('$k =\ 1,\ 2,\ 3$', fontsize=fontsize)
plt.text(-0.3, 3.0,'$(a)$', fontsize=subplot_fontsize)

ax = plt.subplot(222)
for k in range(3,7):
    i = indices[k]
    eigval = w[k]
    eigvec = sp.array(v[:,k])
    psi = eigvec/sp.sqrt(h*sp.sum(eigvec**2))
    plt.plot(xs+.5, psi, label='$\psi_%d$'%k, linewidth=1, color=black, alpha=0.5**(k-3))
    
plt.xticks([0,1], fontsize=fontsize)
plt.yticks([-2,-1,0,1,2], fontsize=fontsize)
#plt.legend(loc=2)
plt.xlim([0,1])
plt.ylim([-2.5,2.5])
ax.set_aspect(1/5.)
plt.xlabel('$(x-x_\mathrm{min})/L$', fontsize=fontsize, labelpad=-2)
plt.ylabel('$\psi_k$', fontsize=fontsize, labelpad=0)
plt.title('$k =\ 4,\ 5,\ 6,\ 7$', fontsize=fontsize)
plt.text(-0.3, 3.0,'$(b)$', fontsize=subplot_fontsize)

# Plot eigenvalues
plt.subplot(212)
ns = sp.arange(8)
lambdas_bl = w[ns]
lambdas_bl[0:3] = 0
lambdas_per = ((sp.pi/L)*(2*sp.floor((ns+1)/2)))**6
lambdas_dir = ((sp.pi/L)*(ns+1))**6
lambdas_neu = ((sp.pi/L)*(ns))**6

y = lambda(x): (x**(1./6.))*(L/sp.pi)

plt.plot(ns+1, y(lambdas_neu), ':o', color = [.8,.8,.8], markeredgecolor='none')
plt.plot(ns+1, y(lambdas_dir), ':o', color = [.6,.6,.6], markeredgecolor='none')
plt.plot(ns+1, y(lambdas_per), ':o', color = [.4,.4,.4], markeredgecolor='none')
plt.plot(ns+1, y(lambdas_bl), ':sk', markeredgecolor='none')
plt.xticks(range(1,9), fontsize=fontsize)
plt.yticks(range(9), fontsize=fontsize)
plt.ylim([-0.5,6.5])
plt.xlim([0.5,7.5])
plt.xlabel(r'$k$', fontsize=fontsize)
#plt.ylabel(r'$ L\,\pi^{-1}\, \lambda_b^{1/6}$', fontsize=fontsize)
plt.ylabel(r'$\lambda_k^{1/6} \times\, L / \pi$', fontsize=fontsize)
plt.text(-0.5, 6.5,'$(c)$', fontsize=subplot_fontsize)

plt.tight_layout(pad=.5, w_pad=1, h_pad=1)
plt.show()
plt.savefig('fig_4.pdf')