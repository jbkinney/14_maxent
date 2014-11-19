import scipy as sp
import matplotlib.pyplot as plt
import time
from deft_nobc import deft_nobc_1d    # Import the density estimation function

################################################################################
# Set simulation parameters 
################################################################################

# Set the number of data points
N = 100

# Set the number of grid points      
G = 100       

# Set the order of the constrained derivative
alpha = 3     

################################################################################
# Generate data
################################################################################
x_min = -10.
x_max = 5.
bbox = [x_min, x_max]
L = x_max-x_min
h = L/G
xgrid = sp.linspace(x_min,x_max,G+1)[:-1]+(h/2)

# Define Q_true as a mixture of two Gaussians
mu1 = -3
mu2 = 2
sigma1 = 1
sigma2 = 2
f1 = 1./2.
f2 = 1-f1
Q_true = (f1/sigma1)*sp.exp(-0.5*((xgrid-mu1)/sigma1)**2) + \
         (f2/sigma2)*sp.exp(-0.5*((xgrid-mu2)/sigma2)**2)
Q_true = Q_true/sp.sum(h*Q_true)

# Draw data from Q_true
Q_true_binned = Q_true/sp.sum(Q_true)
data = sp.random.choice(xgrid,N,replace=True,p=Q_true_binned)

################################################################################
# Do density estimation 
################################################################################
start_time = time.clock()
Q_star, xgrid, res = deft_nobc_1d(data, G, bbox, alpha=alpha, details=True)
stop_time = time.clock()
print 'Execution time: %f sec'%(stop_time-start_time)

################################################################################
# Plot results
################################################################################
plt.close('all')
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.figure()

# Plot the data histogram
gray = [0.6, 0.6, 0.6]
plt.bar(xgrid-h/2.0, res.R, width=h, edgecolor='none', color='gray')

# Plot the Q_true
plt.plot(xgrid, Q_true, linewidth=3, color='k')

# Plot the DEFT-estimated density
orange = [1.,0.5,0.]
plt.plot(xgrid, Q_star, linewidth=3, color='b')

# Show plot
plt.xlim([x_min, x_max])
plt.ylim([0, 1.5*max(Q_true)])
plt.yticks([])
plt.xlabel('$x$')
plt.show()