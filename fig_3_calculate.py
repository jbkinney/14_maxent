import scipy as sp
from deft_nobc import deft_nobc_1d
from numpy.random import choice
import pickle

class Results: pass;
#
# Set parameters
#

N = 100
alpha = 3
G = 100
epsilon = 1E-2
num_trials = 20 # 100 used in paper

ds = [0, 2.5, 5]
sigma = 1
x_min = -5.
x_max = 5.

bbox = [x_min, x_max]
L = x_max-x_min
h = L/G
xs = sp.linspace(x_min,x_max,G+1)[:-1]+h/2

all_examples = []
for d_num, d in enumerate(ds):

    # Define Q_true
    Q_true = 0.5*sp.exp(-0.5*((xs-d/2.)/sigma)**2) + 0.5*sp.exp(-0.5*((xs+d/2.)/sigma)**2)
    Q_true = Q_true/sp.sum(h*Q_true)

    # Do this 10x$
    trial_results = []
    is_max_ent = []
    print 'Running trials for d = %0.1f..'%d,
    for k in range(num_trials):
        
        # Generate data according to Q_true
        data_indices = choice(sp.arange(G),size=N,replace=True,p=Q_true/sp.sum(Q_true))
        data = xs[data_indices]
        
        # Estimate density
        Q_star, xgrid, res = deft_nobc_1d(data, G, bbox, alpha=alpha, epsilon=epsilon, details=True)
        trial_results.append(res) 
        print '.', 
    
    example = Results()
    example.results = trial_results
    example.d = d
    example.Q_true = Q_true
    example.xs = xs      
    example.N = N
    example.G = G
    example.alpha = alpha
    example.epsilon = epsilon
    example.bbox = bbox
    example.L = L
    example.h = h                               
        
    print '...Done.'
    
    all_examples.append(example)

pickle.dump(all_examples, open( 'fig_3_20.pickle', 'wb' ) )
