import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import eigh
import pickle
from deft_nobc import compute_K_coeff
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
#figsize = [7.0,2.25]
figsize = [7.0,4.0]
label_xpad = 0.26
label_ypad = -0.05

#plt.rc('text', usetex=True)
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
    
def label_image_sublot(ax,label,label_xpad=label_xpad,label_ypad=label_ypad):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    Lx = xlim[1]-xlim[0]
    Ly = ylim[1]-ylim[0]
    x0 = xlim[0]
    y0 = ylim[1]
    
    x = x0 - label_xpad*Lx
    y = y0 + label_ypad*Ly
    ax.text(x,y,label,fontsize=subplot_fontsize)

# Load info
all_examples = pickle.load(open('fig_3_20.pickle','rb'))

# Plot results
plt.figure(figsize=figsize)

num_examples = len(all_examples)

for n, example in enumerate(all_examples):


    res = example.results[0]
    Q_true = example.Q_true
    d = example.d
    num_trials = len(example.results)
    
    xgrid = res.xgrid
    h = example.h
    bbox = example.bbox
    
    max_ell = min([max(res.ells) for res in example.results])
    min_ell = max([min(res.ells) for res in example.results])

    # Plot Q_true, R, and Q_Star for last trial
    subplot_num = n
    ax = plt.subplot(3,3,subplot_num+1)
    plt.bar(xgrid-h/2.0, res.R, width=h, edgecolor='none', color=gray)
    plt.plot(xgrid, Q_true, 'k', linewidth=2)
    plt.xlim(bbox)
    plt.ylim([0, 1.5*max(Q_true)])
    plt.yticks([])
    plt.xlabel('$x$',labelpad=0,fontsize=fontsize)
    label_subplot(ax,'$(%s)$'%subplot_labels[subplot_num])

    # Plot evidence vs. ell
    subplot_num = n+3
    ax = plt.subplot(3,3,subplot_num+1)
    yl = [-40,40]
    xl = [0.3, max_ell]
    plt.semilogx(xl, [0, 0], '--', color='k')
    yrange = yl[1]-yl[0]
    num_max_ent = 0
    num_called_max_ent = 0
    num_called_max_ent_and_right = 0
    
    #
    # Compute K coefficients
    #
    K_coeffs = sp.zeros(len(example.results))
    flags = sp.zeros(len(example.results))
    max_ent = sp.array([False]*len(example.results))
    result_matrix = sp.zeros([2,2])
    # row: 0 if K > 0, 1 if K < 0
    # col: 0 if l = finite, 1 if l = infinite
    num_tests = len(example.results)
    for res_num, res in enumerate(example.results):

        # Compute the maximum of log_ptgd
        shifted_log_ptgd = res.log_ptgd - res.log_ptgd_at_infty                    
        indices = sp.isfinite(shifted_log_ptgd)
        max_ptgd =  max(shifted_log_ptgd[indices])
        
        # Compute whether data was assigned the MaxEnt density estiamte
        max_ent[res_num] = (max_ptgd < 0.0)

        # Compute the K coefficient
        K_coeff = compute_K_coeff(res)         
        K_coeffs[res_num] = K_coeff
        
        # Flag how K_coeff performs relative to full computation
        if (K_coeff > 0.0) and (max_ptgd > 0.0):  # True positives
            flag = 1 
        elif (K_coeff < 0.0) and (max_ptgd < 0.0):   # True negatives
            flag = -1
        else:
            flag = 0
        flags[res_num] = flag
        
        row = (K_coeff < 0.0)  # 0 if K is positive, 1 if K is negative
        col = (max_ptgd < 0.0) # 0 if MaxEnt N, 1 if not MaxEnt Y
        result_matrix[row,col] += 1.0
        

    # Summarize results
    num_max_ent = sum(max_ent)
    num_trials = len(example.results)
    pct_max_ent = num_max_ent*100./num_trials
    pct_true_pos = sum(flags == 1)*100./num_trials + 1E-10
    pct_true_neg = sum(flags == -1)*100./num_trials + 1E-10
    pct_pos = 100.-pct_max_ent
    pct_neg = pct_max_ent        
             
    print '=== Results for d = %f ===='%d                                 
    print 'Positive (not MaxEnt) rate: %d%%'%pct_pos
    if pct_pos > 0:
        print 'TP rate: %d%%'%(100.*pct_true_pos/pct_pos)
    print 'Negative (MaxEnt) rate: %d%%'%pct_neg
    if pct_neg > 0:
        print 'TN rate: %d%%'%int(100.*pct_true_neg/pct_neg)                      
                                                                                                                      
    plt.ylim(yl)
    plt.yticks([-40, -20, 0, 20, 40])
    plt.xlim(xl)

    plt.xlabel('$\ell$',labelpad=-1,fontsize=fontsize)
    plt.ylabel('$\ln\ E$',labelpad=0,fontsize=fontsize)
    plt.title('MaxEnt: %d%%'%pct_max_ent, fontsize=fontsize)
    label_semilogx_subplot(ax,'$(%s)$'%subplot_labels[subplot_num], label_ypad=0.1)

    for res_num, res in enumerate(example.results):   
        ell_star = 2*xl[1] if max_ent[res_num] else res.ells[res.istar]
        this_yl = [yl[0]-sp.rand()*yrange, yl[1]+sp.rand()*yrange]  
    
    for res_num, res in enumerate(example.results): 
        color = lightblue if max_ent[res_num] else orange
        plt.semilogx(res.ells, res.log_ptgd - res.log_ptgd_at_infty, color=color, linewidth=0.5, alpha=0.2)

    # Plot K coefficient results    
    subplot_num = n+6
    ax = plt.subplot(3,3,subplot_num+1)  
    plt.imshow(result_matrix, interpolation='nearest', cmap=plt.cm.cool)  
    plt.xticks([0, 1], ['N', 'Y'])
    plt.xlabel('MaxEnt')
    plt.yticks([0, 1], ['+', '-'])
    plt.ylabel('sign of $K$')
    #plt.colorbar()
    plt.clim([0,result_matrix.max()])
    for row in [0,1]:
        for col in [0,1]:
            plt.text(col,row,'%d'%result_matrix[row,col],horizontalalignment='center',verticalalignment='center',fontsize=10)
    plt.clim([0, result_matrix.max()])
    plt.colorbar(ticks = [0, result_matrix.max()])
    label_subplot(ax,'$(%s)$'%subplot_labels[subplot_num], label_ypad=0.1, label_xpad=0.75)
    
    
plt.subplots_adjust(hspace=0.8, wspace=0.4, left=0.07, right=0.98, top=0.95, bottom=0.15)
plt.show()
plt.savefig('fig_3.pdf')