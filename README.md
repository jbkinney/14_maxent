14_maxent
=========
Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 19 November 2014 

Reference: 
    Kinney, J.B. (2014) Unification of Field Theory and Maximum Entropy Methods for Learning Probability Densities
    
Code: https://github.com/jbkinney/14_maxent

=== Instructions ===

All computations were performed using the Canopy Python Environment by Enthought, available at https://www.enthought.com/products/canopy/ 

Dependencies: scipy

The file "deft_nobc.py" and "deft_utils.py" contain all the density estmation routines used in this paper. Just copy this file to any directory in which you want to peform density estimation, and do execute "from deft\_nobc.py import deft\_nobc\_1d" in your Python script

To see a simple demsontration of deft_nobc_1d in 1 dimension, run "demo.py"

To recreate Fig. 1, run "fig_1.py".

To recreate Fig. 2, run "fig_2_calculate.py", then "fig_2_draw.py"

=== FILES ===

*demo.py:*
  Contains a quick example of DEFT without boundary conditions in 1D. 

*deft_nobc.py:*
	Contains primary functions for field theory density estimation in 1D without boundary conditions. 
	
*deft_utils.py*
  Contains helper functions for field theory density estimation in 1D without boundary conditions.

*fig_1.py:*
	Performs computations for and plots Fig. 1. Saves as fig_1.pdf
	
*fig_2_calculate.py:*
	Performs computations for Fig. 2. Saves results in fig_2.pickle.
	
*fig_2_draw.py:*
	Draws Fig. 2, saves as fig_2.pdf
	
*fig_2.pickle:*
  Contains results of simulations performed for fig_2.pdf
