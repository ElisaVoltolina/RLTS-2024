import os
in_file = os.path.join(os.path.dirname(__file__), '..', 'files', 'myciel6.col')
out_file = os.path.join(os.path.dirname(__file__), '..', 'files', 'experiment_TL10_myciel6.xlsx')

# Constants
MAX_VAL = 999999
cutting_time = 10 # Termination condition (TIME LIMIT)

# Parameters
tt = 10  # Tabu length
pertubation_depth = 0.01  # Perturbation strength    
restart_threshold = 100  # Restart trigger threshold   
pertubation_threshold = 500  # Perturbation trigger threshold (T_p) 
no_improve = 0  # Consecutive non-improvement count
pertubation_time = 0  # Count of small perturbations
alpha = 0.1
beta = 0.2  # Values to experiment with
gamma1 = 0.3

