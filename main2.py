from GenerativeModel import getInitialState
from GenerativeModel import getGoalState

from solvers.UCT               import UCT_like
from solvers.UCT_FiniteHorizon import UCT_like_FH
from solvers.UCT_EBC           import UCT_adativeCoefficient
from solvers.UCT_EBC_FH        import UCT_adativeCoefficient_FH
from solvers.maxUCT_FH         import maxUCT_like
from solvers.maxUCT_EBC_FH     import maxUCT_adaptive

from simulation.sim_ToolBox     import runSim_FH
from simulation.CrossingTraffic import sim_CrossingTraffic
from simulation.Elevators       import sim_Elevators
from simulation.GameOfLife      import sim_GameOfLife
from simulation.Reconnaissance  import sim_Reconnaissance
from simulation.Maze            import sim_Maze
from simulation.SysAdmin        import sim_SysAdmin

#import tracemalloc
import matplotlib.pyplot as plt
import statistics
import pickle
import json

s0 = getInitialState()
s_goal = getGoalState()

#MaxTrials = 20000
#timeOut = 300.0
#exploration_c = 10
#horizon = 40

# CALL THE ALGORITHM TO SOLVE THE PROBLEM ------------------------------------
# For Goal-oriented problems only
#[G, Vs0] = UCT_like(s0, s_goal, MaxTrials, exploration_c)
#[G, Vs0] = UCT_adativeCoefficient(s0, s_goal, MaxTrials, 2)  

# For infinite horizon domains, metric max reward
#[G, Vs0] = UCT_like_FH(s0, horizon, MaxTrials, timeOut, exploration_c,False)                   
#[G, Vs0] = UCT_adativeCoefficient_FH(s0, horizon, MaxTrials, timeOut, 2, False)    
#[G, Vs0] = maxUCT_like(s0, horizon, MaxTrials, timeOut, exploration_c)    
#[G, Vs0] = maxUCT_adaptive(s0, horizon, MaxTrials, timeOut, 2)    


# PLOT THE EVOLUTION OF V(s0) ------------------------------------------------
#plt.plot(Vs0)

# RUN A SIMULATION -----------------------------------------------------------
#cost = runSim_FH(s0, G, horizon, True, 6)

# RUN ANIMATED SIMULATIONS ---------------------------------------------------
#sim_CrossingTraffic(s0, G, horizon)
#sim_Elevators(s0, G, horizon, 1)
#sim_GameOfLife(s0, G, horizon, 1)
#sim_Reconnaissance(s0, G, horizon, 1)
#sim_Maze(s0, G, horizon, 1)
#sim_SysAdmin(s0, G, horizon, 1)

#%%
# EXPERIMENTS ----------------------------------------------------------------

results = {}

results['maxUCT-EBC'] = {}
results['maxUCT-EBC']['max'] = []
results['maxUCT-EBC']['pair'] = []
results['maxUCT-EBC']['max_estim'] = []
results['maxUCT-EBC']['pair_estim'] = []

results['maxUCT-UCB'] = {}
results['maxUCT-UCB'][0.1] = []
results['maxUCT-UCB'][1.0] = []
results['maxUCT-UCB'][10.0]  = []
results['maxUCT-UCB'][100.0] = []

results['UCT-EBC'] = {}
results['UCT-EBC']['expanded'] = {}
results['UCT-EBC']['expanded']['max'] = []
results['UCT-EBC']['expanded']['pair'] = []
results['UCT-EBC']['simple'] = {}
results['UCT-EBC']['simple']['max'] = []
results['UCT-EBC']['simple']['pair'] = []

results['UCT-UCB'] = {}
results['UCT-UCB']['expanded'] = {}
results['UCT-UCB']['expanded'][0.1] = []
results['UCT-UCB']['expanded'][1.0] = []
results['UCT-UCB']['expanded'][10.0] = []
results['UCT-UCB']['expanded'][100.0] = []
results['UCT-UCB']['simple'] = {}
results['UCT-UCB']['simple'][0.1] = []
results['UCT-UCB']['simple'][1.0] = []
results['UCT-UCB']['simple'][10.0] = []
results['UCT-UCB']['simple'][100.0] = []


MaxTrials = 20000
timeOut = 300.0   
horizon = 40



# TEST maxUCT-EBC algorithm
opts_dict = {}
opts_dict['max'] = 0
opts_dict['pair'] = 2
opts_dict['max_estim'] = 3
opts_dict['pair_estim'] = 5


for opt,ref in opts_dict.items():   # Test all the action selection strategies

    optimizedVs0 = []               # list to store the final V(s0). 
    executedVs0  = []               # list to store the simulated costs

    for i in range(0,5):           # Run the algorithm 10 times
    
        [G, Vs0] = maxUCT_adaptive(s0, horizon, MaxTrials, timeOut, ref)
        
        # Save the optimized V(s0)
        optimizedVs0.append( Vs0[-1] )
        
        # Run 100 simulations and save the results
        for i in range(0,100):
            executedVs0.append(runSim_FH(s0, G, horizon, True, ref))
         

    
    meanOptimized = statistics.mean(optimizedVs0)
    stdOptimized = statistics.pstdev(optimizedVs0)
    
    meanExecuted = statistics.mean(executedVs0)
    stdExecuted = statistics.pstdev(executedVs0)
    
    results['maxUCT-EBC'][opt].append( (meanOptimized,stdOptimized) )
    results['maxUCT-EBC'][opt].append( (meanExecuted,stdExecuted) )    




# TEST maxUCT-UCB
c = [0.1, 1.0, 10.0, 100.0]

for exploration_c in c:
    
    optimizedVs0 = []               # list to store the final V(s0). 
    executedVs0  = []               # list to store the simulated costs

    for i in range(0,5):           # Run the algorithm 10 times
    
        [G, Vs0] = maxUCT_like(s0, horizon, MaxTrials, timeOut, exploration_c)    
        
        # Save the optimized V(s0)
        optimizedVs0.append( Vs0[-1] )
        
        # Run 100 simulations and save the results
        for i in range(0,100):
            executedVs0.append(runSim_FH(s0, G, horizon, True, 6))
         

    
    meanOptimized = statistics.mean(optimizedVs0)
    stdOptimized = statistics.pstdev(optimizedVs0)
    
    meanExecuted = statistics.mean(executedVs0)
    stdExecuted = statistics.pstdev(executedVs0)
    
    results['maxUCT-UCB'][exploration_c].append( (meanOptimized,stdOptimized) )
    results['maxUCT-UCB'][exploration_c].append( (meanExecuted,stdExecuted) )    


# TEST UCT-EBC
check_options = {}
check_options['expanded'] = True
check_options['simple'] = False

opts_dict = {}
opts_dict['max'] = 0
opts_dict['pair'] = 2


for checkOpt,flag in check_options.items():
    
    for actOpt,ref in opts_dict.items(): 
    
    
        optimizedVs0 = []               # list to store the final V(s0). 
        executedVs0  = []               # list to store the simulated costs
    
        for i in range(0,5):           # Run the algorithm 10 times
        
            [G, Vs0] = UCT_adativeCoefficient_FH(s0, horizon, MaxTrials, timeOut, ref, flag)
            
            # Save the optimized V(s0)
            optimizedVs0.append( Vs0[-1] )
        
            # Run 100 simulations and save the results
            for i in range(0,100):
                executedVs0.append(runSim_FH(s0, G, horizon, flag, ref))
         

    
        meanOptimized = statistics.mean(optimizedVs0)
        stdOptimized = statistics.pstdev(optimizedVs0)
        
        meanExecuted = statistics.mean(executedVs0)
        stdExecuted = statistics.pstdev(executedVs0)
        
        results['UCT-EBC'][checkOpt][actOpt].append( (meanOptimized,stdOptimized) )
        results['UCT-EBC'][checkOpt][actOpt].append( (meanExecuted,stdExecuted) )    




# TEST UCT-UCB
check_options = {}
check_options['expanded'] = True
check_options['simple'] = False

c = [0.1, 1.0, 10.0, 100.0]

for checkOpt,flag in check_options.items():
    
    for exploration_c in c:
        
        optimizedVs0 = []               # list to store the final V(s0). 
        executedVs0  = []               # list to store the simulated costs
    
        for i in range(0,5):           # Run the algorithm 10 times
        
            [G, Vs0] = UCT_like_FH(s0, horizon, MaxTrials, timeOut, exploration_c, flag)  
            
            # Save the optimized V(s0)
            optimizedVs0.append( Vs0[-1] )
        
            # Run 100 simulations and save the results
            for i in range(0,100):
                executedVs0.append(runSim_FH(s0, G, horizon, flag, 6))
         

    
        meanOptimized = statistics.mean(optimizedVs0)
        stdOptimized = statistics.pstdev(optimizedVs0)
        
        meanExecuted = statistics.mean(executedVs0)
        stdExecuted = statistics.pstdev(executedVs0)
        
        results['UCT-UCB'][checkOpt][exploration_c].append( (meanOptimized,stdOptimized) )
        results['UCT-UCB'][checkOpt][exploration_c].append( (meanExecuted,stdExecuted) ) 





# Save the results in a file
# OPTION 1: Pickle
filename1 = 'ReconP1.pickle'
with open(filename1, 'wb') as f:
    pickle.dump(results, f)

#with open('.\experiments\ElevatorsP1.pickle', 'rb') as f:
#    results_pickle = pickle.load(f)

# OPTION2: JSON
filename2 = 'ReconP1.json'
with open(filename2, 'w') as f2:
    json.dump(results, f2)

#f = open(filename2)
#results_json = json.load(f)
#f.close()




