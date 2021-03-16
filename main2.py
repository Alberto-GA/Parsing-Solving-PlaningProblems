from GenerativeModel import getInitialState
from GenerativeModel import getGoalState
from solvers.UCT import UCT_like
from solvers.UCT_FiniteHorizon import UCT_like_FH
from simulation import runSim_FH
from simulation import sim_CrossingTraffic
from simulation import sim_Elevators
from simulation import sim_GameOfLife
from simulation import sim_Reconnaissance
from simulation import sim_Maze
from simulation import sim_SysAdmin

import matplotlib.pyplot as plt


s0 = getInitialState()
s_goal = getGoalState()

MaxTrials = 5000
exploration_c = 10
horizon = 40


# CALL THE ALGORITHM TO SOLVE THE PROBLEM
#[G, Vs0] = UCT_like(s0, s_goal, MaxTrials, exploration_c)                     # For infinite horizon domains
[G, Vs0] = UCT_like_FH(s0, horizon, MaxTrials, exploration_c)                  # For finite horizon domains


# PLOT THE EVOLUTION OF V(s0)
plt.plot(Vs0)

# RUN A SIMULATION:
runSim_FH(s0, G, horizon)

# RUN ANIMATED SIMULATIONS
#sim_CrossingTraffic(s0, G, horizon)
#sim_Elevators(s0, G, horizon, 1)
#sim_GameOfLife(s0, G, horizon, 1)
#sim_Reconnaissance(s0, G, horizon, 3)
#sim_Maze(s0, G, horizon, 1)
#sim_SysAdmin(s0, G, horizon, 3)



"""
noop = s0.actions[0]
[s1, cost] = s0.SampleChild(noop)
goUp = s1.actions[1]
[s2, cost2] = s1.SampleChild(goUp)
openDown = s2.actions[3]
[s3,cost3] = s2.SampleChild(openDown)
closeDoor = s3.actions[0]
[s4,cost4] = s3.SampleChild(closeDoor)
goDown = s4.actions[1]
[s5,cost5] = s4.SampleChild(goDown)
openUp = s5.actions[4]
[s6,cost6] = s5.SampleChild(openUp)
[s7,cost7] = s5.SampleChild(s5.actions[2])
[s8,cost8] = s7.SampleChild(s7.actions[2])
[s9,cost9] = s8.SampleChild(s8.actions[2])
[s10,cost10] = s9.SampleChild(s9.actions[2])
[s11,cost11] = s10.SampleChild(s10.actions[2])
[s12,cost12] = s11.SampleChild(s11.actions[2])
[s13,cost13] = s12.SampleChild(s12.actions[2])
[s14,cost14] = s13.SampleChild(s13.actions[1])
[s15,cost15] = s14.SampleChild(s14.actions[4])
[s16,cost16] = s15.SampleChild(s15.actions[1])
[s17,cost17] = s16.SampleChild(s16.actions[0])
[s18,cost18] = s17.SampleChild(s17.actions[1])
[s19,cost19] = s18.SampleChild(s18.actions[4])
[s20,cost20] = s19.SampleChild(s19.actions[0])
[s21,cost21] = s20.SampleChild(s20.actions[1])
[s22,cost22] = s21.SampleChild(s21.actions[2])
print(s22)

"""