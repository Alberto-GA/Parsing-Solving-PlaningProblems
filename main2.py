from GenerativeModel import getInitialState
from GenerativeModel import getGoalState
from solvers.UCT import UCT_like
from solvers.UCT_FiniteHorizon import UCT_like_FH
from simulation import runSim_FH
from simulation import sim_CrossingTraffic

s0 = getInitialState()
s_goal = getGoalState()

MaxTrials = 10000
exploration_c = 2
horizon = 40


#[G, Vs0] = UCT_like(s0, s_goal, MaxTrials, exploration_c)
[G, Vs0] = UCT_like_FH(s0, horizon, MaxTrials, exploration_c)


# RUN A SIMULATION:
runSim_FH(s0, G, horizon)
sim_CrossingTraffic(s0, G, horizon)