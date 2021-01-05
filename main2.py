from GenerativeModel import getInitialState
from GenerativeModel import getGoalState
from solvers.UCT import UCT_like


s0 = getInitialState()
s_goal = getGoalState()
s0bis = getInitialState()

MaxTrials = 500
exploration_c = 2

[G, Vs0] = UCT_like(s0, s_goal, MaxTrials, exploration_c)



                
                
            