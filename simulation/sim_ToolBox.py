import math
import operator
from random import choice

# ----------------------------------------------------------------------------
#                               UCT LEGACY METHODS 

def StateEquality(s1,s2):
    '''
    Parameters
    ----------
    s1 : State Object from GenerativeModel.py
        A state to be compared with the other one.
    s2 : State Object from GenerativeModel.py
        Another state to be compared.

    Returns
    -------
    rv : Bool
        The return value is True if s1 and s2 refer to the same state, 
        otherwise this function returns False. Note that s1 and s2 may be two
        different instances of the same state. In this case the return value 
        will be True. In fact, this is the main functionality of the function.
        
    IMPORTANT REMARK: This function takes into account solely the predicates 
    of each state. The remaining decision epochs are not considered.
    '''
    
    rv= True                                         # Init return value
    if len(s1.predicates) == len(s2.predicates):     # Check if the number of predicates is the same
        for pred in s1.predicates:
            if pred not in s2.predicates:            # Check if every predicate of s1 is in s2.
                rv = False
                break                                # One mismatch is enough to return False
    else:
        rv = False
    
    return rv



def StateEquality_FH(s1,s2):
    '''
    Parameters
    ----------
    s1 : State Object from GenerativeModel.py
        A state to be compared with the other one.
    s2 : State Object from GenerativeModel.py
        Another state to be compared.

    Returns
    -------
    rv : Bool
        The return value is True if s1 and s2 refer to the same state, 
        otherwise this function returns False. Note that s1 and s2 may be two
        different instances of the same state. In this case the return value 
        will be True. In fact, this is the main functionality of the function.
        
    IMPORTANT REMARK: This function takes into account, not only the predicates 
    of each state but also the remaining decision epochs.

    '''
    
    rv= True                                             # Init return value
    
    if s1.remaining_steps == s2.remaining_steps:         # First, check the remaining steps      
        
        if len(s1.predicates) == len(s2.predicates):     # Check if the number of predicates is the same
        
            for pred in s1.predicates:
                if pred not in s2.predicates:            # Check if every predicate of s1 is in s2.
                    rv = False
                    break                                # One mismatch is enough to return False
        
        else:
            rv = False
            
    else: 
        rv = False
        
    return rv



def checkState(s,G):
    '''
    Parameters
    ----------
    s : State object from GenerativeModel.py
        This function checks if s is already a key of G. In other words, This
        function checks if s has been already visited.
    G : dict
        This dictionary depicts the graph that holds the solution of the 
        planning problem. 
        
    Returns
    -------
    s : State object from GenerativeModel.py
        If s is a new state, the output of the function will be exactly the 
        input. Hereafter we will use this reference to talk about the state
        encoded by the predicates of s.
        By contrast, if s is already in the graph, the function will return 
        the existing reference (intance) of the state encoded by s.
    
    IMPORTANT REMARK: This function relies on StateEquality(s1,s2) to make 
    the comparisons. Hence, the remaining decision epochs are NOT considered.

    '''
    
    # state by state check if the predicates of the analysed state matches with 
    # the predicates of already visited states.
    for state in G.keys():
        
        if StateEquality(s,state) :
            # Overwrite s because it is a new instance 
            # of an already visited state
            s = state
            break
        
    return s



def checkState_FH(s,G):
    '''
    Parameters
    ----------
    s : State object from GenerativeModel.py
        This function checks if s is already a key of G. In other words, This
        function checks if s has been already visited.
    G : dict
        This dictionary depicts the graph that holds the solution of the 
        planning problem. 
        
    Returns
    -------
    s : State object from GenerativeModel.py
        If s is a new state, the output of the function will be exactly the 
        input. Hereafter we will use this reference to talk about the state
        encoded by the predicates of s.
        By contrast, if s is already in the graph, the function will return 
        the existing reference (intance) of the state encoded by s.
    
    IMPORTANT REMARK: This function relies on StateEquality_FH(s1,s2) to make 
    the comparisons. Hence, the remaining decision epochs are considered.

    '''
    
    # state by state check if the predicates of the analysed state matches with 
    # the predicates of already visited states.
    for state in G.keys():
        
        if StateEquality_FH(s,state) :
            # Overwrite s because it is a new instance 
            # of an already visited state
            s = state
            break
        
    return s


# ----------------------------------------------------------------------------
"""
                     DEFINE THE ACTION SELECTION STRATEGIES
"""
from solvers.maxUCT_FH     import ActionSelection as UCB1
from solvers.maxUCT_EBC_FH import ActionSelection_Max as maxEntropy
from solvers.maxUCT_EBC_FH import ActionSelection_Mean as meanEntropy
from solvers.maxUCT_EBC_FH import ActionSelection_Pair as pairEntropy
from solvers.maxUCT_EBC_FH import ActionSelection_Max_Estimated as maxEntropy_estimated
from solvers.maxUCT_EBC_FH import ActionSelection_Mean_Estimated as meanEntropy_estimated
from solvers.maxUCT_EBC_FH import ActionSelection_Pair_Estimated as pairEntropy_estimated

def ActionSelection_maxQvalue(s,G):
    '''
    Parameters
    ----------
    s : State Object
        This is the state of the decision node where we have to choose the 
        best action to play
    G : dict
        Tree where we can find the information about the decision node

    Returns
    -------
    a : action
        The action that maximizes Q(s,a)

    '''
    
    for a in G[s].keys():
        if (a != 'N' and a != 'V'):        
            if (G[s][a]['Q-value'] == G[s]['V']):       # look for the action that satisfies V(s) = Q(s,a)
                return a
    
    
    
# ---------------------------------------------------------------------------
def runSim_FH(s0, G, horizon, FH_Flag, ActionSelection_op):
    '''
    Parameters
    ----------
    s0 : state object
        This is the initial state of the simulation
    G : dictionary
        This is the graph that contains the plan or policy solution.
    horizon : int
        This is the max. number of decission epochs. In other words, the 
        finite horizon of the MDP
    FH_Flag : Bool
        This flag tells the simulator how it should check the states in the 
        Graph. True means that this function will use CheckState_FH() and 
        therefore, the remaining decision epochs will be considered.
        By contrast, False means that the fucntion will use CheckState() so
        checking the predicates is more than enough to check the state     
    ActionSlection_op:  int
        There is a mapping between the option choosen and the action selection
        strategy:
            
            0-> ActionSelection_Max
            1-> ActionSelection_Mean
            2-> ActionSelection_Pair
            3-> ActionSelection_Max_Estimated
            4-> ActionSelection_Mean_Estimated
            5-> ActionSelection_Pair_Estimated
            6-> UCB1
            default -> ActionSelection_maxQvalue 
    Returns
    -------
    None. 
    
    The goal of this function is to run a simulation using the generative model
    of the problem. In each state, the function looks for the best action in G,
    then this action is triggered and the next state is generated according to 
    the probabilty dsitributuions of the generative model. The simulation runs
    until the horizon.

    '''
    s = s0                           # assign s0 to the current state
    count = 0                        # init a counter to count decission epochs
    accrualCost = 0                  # int a cost/reward adder
    while (count < horizon):         
        
        #print(s)                     # print current state
        
        if s in G:                   # look for the best action in the Tree
            
            if   ActionSelection_op == 0: action = maxEntropy(s,G)
            elif ActionSelection_op == 1: action = meanEntropy(s,G)
            elif ActionSelection_op == 2: action = pairEntropy(s,G)
            elif ActionSelection_op == 3: action = maxEntropy_estimated(s,G)
            elif ActionSelection_op == 4: action = meanEntropy_estimated(s,G)
            elif ActionSelection_op == 5: action = pairEntropy_estimated(s,G)
            elif ActionSelection_op == 6: action = UCB1(s,G,1.0)
            else: action = ActionSelection_maxQvalue(s,G)
                
            
        elif FH_Flag: # if s not in G but we are using expanded Space-State
                      # try to look for the same state with different remaining
                      # decision epochs

            s_twin = checkState(s,G)
            if s_twin in G:                   # look for the best action in the Tree
                
                
                if   ActionSelection_op == 0: action = maxEntropy(s_twin,G)
                elif ActionSelection_op == 1: action = meanEntropy(s_twin,G)
                elif ActionSelection_op == 2: action = pairEntropy(s_twin,G)
                elif ActionSelection_op == 3: action = maxEntropy_estimated(s_twin,G)
                elif ActionSelection_op == 4: action = meanEntropy_estimated(s_twin,G)
                elif ActionSelection_op == 5: action = pairEntropy_estimated(s_twin,G)
                elif ActionSelection_op == 6: action = UCB1(s_twin,G,1.0)
                else: action = ActionSelection_maxQvalue(s_twin,G)
                
            elif s.actions:  # The plan is incomplete! Sample a random action and pray  
            
                action = s.SampleAction()
            else :
                accrualCost += -5.0
                return
                
            
        elif s.actions :             # The plan is incomplete! Sample a random action and pray  
            action = s.SampleAction()
        
        else:                        # This is a Dead-End state. Imposible to continue
            accrualCost += -5.0
            print('Dead-End reached, simulation finished')
            return
            
          
        #print(action.name)                           # print selected action
        
        [successor,cost] = s.SampleChild(action)     # Generate new state
        accrualCost += cost                          # update accrual cost
        
        if FH_Flag:                                  # check if the successor is already in the graph
            successor = checkState_FH(successor,G)   
        else : 
            successor = checkState(successor,G)
            
        s = successor                                # assing successor to the current state
        count +=1                                    # update counter

    #print("Terminal State - Finite Horizon reached")       
    #print(accrualCost) 
    return  accrualCost
# ----------------------------------------------------------------------------