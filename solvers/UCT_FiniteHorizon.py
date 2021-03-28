"""
THIS ALGORITHM WITHIN THE TRIAL-BASED HEURISTIC TREE-SEARCH METHOD FRAMEWORK
 
HEURISTIC: Rollout legacy from plain UCT
ACTION SELECTION STRATEGY: Upper Confidence Bounds applied to tres (UCB)
                           minimization of the regret of choosing the wrong 
                           action
BACK-UP : classical Monte Carlo planning
OUTCOME SELECTION : succesors are sampled according to P(s'|s,a)
TRIAL LENGTH: the trial continues until the Finite Horizon of the MDP

NOTE:
    State-Space S = {predicates}  or  {predicates} x {0,...,H} depending
    on the checkstate function.

"""
#-------------------------------LIBRAIRES------------------------------------#
import math
import operator
from random import choice
import time
#-------------------------------FUNCTIONS------------------------------------#
"""
                        DEFINE THE HEURISTIC TO INIT V(s)
"""
def Rollout(s, horizon):
    '''
    Parameters
    ----------
    s : State object from GenerativeModel.py
        This is the sate where the rollout starts. This function will sample 
        a random applicable action 'a' for 's' and then a succesor s' will be 
        sampled according to P(s'|s,a). When this transition is triggered, it
        will return a cost that will be added up to the accrual cost. This 
        process continues until the maximum depth is reached, or until there 
        are no more decision steps left. Whichever comes first.
    horizon : int
        This is the remaining decision steps for state s. This argument is 
        a little bit redundant because in a general use case this 'horizon' 
        must be equal to s.remaining_steps. However, I leave this argument
        to allow the user to provide a different value.

    Returns
    -------
    payoff : float
        This is the accrual cost of the rollout that starts in s and continues
        until the horizon or until the maximum depth. In other words, this is
        a first estimate of V(s).
    
    Note also that the childs are never included in the graph. 
    state.SampleChild(a) instansciates locally a successor state but it is not
    stored in the graph.

    '''
    depth = 40     # Define the depth parameter, how deep do you want to go?
    nRollout = 0    # initialise the rollout counter
    payoff = 0.0    # initialise the cummulative cost/reward
    while nRollout < depth:
        
        # 1) Stop the rollout if the state is terminal -> horizon reached
        # 2) Stop the rollout if a dead-end is reached. There are several ways
        #    to model a Dead-End.
        #    a) Maze -> Dead-end if s has no applicable actions.
        #    b) Navigation -> Dead-end when the robot is lost and s.predicates
        #                     is an empty set. CAUTION for other problems such
        #                     as SysAdmin or Skillteaching , empty predicates
        #                     do NOT mean dead-end. So just in case do not 
        #                     uncomment the last elif.
        if ( (horizon-nRollout) == 0): return payoff
        elif not s.actions: return payoff - (horizon-nRollout) * 0.5
        #elif not s.predicates: return payoff - (horizon-nRollout) * (0.8)     # max cost for the rest of decission epochs
                
        # The rollouts progress with random actions -> sample an action
        a = s.SampleAction()

        # Sample a state according to P(s'|s,a)
        [successor, cost] = s.SampleChild(a)
        
        # Compute the inmediate cost/reward and update the payoff
        payoff += cost
       
        # update the current state with the sampled successor
        s = successor
        
        # increase the rollout counter
        nRollout += 1
        
    return payoff

#----------------------------------------------------------------------------#    
"""
                     DEFINE THE ACTION SELECTION STRATEGY
"""
def ActionSelection(s,G,c):
    '''
    Parameters
    ----------
    s : state object from GenerativeModel.py
        This is the state that will be considered to choose the action.
    G : dict
        This dictionary is the Graph where the information about the partial 
        tree is stored. Each entry contains the data of state s that is needed
        to apply the modified UCB formula
    c : float
        This is the exploration coefficient. Higher c means more exploration.

    Returns
    -------
    a_UCB : action object
        This action selection strategy returns the action that maximize the
        UCB formula. This formula ensure the minimization of the regret of 
        choosing the wrong action. The action selection method is the heart of 
        UCT. It is the way the algorithm deals with the exploration-exploitation 
        dilemma. Namely, UCT takes the ideas of bandit problems and applies 
        the UCB formula to solve the conflict. In this formula there is a term 
        that votes for exploitation of the best current policy. By contrast, 
        the other term is devoted to exploring less visited nodes.

    '''

    UCB = {}            # Dictionary to save the result of UCB for each action
    
    for a in s.actions: # UCB formula
        
        UCB[a] = G[s][a]["Q-value"] + c * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])
      
    # choose the relevant action that maximize the UCB formula    
    #a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    
    maxUCB = max(UCB.items(), key=operator.itemgetter(1))[1]
    UCB_actions = []    
    # Just in case there are several actions with the same UCB evaluation, take
    # one of them randomly
    for a in UCB.keys():
        if UCB[a] == maxUCB :
            UCB_actions.append(a)
    
    a_UCB = choice(UCB_actions)    
        
    return a_UCB
    
#----------------------------------------------------------------------------#
"""
                     DEFINE SOME USEFUL TOOLS
"""    

from simulation.sim_ToolBox import checkState
from simulation.sim_ToolBox import checkState_FH

#----------------------------------------------------------------------------#     
"""
            DESCRIPTION OF ALL THE PROCESSES WITHIN A TRIAL
""" 
def UCT_Trial(s, H, c, FH_Flag):
    
    global G           # Make sure that I have access to the graph
    K = -0.5           # Internal parameter -> asociated cost to dead-ends
                       # This cost is also defined in rollout method. 
                       # Please, make coherent modifications !
    
    # 1) CHECK IF THE STATE IS TERMINAL---------------------------------------
    # as a reminder: in finite horizion MDP terminal means that the final
    # decision epoch has been reached. In infinte horizon (disc. reward) 
    # MDP, the terminal states are the goals and the dead-ends.
        
    if H == 0 : return 0
    elif not s.actions: return H*K             # Penalty for dead-ends (no applicable actions)
    #elif not s.predicates: return H * K
        
    # 2) CHECK IF THE CURRENT STATE IS NEW -----------------------------------
        # If this state have been visited before, overwrite it with the first
        # instance of that state. Otherwise continue an initialise the node.
    if FH_Flag:
        s = checkState_FH(s,G)
    else:
        s = checkState(s,G)
    
    if s not in G:
        
        # INIT NODE
        # Create a new node in the graph if this is a new state
        G[s] = {}        # intialise node's dictionary
        G[s]["N"] = 0    # Count the first visit to the node (as the number of initialised actions) 
        G[s]["V"] = 0    # Initialise the Value function of the decission Node
        
        # Initialise the Q-values based on rollouts
        # NOTE that (all the possible/only relevant) actions are tested.
        # NOTE that the childs are not created in the graph.
        aux = []          # empty list to ease the maximization
    
        for a in s.actions:      # Init all the applicabel actions
            
            # Count the initialisation of this action as a visit to Node s
            G[s]["N"]+=1 
            
            # Sample a successor according to the generative model
            [successor, cost]= s.SampleChild(a)
            
            # the Qvalue is the inmediate cost/reward plus the long term
            # cost/reward that is estimated through a rollout
            G[s][a]={}
            G[s][a]["Q-value"] = cost + Rollout(successor, H-1)
            aux.append(G[s][a]["Q-value"])  
            
            # Register the visit for this pair s-a
            G[s][a]["Na"]= 1
                
        # Compute the Qvalue of the decision node (V(s)).Two approaches are valid.
        # OPTION1: Averaging the Qvalues ofits successor chance nodes.
        #          V(s) <- SUM[Na(s,a) . Q(s,a)]/N(s)
        """

        """    
        # OPTION2: Taking into account only the optimal Q(s,a)
        #          V(s) <- max(Q(s,a)) | a in A
        G[s]["V"] = max(aux)  
        
        #Return and finish the trial.
        rv = max(aux)        # the return value is the max Q(s,a)
        aux = []             # clear the auxiliary list
        return rv
    
    # 3) EXPAND THE NODE IF IT'S ALREADY IN THE GRAPH ------------------------
    a_UCB = ActionSelection(s,G,c)
    
    # 4) SAMPLE A CHILD  PLAYING THIS ACTION ---------------------------------
    [successor,cost] = s.SampleChild(a_UCB)
    
    
    # 6) UPDATE THE COUNTERS -------------------------------------------------
    G[s]["N"] += 1
    G[s][a_UCB]["Na"] += 1   
    
    # 5) COMPUTE AN ESTIMATE OF Q(s,a_UCB)------------------------------------
    # The importance of this first estimate is twofold. First it will be 
    # used to Compute the final estimate of the Q-value. Second, its
    # recursive architecture allows to expand the state including the 
    # child in the graph, and it also performs a subsequent backup in 
    # reverse order so the trial finishes when the backup is done in the 
    # root node.
    
    QvaluePrime =  cost + UCT_Trial(successor, H-1, c, FH_Flag)
  
    # 7) UPDATE THE Q-VALUE OF THE PAIR (s,a_UCB)-----------------------------
    # classical POMCP-GO approach
    G[s][a_UCB]["Q-value"] += (QvaluePrime - G[s][a_UCB]["Q-value"]) / G[s][a_UCB]["Na"]
      
      
    # 8) UPDATE THE VALUE FUNCTION OF THE DECISION NODE
    
    # OPTION 1: V(s) <- SUM[Na(s,a) . Q(s,a)]/N(s)
    """
        This is not the best approach
    """
    # OPTION 2: V(s) <- max Q(s,a) | a in A
    aux = []              
    for a in G[s].keys(): 
        if a=="N" or a=="V": continue
        else : aux.append(G[s][a]["Q-value"])
    G[s]["V"] = max(aux)
    aux = []
             
    
    return QvaluePrime 
    
#----------------------------------------------------------------------------#    
"""
            DESCRIPTION OF THE MAIN BODY OF THE ALGORITHM
"""
def UCT_like_FH(s0, horizon, maxTrials, timeOut, c, FH_Flag):
    '''
    Parameters
    ----------
    s0 : State object from GenerativeModel.py
        This is the initial state, the root of the tree. All the trials will
        start in this state.
    horizon : int
        This is the finite horizon of the MDP. The planning problem has only
        "horizon" decision epochs.
    maxTrials : int
        This is the "time-out", the stop condition. The algorithm will run
        trials until the number of trials reaches maxTrials
    c : float
        Exploration coefficient for the action selection strategy

    Returns
    -------
    G : dict
        This dictionary is the Graph where the information about the partial 
        tree is stored. Each entry contains the data of all the states that
        have been discovered through trials.
         G = { s1: { N  : Number of times this State has been visited N(s)
                     V  : Value function in the decission Node s. 
                     a1 : { "Na"       : number of times chance node s1,a1 has 
                                         been visited
                            "Q-value"  : current estimation for Q(s1,a1)
                          } 
                     
                     a2 : {...}
                   }
              
              s2: {...}         
            }
        
    Vs0 : list
        This list contains the evolution of V(s0) along trials.

    '''
    
    nTrial = 0                         # Initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # Initialize a graph
    Vs0 = []                           # Stores the evolution of Vs0 along trials
    
    k=1                                # Display counter
    
    elapsedTime = 0.0                  # Init elapsed Time
    tic = time.perf_counter()          # Reference time
    
    while (nTrial < maxTrials) and (elapsedTime < timeOut) :         # Perform trials while possible
        
        if (nTrial >= k*maxTrials/10): # Display progress every 10%
            print( str(k*10) + "%")
            k+=1
            
        nTrial += 1
        UCT_Trial(s0, horizon, c, FH_Flag)
        Vs0.append(G[s0]["V"])
        
        toc =  time.perf_counter()    # Timeout control
        elapsedTime = toc-tic
        
    return G,Vs0


