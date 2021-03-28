"""
THIS ALGORITHM WITHIN THE TRIAL-BASED HEURISTIC TREE-SEARCH METHOD FRAMEWORK
 
HEURISTIC: Rollout legacy from plain UCT
ACTION SELECTION STRATEGY: Upper Confidence Bounds applied to tres (UCB)
                           minimization of the regret of choosing the wrong 
                           action
BACK-UP : maxUCT like. 
OUTCOME SELECTION : succesors are sampled according to P(s'|s,a)
TRIAL LENGTH: the trial continues until the Finite Horizon of the MDP

NOTE:
    State-Space S = {predicates} x {0,...,H}

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
    
    # Give one extra step for new nodes that have been discovered at the end 
    # of the trial. 
    if horizon < 1:
        horizon = 1
    
    depth = 40      # Define the depth parameter, how deep do you want to go?
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
        choosing the wrong action.

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

from simulation.sim_ToolBox import checkState_FH

#----------------------------------------------------------------------------#        

def initNode(s, horizon):
    '''
    Parameters
    ----------
    s : State object from GenerativeModel.py
        This is the node that is going to be initialised in the Graph.
    horizon : int
        Remaining decision steps for the initialised node.

    Returns
    -------
    rv : float
        This function returns the first estimate of V(s) but this return is
        not used by the algorithm because the backup function works 
        differently now.

    '''
    global G
    
    # Create a new node in the graph if this is a new state
    G[s] = {}        # intialise node's dictionary
    G[s]["N"] = 0    # Count the first visit to the node (as the number of initialised actions) 
    G[s]["V"] = 0    # Initialise the Value function of the decission Node
    
    # Initialise the Q-values based on rollouts
    # NOTE that the childs are not created in the graph.
    aux = []          # empty list to ease the maximization
    for a in s.actions:
        
        # Count the initialisation of this action as a visit to Node s
        G[s]["N"] += 1 
        
        # Sample a successor according to the generative model
        [successor, cost]= s.SampleChild(a)
        
        # Create a dictionary to store a lot of information
        G[s][a]={}
        G[s][a]["Cost"] = cost                # Init expected cost of C(s,a)
        G[s][a]["Successors"] = {}            # Keep track of the children of s
        G[s][a]["Q-value"] = cost + Rollout(successor, horizon-1)
        aux.append(G[s][a]["Q-value"])  
        G[s][a]["Na"] = 1               # Register the visit for this pair s-a
            
    # Compute the Qvalue of the decision node (V(s)).
    G[s]["V"] = max(aux)  
    aux = []          # clear the auxiliary list
    
    #Return and finish the trial.
    rv = G[s]["V"]       # the return value is the max Q(s,a)  
    return rv

#-----------------------------------------------------------------------------    
"""
            DESCRIPTION OF ALL THE PROCESSES WITHIN A TRIAL
"""                
def UCT_Trial(s,H,c):
    '''
    Parameters
    ----------
    s : State object from GenerativeModel.py
        This is the current sate.
    H : int
        This is the remaing decision epochs. It must be equal to s.remaining_steps
    c : float
       This is the exploration coefficient for the action selection strategy

    Returns
    -------
    None.

    '''
    
    global G           # Make sure that I have access to the graph
    K = -0.5             # Internal parameter -> asociated cost to dead-ends
    
    
    # 0) CHECK IF THE CURRENT STATE IS NEW -----------------------------------
    # If this state have been visited before, overwrite it with the first
    # instance of that state. Otherwise continue an initialise the node.
    s = checkState_FH(s,G)
    
    # 0.5) ESPECIAL CHECK FOR DEAD_ENDS (No Application for most problems)
    if not s.actions:          
        
        if s not in G :                #Include dead-end node in the Graph
            G[s] = {}
            G[s]["V"] = K*40
            G[s]["N"] = 1
            return
        else :
            G[s]["N"] += 1
            return         
        
            
    # 2) CHECK IF THE STATE IS ALREADY IN THE GRAPH --------------------------
    if s not in G: return initNode(s,H)
    
    # 1) CHECK IF THE STATE IS TERMINAL---------------------------------------        
    if  H == 0 : return
    
    # 3) EXPAND THE NODE IF IT'S ALREADY IN THE GRAPH ------------------------
    a_UCB = ActionSelection(s,G,c)
    
    # 4) SAMPLE A CHILD  PLAYING THIS ACTION ---------------------------------
    [successor,cost] = s.SampleChild(a_UCB)
    
    successor = checkState_FH(successor,G)
    # 6) UPDATE THE COUNTERS -------------------------------------------------
    G[s]["N"] += 1
    G[s][a_UCB]["Na"] += 1
    
    if successor in G[s][a_UCB]["Successors"]:   
        G[s][a_UCB]["Successors"][successor] += 1    
    else :                           
        G[s][a_UCB]["Successors"][successor] = 1
        
    # 5) CONTINUE THE TRIAL---------------------------------------------------     
    UCT_Trial(successor,H-1,c)  
    

    # 7) BACK-UP FUNCTIONS --------------------------------------------------- 
    G[s][a_UCB]["Cost"] += (cost - G[s][a_UCB]["Cost"] ) /  G[s][a_UCB]["Na"]
    
    
    aux = 0
    if G[s][a_UCB]["Successors"]:

        for child in G[s][a_UCB]["Successors"].keys():
        
            aux += G[s][a_UCB]["Successors"][child] * G[child]["V"]
        
    
    G[s][a_UCB]["Q-value"] = G[s][a_UCB]["Cost"] + (aux) / G[s][a_UCB]["Na"] 
    
    
    # 8) UPDATE THE VALUE FUNCTION OF THE DECISION NODE------------------------
    # V(s) <- max Q(s,a) | a in A
    aux = []              
    for a in G[s].keys(): 
        if a=="N" or a=="V": continue
        else : aux.append(G[s][a]["Q-value"])
    G[s]["V"] = max(aux)
    aux = []
               
    return
    
#----------------------------------------------------------------------------#    
"""
            DESCRIPTION OF THE MAIN BODY OF THE ALGORITHM
"""
def maxUCT_like(s0, horizon, maxTrials, timeOut, c):
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
                     a1 : { "cost"     : estimate of C(s1,a1)
                            "Na"       : number of times chance node s1,a1 has 
                                         been visited
                            "Q-value"  : current estimation for Q(s1,a1)
                            "successors" : { s2: number of times s2 has been
                                                 sampled from chance node s1,a1
                                             s3: ...
                                           }
                           } 
                     a2 : {...}
                   }
              
              s2: {...}         
            }
        
    Vs0 : list
        This list contains the evolution of V(s0) along trials.

    '''
    
    nTrial = 0                         # initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # initialize a graph
    Vs0 = []
    k=1                                # Display counter
    
    elapsedTime = 0.0                  # Init elapsed Time
    tic = time.perf_counter()          # Reference time
    
    while (nTrial < maxTrials) and (elapsedTime < timeOut) :         # perform trials while possible
        
        if (nTrial >= k*maxTrials/10): # Display progress every 10%
            print( str(k*10) + "%")
            k+=1    
    
        nTrial += 1
        UCT_Trial(s0,horizon,c)
        Vs0.append(G[s0]["V"])
        
        toc =  time.perf_counter()    # Timeout control
        elapsedTime = toc-tic
        
    return G,Vs0     