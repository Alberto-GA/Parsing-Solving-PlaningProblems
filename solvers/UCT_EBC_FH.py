"""
THIS ALGORITHM WITHIN THE TRIAL-BASED HEURISTIC TREE-SEARCH METHOD FRAMEWORK
 
HEURISTIC: Rollout legacy from plain UCT
ACTION SELECTION STRATEGY: enhanced UCB with adaptive exploration coefficient
                           based on entropy. 3 different options are provided
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
    
    depth = 1      # Define the depth parameter, how deep do you want to go?
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
def ActionSelection_Max(s,G):
    '''
    Parameters
    ----------
    s : state object from GenerativeModel.py
        This is the state that will be considered to choose the action.
    G : dict
        This dictionary is the Graph where the information about the partial 
        tree is stored. Each entry contains the data of state s that is needed
        to apply the modified UCB formula

    Returns
    -------
    a_UCB : action object
        This action selection strategy returns the action that maximize the
        modified UCB formula. This new formula relies on an adaptive explora-
        tion coefficient based on the EXACT max entropy among the actions of s

    '''
    c = [0.1, 2.0]      # Exploration coefficient bounds
    maxCost = 40.0       # max|C(s,a)| 
    UCB = {}            # Dictionary to save the result of UCB for each action
    
    # Compute normalised entropy with MaxEntropy
    en = (c[1] - c[0]) * s.max_entropy + c[0]
    
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = maxCost * en 
    
    # Compute UCB values for each applicable action
    for a in s.actions:
    
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula
    # OPTION1:
    #a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    
    # OPTION 2:
    maxUCB = max(UCB.items(), key=operator.itemgetter(1))[1]
    UCB_actions = []    
    # Just in case there are several actions with the same UCB evaluation, take
    # one of them randomly
    for a in UCB.keys():
        if UCB[a] == maxUCB :
            UCB_actions.append(a)
    
    a_UCB = choice(UCB_actions)
    
    
    
    return a_UCB


def ActionSelection_Mean(s,G):
    '''
    Parameters
    ----------
    s : state object from GenerativeModel.py
        This is the state that will be considered to choose the action.
    G : dict
        This dictionary is the Graph where the information about the partial 
        tree is stored. Each entry contains the data of state s that is needed
        to apply the modified UCB formula

    Returns
    -------
    a_UCB : action object
        This action selection strategy returns the action that maximize the
        modified UCB formula. This new formula relies on an adaptive explora-
        tion coefficient based on the EXACT mean entropy among the actions of 
        the state s.

    '''
       
    c = [0.1, 2.0]   # Exploration coefficient bounds 
    maxCost = 40.0    # max|C(s,a)|
    UCB = {}         # Dictionary to save the result of UCB for each action
    
            
    # Compute normalised entropy with MeanEntropy
    en = (c[1]-c[0]) * s.mean_entropy + c[0]
    
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = maxCost * en 
        
    # Compute UCB values for each applicable action
    for a in s.actions:

        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    # OPTION 1:
    #a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    
    # OPTION 2:
    maxUCB = max(UCB.items(), key=operator.itemgetter(1))[1]
    UCB_actions = []    
    # Just in case there are several actions with the same UCB evaluation, take
    # one of them randomly
    for a in UCB.keys():
        if UCB[a] == maxUCB :
            UCB_actions.append(a)
    
    a_UCB = choice(UCB_actions)
    
    return a_UCB


def ActionSelection_Pair(s,G):
    '''
    Parameters
    ----------
    s : state object from GenerativeModel.py
        This is the state that will be considered to choose the action.
    G : dict
        This dictionary is the Graph where the information about the partial 
        tree is stored. Each entry contains the data of state s that is needed
        to apply the modified UCB formula

    Returns
    -------
    a_UCB : action object
        This action selection strategy returns the action that maximize the
        modified UCB formula. This new formula relies on an adaptive explora-
        tion coefficient based on the EXACT entropy of the pairs state-action.

    '''
    
    c = [0.1, 2.0]     # Exploration coefficient bounds 
    maxCost = 40.0       # max|C(s,a)| 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.actions:
        
        # Compute normalised entropy based s.Entropy(a)
        en = (c[1]-c[0]) * s.entropy[a] + c[0]
        
        # Compute the adaptive explotration coefficient by rescaling with the 
        # higher cost/reward
        c_ebc = maxCost * en 
        
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    # OPTION 1:
    #a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    
    # OPTION 2:
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
              
def Trial(s, H, option, FH_Flag):
    '''
    Parameters
    ----------
    s : State object from GenerativeModel.py
        This is the current sate.
    H : int
        This is the remaing decision epochs. It must be equal to s.remaining_steps
    option : int
        Set the desired action selection strategy:
            0-> max entropy exact
            1-> mean entropy exact
            2-> s-a pair entropy exact
    Returns
    -------
    None.

    '''
    
    global G           # Make sure that I have access to the graph
    K = -0.5          # Internal parameter -> asociated cost to dead-ends
    
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
        # First of all init the entropy of the state:
        s.set_entropy()
        
        # Create a new node in the graph if this is a new state
        G[s] = {}        # intialise node's dictionary
        G[s]["N"] = 0    # Count the first visit to the node (as the number of initialised actions) 
        G[s]["V"] = 0    # Initialise the Value function of the decission Node
        
        # Initialise the Q-values based on rollouts
        # NOTE that the children are not created in the graph.
        aux = []          # empty list to ease the maximization
        
        for a in s.actions:    # Init all the applicable actions
            
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
    
    if   option == 0 : a_UCB = ActionSelection_Max(s,G)    
    elif option == 1 : a_UCB = ActionSelection_Mean(s,G)
    elif option == 2 : a_UCB = ActionSelection_Pair(s,G)
    
    # 4) SAMPLE A CHILD  PLAYING THIS ACTION ---------------------------------
    [successor,cost] = s.SampleChild(a_UCB)
    
    # 6) UPDATE THE COUNTERS -------------------------------------------------
    # The order between this step and step 5 could be reversed.  
    G[s]["N"] += 1
    G[s][a_UCB]["Na"] += 1   
    
      
    
    # 5) COMPUTE AN ESTIMATE OF Q(s,a_UCB)------------------------------------
    # The importance of this first estimate is twofold. First it will be 
    # used to Compute the final estimate of the Q-value. Second, its
    # recursive architecture allows to expand the state including the 
    # child in the graph, and it also performs a subsequent backup in 
    # reverse order so the trial finishes when the backup is done in the 
    # root node.
         
    QvaluePrime =  cost + Trial(successor, H-1, option, FH_Flag)  
        

    # 7) UPDATE THE Q-VALUE OF THE PAIR (s,a_UCB)-----------------------------   
    # classical POMCP-GO approach
    G[s][a_UCB]["Q-value"] += (QvaluePrime - G[s][a_UCB]["Q-value"]) / G[s][a_UCB]["Na"]
    
    
    # 8) UPDATE THE VALUE FUNCTION OF THE DECISSION NODE
    
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
def UCT_adativeCoefficient_FH(s0, horizon, maxTrials, option, FH_Flag):
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
    option : int
        Set the desired action selection strategy:
            0-> max entropy exact
            1-> mean entropy exact
            2-> s-a pair entropy exact

    Returns
    -------
    G : dict
        This dictionary is the Graph where the information about the partial 
        tree is stored. Each entry contains the data of all the states that
        have been discovered through trials
       
        G = { s1: {a1 : {"Q-value" : current estimation for Q(s1,a1)
                        "Na"   : number of times we have played a1 in s1}
               a2 : {...}
               N  : Number of times this State has been visited
               V  : Value function in the decission Node s.
               } 
         
         s2:{...}
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
    
    
    while nTrial < maxTrials :         # perform trials while possible
        
        if (nTrial >= k*maxTrials/10): # Display progress every 10%
            print( str(k*10) + "%")
            k+=1
    
    
        nTrial += 1
        Trial(s0,horizon, option, FH_Flag)       
        Vs0.append(G[s0]["V"])  
        
    return G,Vs0      