"""
THIS ALGORITHM WITHIN THE TRIAL-BASED HEURISTIC TREE-SEARCH METHOD FRAMEWORK
 
HEURISTIC: Rollout legacy from plain UCT
ACTION SELECTION STRATEGY: enhanced UCB with adaptive exploration coefficient
                           based on entropy. 6 different options are provided
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
    c = [0.5, 50.0]      # Exploration coefficient bounds
    maxCost = 0.75       # max|C(s,a)|
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
    
    c = [0.5, 50.0]      # Exploration coefficient bounds
    maxCost = 0.75       # max|C(s,a)|
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
    
    c = [0.5, 50.0]      # Exploration coefficient bounds
    maxCost = 0.75       # max|C(s,a)|
    UCB = {}            # Dictionary to save the result of UCB for each action
    
    # Compute UCB values for each applicable action
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


"""
The previous functions took advantage of some domain-dependent information. 
They open a small window to see what is inside the transition model. It could
seem that they are cheating and breaking the spirit of a generative model;
however, since this is not Reinforced Learning but probabilistic planning 
(let's say something like model-based RL) this is completely correct.

Nevertheless, let's propose more functions that will use learnt/estimated 
probabilities to compute the entropy of the actions. Then, the following 
functions will try to accomplish the same tasks but using estimated probabi-
lities.
 
Warning!! at the begining there are not enough sampled data and the transitions
may seem deterministic. This leads to a low entropy and consequently to a low 
exploration. There are two options to overcome this issue:
    a) Wait several Trials before using this action selection estrategy. First
       Trials should use plain UCB formula with a fixed exploration 'c'
    b) Inside this method, define a fixed default value for the exploration
       coefficient that will be used in case there are not enoguh sampled data
Both options are basically the same thing, here we propose to use opt b.
"""

def ActionSelection_Pair_Estimated(s,G):
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
        tion coefficient based on the ESTIMATED entropy of the pairs 
        state-action.

    '''
    
    c = [0.5, 50.0]      # Exploration coefficient bounds
    maxCost = 0.75       # max|C(s,a)|
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    # Compute UCB values for each applicable action
    for a in s.actions:
        
        # Estimate the entropy. If the action does not have enough visits to 
        # estimate the probabilities correctly, set entropy to 1.0 to push for 
        # exploration.
        if G[s][a]["Na"] > 10: 
            e = 0.0
            for child in G[s][a]["Successors"].keys():
                prob = G[s][a]["Successors"][child]/G[s][a]["Na"]
                e += - prob * math.log2(prob)
        else : 
            e = 1.0
        
        # Compute normalised entropy based on the estimated entropy
        en = (c[1]-c[0]) * e + c[0]
        
        # Compute the adaptive explotration coefficient by rescaling with the 
        # higher cost/reward
        c_ebc = maxCost * en 
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    
    return a_UCB



def ActionSelection_Mean_Estimated(s,G):
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
        tion coefficient based on the ESTIMATED mean entropy among the actions 
        
    '''
    
    c = [0.5, 50.0]      # Exploration coefficient bounds
    maxCost = 0.75       # max|C(s,a)|
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    
    # Estimate the mean entropy
    entropies = []
    for a in s.actions:
        
        # Estimate the entropy. If the action does not have enough visits to 
        # estimate the probabilities correctly, set entropy to 1.0 to push for 
        # exploration.
        if G[s][a]["Na"] > 10: 
            e = 0.0
            for child in G[s][a]["Successors"].keys():
                prob = G[s][a]["Successors"][child]/G[s][a]["Na"]
                e += - prob * math.log2(prob)
        else : 
            e = 1.0
        
        entropies.append(e)
    
    e = (1/len(entropies)) * sum(entropies)
    
    # Compute normalised entropy based on the estimated entropy
    en = (c[1]-c[0])* e + c[0]
    
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = maxCost * en 
    
    # Compute UCB values for each applicable action
    for a in s.actions:         
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB



def ActionSelection_Max_Estimated(s,G):
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
        tion coefficient based on the ESTIMATED max entropy among the actions 
        
    '''
    
    c = [0.5, 50.0]      # Exploration coefficient bounds
    maxCost = 0.75       # max|C(s,a)|
    UCB = {}          # Dictionary to save the result of UCB for each action
    
    
    # Estimate the max entropy
    entropies = []
    for a in s.actions:
        
        # Estimate the entropy. If the action does not have enough visits to 
        # estimate the probabilities correctly, set entropy to 1.0 to push for 
        # exploration.
        if G[s][a]["Na"] > 10: 
            e = 0.0
            for child in G[s][a]["Successors"].keys():
                prob = G[s][a]["Successors"][child]/G[s][a]["Na"]
                e += - prob * math.log2(prob)
        else : 
            e = 1.0
        
        entropies.append(e)
    
    e = max(entropies)
    # Compute normalised entropy based on the estimated entropy
    en = (c[1]-c[0])* e + c[0]
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = maxCost * en 
    
    for a in s.actions:         
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
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
    
    # First of all init the entropy of the state:
    s.set_entropy()

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
def Trial(s,H,option):
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
            3-> max entropy estimated
            4-> mean entropy estimated
            5-> s-a pair entropy estimated

    Returns
    -------
    None.

    '''
    
    global G           # Make sure that I have access to the graph
    K = -0.5           # Internal parameter -> asociated cost to dead-ends
    
    
    # 0) CHECK IF THE CURRENT STATE IS NEW -----------------------------------
    # If this state have been visited before, overwrite it with the first
    # instance of that state. Otherwise continue an initialise the node.
    s = checkState_FH(s,G)
    
    # 0.5) ESPECIAL CHECK FOR DEAD_ENDS (Only applicable to MAZE)
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
    
    # 1) CHECK IF THE STATE IS TERMINAL --------------------------------------        
    if  H == 0 : return
    
    # 3) EXPAND THE NODE IF IT'S ALREADY IN THE GRAPH ------------------------
    if   option == 0 : a_UCB = ActionSelection_Max(s,G)    
    elif option == 1 : a_UCB = ActionSelection_Mean(s,G)
    elif option == 2 : a_UCB = ActionSelection_Pair(s,G)
    elif option == 3 : a_UCB = ActionSelection_Max_Estimated(s,G)
    elif option == 4 : a_UCB = ActionSelection_Mean_Estimated(s,G)
    elif option == 5 : a_UCB = ActionSelection_Pair_Estimated(s,G)
    
    
    # 4) SAMPLE A CHILD  PLAYING THIS ACTION ---------------------------------
    [successor,cost] = s.SampleChild(a_UCB)   
    successor = checkState_FH(successor,G)   # Check because this object will
                                             # be used as a key later.
    
    # 6) UPDATE THE COUNTERS -------------------------------------------------
    G[s]["N"] += 1
    G[s][a_UCB]["Na"] += 1
    
    if successor in G[s][a_UCB]["Successors"]:   
        G[s][a_UCB]["Successors"][successor] += 1    
    else :                           
        G[s][a_UCB]["Successors"][successor] = 1
        
    # 5) CONTINUE THE TRIAL---------------------------------------------------     
    Trial(successor, H-1, option)  
    

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
def maxUCT_adaptive(s0, horizon, maxTrials, timeOut, option):
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
            3-> max entropy estimated
            4-> mean entropy estimated
            5-> s-a pair entropy estimated

    Returns
    -------
    G : dict
        This dictionary is the Graph where the information about the partial 
        tree is stored. Each entry contains the data of all the states that
        have been discovered through trials
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
        Trial(s0, horizon, option)
        Vs0.append(G[s0]["V"])  
        
        toc =  time.perf_counter()    # Timeout control
        elapsedTime = toc-tic
        
    return G,Vs0     