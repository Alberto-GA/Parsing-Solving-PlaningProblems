"""
       from UCT like algorithm V2 -> NEW BACKUP FUNCTION -> MAX UCT


"""
#-------------------------------LIBRAIRES------------------------------------#
import math
import operator
from random import choice
#-------------------------------FUNCTIONS------------------------------------#
"""
"""
def Rollout(s, horizon):
    
    depth = 40      # Define the depth parameter, how deep do you want to go?
    nRollout = 0    # initialise the rollout counter
    payoff = 0      # initialise the cummulative cost/reward
    while nRollout < depth:
        
        # NOTE: "the first state will never be a dead-end so payoff not 0"
        # 1) Stop the rollout if the state is terminal -> horizon reached
        # 2) Stop the rollout if a dead-end is reached.
        if ( (horizon-nRollout) == 0): return payoff
        elif not s.actions: return payoff - 5.0
        #elif not s.predicates: return payoff - (horizon-nRollout) * (0.8) # max cost for the rest of decission epochs        
        
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

"""
def ActionSelection(s,G,c):
    
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
"""
def StateEquality(s1,s2):
    rv= True                                         # Init return value
    if len(s1.predicates) == len(s2.predicates):     # Check if the number of predicates is the same
        for pred in s1.predicates:
            if pred not in s2.predicates:            # Check if every predicate of s1 is in s2.
                rv = False
                break                                # One mismatch is enough to return False
    else:
        rv = False
    
    return rv
      
    
def checkState(s):
    
    global G              # Get access to the graph
    # state by state check if the predicates of the analysed state matches with 
    # the predicates of already visited states.
    for state in G.keys():
        
        if StateEquality(s,state) :
            # Overwrite s because it is a new instance 
            # of an already visited state
            s = state
            break
    return s


def CheckGoal(s1, s_g):
    
    rv = True
    for pred in s_g.predicates:
        if pred not in s1.predicates:
           rv = False
           break
       
    return rv

#----------------------------------------------------------------------------#        

def initNode(s, horizon):
    global G
    
    # Create a new node in the graph if this is a new state
    G[s] = {}        # intialise node's dictionary
    G[s]["N"] = 0    # Count the first visit to the node (as the number of initialised actions) 
    G[s]["V"] = 0    # Initialise the Value function of the decission Node
    
    # Initialise the Q-values based on rollouts
    # NOTE that (all the possible/only relevant) actions are tested.
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
              
def UCT_Trial(s,H,c):
    
    global G           # Make sure that I have access to the graph
    K = -0.5             # Internal parameter -> asociated cost to dead-ends
    
    
    # 0) CHECK IF THE CURRENT STATE IS NEW -----------------------------------
    # If this state have been visited before, overwrite it with the first
    # instance of that state. Otherwise continue an initialise the node.
    s = checkState(s)
    
    # 1) CHECK IF THE STATE IS TERMINAL---------------------------------------        
    if  H == 0 : return
    elif not s.actions:          
        
        if s not in G :                #Include dead-end node in the Graph
            G[s] = {}
            G[s]["V"] = K*40
            G[s]["N"] = 1
            return
        else :
            G[s]["N"] += 1
            return         
        
            
    # 2) CHECK IF THE STATE IS ALREADY IN THE GRAPH
    if s not in G: return initNode(s,H)
    
    # 3) EXPAND THE NODE IF IT'S ALREADY IN THE GRAPH ------------------------
    a_UCB = ActionSelection(s,G,c)
    
    # 4) SAMPLE A CHILD  PLAYING THIS ACTION ---------------------------------
    [successor,cost] = s.SampleChild(a_UCB)
    
    successor = checkState(successor)
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

"""
def maxUCT_like(s0, horizon, maxTrials,c):
    
    nTrial = 0                         # initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # initialize a graph
    Vs0 = []
    k=1
    
    while nTrial < maxTrials :         # perform trials while possible
        
        if (nTrial >= k*maxTrials/10):
            print( str(k*10) + "%")
            k+=1    
    
        nTrial += 1
        UCT_Trial(s0,horizon,c)
        Vs0.append(G[s0]["V"])  
        
    return G,Vs0     