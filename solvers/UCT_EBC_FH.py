"""
              Second enhancement of UCT like algorithm V2

This is the second attempt to improve the performances of UCT. The approach of 
this algorithm is based on tuning the exploration coefficient in accordance with
an entropic criteria. There are several ways to define the entropy of a state, 
and some of them are considered here. However, all of them agree that the 
entropy of a particular state is higher when the outcome of an action is more 
uncertain. All in all, in this first approach, the solver will compute the en-
tropy using information about the domain so it is not using a pure generative
model. In future approaches we will try to estimate this entropy with enough 
sampled data.

Again, the only difference with UCT relies on the way the action selection is
made.

Note: this code can be used with different grid sizes.


"""
#-------------------------------LIBRAIRES------------------------------------#
import math
import operator
#-------------------------------FUNCTIONS------------------------------------#
"""
The Rollout function is used to initialise the Q-value of a new node in the 
Graph. It basically returns an estimation of the long term cost/reward starting
from the child "s".
Note that the rollout do not need to end in the goal.
Note also that the childs are never included in the graph... 
state.SampleChild(a) instansciates locally a successor state but it is not
stored in the graph.
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
Three different methods to sample an action according to UCB-EBC. All of them 
will only take into account relevant actions... if some experiments must be 
undertaken considering all actions, please replace s.relevActions by s.actions

"""
def ActionSelection_Max(s,G):
    c = [0,2]           # Exploration coefficient bounds 
    UCB = {}            # Dictionary to save the result of UCB for each action
    
    # Compute normalised entropy with MaxEntropy
    en = (c[1] - c[0]) * s.max_entropy + c[0]
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = 5 * en 
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.actions:
    
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""

"""
def ActionSelection_Mean(s,G):
    c = [0,2]           # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
            
    # Compute normalised entropy with MeanEntropy
    en = (c[1]-c[0]) * s.mean_entropy + c[0]
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = 5 * en 
        
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.actions:

        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""

"""
def ActionSelection_Pair(s,G):
    c = [0.0, 5.0]           # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.actions:
        
        # Compute normalised entropy based s.Entropy(a)
        en = (c[1]-c[0]) * s.entropy[a] + c[0]
        # Compute the adaptive explotration coefficient by rescaling with the 
        # higher cost/reward
        c_ebc = 4.0 * en 
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB
    
#----------------------------------------------------------------------------#
"""
checkState(s): is a function that allows us to check if the current "new" 
state "s" is actually new or have been visited before. How does it work? 
Simple, it takes a state as an imput, if there is an state "s" in the graph 
with the same predicates s is overwritten with that state and the recently
created instance is never used again. Otherwise, the function returns the 
reference to the state object without any modification to inlcude it in the 
graph.

This function, in turn, relies on StateEquality(s1,s2) that compares the predi-
cates of two input states and returns TRUE if both states have the same predi-
cates. This means that they are two different instances (references) of the 
same State. Otherwise it returns FALSE.

BONUS: in some problem definitions, the goal state is not defined with all the
predicates. In fact, they usually specify only some requiered predicates while
the rest can be either true or false. Consequently, using StateEquality(s,goal)
may fail to indetify goal states. Use CheckGoal(s,goal) instead!
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
    
              
def Trial(s, H, option):
    
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
    s = checkState(s)
    
    if s not in G:
        
        # First of all init the entropy of the state:
        s.set_entropy()
        
        # Create a new node in the graph if this is a new state
        G[s] = {}        # intialise node's dictionary
        G[s]["N"] = 0    # Count the first visit to the node (as the number of initialised actions) 
        G[s]["V"] = 0    # Initialise the Value function of the decission Node
        
        # Initialise the Q-values based on rollouts
        # NOTE that (all the possible/only relevant) actions are tested.
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
    # The order between this step and step 5 could be reversed. This
    # is so because the target problem allows to play actions that lead
    # the agent to the same state. Taking into account the recursivity of
    # the following step, it could generate an infinte loop of 
    # actionSelection-childSampling if G is not modified so that the UCB
    # formula is affected. 
    # The objective of this strategy is not to remove the loops but to 
    # make the loops finite. To do it, the "lazy" action mustn't be the
    # result of the action selection (UCB) forever. Updating the counters
    # in combination with a high enough exploration coefficient seems to 
    # be a promising strategy...   
    G[s]["N"] += 1
    G[s][a_UCB]["Na"] += 1   
    
      
    
    # 5) COMPUTE AN ESTIMATE OF Q(s,a_UCB)------------------------------------
        # The importance of this first estimate is twofold. First it will be 
        # used to Compute the final estimate of the Q-value. Second, its
        # recursive architecture allows to expand the state including the 
        # child in the graph, and it also performs a subsequent backup in 
        # reverse order so the trial finishes when the backup is done in the 
        # root node.
    
    '''
    if successor == s :
        
        # Kill possible loop (s'=s) with current Qvalue estimate
        # This condition never applies if only relevant actions are considered        
        QvaluePrime = cost + G[s]["V"]
        
    else :
    '''      
    QvaluePrime =  cost + Trial(successor, H-1, option)  
        

    # 7) UPDATE THE Q-VALUE OF THE PAIR (s,a_UCB)-----------------------------
    
        # OPTION 1 : classical POMCP-GO approach
    G[s][a_UCB]["Q-value"] += (QvaluePrime - G[s][a_UCB]["Q-value"]) / G[s][a_UCB]["Na"]
    
        # OPTION 2 : MinPOMCP-GO approach
    
    
    
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
This is the skeleton of the UCT: it relies on the UCT_Trial mnethod wich will
update and refine the information in G, a global variable which represents
the current partial tree. The desired architecture for this variable is:
    
    G = { s1: {a1 : {"Q-value" : current estimation for Q(s1,a1)
                        "Na"   : number of times we have played a1 in s1}
               a2 : {...}
               N  : Number of times this State has been visited
               V  : Value function in the decission Node s. This computation 
                    is not essential but could be useful if "lazy" actions are playing
               } 
         
         s2:{...}
         }
"""
def UCT_adativeCoefficient_FH(s0, horizon, maxTrials, option):
    
    nTrial = 0                         # initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # initialize a graph
    Vs0 = []
    
    # safety check
    """
    print("option:" , option)
    if option == 0:
        print("Adaptive coefficient based on Max Entropy")
    elif option == 1:
        print("Adaptive coefficient based on Mean Entropy")
    elif option == 2:
        print("Adaptive coefficient based on state-action pairs Entropy") 
    else: 
        print("Option error, choose an integer in [0,2]")
        return
    """
    k=1
    
    
    while nTrial < maxTrials :         # perform trials while possible
        
        if (nTrial >= k*maxTrials/10):
            print( str(k*10) + "%")
            k+=1
    
    
        nTrial += 1
        Trial(s0,horizon, option)       
        Vs0.append(G[s0]["V"])  
        
    return G,Vs0      