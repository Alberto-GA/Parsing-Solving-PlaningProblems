"""
                        UCT like algorithm V2
This is the second versiond of the original code. The previous version was 
adapted to work with a declarative model of the Obstacles-Maze-SSP_MDP. 
However, this algorithm doesn't need to know the whole transition model. In
fact, it doesn't need to instanciate all the possible states before starting.
For these reasons this code has been enhanced to work only with a generative
model that is described in SSP_GenerativeModel.py. Then, the algorithm will 
take actions at given states that eventually will lead it to new states. As a
consequence, this algorithm has an additional mission which consists in 
identifying if the "new" state have been already visited. This is important 
because new states are instanciated when a child is sampled but we only want to 
keep one instance of each particular state.

This algorithm is an adptation of the classical UCT algorithm. It is based on 
RTDP. Credits to Caroline Chanel.

"""
#-------------------------------LIBRAIRES------------------------------------#
import math
import operator
from random import choice
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
def Rollout(s,horizon):
    
    depth = 3
    nRollout = 0    # initialise the rollout counter
    payoff = 0      # initialise the cummulative cost/reward
    while nRollout < depth:
        
        # Stop the rollout if a dead-end is reached.
        # NOTE: "the first state will never be a dead-end so payoff not 0"
        if ( (horizon-nRollout) == 0): return payoff
        
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
The action selection method is the heart of UCT. It is the way the algorithm 
deals with the exploration-exploitation dilemma. Namely, UCT takes the ideas 
of bandit problems and applies the UCB formula to solve the conflict. In this
formula there is a term that votes for exploitation of the best current policy.
By contrast, the other term is devoted to exploring less visited nodes.

It receives the current Graph as input because this function doesn't modify 
the graph.

"""


"""
There is a problem with the action selection when "lazy" actions are chosen 
from the UCB formula because the successor is the same state and this issue 
comes into conflict with the recursivity. For this reason I've suggested to 
take into account only relevant actions using the function below. 

WARNING: This choice doesn't seem to be a great idea because the nodes are 
completely initialised. This means that all the actions are tried and they
are provided with a first stimate of the Q-value(s,a) thanks to a rollout.
And where is the problem? Well, this values are initialised but the actions 
are no longer taken into account through action selection (UCB) resulting in
a biased situation between relevant and lazy actions.

CONCLUSION: if you want to remove the "lazy" actions to kill loops of type 
s'=s, only relevant actions must be initialised!!! -> LOOK STEP 2

"""
def ActionSelection(s,G,c):

    UCB = {}         # Dictionary to save the result of UCB for each action
    
    for a in s.actions:  # UCB formula
        
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
def UCT_Trial(s, H, c):
    
    global G           # Make sure that I have access to the graph
    K = -5             # Internal parameter -> asociated cost to dead-ends
    
    # 1) CHECK IF THE STATE IS TERMINAL---------------------------------------
        # as a reminder: in finite horizion MDP terminal means that the final
        # decision epoch has been reached. In infinte horizon (disc. reward) 
        # MDP, the terminal states are the goals and the dead-ends.
        
    if H == 0 : return 0
    elif not s.actions: return K             # Penalty for dead-ends (no applicable actions)
        
    # 2) CHECK IF THE CURRENT STATE IS NEW -----------------------------------
        # If this state have been visited before, overwrite it with the first
        # instance of that state. Otherwise continue an initialise the node.
    s = checkState(s)
    
    if s not in G:
        #print("New state detected -> Initialisation")
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
        # To expand a node, UCT applies the action selection  
        # strategy that is based on the UCB formula, this code provide two
        # different functions to return the 'best' action:
        #    -Actionselection(s,G)-> all actions, including "lazy" actions, are considered.
        #    -RelevantActionSelection(s,G)-> only relevant actions are considered.
    a_UCB = ActionSelection(s,G,c)
    
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
    
    if successor == s :
        
        # Kill possible loop (s'=s) with current Qvalue estimate
        # This condition never applies if only relevant actions are considered        
        QvaluePrime = cost + G[s]["V"]
        
    else :
            
        QvaluePrime =  cost + UCT_Trial(successor, H-1, c)  
        
  
    # 7) UPDATE THE Q-VALUE OF THE PAIR (s,a_UCB)-----------------------------
    
        # OPTION 1 : classical POMCP-GO approach
    G[s][a_UCB]["Q-value"] += (QvaluePrime - G[s][a_UCB]["Q-value"]) / G[s][a_UCB]["Na"]
    
        # OPTION 2 : MinPOMCP-GO approach
    
    
    
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
This is the skeleton of the UCT: it relies on the UCT_Trial method wich will
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

Note that I've decided to use a dictionary becasue it is easy to visualize in 
the variable explorer. However other data structures could be useful as well...

The Main function needs three arguments:
    1) Initial state
    2) max number of trials
    3) exploration coefficient

The basic return is the whole tree so that it is possible to se the estimated
value of the initial state, and the suggested policy solution.
"""
def UCT_like_FH(s0, horizon, maxTrials,c):
    
    nTrial = 0                         # Initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # Initialize a graph
    Vs0 = []                           # Stores the evolution of Vs0 along trials
    
    k=1
    
    while nTrial < maxTrials :         # Perform trials while possible
        
        if (nTrial >= k*maxTrials/10):
            print( str(k*10) + "%")
            k+=1
            
        nTrial += 1
        UCT_Trial(s0, horizon, c)
        Vs0.append(G[s0]["V"])
        
        
    return G,Vs0


