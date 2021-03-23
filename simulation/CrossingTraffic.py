import numpy as np
import matplotlib.pyplot as plt      
import matplotlib.animation as ani

from ppddl_parser import PDDLParser
from simulation.sim_ToolBox import checkState

def UpdateMap(s,H):
    '''
    Parameters
    ----------
    s : state object
        This is the state object whose predicates are going to be translated
        into a matrix representation.
    H : int
        This is the size of the grid map HxH.

    Returns
    -------
    newMap : np.array (H,H)
        The function returns a new matrix that contains a representation of the
        state's predicates.
        robot position <- 1
        car positions <- -1
        empty <- 0

    '''
    
    newMap = np.zeros((H,H))             # init a new matrix
    
    for pred in s.predicates:            # proccess all the state predicates
        words = pred.split('_')          # split the string
        if words[0]=='robot':            # this predicate depicts robot position
            val = 1
        else:                            # else obstacle positon
            val = -1
        x = int(words[3][1:]) - 1        # compute the  matirx index from predicate info
        y = H - int(words[4][1:])
        
        newMap[y][x] = val               # Update the map with this info
    
    return newMap
        

def sim_CrossingTraffic(s0, G, horizon):
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

    Returns
    -------
    None. 
    
    This function has the same goal as runSim_FH, in fact, it works with the 
    same idea. The difference between this functions is that this function is 
    specifically designed for 'crossing traffic mdp' and it is able to find
    a graphic represention of the simulation and save it in a .gif file.

    '''
    
    # Build grid -> it is necessary to parse again the Domain.ppddl file in 
    #               order to read the size of the grid map
    directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\crossing_traffic_inst_mdp\p6\Domain.ppddl'
    MyDomain = PDDLParser.parse(directory1)
    
    nb_states = len(MyDomain.predicates) / 2   # Only half of the predicates encode robot position
    H = int(np.sqrt(nb_states))                # Assume square grid map
    
    gridMap = UpdateMap(s0,H)                  # build the representation of the initial state
    
    simulation = []                      # create a list to save the evolution
    simulation.append(gridMap)           # of the grid map
    
    # Run simulation -> same idea as runSim_FH
    s = s0                          # assign s0 to the current state
    count = 0                       # init a counter to count decission epochs
    accrualCost = 0                 # int a cost/reward adder
    while (count < horizon):
        
        print(s)
        simulation.append(UpdateMap(s,H))
        
        if s in G:                   # look for the best action in the policy solution
            for a in G[s].keys():
                if (a != 'N' and a != 'V'):        
                    if (G[s][a]['Q-value'] == G[s]['V']):    # look for the action that satisfies V(s) = Q(s,a)
                        action = a
                        break
        else:                        # The plan is incomplte! Sample a random action and pray
            action = s.SampleAction()
        
            
          
        print(action.name)                         # print selected action
        [successor,cost] = s.SampleChild(action)   # Generate new state
        accrualCost += cost                        # update accrual cost
        successor = checkState(successor,G)        # check if the successor is already in the graph
        s = successor                              # assing successor to the current state
        count +=1                                  # update counter

    print("Terminal State")       
    print(accrualCost) 
        
    if (accrualCost == -40.0):
        return
    # Print and save animation
    fig, ax = plt.subplots()
    
    def newFrame(i=int):
        ax.matshow(simulation[i], cmap=plt.cm.Spectral, vmin = -1.0 , vmax = 1.0)
        
    animator = ani.FuncAnimation(fig, newFrame, interval = 200)
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\FroggerP6_MX.gif')
    
    return