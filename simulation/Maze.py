import numpy as np
import matplotlib.pyplot as plt      
import matplotlib.animation as ani

from simulation.sim_ToolBox import checkState


def update_Maze(s, H, W, obs_lst):
    '''
    Parameters
    ----------
    s : state object
        This is the state object whose predicates are going to be translated
        into a matrix representation.
    H : int
        This is the vertical size (height) of the grid map.
    W : int
        This is the horizontal size (width) of the grid map.
    obs_lst : list of duples
        This is the list of obstacles (Dead-Ends). Each element is a duple made
        of two integers giving the position of each obsstacle.

    Returns
    -------
    newMaze : np.array (H,W)
        The function returns a new matrix that contains a representation of the
        state's predicates.
        robot position <- 1
        obs positions <- -1
        empty <- 0

    '''
    
    newMaze = np.zeros((H,W))            # init a new matrix
    
    for i,j in obs_lst:                  # Set the position of the obstacles
        newMaze[i,j] = -1.0
    
    for pred in s.predicates:            # Set the position of the agent
        
        words = pred.split('_')          # split the string
        
        if words[1] == 'at':             # this predicate depicts robot position
                   
            x = int(words[2][1:])        # compute the  matirx index from predicate info
            y = int(words[3][1:])
            newMaze[x][y] = 1.0          # Update the map with this info
    
    return newMaze  


def sim_Maze(s0, G, horizon, problem):
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
    problem : int
        It is simply an identifier to decided the configuration of the problem
        be ware of the fact that this is a hardcoded customisation.

    Returns
    -------
    none.
    
    This function has the same goal as runSim_FH, in fact, it works with the 
    same idea. The difference between them is that this function is 
    specifically designed for 'Maze mdp' and it is able to find
    a graphic represention of the simulation and save it in a .gif file.

    '''
    
    # init the size of the grid map and the obstacles according to ppddl problem
    height = 0
    width = 0
    obstacles = []
    
    if problem == 1:
        height = 3
        width = 4
        obstacles.append((0,2))
        obstacles.append((2,1))
    
    else:
        print('Problem not supported')
        return
    
    # Define some list to keep track of the evolution of the parameters
    positions = []
    rewards = []
    
    # Start the simulation
    s = s0                          # assign s0 to the current state
    count = 0                       # init a counter to count decission epochs
    accrualCost = 0                 # int a cost/reward adder
    while (count < horizon):
        
        print(s)
        
        # Update the status with the current state predicates
        current_pos = update_Maze(s, height, width, obstacles)
        
        # Append current status to the list
        positions.append(current_pos)

        if s in G:                   # look for the best action in the policy solution
            for a in G[s].keys():
                if (a != 'N' and a != 'V'):        
                    if (G[s][a]['Q-value'] == G[s]['V']):    # look for the action that satisfies V(s) = Q(s,a)
                        action = a
                        break
        
        elif s.actions:                        # The plan is incomplete! Sample a random action and pray
            action = s.SampleAction()
        
        else:                                  # Dead-end reached
            accrualCost += -5.0
            break
            
          
        print(action.name)                         # print selected action
        [successor,cost] = s.SampleChild(action)   # Generate new state
        accrualCost += cost                        # update accrual cost
        rewards.append(accrualCost)                # keep the track of the rewards
        successor = checkState(successor,G)        # check if the successor is already in the graph
        s = successor                              # assing successor to the current state
        count +=1                                  # update counter

    print("Terminal State")       
    print(accrualCost) 
    
    # Print and save animation
    fig, ax = plt.subplots()
    
    def newFrame(i=int):
        ax.matshow(positions[i], cmap=plt.cm.Spectral, vmin = -1.0 , vmax = 1.0)
        
    animator = ani.FuncAnimation(fig, newFrame, interval = 300)
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\MazeP1_OK.gif')
    
    return