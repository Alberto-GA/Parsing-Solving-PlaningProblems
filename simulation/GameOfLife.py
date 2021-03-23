import numpy as np
import matplotlib.pyplot as plt      
import matplotlib.animation as ani

from simulation.sim_ToolBox import checkState


def updateCells(s,H,W):
    '''
    
    Parameters
    ----------
    s : State object
        This is the state object whose predicates are going to be translated
        into a matrix representation.
    H : int
        Height - Vertical size of the grid map
    W : TYPE
        Width - horizontal size of the grid map

    Returns
    -------
    newCells : np.array (H,W)
        This is the matrix representation of s.

    '''
    
    newCells = np.zeros( (H, W) )              # init a new matrix
    
    # A predicate follows this syntax 'alive__xi_yj' i= 1,H j = 1,W
    for pred in s.predicates:            # proccess all the state predicates
        words = pred.split('_')          # split the string
        i = int( words[2][1:] ) - 1      # x index
        j = int( words[3][1:] ) - 1      # y index
        newCells[i,j] = 1.0              # set 1.0 to live cells
    
    return newCells

def sim_GameOfLife(s0, G, horizon, problem):
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
    None.
    
    This function has the same goal as runSim_FH, in fact, it works with the 
    same idea. The difference between them is that this function is 
    specifically designed for 'Game of life mdp' and it is able to find
    a graphic represention of the simulation and save it in a .gif file.

    '''
    # Init the size of the grid map according to the planning problem
    height = 0
    width = 0
     
    if problem <= 3:
        height = 3
        width = 3
    
    elif problem <= 6:
        height = 4
        width = 4
        
    elif problem <= 9:
        height = 5
        width = 5  
    else:
        height = 3
        width = 10
    
    # Init a list to store all the matrix representations 
    generations = []
    
    # Init a list to save the evolution of the reward
    rewards = []                
    
    # Start the simulation
    s = s0                          # assign s0 to the current state
    count = 0                       # init a counter to count decission epochs
    accrualCost = 0                 # int a cost/reward adder
    while (count < horizon):
        
        print(s)
        
        # Create the matrix and dictionary that depicts this state
        currentCells = updateCells(s, height, width)
        
        # Append it to the list
        generations.append(currentCells)
        
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
        rewards.append(accrualCost)                # keep the track of the rewards
        successor = checkState(successor,G)        # check if the successor is already in the graph
        s = successor                              # assing successor to the current state
        count +=1                                  # update counter

    print("Terminal State")       
    print(accrualCost) 
    
    # Print and save animation
    fig, ax = plt.subplots()
    fig, ax_lst = plt.subplots(1, 2)    # Add a 1x(2-3) grid of Axes 
    fig.suptitle('Elevators p' + str(problem))        # Add a title to the figure
    
    
    
    def newFrame(i=int):
        ax_lst[0].matshow(generations[i], cmap=plt.cm.Spectral, vmin = -1.0 , vmax = 1.0)
    
        ax_lst[1].cla()
        ax_lst[1].set_title('Accrual reward')
        ax_lst[1].set_ylim(0, (int(accrualCost/10) + 5)*10 )
        #ax_lst[1].yaxis.set_major_locator(ticker.FixedLocator([-1, 1]))
        #ax_lst[1].bar(['reward'], [rewards[i]] , color = 'cyan')
        # OPT 2 -> plot
        ax_lst[1].set_xlim(0,40)
        ax_lst[1].plot(rewards[:i], color = 'cyan')
        
    animator = ani.FuncAnimation(fig, newFrame, interval = 400)
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\GameOfLifeP1.gif')
    
    return