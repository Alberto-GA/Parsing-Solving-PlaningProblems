import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt      
import matplotlib.animation as ani

from simulation.sim_ToolBox import checkState
from simulation.sim_ToolBox import checkState_FH

def update_elevators(s, N_elev, N_floors):
    '''
    
    Parameters
    ----------
    s : State object
        This is the state object whose predicates are going to be translated
        into a matrix representation.
    N_elev : int
        Number of elevators
    N_floors : int
        Number of floors

    Returns
    -------
    newElevators : dict
        This dictionary stores all the information related to the floor, pax, 
        door position and direction for each elevator
    newFloors : np.array N_floors x (N_elev + 1)
        This is a matrix representation of the current position of the people
        and the elevators

    '''
    newFloors = np.zeros( (N_floors, N_elev + 1) )          # init a new matrix
    
    # The first column represents the people waiting for the elevators
    
    newElevators = {}
    newElevators['e0'] = {}
    newElevators['e0']['up'] = -1.0
    newElevators['e0']['closed'] = -1.0
    newElevators['e0']['person'] = 0.0
    
    if N_elev > 1:
        newElevators['e1']={}
        newElevators['e1']['up'] = 0.0
        newElevators['e1']['closed'] = 0.0
        newElevators['e1']['person'] = 0.0
    
    for pred in s.predicates:            # proccess all the state predicates
        words = pred.split('_')          # split the string
        
        if words[1]=='dir':
            elev = words[-1]
            newElevators[elev]['up'] = 1.0
            
        elif words[1]=='closed':
            elev = words[-1]
            newElevators[elev]['closed']= 1.0
        
        elif words[1] == 'in':
            elev = words[-1]
            if words[4] == 'up': 
                newElevators[elev]['person'] = 1.0
            else: 
                newElevators[elev]['person'] = -1.0
        
        elif words[1] == 'waiting':
            floor = int(words[-1][-1])
            index = N_floors -1 - floor
            if words[2] == 'up': 
                newFloors[index][0] = 1.0
            else : 
                newFloors[index][0] = -1.0
        
        elif words[1] == 'at':
            floor = int(words[-1][-1])
            index = N_floors -1 - floor
            elev = words[4]
            newElevators[elev]['floor'] = floor
            newFloors[index][int(elev[-1])+1] = 1.0
            
        else:
            print('Something went wrong')
            
            
    return newElevators, newFloors
    
    
    

def sim_Elevators(s0, G, horizon, problem):
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
    same idea. The difference between this functions is that this function is 
    specifically designed for 'elevators mdp' and it is able to find
    a graphic represention of the simulation and save it in a .gif file.
    
    '''
    
    # Define the grid size according to the problem
    n_elevators = 0
    n_floors = 0
    
    if problem == 1:
        n_elevators = 1
        n_floors = 3
    
    elif problem <= 3:
        n_elevators = 2
        n_floors = 3
    
    elif problem == 4:
        n_elevators = 1
        n_floors = 4
            
    elif problem <= 6:
        n_elevators = 2
        n_floors = 4
        
    elif problem == 7:
        n_elevators = 1
        n_floors = 5
            
    elif problem <= 9:
        n_elevators = 2
        n_floors = 5
        
    else:
        n_elevators = 1
        n_floors = 6
        
    floorsList = []                 # List of matrix depicting the evolution of elevators' positions
    elevatorList = []               # List of dictionaries depicting the evolution of elevators' states
    
    s = s0                          # assign s0 to the current state
    count = 0                       # init a counter to count decission epochs
    accrualCost = 0                 # int a cost/reward adder
    while (count < horizon):
        
        print(s)
        
        # Create the matrix and dictionary that depicts this state
        [currentElev, currentFloors] = update_elevators(s, n_elevators, n_floors)
        
        # Append them to the lists
        floorsList.append(currentFloors)
        elevatorList.append(currentElev)
        
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
        successor = checkState_FH(successor,G)        # check if the successor is already in the graph
        s = successor                              # assing successor to the current state
        count +=1                                  # update counter

    print("Terminal State")       
    print(accrualCost) 
    
    # The simulation is done -> go on with the graphical representation
    fig = plt.figure()                                # Create an empty figure with no axes
    fig, ax_lst = plt.subplots(1, n_elevators + 1)    # Add a 1x(2-3) grid of Axes 
    fig.suptitle('Elevators p' + str(problem))        # Add a title to the figure
    
            
    def newFrame(i=int):
        e1_state = [ elevatorList[i]['e0']['up'], elevatorList[i]['e0']['closed'], elevatorList[i]['e0']['person'] ]
        
        ax_lst[0].matshow(floorsList[i], cmap=plt.cm.Spectral, vmin = -1.0 , vmax = 1.0)
        ax_lst[1].cla()
        ax_lst[1].set_title('e0')
        ax_lst[1].set_ylim(-1,1)
        ax_lst[1].yaxis.set_major_locator(ticker.FixedLocator([-1, 1]))
        ax_lst[1].bar(['Up','Closed','person'], e1_state , color = 'cyan')
        
        if n_elevators > 1:                                                              
            e2_state = [ elevatorList[i]['e1']['up'], elevatorList[i]['e1']['closed'], elevatorList[i]['e1']['person'] ]
            
            ax_lst[2].cla()
            ax_lst[2].set_title('e1') 
            ax_lst[2].set_ylim(-1,1)
            ax_lst[2].yaxis.set_major_locator(ticker.FixedLocator([-1, 1]))
            ax_lst[2].bar(['Up','Closed','person'], e2_state, color = 'cyan')
    
    animator = ani.FuncAnimation(fig, newFrame, interval = 400)
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\elevatorsP1_FH50k.gif')
    
    return