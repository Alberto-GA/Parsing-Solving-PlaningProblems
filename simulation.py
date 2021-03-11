from ppddl_parser import PDDLParser
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt      
import matplotlib.animation as ani

# ----------------------------------------------------------------------------
#                               UCT LEGACY METHODS 

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

def checkState(s,G):
    
    # state by state check if the predicates of the analysed state matches with 
    # the predicates of already visited states.
    for state in G.keys():
        
        if StateEquality(s,state) :
            # Overwrite s because it is a new instance 
            # of an already visited state
            s = state
            break
    return s


# ----------------------------------------------------------------------------
def runSim_FH(s0, G, horizon):
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
    
    The goal of this function is to run a simulation using the generative model
    of the problem. In each state, the function looks for the best action in G,
    then this action is triggered and the next state is generated according to 
    the probabilty dsitributuions of the generative model. The simulation runs
    until the horizon.

    '''
    s = s0                           # assign s0 to the current state
    count = 0                        # init a counter to count decission epochs
    accrualCost = 0                  # int a cost/reward adder
    while (count < horizon):         
        
        print(s)                     # print current state
        
        if s in G:                   # look for the best action in the policy solution
            for a in G[s].keys():
                if (a != 'N' and a != 'V'):        
                    if (G[s][a]['Q-value'] == G[s]['V']):       # look for the action that satisfies V(s) = Q(s,a)
                        action = a
                        break
        else:                        # The plan is incomplte! Sample a random action and pray
            action = s.SampleAction()
        
            
          
        print(action.name)                           # print selected action
        
        [successor,cost] = s.SampleChild(action)     # Generate new state
        accrualCost += cost                          # update accrual cost
        successor = checkState(successor,G)          # check if the successor is already in the graph
        s = successor                                # assing successor to the current state
        count +=1                                    # update counter

    print("Terminal State")       
    print(accrualCost) 

# ----------------------------------------------------------------------------

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
        ax.matshow(simulation[i], cmap=plt.cm.Spectral)
        
    animator = ani.FuncAnimation(fig, newFrame, interval = 200)
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\simuP6.gif')
    
    
# ----------------------------------------------------------------------------
def update_elevators(s, N_elev, N_floors):
    '''
    
    Parameters
    ----------
    s : TYPE
        DESCRIPTION.
    N_elev : TYPE
        DESCRIPTION.
    N_floors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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
    s0 : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    horizon : TYPE
        DESCRIPTION.
    problem : int
        DESCRIPTION.

    Returns
    -------
    None.

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
        
    floorsList = []
    elevatorList = []
    
    s = s0                          # assign s0 to the current state
    count = 0                       # init a counter to count decission epochs
    accrualCost = 0                 # int a cost/reward adder
    while (count < horizon):
        
        print(s)
        [currentElev, currentFloors] = update_elevators(s, n_elevators, n_floors)
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
        successor = checkState(successor,G)        # check if the successor is already in the graph
        s = successor                              # assing successor to the current state
        count +=1                                  # update counter

    print("Terminal State")       
    print(accrualCost) 
    
 
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
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\elevatorsP1.gif')
    
    return