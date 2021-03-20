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
        
        elif s.actions :             # The plan is incompelte! Sample a random action and pray  
            action = s.SampleAction()
        
        else:                        # This is a Dead-End state. Imposible to continue
            accrualCost += -5.0
            print('Dead-End reached, simulation finished')
            return
            
          
        print(action.name)                           # print selected action
        
        [successor,cost] = s.SampleChild(action)     # Generate new state
        accrualCost += cost                          # update accrual cost
        successor = checkState(successor,G)          # check if the successor is already in the graph
        s = successor                                # assing successor to the current state
        count +=1                                    # update counter

    print("Terminal State - Finite Horizon reached")       
    print(accrualCost) 
    return 
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
        ax.matshow(simulation[i], cmap=plt.cm.Spectral, vmin = -1.0 , vmax = 1.0)
        
    animator = ani.FuncAnimation(fig, newFrame, interval = 200)
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\FroggerP3.gif')
    
    return
    
# ----------------------------------------------------------------------------
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
        successor = checkState(successor,G)        # check if the successor is already in the graph
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
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\elevatorsP1MAX.gif')
    
    return

# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------

def update_Recon(s, H, W, mission_preds):
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
    mission_preds : list of strings
        This list contains only the predicates related to the status of the
        mission. In other words, the predicates that do not describe the
        status of the tools.

    Returns
    -------
    mission_status : dict
        The Keys of this dictionary are the mission predicates and their
        respective values can be either 0 (predicate False in s) or 1 (True).
    tools_status : dict
        The Keys of this dictionary are the tool status predicates and their
        respective values can be either 0 (tool not damaged in s) or 1 (if 
        the tool is broken)
                                        
    newPos : np.array(HxW)
        This is the matrix representation of the position of the robot within
        the grid-map.

    '''
    
    # 1) Init all the output variables
    
    mission_status = {}               # dictionary to report mission status
    for pred in mission_preds:
        mission_status[pred] = 0.0
        
    tools_status = {}                 # dictionary to report tools status
    tools_status['damaged__l1']= 0.0
    tools_status['damaged__w1']= 0.0
    tools_status['damaged__p1']= 0.0
    
    newPos = np.zeros( (H, W) )       # init a new matrix to report position
    
    
    # 2) Check state predicates and update the output variables
    
    for pred in s.predicates:            # Read predicate by predicate
        words = pred.split('_')          # split the string
        
        if words[2]=='a1':                # This predicate depicts robot position
            j = int(words[3][1:])         # x position (from left to right)
            i = H - int(words[4][1:]) - 1 # y position (from bottom to top)
            newPos[i,j] = 1.0             # set 1 to robot position else 0
            
        elif words[0]=='damaged':         # This predicate is a tool status predicate
            tools_status[pred] = 1.0      # Report that this tool is broken
            
        else :                            # This predicate is a mission status predicate
            mission_status[pred]= 1.0     # Report that this mission objective is fulfilled       
        
    return mission_status, tools_status, newPos
    

def sim_Reconnaissance(s0, G, horizon, problem):
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
    This function has the same goal as runSim_FH, in fact, it works with the 
    same idea. The difference between them is that this function is 
    specifically designed for 'Reconnaissance mdp' and it is able to find
    a graphic represention of the simulation and save it in a .gif file.

    '''
    
    # Build the path to the palnning domain, parse again to get domain predicates
    directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\recon_inst_mdp\p' + str(problem) + r'\Domain.ppddl'
    MyDomain = PDDLParser.parse(directory1)
    
    miss_pred_lst = []                          # List to save mission status predicates
    
    for pred in MyDomain.predicates:            # Check predicate by predicate
        words = pred.name.split('_')
                                                          # Check if this is a mission status predicate,
        if not (words[2]=='a1' or words[0]=='damaged'):   # position and damaged will be ploted separately
            miss_pred_lst.append(pred.name)
    
    # Init the size of the grid map according to the planning problem
    height = 0
    width = 0
    
    if problem < 3:
        height = 2
        width = 2
    
    elif problem < 5:
        height = 3
        width = 3
        
    elif problem < 8:
        height = 4
        width = 4  
            
    else:
        height = 5
        width = 5
    
    
    # Define some list to keep track of the evolution of the parameters
    positions = []
    rewards = []
    miss_status_lst = []
    tools_status_lst = []
    
    # Start the simulation
    s = s0                          # assign s0 to the current state
    count = 0                       # init a counter to count decission epochs
    accrualCost = 0                 # int a cost/reward adder
    while (count < horizon):
        
        print(s)
        
        # Update the status with the current state predicates
        [miss, tools, pos] = update_Recon(s, height, width, miss_pred_lst)
        
        # Append current status to the respective list
        positions.append(pos)
        miss_status_lst.append(miss)
        tools_status_lst.append(tools)
        
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
    
    if (accrualCost <= 0):         # Bad simulation...no need to save the .gif
        return
        
    # Print and save animation
    fig = plt.figure()                               # Create an empty figure with no axes 
    fig.suptitle('Recon p' + str(problem))           # Add a title to the figure
    status_ax = fig.add_axes([0.05, 0.03, 0.9, 0.4]) # Add axes for mission plot
    rew_ax = fig.add_axes([0.5, 0.55, 0.4, 0.35])    # Add axes for tools plot
    pos_ax = fig.add_axes([0, 0.55, 0.4, 0.35])      # Add axes for position plot
    
    def newFrame(i=int):
        pos_ax.matshow(positions[i], cmap=plt.cm.Spectral, vmin = -1.0 , vmax = 1.0)
    
        status_ax.cla()
        status_ax.set_title('Mission status')
        status_ax.set_ylim(0, 1)
        status_ax.yaxis.set_major_locator(ticker.FixedLocator([0, 1]))
        status_ax.bar(miss_status_lst[i].keys(), miss_status_lst[i].values(), color = 'cyan')
        status_ax.bar(['water', 'life', 'pic'], [ tools_status_lst[i]['damaged__w1'],
                                                  tools_status_lst[i]['damaged__l1'],
                                                  tools_status_lst[i]['damaged__p1'] ],
                         color = 'magenta')
        
        status_ax.xaxis.set_tick_params(which='major', size=5, width=1,
                                        direction='in', pad = -110.0, labelsize= 10.0,
                                        labelrotation = 90)
        status_ax.margins(0.01)
        
        rew_ax.cla()
        rew_ax.set_title('Reward')                
        rew_ax.set_ylim(0, 15)
        rew_ax.set_xlim(0,40)
        rew_ax.plot(rewards[:i], color = 'cyan')
        
        
    animator = ani.FuncAnimation(fig, newFrame, interval = 400)
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\ReconP3.gif')
    
    
    
    return
    
    
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
def update_sysadmin(s, predicates):
    '''
    Parameters
    ----------
   s : state object
        This is the state object whose predicates are going to be translated
        into a graphical representation.
    predicates : set of strings
        This set contains the domain predicates. It is used in order to define
        the entries of a dictionary whose values can be either 0 or 1, depending
        on whether the predicates of state 's' are False or True. 
        
    Returns
    -------
    status : dict
        A dictionary whose entries are the predicates of the planning domain 
        and the values can be 0 or 1 depending of s.predicates.

    '''
    
    # Create a new dictionary to encode the state
    status = {}
    for pred in predicates:
        status[pred]= 0.0
        
    # Check state predicates and update the status
    for pred in s.predicates:
        status[pred] = 1.0        # Only predicates of s will be set to True
        
    return status

def sim_SysAdmin(s0, G, horizon, problem):
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
    specifically designed for 'Maze mdp' and it is able to find
    a graphic represention of the simulation and save it in a .gif file.

    '''
    
    # Get the predicates of the problem -> input for update_sysadmin function
    setOfPred = s0.predicates
    
    # Create a list to store the evolution of status dictionaries
    evolution = []
    rewards = []
    
    # Start the simulation
    s = s0                          # assign s0 to the current state
    count = 0                       # init a counter to count decission epochs
    accrualCost = 0                 # int a cost/reward adder
    while (count < horizon):
        
        print(s)
        
        # Update the status with the current state predicates
        newStatus= update_sysadmin(s, setOfPred)
        
        # Append current status to the list
        evolution.append(newStatus)
    
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
    
    
    # Pre-proccess data before drawing the plot
    evolution_subnet1 = []          # list to store the evolution of subnet1
    evolution_subnet2 = []          # list to store the evolution of subnet2
    num_comp = len(setOfPred)       # number of computers
    
    for status in evolution:        # For each decision epoch factorise the status
                                    # of the network into two subnets
                                    
        # Prepare the dictionaries before checking the values of the status
                            
        subnet1 = {}          # Dictionary to take the first half of computers
        subnet2 = {}          # Dictionary to take the first half of computers
        
        for pred in setOfPred:      # Init the entries of each subnet dictionary
           words = pred.split('c')  # get the identifier of the computer
           
           if float(words[1]) <= num_comp/2 :  # if the computer belongs to the first half
               subnet1[pred] = 0.0             # init the entry in subnet1
               
           else :                              # if the computer belongs to the second half          
               subnet2[pred] = 0.0             # init the entry in subnet2
         
        # when the dictionaries are ready...
        # put the values of the status in the correct subnet
        for pred in status.keys():  
           
           if pred in subnet1:                  # this predicate (computer) belongs to subnet1
               subnet1[pred] = status[pred]
               
           else:                                # this predicate (computer) belongs to subnet2
               subnet2[pred] = status[pred]
        
        # Update the evoluton lists
        evolution_subnet1.append(subnet1)
        evolution_subnet2.append(subnet2)
        
    
    # Print and save animation
    fig, ax = plt.subplots()
    fig, ax_lst = plt.subplots(1, 3, constrained_layout=True)    # Add a 1x(2-3) grid of Axes 
    fig.suptitle('SysAdmin p' + str(problem))        # Add a title to the figure
        
    
    def newFrame(i=int):
        
        # First plot -> subnet 1
        ax_lst[0].cla()
        ax_lst[0].set_xlim(0, 1)
        ax_lst[0].xaxis.set_major_locator(ticker.FixedLocator([0, 1]))
        
        # prepare data to plot (Because barh is not able to print form dictionary items)
        a = []
        b = []
        for key, val in evolution_subnet1[i].items():
            a.append(key)
            b.append(val)

        ax_lst[0].barh(a, b, color = 'cyan')
        ax_lst[0].yaxis.set_tick_params(which='major', size=5, width=1,
                                        direction='in', pad = -80.0, labelsize= 10.0,
                                        labelrotation = 0)
        ax_lst[0].margins(0.01)
        
        # First plot -> subnet 2
        ax_lst[1].cla()
        ax_lst[1].set_xlim(0, 1)
        ax_lst[1].xaxis.set_major_locator(ticker.FixedLocator([0, 1]))
        
        # prepare data to plot (Because barh is not able to print form dictionary items)
        c = []
        d = []
        for key, val in evolution_subnet2[i].items():
            c.append(key)
            d.append(val)
        ax_lst[1].barh(c, d, color = 'cyan')
        ax_lst[1].yaxis.set_tick_params(which='major', size=5, width=1,
                                        direction='in', pad = -80.0, labelsize= 10.0,
                                        labelrotation = 0)
        ax_lst[1].margins(0.01)        
        
        
        # First plot -> accrual reward
        ax_lst[2].cla()
        ax_lst[2].set_title('Reward')                
        ax_lst[2].set_ylim(0, 300)
        ax_lst[2].set_xlim(0,40)
        ax_lst[2].plot(rewards[:i], color = 'cyan')
        
        
    animator = ani.FuncAnimation(fig, newFrame, interval = 400)
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\SysAdminP3.gif')
     
    return
    

# ----------------------------------------------------------------------------



















