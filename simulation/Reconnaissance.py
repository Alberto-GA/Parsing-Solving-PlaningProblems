import numpy as np
import matplotlib.pyplot as plt      
import matplotlib.animation as ani
from matplotlib import ticker

from ppddl_parser import PDDLParser
from simulation.sim_ToolBox import checkState
from simulation.sim_ToolBox import checkState_FH

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
        successor = checkState_FH(successor,G)        # check if the successor is already in the graph
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
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\ReconP1_FH_MX.gif')
    
    
    
    return