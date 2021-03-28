import matplotlib.pyplot as plt      
import matplotlib.animation as ani
from matplotlib import ticker

from simulation.sim_ToolBox import checkState

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
    
    animator.save(r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\output\SysAdminP1_test.gif')
     
    return