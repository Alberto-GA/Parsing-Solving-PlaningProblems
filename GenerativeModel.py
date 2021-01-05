from ppddl_parser import PDDLParser
from predicate    import Predicate
from literal      import Literal
from action       import Action
from random       import random
from logic        import XOR


#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\rover\Domain.pddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\rover\p01.pddl'

directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\blocksworld\domain.pddl'
directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\blocksworld\p01.pddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\navigation_inst_mdp\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\navigation_inst_mdp\Problem.ppddl'

#(:metric maximize (reward))
#	;; (:horizon 40)
#	;; (:discount 1.0)

MyDomain = PDDLParser.parse(directory1)
MyProblem = PDDLParser.parse(directory2)

def getInitialState():
    s0 = State(MyProblem.init)
    return s0

def getGoalState():
    s_goal = State(MyProblem.goal)
    return s_goal


#%%

# The Domain contains the general description of all the actions.
#  action (?x - type1, ?y - type2)
#       preconditions
#       effects
#
# The Problem instantiates all the different objects for each type. So every
# general action can be expanded by giving details about the arguments.
#
# OBJECTIVE: Create a list with all the possible actions for this problem

actions = []

for g_action in MyDomain.operators:
    
    # Save the name of the function because it won't change
    act_name = g_action.name
    
    # Make two lists with the types and names of the parameters
    parameter_types = []          # [type1, type2, ...]
    parameter_names = []          # [ ?x, ?y, ...]
    for parameter in g_action.params:
        parameter_types.append( parameter.type )
        parameter_names.append( parameter.name )
    
    # Once the types are known, find in the Problem description all the related objects
    object_list = []                  # [ [obj11, obj12], [obj21], ... ] 
    for kind in parameter_types:
        row = []
        for obj in MyProblem.objects[kind]:
            row.append(obj)       
        object_list.append(row)
   
    # The goal now is to store all the possible combination taking one element
    # of each list in object_list.
    
    # Init the list to save all the combinations
    parameters_list = []
    
    # Compute the size of this list considering the number of objects of each type
    size = 1
    for row in object_list:
        size *= len(row)
    
    # The combinations will be ordered as follows:
    # [ [obj11, obj21, ..., obj(N-1)1, objN1]
    #   [obj11, obj21, ..., obj(N-1)1,objN2]
    #                  ...
    #   [obj11, obj21, ..., obj(N-1)1, objNM] <- argument N of type N has M objects
    #   [obj11, obj21, ..., obj(N-1)2, objN1] <- try next object of type (N-1)
    
    # First we have to compute the change interval of each argument
    # each argument changes every C lines where Ci = PI[_j=i+1:^N](nbObj_j)
    changes = []
    aux1 = 1
    aux2 = 1
    for row in object_list:
        aux1 *= len(row)
        aux2  = int( size/ aux1 )
        changes.append(aux2)
    
    # Let's start building each combination line by line
    for i in range(0,size):
        
         parameters_list.append([])         
         for j in range(0,len(object_list)):    # iterate between different arg
             
             if (j>0):  # the objects of this argument are iterated several times
                 k = int( i / changes[j]) - int( i / changes[j-1]) * len(object_list[j])
             else:      # the objects of the first arguments are iterated only once
                 k = int( i / changes[j])    
                 
             parameters_list[i].append(object_list[j][k])
             
             
    # Believe me, this works and now we have a list with all the combinations
    # So now we take each combination and create a new action based on these args
    for parameters in parameters_list:
        
        # Build the preconditions
        preconditions = []
        for precond in g_action.precond:      #Custom each general preconditon
            
            arguments = precond.predicate.args          # Read the arguments of the precondition
            my_arguments = parameters.copy()            # Create a copy of the current combination of parameters
            my_names = parameter_names.copy()           # Create a copy of the generic names of the args of the function
            args2pop = []                               # List with arguments to remove
            for i in range(0, len(parameter_names)):    # Check one by one if all the params names appear in precond args
                if parameter_names[i] not in arguments:
                    args2pop.append(i)                  # Remember the position of the argument to pop
           
            if args2pop:                                # If we have to pop sth
                # revert the list to pop elements from the end to the top
                # Hence, we avoid modifying the actual position of the arguments
                # that we want to pop
                args2pop.reverse()     
                for index in args2pop:
                    my_arguments.pop(index)
                    my_names.pop(index)
            
            # Check the order- Sometimes args of some precondition and effect predicates has different order from action args
            my_arguments_check = []
            aux = []
            for pname in my_names:                              # for each name following the order of action args,
                aux.append(precond.predicate.args.index(pname)) # append its position in precondition predicate
            
            for i in range(0,len(my_names)):   #let's build the checked-arguments from elemt 0
                index = aux.index(i)           #what is the position in "my_arguments" of the parameter that goes in the "ith" position
                my_arguments_check.append(my_arguments[index]) # append the correct parameter that goes in the ith position
                
            
            #!----------------------------------------------------------------
            # Important remark. This code does support name identification.
            # I mean. If the args of the action are (?x, ?y, ?z) the preconditions
            # and effects can change the order. (?x,?z) (?y,?z) 
            #!----------------------------------------------------------------
            
            my_pred = Predicate( precond.predicate.name, my_arguments_check)
            my_precond = Literal (my_pred, precond._positive)
            preconditions.append(my_precond)
        
        # Build the effects
        effects = []
        for effect in  g_action.effects:          # Custom each general effect
            
            arguments = effect[1].predicate.args
            my_arguments = parameters.copy()
            my_names = parameter_names.copy()
            args2pop = []
            for i in range(0, len(parameter_names)):
                if parameter_names[i] not in arguments:
                    args2pop.append(i)
                    
            if args2pop:
                args2pop.reverse()
                for index in args2pop:
                    my_arguments.pop(index)
                    my_names.pop(index)
            
            my_arguments_check = []
            aux = []
            for pname in my_names:
                aux.append(effect[1].predicate.args.index(pname))
            for i in range(0,len(my_names)):
                index = aux.index(i)
                my_arguments_check.append(my_arguments[index])
                    
            my_prob = effect[0]
            my_pred = Predicate (effect[1].predicate.name, my_arguments_check)
            my_effect = Literal(my_pred, effect[1]._positive)
            effects.append( (my_prob, my_effect) )
        
        # Create a new action and append it to the list
        my_action = Action(act_name, parameters, preconditions, effects)
        actions.append(my_action)


#%%        
#------------------------------------------------------------------------------
def set_applicable_actions (actions, predicates):
    
    app_actions = actions.copy()            # Make a copy of the list of actions
    act2pop = []                            # Make a list with de index of the actions to pop
    for i in range(0,len(app_actions)):     # Analyse all the actions
        
        act = app_actions[i]    
        for precond in act.precond:         # Check if all the preconditions are met
                                            
            precond_str = str(precond)
            
            if ( precond.predicate.name == '=' ):    # First check if this is an equality precondition
                proposition1 = precond.predicate.args[0] == precond.predicate.args[1]
                proposition2 = precond._positive
                if XOR(proposition1, proposition2):  # Use exclusive OR gate to discard the action
                    act2pop.append(i)
                    break  
                
            elif precond_str not in predicates:   # Otherwise check if the precondition is within the predicates of the state
                act2pop.append(i)
                break                       # If at least one precondition is not met, the action is discarded
    
    act2pop.reverse()                       # Reverse the list and
    for i in act2pop:
        app_actions.pop(i)                  # pop from the end of the list
        
    return app_actions                      # Save the result in the state attribute
        

#-----------------------------------------------------------------------------
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

def CheckGoal(s1):
    s_g = getGoalState()
    rv = True
    
    for pred in s_g.predicates:
        if pred not in s1.predicates:
           rv = False
           break
       
    return rv


#-----------------------------------------------------------------------------  
def Cost(state):
    
    nominalCost = -0.5
    DeadEndCost = -5
    goalCost = 0
    
    #if StateEquality(state, getGoalState()): return goalCost
    if CheckGoal(state)   : return goalCost
    elif not state.actions: return DeadEndCost
    else:                   return nominalCost   
    
#-----------------------------------------------------------------------------
class State:
    
    def __init__(self, predicates):
        self.predicates = predicates
        self.actions = set_applicable_actions (actions, predicates)
        
    def SampleAction(self):
        """
        This method returns a random action according to the actions 
        atribute. this method is used in the rollouts of the UCT algorithm
        """
        r = random()                # sample a random number in the range [0,1]
        nA = len(self.actions)      # compute the number of actions
        deltaP = 1/nA               # compute an increment of probability
        accrual = 0                 # accrual probability
        for action in self.actions:
            
            accrual += deltaP       # addup the increment to accrual
            if r <= accrual: return action
            else: continue
        
        
    def SampleChild(self, action):
     
        new_pred = list(self.predicates)  # Copy from set to list
        for eff in action.effects:
            r = random()                # Create a random number in the range [0,1]
            if (r <= eff[0]):           # Consider the effect according to probability
                if eff[1]._positive:    # This predicate becomes (remains) true
                    new_pred.append(str(eff[1]))    
                else:                   # This predicate becomes false
                                        # But... What if it is already false and it is not in the list? try-catch or check before remove
                    aux = str(eff[1])[4:]
                    if aux in new_pred:  new_pred.remove(aux)
                        
        
                         
        child = State(set(new_pred))
        cost = Cost(child)
        return [child, cost]
       
            
    def __str__(self):
        return str(self.predicates)
#-----------------------------------------------------------------------------



