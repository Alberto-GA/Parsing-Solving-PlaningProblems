import math
from ppddl_parser import PDDLParser
from predicate    import Predicate
from literal      import Literal
from action       import Action
from random       import random
from logic        import XOR




#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\blocksworld\domain.pddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\blocksworld\p01.pddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\rover\Domain.pddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\rover\p01.pddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\crossing_traffic_inst_mdp\p3\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\crossing_traffic_inst_mdp\p3\p03.ppddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\elevators_inst_mdp\p1\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\elevators_inst_mdp\p1\p01.ppddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\game_of_life_inst_mdp\p1\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\game_of_life_inst_mdp\p1\p01.ppddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\recon_inst_mdp\p3\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\recon_inst_mdp\p3\p03.ppddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\Maze\p1\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\Maze\p1\p01.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\Maze\p1\p01B.ppddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\sysadmin_inst_mdp\p3\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\sysadmin_inst_mdp\p3\p03.ppddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\navigation_inst_mdp\p1\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\navigation_inst_mdp\p1\p01.ppddl'

#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\traffic_inst_mdp\p1\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\traffic_inst_mdp\p1\p01.ppddl'

directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\skill_teaching_inst_mdp\p5\Domain.ppddl'
directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\skill_teaching_inst_mdp\p5\p05.ppddl'

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
            
            # each effect is a tuple -> (list of mutex outcomes, list of conditions)
            # part 1: edit the list of mutually exlcusive outcomes -> effect[0]
            mutex_effects_list = []
            for mutex_effect in effect[0]:
                
                if type(mutex_effect[1]) is tuple:  # We won't custom fluent effects
                     mutex_effects_list.append( mutex_effect )
                else:
                    arguments = mutex_effect[1].predicate.args
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
                        aux.append(mutex_effect[1].predicate.args.index(pname))
                    for i in range(0,len(my_names)):
                        index = aux.index(i)
                        my_arguments_check.append(my_arguments[index])
                            
                    my_prob = mutex_effect[0]
                    my_pred = Predicate (mutex_effect[1].predicate.name, my_arguments_check)
                    my_effect = Literal(my_pred, mutex_effect[1]._positive)
                    mutex_effects_list.append( (my_prob, my_effect) )
            
            # part 2: edit the list of conditions for conditional effects -> effect[1]
            conditions = []
            for cond_eff in effect[1]:
                arguments = cond_eff.predicate.args         # Read the arguments of the condition
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
                    aux.append(cond_eff.predicate.args.index(pname)) # append its position in precondition predicate
                
                for i in range(0,len(my_names)):   #let's build the checked-arguments from elemt 0
                    index = aux.index(i)           #what is the position in "my_arguments" of the parameter that goes in the "ith" position
                    my_arguments_check.append(my_arguments[index]) # append the correct parameter that goes in the ith position
                    
                
                my_pred = Predicate( cond_eff.predicate.name, my_arguments_check)
                my_cond = Literal (my_pred, cond_eff._positive)
                conditions.append(my_cond)
            
            # The 2 parts of the effect have been customised, apend the effect
            # to the list and continue with the following effect
            effects.append( (mutex_effects_list, conditions) )
        
        # Create a new action and append it to the list
        my_action = Action(act_name, parameters, preconditions, effects)
        actions.append(my_action)


#%%        
#------------------------------------------------------------------------------
def set_applicable_actions (actions, predicates):
    
    # PART I
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
                
            elif not precond._positive:   # Check that this precondition is NOT within the predicates of the state
                precond_str = precond_str[4:]   # remove the not at the beginning
                if precond_str in predicates:
                    act2pop.append(i)
                    break 
            
            elif precond_str not in predicates:   # Otherwise check if the precondition is within the predicates of the state
                act2pop.append(i)
                break                       # If at least one precondition is not met, the action is discarded
        
    act2pop.reverse()                       # Reverse the list and
    for i in act2pop:
        app_actions.pop(i)                  # pop from the end of the list
    
    '''
    # PART II
    # Now we have a list of actions whose preconditions are meet with the 
    # current predicates. The objective now is to, for each applicable action,
    # remove the effects whose conditions are not satisfied with the current
    # predicates.
    final_actions = []
    ...
    
    '''
             
    return app_actions     #final_actions           # Save the result in the state attribute
  

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
def get_clean_effect_list(s_predicates, appli_action):
    '''    
    Parameters
    ----------
    s_predicates : set
        This is the set of predicates that will be used to assess the relevance
        of appli_action's effects.
    appli_action : action object
        This is an applicable action in the state defined by the predicates
        in s_predicates.

    Returns
    -------
    active_effects : list
        This is the list of effects that is the most condensed representation
        of the action effects. This list contains only relevant effects that
        may lead to a new state. In other words, effects whose conditions hold
        in s and which may add or remove a predicate from s_predicates.

    '''
    
    # PART 1: separate the applicable effects from the effects that do not 
    # satisfy the conditions.
    
    appli_effects = []            # List of applicable effects that satisfy 
                                  # the conditions.

    for eff,cond_lst in appli_action.effects:       # Analyse effect by effect
            
            cond_ok = True              # Boolean flag to accept this effect
            for cond in cond_lst:       # Check all the coditions of the effect
                               
                cond_str = str(cond)
            
                if ( cond.predicate.name == '=' ):    # First check if this is an equality condition
                    proposition1 = cond.predicate.args[0] == cond.predicate.args[1]
                    proposition2 = cond._positive
                    if XOR(proposition1, proposition2):  # Use exclusive OR gate to discard the action
                        cond_ok = False
                        break
                    
                elif not cond._positive:                # else check if the negative condition is NOT within the predicates of the state
                    cond_str = cond_str[4:]             
                    if cond_str in s_predicates:
                        cond_ok = False
                        break
                                       
                elif cond_str not in s_predicates:   # else check if the condition is within the predicates of the state
                    cond_ok = False
                    break
    
            if cond_ok:                   # If all the conditions are OK we can append this effect to our list
                            
                appli_effects.append( (eff, cond_lst) )
    
        
    # PART 2: Once we have the applicable effects, let's look for the effects 
    # that actually may have an impact on s.predicates. We want to remove 3
    # kind of effects:
    #   1) Decrease/increase reward effect -> they are not taken into account
    #      to generate succesor's predicates.
    #   2) Effects that attempt to remove a predicate that is not within
    #      s.predicates.
    #   3) Effects that attempt to add a predicate that is is already in
    #      s.predicates.
    
    active_effects = []           # Final effect list to return
    for mutex_lst,cond_lst in appli_effects:
    
        my_mutex_lst = []         # Mutually exclusive effects that satisfy
                                  # the aforementioned requirement will be 
                                  # added to this list.
        for prob,eff in mutex_lst:
            
            if not (type(eff) == tuple):         # Requirement 1)
                
                if eff._positive:                # Requirement 3)
                    pred = str(eff)
                    if not(pred in s_predicates):
                        my_mutex_lst.append((prob,eff))
                else: 
                    pred = str(eff)[4:]          # requirement 2)
                    if pred in s_predicates:
                        my_mutex_lst.append((prob,eff))
        
        if my_mutex_lst:
            active_effects.append((my_mutex_lst,cond_lst))
       
    return active_effects
# ----------------------------------------------------------------------------

def compute_probabilities(active_eff_lst, index, sequence_p, sequence_p_list):
    '''
    Parameters
    ----------
    active_eff_lst : list
        This is the list of effects that is the most condensed representation
        of the action effects. This list contains only relevant effects that
        may lead to a new state. In other words, effects whose conditions hold
        in s and which may add or remove a predicate from s_predicates. It can
        be computed with the function get_clean_effect_list()
    index : int
        This argument is the current depth of the probability tree, that is to
        say, the index of the current effect from the list.
    sequence_p : float
        This is the current probability of the sequence of events, the proba-
        bility of the branch at depth index.
    sequence_p_list : list
        This list will be used to store the probabilities of all possible
        branches.

    Returns
    -------
    None. The function works with the list object provided in the arguments

    '''
    
    mutex_effects_lst = active_eff_lst[index][0]   # Get the current effect
    
    accrual_p = 0.0            # Set a counter because the accrual probability
                               # must be excatly 1.0 within a mutually exclu-
                               # sive effect list.
                               
    for prob,eff in mutex_effects_lst:   # Create a branch for each mutex eff.
    
        accrual_p += prob              # add up the probability of this branch
        current_p = sequence_p * prob  # Update the current probability of the branch
        
        if index < len(active_eff_lst)-1:  # Go deeper if there are more effects to branch          
            compute_probabilities(active_eff_lst, index+1, current_p, sequence_p_list)
        
        else:                              # Branch has reach the leaf, add the probability
            sequence_p_list.append(current_p)
     
    if accrual_p < 1.0:                  # There is one more mutex effect to consider
                                         # if this effect does not occur I still have to expand the tree
                                         # and s' == s is also a succesor of s!
         prob = 1.0 - accrual_p
         current_p = sequence_p * prob
         if index < len(active_eff_lst)-1:
             compute_probabilities(active_eff_lst, index+1, current_p, sequence_p_list)
         else:
             sequence_p_list.append(current_p)  
            


# ----------------------------------------------------------------------------
class State:
    
    def __init__(self, predicates):
        self.predicates = predicates
        self.actions = set_applicable_actions (actions, predicates)
        self.entropy = {}
        self.max_entropy = 0.0
        self.mean_entropy = 0.0
        
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
        """
        The successor state will feature a new list of predicates.
        The objective of this method is to build this new list by applying 
        the effects of the action.
        """
        new_pred = list(self.predicates)  # Copy from set to list
        cost = 0
        
        for eff_list, cond_list in action.effects:
            
            # Check if the conditions of this effect are met.
            cond_ok = True        
            for cond in cond_list:
                
                cond_str = str(cond)
            
                if ( cond.predicate.name == '=' ):    # First check if this is an equality condition
                    proposition1 = cond.predicate.args[0] == cond.predicate.args[1]
                    proposition2 = cond._positive
                    if XOR(proposition1, proposition2):  # Use exclusive OR gate to discard the action
                        cond_ok = False
                        break
                    
                elif not cond._positive:                # Otherwise check if the negative condition is NOT within the predicates of the state
                    cond_str = cond_str[4:]             
                    if cond_str in self.predicates:
                        cond_ok = False
                        break
                                       
                elif cond_str not in self.predicates:   # Otherwise check if the condition is within the predicates of the state
                    cond_ok = False
                    break
            
            # If the conditions are met, evaluate the effect, otherwise continue with the following effect.
            if cond_ok:
                
                r = random()               # Generate a random number in [0,1]
                accrualProb = 0
                
                for prob, mutex_effect in eff_list:
                    
                    accrualProb += prob
                    if r <= accrualProb:
                        
                        # Apply solely this effect.
                        if type(mutex_effect) is tuple:              # For problems that uses rewards -> this effect is not a literal
                            if mutex_effect[0] == "decrease":
                                cost += (-1.0)*mutex_effect[2]
                            else:
                                cost += mutex_effect[2]
                                
                        elif mutex_effect._positive:                 # This predicate becomes (remains) true
                            new_pred.append(str(mutex_effect)) 
                            
                        else:                                        # This predicate becomes false (remove it if it was true)
                            aux = str(mutex_effect)[4:]
                            if aux in new_pred:  new_pred.remove(aux)
                        break
                    else: 
                        continue
            
            else: 
                continue
                        
        
        # all the effects have been assessed and we have the new predicates
        # then, instanciate a new state object encoding these predicates
        child = State(set(new_pred))
        
        # If the action didn't have any effect on the reward, compute a default
        # cost with the Cost function. These kind of function are common in
        # problems that don't have a :metric maximise/minimise reward. In other 
        # words, the goal-oriented problems.
        if MyProblem.metric == "Goal-oriented": cost = Cost(child)
        
        return [child, cost]
       
    
    def set_entropy(self):
        '''
        This method compute the entropy attributes of the state. These attri-
        butes are not initialised by default. The user must decide when is
        the best moment to use this method.
        '''
        entropy_lst = []
        
        for act in self.actions:    # Compute the entropy for each pair s,a
            
            # First, proccess the effects of the action to make sure that each
            # brach of the probability tree truly leads to a different succesor
            active_eff = get_clean_effect_list(self.predicates, act)
            
            # Check if the list of active effects is not empty. If the problem
            # has goal states where any action is applicable but no effect is
            # triggered, we can face this situation. Simply init the entropy to
            # zero because the action is deterministic.
            if not active_eff: 
                self.entropy[act] = 0.0
                entropy_lst.append(0.0)
                continue                     # go for the next action
            
            # Define and compute a list with the probabilities of each branch
            branch_prob = []
            compute_probabilities(active_eff, 0, 1.0, branch_prob )
            
            # Now compute the entropy for the pair state-action defined as:       
            #     e(s,a) = -SUM (s') {P(s'|s,a)*log2(P(s'|s,a))
            
            e = 0.0
            
            for p in branch_prob:
                e -= p * math.log2(p)     
            
            entropy_lst.append(e)            
            self.entropy[act] = e
        
        
        # Now compute the max entropy of the sate defined as
        #    e(s) = max (a in A) {-SUM (s') {P(s'|s,a)*log2(P(s'|s,a))}

        self.max_entropy = max(entropy_lst)
        
        # Now compute the mean entropy of the sate defined as:
        # e(s) = 1/|A| SUM(a in A) {-SUM (s') {P(s'|s,a)*log2(P(s'|s,a))}
        
        self.mean_entropy = (1.0/len(self.actions))*sum(entropy_lst)
        
        
        
    def __str__(self):
        return str(self.predicates)
#-----------------------------------------------------------------------------



