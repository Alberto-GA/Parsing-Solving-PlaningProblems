#-----------------------------------------------------------------------------
"""
This code tries to provide a function to read PPDDL domains and problems and 
compile them to generate a generative transition model.

"""

# Let's start by importing "lex" and "yacc" compiler construction tools from 
# "PLY". 
# - lex.py module is used to break input text into a collection of tokens 
#   specified by a collection of regular expression rules.
# - yacc.py is used to recognize language syntax that has been specified in 
#   the form of a context free grammar.
from ply import lex
from ply import yacc

# Now import some classes that will help to parse the files.
from domain    import Domain
from problem   import Problem
from term      import Term
from literal   import Literal
from predicate import Predicate
from action    import Action



#-----------------------------------------------------------------------------
# 1 ----------------------------DEFINE LEXER----------------------------------

# 1.1) List of token names. This is always required.
# Defines all of the possible token names that can be produced by the lexer
# and it is used by yacc to identify terminals
tokens = (
    'NAME',
    'VARIABLE',
    'NUMBER',
    'LPAREN',
    'RPAREN',
    'HYPHEN',
    'EQUALS',
    'DEFINE_KEY',
    'DOMAIN_KEY',
    'REQUIREMENTS_KEY',
    'STRIPS_KEY',
    'EQUALITY_KEY',
    'TYPING_KEY',
    'PROBABILISTIC_EFFECTS_KEY',
    'ADL_KEY',                                                               #
    'REWARDS_REQ_KEY',                                                       #
    'REWARD_KEY',                                                            #
    'METRIC_KEY',                                                            #
    'MAXIMIZE_KEY',                                                          #
    'TYPES_KEY',
    'PREDICATES_KEY',
    'ACTION_KEY',
    'PARAMETERS_KEY',
    'PRECONDITION_KEY',
    'EFFECT_KEY',
    'AND_KEY',
    'NOT_KEY',
    'PROBABILISTIC_KEY',
    'WHEN_KEY',                                                              #
    'INCREASE_KEY',                                                          #
    'DECREASE_KEY',                                                          # 
    'PROBLEM_KEY',
    'OBJECTS_KEY',
    'INIT_KEY',
    'GOAL_KEY'
)

# 1.2) Specification of tokens. "t_" followed by a token 
#      -> Regular expression rules for simple tokens
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_HYPHEN = r'\-'
t_EQUALS = r'='


#     -> tell lex that spaces and tabs are not important
t_ignore = ' \t'


#     -> To handle reserved words, we have to write a single rule to match an 
#        identifier and do a special name lookup in a function.
reserved = {
    'define'                    : 'DEFINE_KEY',
    'domain'                    : 'DOMAIN_KEY',
    ':requirements'             : 'REQUIREMENTS_KEY',
    ':strips'                   : 'STRIPS_KEY',
    ':equality'                 : 'EQUALITY_KEY',
    ':typing'                   : 'TYPING_KEY',
    ':probabilistic-effects'    : 'PROBABILISTIC_EFFECTS_KEY',
    ':adl'                      : 'ADL_KEY',                                 #
    ':rewards'                  : 'REWARDS_REQ_KEY',                         #
    'reward'                    : 'REWARD_KEY',                              #
    ':metric'                   : 'METRIC_KEY',                              #
    'maximize'                  : 'MAXIMIZE_KEY',                            #
    ':types'                    : 'TYPES_KEY',
    ':predicates'               : 'PREDICATES_KEY',
    ':action'                   : 'ACTION_KEY',
    ':parameters'               : 'PARAMETERS_KEY',
    ':precondition'             : 'PRECONDITION_KEY',
    ':effect'                   : 'EFFECT_KEY',
    'and'                       : 'AND_KEY',
    'not'                       : 'NOT_KEY',
    'probabilistic'             : 'PROBABILISTIC_KEY',
    'when'                      : 'WHEN_KEY',                                #
    'increase'                  : 'INCREASE_KEY',                            #
    'decrease'                  : 'DECREASE_KEY',                            #  
    'problem'                   : 'PROBLEM_KEY',
    ':domain'                   : 'DOMAIN_KEY',
    ':objects'                  : 'OBJECTS_KEY',
    ':init'                     : 'INIT_KEY',
    ':goal'                     : 'GOAL_KEY'
}

def t_KEYWORD(t):
    r':?[a-zA-z_][a-zA-Z_0-9\-]*'
    t.type = reserved.get(t.value, 'NAME')
    return t


#   -> If some kind of action needs to be performed, some token specifications 
#      can rely on functions:
def t_NAME(t):
    r'[a-zA-z_][a-zA-Z_0-9\-]*'
    return t

def t_VARIABLE(t):
    r'\?[a-zA-z_][a-zA-Z_0-9\-]*'
    return t

def t_NUMBER(t):
    r'[0-9]\.\d+'
    t.value = float(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lineno += len(t.value)


#   -> error handling:
def t_error(t):
    print("Error: illegal character '{0}'".format(t.value[0]))
    t.lexer.skip(1)


#1.3) build the lexer
lex.lex()


#-----------------------------------------------------------------------------
# 2 ----------------------------DEFINE PARSER---------------------------------

# - To build the parser we have to define all the grammar rules.
# - Each grammar rule is defined by a Python function. The docstring to 
# that function contains the appropriate context-free grammar specification.
# this grammar may contain terminals (basically the tokens of the lexer) and 
# non-terminals (that consist of terminals and rules).
# - The body of the functions implements the semantic actions of the rule.
# - Each function accepts a single argument p that is a sequence containing  
# the values of each grammar symbol in the corresponding rule.

# Grammar rule 1:
def p_pddl(p):
    '''pddl : domain
            | problem'''
    p[0] = p[1]

#-----------------------------------------------------------------------------
def p_problem(p):
    '''problem : LPAREN DEFINE_KEY problem_def domain_def objects_def init_def goal_def RPAREN
               | LPAREN DEFINE_KEY problem_def domain_def init_def metric_def RPAREN
               | LPAREN DEFINE_KEY problem_def domain_def init_def goal_def RPAREN'''
               
    if ( (len(p) == 8) and (type(p[6]) is list) ): 
         p[0] = Problem(p[3], p[4], [], p[5], p[6],'Goal-oriented')
         
    elif len(p) == 8:
        p[0] = Problem(p[3], p[4], [], p[5], [], p[6])
        
    elif len(p) == 9:  
        p[0] = Problem(p[3], p[4], p[5], p[6], p[7], 'Goal-oriented')
        

        
#(define (problem test-problem)
def p_problem_def(p):
    '''problem_def : LPAREN PROBLEM_KEY NAME RPAREN'''
    p[0] = p[3]


def p_objects_def(p):
    '''objects_def : LPAREN OBJECTS_KEY typed_constants_lst RPAREN'''
    p[0] = p[3]


def p_init_def(p):
    '''init_def : LPAREN INIT_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPAREN
                | LPAREN INIT_KEY ground_predicates_lst RPAREN
                | LPAREN INIT_KEY  RPAREN'''
        
    if len(p) == 5:
        p[0] = p[3]
        
    elif len(p) == 8:
        p[0] = p[5]
        
    elif len(p) == 4:
        p[0] = []


def p_goal_def(p):
    '''goal_def : LPAREN GOAL_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPAREN'''
    p[0] = p[5]
    

def p_metric_def(p):
    '''metric_def : LPAREN METRIC_KEY MAXIMIZE_KEY LPAREN REWARD_KEY RPAREN RPAREN '''   
    p[0] = p[3]
#--------------------------------------------------


# Grammar rule 6:
# (define (domain ... )        -> see rule 7
#         (:requirements ... ) -> see rule x
#         (:types ... )        -> see rule x
#         (:predicates ... )   -> see rule x
#         (:action ... )       -> see rule x
# ) 
def p_domain(p):
    '''domain : LPAREN DEFINE_KEY domain_def require_def types_def predicates_def action_def_lst RPAREN
              | LPAREN DEFINE_KEY domain_def require_def predicates_def action_def_lst RPAREN'''
    
    if len(p)==8:           # For Planning Domains without typing requirment
        p[0] = Domain(p[3], p[4], [], p[5], p[6])
    elif len(p)==9:         # For Planning Domains with typing requirement
        p[0] = Domain(p[3], p[4], p[5], p[6], p[7])


# Grammar rule 7:
# (domain DomainName)
def p_domain_def(p):
    '''domain_def : LPAREN DOMAIN_KEY NAME RPAREN'''
    p[0] = p[3]


# Grammar rule 8:
# (:requirements ListOfRequirents)
def p_require_def(p):
    '''require_def : LPAREN REQUIREMENTS_KEY require_key_lst RPAREN'''
    p[0] = p[3]


# Grammar rule 9:
# ListOfRequiremts
def p_require_key_lst(p):
    '''require_key_lst : require_key require_key_lst
                       | require_key'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


# Grammar rule 10:
# SingleRequirement ¡ only strips,equality,typing and prob.effects are supported
def p_require_key(p):
    '''require_key : STRIPS_KEY
                   | EQUALITY_KEY
                   | TYPING_KEY
                   | PROBABILISTIC_EFFECTS_KEY
                   | ADL_KEY
                   | REWARDS_REQ_KEY'''
    p[0] = str(p[1])


# Grammar rule 11:
# (:types type1, type2, ...)
def p_types_def(p):
    '''types_def : LPAREN TYPES_KEY names_lst RPAREN'''
    p[0] = p[3]


# Grammar rule 12:
# (:predicates
#     predicate1
#     predicate2
#     ... 
# )
def p_predicates_def(p):
    '''predicates_def : LPAREN PREDICATES_KEY predicate_def_lst RPAREN'''
    p[0] = p[3]


# Grammar rule 13:
# ListOfPredicates
def p_predicate_def_lst(p):
    '''predicate_def_lst : predicate_def predicate_def_lst
                         | predicate_def'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


# Grammar rule 14:
# SinglePredicate 
# ( PredicateName ?variable1 - type1 ?variable2 - type2)
def p_predicate_def(p):
    '''predicate_def : LPAREN NAME typed_variables_lst RPAREN
                     | LPAREN NAME RPAREN'''
    if len(p) == 4:
        p[0] = Predicate(p[2])
    elif len(p) == 5:
        p[0] = Predicate(p[2], p[3])


# Grammar rule 15:
# List of actions
def p_action_def_lst(p):
    '''action_def_lst : action_def action_def_lst
                      | action_def'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]
        
        
# Grammar rule 16:
# Single action
# (:action action1
#       :parameters (?variable1 - type1 ?variable2 - type2 ...) 
#       :precondition (and ListOfPredicates
#	                  )
#       :effect (and ListOfPredicates
#		        )
# )    
def p_action_def(p):
    '''action_def : LPAREN ACTION_KEY NAME parameters_def action_def_body RPAREN
                  | LPAREN ACTION_KEY NAME action_def_body RPAREN '''
                  
    if len(p) == 6:    # For Domains without parameters nor preconditions that rely on conditional effects        
        p[0] = Action(p[3], [],  p[4][0],  p[4][1])
    elif len(p) == 7:  # For Domains following the grammar above
        p[0] = Action(p[3], p[4], p[5][0], p[5][1])


# Grammar rule 17:
# List of parameters
def p_parameters_def(p):
    '''parameters_def : PARAMETERS_KEY LPAREN typed_variables_lst RPAREN
                      | PARAMETERS_KEY LPAREN RPAREN'''
    if len(p) == 4:
        p[0] = []
    elif len(p) == 5:
        p[0] = p[3]


# Grammar rule 18:
# action body
#       :precondition (and ListOfPredicates
#	                  )
#       :effect (and ListOfPredicates
#		        )
def p_action_def_body(p):
    '''action_def_body : precond_def effects_def
                       | effects_def'''
    if len(p) == 2:
        p[0] = ([], p[1])
    elif len(p) == 3:
        p[0] = (p[1], p[2])

# Grammar rule 19:
# preconditions 
def p_precond_def(p):
    '''precond_def : PRECONDITION_KEY LPAREN AND_KEY literals_lst RPAREN
                   | PRECONDITION_KEY literal'''
    if len(p) == 3:
        p[0] = [p[2]]
    elif len(p) == 6:
        p[0] = p[4]

# Grammar rule 19:
# effects 
def p_effects_def(p):
    '''effects_def : EFFECT_KEY LPAREN AND_KEY effects_lst RPAREN
                   | EFFECT_KEY effect'''
    if len(p) == 3:
        p[0] = [p[2]]
    elif len(p) == 6:
        p[0] = p[4]

# Grammar rule 20:
# list of effects 
def p_effects_lst(p):
    '''effects_lst : effect effects_lst
                   | effect'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]

# Grammar rule 21:
# single effect -> ( list of probability-effects pairs, list of conditions)
# Preconditions and conditional effects are not the same thing. 
# if all the preconditions of the action are true, then the action can be triggered
# Then, we have to assess each effect. Independent probabilistic effects are 
# not mutually exclusive. Within each effect, we have to check if they are conditional 
# effects and whether their conditions are true or not.
def p_effect(p):
    '''effect : literal
              | LPAREN PROBABILISTIC_KEY prob_effect_list RPAREN
              | LPAREN WHEN_KEY literal fluent_def RPAREN
              | LPAREN WHEN_KEY LPAREN AND_KEY literals_lst RPAREN fluent_def RPAREN
              | LPAREN WHEN_KEY literal LPAREN PROBABILISTIC_KEY prob_effect_list RPAREN RPAREN
              | LPAREN WHEN_KEY LPAREN AND_KEY literals_lst RPAREN LPAREN PROBABILISTIC_KEY prob_effect_list RPAREN RPAREN'''
              
    if len(p) == 2:
        p[0] = ( [(1.0, p[1])] , [])
    
    elif len(p) == 5:
        p[0] = (p[3], [])
        
    elif len(p) == 6:                               # Increase/Decrease reward           
        p[0] = ( [(1.0, p[4])] , [p[3]])
    
    elif ( len(p) == 9 and  p[4]== r'and' ):          # Increas/Decrease reaward several conditions
        p[0] = ( [(1.0, p[7])] , p[5])
    
    elif len(p) == 9:                               # conditional effects -> WHEN
        p[0] = (p[6], [p[3]])
    
    elif len(p) == 12 :                             # conditional effects -> WHEN            
        p[0] = (p[9], p[5])
    
           
def p_prob_effect_list(p):
    '''prob_effect_list : prob_effect prob_effect_list
                        | prob_effect'''
                       
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]

def p_prob_effect(p):
    '''prob_effect : NUMBER literal'''
    p[0] = (p[1], p[2])                                                        # before i had [(p[1], p[2])]


def p_fluent_def(p):
    '''fluent_def : LPAREN DECREASE_KEY LPAREN REWARD_KEY RPAREN NUMBER RPAREN
                  | LPAREN INCREASE_KEY LPAREN REWARD_KEY RPAREN NUMBER RPAREN'''
                 
    p[0] = (p[2], p[4], p[6])
                  
                  
# ----------------------------------------------------------------------------

def p_literals_lst(p):
    '''literals_lst : literal literals_lst
                    | literal'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_literal(p):
    '''literal : LPAREN NOT_KEY predicate RPAREN
               | predicate'''
    if len(p) == 2:
        p[0] = Literal.positive(p[1])
    elif len(p) == 5:
        p[0] = Literal.negative(p[3])


def p_ground_predicates_lst(p):
    '''ground_predicates_lst : ground_predicate ground_predicates_lst
                             | ground_predicate'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_predicate(p):
    '''predicate : LPAREN NAME variables_lst RPAREN
                 | LPAREN EQUALS VARIABLE VARIABLE RPAREN
                 | LPAREN NAME RPAREN'''
    if len(p) == 4:
        p[0] = Predicate(p[2])
    elif len(p) == 5:
        p[0] = Predicate(p[2], p[3])
    elif len(p) == 6:
        p[0] = Predicate('=', [p[3], p[4]])


def p_ground_predicate(p):
    '''ground_predicate : LPAREN NAME constants_lst RPAREN
                        | LPAREN NAME RPAREN'''
    if len(p) == 4:
        p[0] = Predicate(p[2])
    elif len(p) == 5:
        p[0] = Predicate(p[2], p[3])


def p_typed_constants_lst(p):
    '''typed_constants_lst : constants_lst HYPHEN type typed_constants_lst
                           | constants_lst HYPHEN type'''
    if len(p) == 4:
        p[0] = [ Term.constant(value, p[3]) for value in p[1] ]
    elif len(p) == 5:
        p[0] = [ Term.constant(value, p[3]) for value in p[1] ] + p[4]


def p_typed_variables_lst(p):
    '''typed_variables_lst : variables_lst HYPHEN type typed_variables_lst
                           | variables_lst HYPHEN type'''
    if len(p) == 4:
        p[0] = [ Term.variable(name, p[3]) for name in p[1] ]
    elif len(p) == 5:
        p[0] = [ Term.variable(name, p[3]) for name in p[1] ] + p[4]


def p_constants_lst(p):
    '''constants_lst : constant constants_lst
                     | constant'''
    if len(p) == 2:
        p[0] = [ Term.constant(p[1]) ]
    elif len(p) == 3:
        p[0] = [ Term.constant(p[1]) ] + p[2]


def p_variables_lst(p):
    '''variables_lst : VARIABLE variables_lst
                     | VARIABLE'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_names_lst(p):
    '''names_lst : NAME names_lst
                 | NAME'''
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_type(p):
    '''type : NAME'''
    p[0] = p[1]


def p_constant(p):
    '''constant : NAME'''
    p[0] = p[1]


def p_error(p):
    print("Error: syntax error when parsing '{}'".format(p))


# Build the parser
yacc.yacc()

class PDDLParser(object):

    @classmethod
    def parse(cls, filename):
        data = cls.__read_input(filename)
        return yacc.parse(data)

    @classmethod
    def __read_input(cls, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = ''
            for line in file:
                line = line.rstrip().lower()
                line = cls.__strip_comments(line)
                data += '\n' + line
        return data

    @classmethod
    def __strip_comments(cls, line):
        pos = line.find(';')
        if pos != -1:
            line = line[:pos]
        return line



# Test it out --> basurilla
#read text
#directory = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\rover\Domain.pddl'
#infile = open(directory)
#data = infile.read()                    
#infile.close()
#domain = PDDLParser.parse(directory)
# Give the lexer some input
#lexer.input(data)
#parser.parse(data)