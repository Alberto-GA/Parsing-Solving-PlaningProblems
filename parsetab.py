
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'ACTION_KEY AND_KEY DEFINE_KEY DOMAIN_KEY EFFECT_KEY EQUALITY_KEY EQUALS GOAL_KEY HYPHEN INIT_KEY LPAREN NAME NOT_KEY OBJECTS_KEY PARAMETERS_KEY PRECONDITION_KEY PREDICATES_KEY PROBABILISTIC_EFFECTS_KEY PROBABILISTIC_KEY PROBABILITY PROBLEM_KEY REQUIREMENTS_KEY RPAREN STRIPS_KEY TYPES_KEY TYPING_KEY VARIABLEpddl : domain\n            | problemproblem : LPAREN DEFINE_KEY problem_def domain_def objects_def init_def goal_def RPAREN\n               | LPAREN DEFINE_KEY problem_def domain_def init_def RPARENproblem_def : LPAREN PROBLEM_KEY NAME RPARENobjects_def : LPAREN OBJECTS_KEY typed_constants_lst RPARENinit_def : LPAREN INIT_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPAREN\n                | LPAREN INIT_KEY ground_predicates_lst RPARENgoal_def : LPAREN GOAL_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPARENdomain : LPAREN DEFINE_KEY domain_def require_def types_def predicates_def action_def_lst RPARENdomain_def : LPAREN DOMAIN_KEY NAME RPARENrequire_def : LPAREN REQUIREMENTS_KEY require_key_lst RPARENrequire_key_lst : require_key require_key_lst\n                       | require_keyrequire_key : STRIPS_KEY\n                   | EQUALITY_KEY\n                   | TYPING_KEY\n                   | PROBABILISTIC_EFFECTS_KEYtypes_def : LPAREN TYPES_KEY names_lst RPARENpredicates_def : LPAREN PREDICATES_KEY predicate_def_lst RPARENpredicate_def_lst : predicate_def predicate_def_lst\n                         | predicate_defpredicate_def : LPAREN NAME typed_variables_lst RPAREN\n                     | LPAREN NAME RPARENaction_def_lst : action_def action_def_lst\n                      | action_defaction_def : LPAREN ACTION_KEY NAME parameters_def action_def_body RPARENparameters_def : PARAMETERS_KEY LPAREN typed_variables_lst RPAREN\n                      | PARAMETERS_KEY LPAREN RPARENaction_def_body : precond_def effects_defprecond_def : PRECONDITION_KEY LPAREN AND_KEY literals_lst RPAREN\n                   | PRECONDITION_KEY literaleffects_def : EFFECT_KEY LPAREN AND_KEY effects_lst RPAREN\n                   | EFFECT_KEY effecteffects_lst : effect effects_lst\n                   | effecteffect : literal\n              | LPAREN PROBABILISTIC_KEY PROBABILITY literal RPARENliterals_lst : literal literals_lst\n                    | literalliteral : LPAREN NOT_KEY predicate RPAREN\n               | predicateground_predicates_lst : ground_predicate ground_predicates_lst\n                             | ground_predicatepredicate : LPAREN NAME variables_lst RPAREN\n                 | LPAREN EQUALS VARIABLE VARIABLE RPAREN\n                 | LPAREN NAME RPARENground_predicate : LPAREN NAME constants_lst RPAREN\n                        | LPAREN NAME RPARENtyped_constants_lst : constants_lst HYPHEN type typed_constants_lst\n                           | constants_lst HYPHEN typetyped_variables_lst : variables_lst HYPHEN type typed_variables_lst\n                           | variables_lst HYPHEN typeconstants_lst : constant constants_lst\n                     | constantvariables_lst : VARIABLE variables_lst\n                     | VARIABLEnames_lst : NAME names_lst\n                 | NAMEtype : NAMEconstant : NAME'
    
_lr_action_items = {'LPAREN':([0,5,7,8,12,14,19,21,23,24,33,35,37,39,43,46,53,56,60,64,67,69,72,75,82,85,89,92,93,94,99,101,104,106,109,116,117,118,123,127,131,135,136,139,140,145,146,],[4,6,11,13,18,20,32,36,-11,-5,44,51,54,-12,58,44,71,-19,58,-6,71,-8,83,-20,-49,-24,100,-48,71,-23,107,-7,-27,114,-42,-37,125,128,133,125,-47,133,125,-41,-45,-46,-38,]),'$end':([1,2,3,38,62,73,],[0,-1,-2,-4,-10,-3,]),'DEFINE_KEY':([4,],[5,]),'DOMAIN_KEY':([6,13,],[9,9,]),'PROBLEM_KEY':([6,],[10,]),'NAME':([9,10,31,34,42,49,50,51,58,61,65,68,71,78,79,95,107,114,125,128,133,],[15,16,42,50,42,50,-61,68,74,77,79,50,68,50,-60,79,119,119,119,119,119,]),'REQUIREMENTS_KEY':([11,],[17,]),'RPAREN':([15,16,22,25,26,27,28,29,30,40,41,42,45,46,47,49,50,52,53,55,57,59,60,63,66,68,69,70,74,76,78,79,80,81,82,84,85,87,90,91,92,94,96,97,100,101,102,103,104,105,109,110,112,113,115,116,119,122,126,127,129,130,131,134,135,138,139,140,141,142,143,144,145,146,],[23,24,38,39,-14,-15,-16,-17,-18,-13,56,-59,62,-26,64,-55,-61,69,-44,73,-58,75,-22,-25,-54,82,-8,-43,85,-21,-51,-60,91,92,-49,94,-24,-57,-50,101,-48,-23,-56,104,111,-7,112,-53,-27,-30,-42,121,122,-52,-34,-37,131,-9,137,-40,139,140,-47,142,-36,-39,-41,-45,145,-33,-35,146,-46,-38,]),'STRIPS_KEY':([17,26,27,28,29,30,],[27,27,-15,-16,-17,-18,]),'EQUALITY_KEY':([17,26,27,28,29,30,],[28,28,-15,-16,-17,-18,]),'TYPING_KEY':([17,26,27,28,29,30,],[29,29,-15,-16,-17,-18,]),'PROBABILISTIC_EFFECTS_KEY':([17,26,27,28,29,30,],[30,30,-15,-16,-17,-18,]),'TYPES_KEY':([18,],[31,]),'OBJECTS_KEY':([20,],[34,]),'INIT_KEY':([20,36,],[35,35,]),'PREDICATES_KEY':([32,],[43,]),'ACTION_KEY':([44,],[61,]),'HYPHEN':([48,49,50,66,86,87,96,],[65,-55,-61,-54,95,-57,-56,]),'AND_KEY':([51,83,107,114,],[67,93,117,123,]),'GOAL_KEY':([54,],[72,]),'VARIABLE':([74,79,87,100,103,119,120,132,],[87,-60,87,87,87,87,132,141,]),'PARAMETERS_KEY':([77,],[89,]),'PRECONDITION_KEY':([88,111,121,],[99,-29,-28,]),'EFFECT_KEY':([98,108,109,131,137,139,140,145,],[106,-32,-42,-47,-31,-41,-45,-46,]),'NOT_KEY':([107,114,125,133,],[118,118,118,118,]),'EQUALS':([107,114,125,128,133,],[120,120,120,120,120,]),'PROBABILISTIC_KEY':([114,133,],[124,124,]),'PROBABILITY':([124,],[136,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'pddl':([0,],[1,]),'domain':([0,],[2,]),'problem':([0,],[3,]),'domain_def':([5,8,],[7,14,]),'problem_def':([5,],[8,]),'require_def':([7,],[12,]),'types_def':([12,],[19,]),'objects_def':([14,],[21,]),'init_def':([14,21,],[22,37,]),'require_key_lst':([17,26,],[25,40,]),'require_key':([17,26,],[26,26,]),'predicates_def':([19,],[33,]),'names_lst':([31,42,],[41,57,]),'action_def_lst':([33,46,],[45,63,]),'action_def':([33,46,],[46,46,]),'typed_constants_lst':([34,78,],[47,90,]),'constants_lst':([34,49,68,78,],[48,66,81,48,]),'constant':([34,49,68,78,],[49,49,49,49,]),'ground_predicates_lst':([35,53,67,93,],[52,70,80,102,]),'ground_predicate':([35,53,67,93,],[53,53,53,53,]),'goal_def':([37,],[55,]),'predicate_def_lst':([43,60,],[59,76,]),'predicate_def':([43,60,],[60,60,]),'type':([65,95,],[78,103,]),'typed_variables_lst':([74,100,103,],[84,110,113,]),'variables_lst':([74,87,100,103,119,],[86,96,86,86,130,]),'parameters_def':([77,],[88,]),'action_def_body':([88,],[97,]),'precond_def':([88,],[98,]),'effects_def':([98,],[105,]),'literal':([99,106,117,123,127,135,136,],[108,116,127,116,127,116,144,]),'predicate':([99,106,117,118,123,127,135,136,],[109,109,109,129,109,109,109,109,]),'effect':([106,123,135,],[115,135,135,]),'literals_lst':([117,127,],[126,138,]),'effects_lst':([123,135,],[134,143,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> pddl","S'",1,None,None,None),
  ('pddl -> domain','pddl',1,'p_pddl','ppddl_parser.py',151),
  ('pddl -> problem','pddl',1,'p_pddl','ppddl_parser.py',152),
  ('problem -> LPAREN DEFINE_KEY problem_def domain_def objects_def init_def goal_def RPAREN','problem',8,'p_problem','ppddl_parser.py',157),
  ('problem -> LPAREN DEFINE_KEY problem_def domain_def init_def RPAREN','problem',6,'p_problem','ppddl_parser.py',158),
  ('problem_def -> LPAREN PROBLEM_KEY NAME RPAREN','problem_def',4,'p_problem_def','ppddl_parser.py',166),
  ('objects_def -> LPAREN OBJECTS_KEY typed_constants_lst RPAREN','objects_def',4,'p_objects_def','ppddl_parser.py',171),
  ('init_def -> LPAREN INIT_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPAREN','init_def',7,'p_init_def','ppddl_parser.py',176),
  ('init_def -> LPAREN INIT_KEY ground_predicates_lst RPAREN','init_def',4,'p_init_def','ppddl_parser.py',177),
  ('goal_def -> LPAREN GOAL_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPAREN','goal_def',7,'p_goal_def','ppddl_parser.py',186),
  ('domain -> LPAREN DEFINE_KEY domain_def require_def types_def predicates_def action_def_lst RPAREN','domain',8,'p_domain','ppddl_parser.py',199),
  ('domain_def -> LPAREN DOMAIN_KEY NAME RPAREN','domain_def',4,'p_domain_def','ppddl_parser.py',206),
  ('require_def -> LPAREN REQUIREMENTS_KEY require_key_lst RPAREN','require_def',4,'p_require_def','ppddl_parser.py',213),
  ('require_key_lst -> require_key require_key_lst','require_key_lst',2,'p_require_key_lst','ppddl_parser.py',220),
  ('require_key_lst -> require_key','require_key_lst',1,'p_require_key_lst','ppddl_parser.py',221),
  ('require_key -> STRIPS_KEY','require_key',1,'p_require_key','ppddl_parser.py',231),
  ('require_key -> EQUALITY_KEY','require_key',1,'p_require_key','ppddl_parser.py',232),
  ('require_key -> TYPING_KEY','require_key',1,'p_require_key','ppddl_parser.py',233),
  ('require_key -> PROBABILISTIC_EFFECTS_KEY','require_key',1,'p_require_key','ppddl_parser.py',234),
  ('types_def -> LPAREN TYPES_KEY names_lst RPAREN','types_def',4,'p_types_def','ppddl_parser.py',241),
  ('predicates_def -> LPAREN PREDICATES_KEY predicate_def_lst RPAREN','predicates_def',4,'p_predicates_def','ppddl_parser.py',252),
  ('predicate_def_lst -> predicate_def predicate_def_lst','predicate_def_lst',2,'p_predicate_def_lst','ppddl_parser.py',259),
  ('predicate_def_lst -> predicate_def','predicate_def_lst',1,'p_predicate_def_lst','ppddl_parser.py',260),
  ('predicate_def -> LPAREN NAME typed_variables_lst RPAREN','predicate_def',4,'p_predicate_def','ppddl_parser.py',271),
  ('predicate_def -> LPAREN NAME RPAREN','predicate_def',3,'p_predicate_def','ppddl_parser.py',272),
  ('action_def_lst -> action_def action_def_lst','action_def_lst',2,'p_action_def_lst','ppddl_parser.py',282),
  ('action_def_lst -> action_def','action_def_lst',1,'p_action_def_lst','ppddl_parser.py',283),
  ('action_def -> LPAREN ACTION_KEY NAME parameters_def action_def_body RPAREN','action_def',6,'p_action_def','ppddl_parser.py',300),
  ('parameters_def -> PARAMETERS_KEY LPAREN typed_variables_lst RPAREN','parameters_def',4,'p_parameters_def','ppddl_parser.py',307),
  ('parameters_def -> PARAMETERS_KEY LPAREN RPAREN','parameters_def',3,'p_parameters_def','ppddl_parser.py',308),
  ('action_def_body -> precond_def effects_def','action_def_body',2,'p_action_def_body','ppddl_parser.py',322),
  ('precond_def -> PRECONDITION_KEY LPAREN AND_KEY literals_lst RPAREN','precond_def',5,'p_precond_def','ppddl_parser.py',329),
  ('precond_def -> PRECONDITION_KEY literal','precond_def',2,'p_precond_def','ppddl_parser.py',330),
  ('effects_def -> EFFECT_KEY LPAREN AND_KEY effects_lst RPAREN','effects_def',5,'p_effects_def','ppddl_parser.py',339),
  ('effects_def -> EFFECT_KEY effect','effects_def',2,'p_effects_def','ppddl_parser.py',340),
  ('effects_lst -> effect effects_lst','effects_lst',2,'p_effects_lst','ppddl_parser.py',349),
  ('effects_lst -> effect','effects_lst',1,'p_effects_lst','ppddl_parser.py',350),
  ('effect -> literal','effect',1,'p_effect','ppddl_parser.py',359),
  ('effect -> LPAREN PROBABILISTIC_KEY PROBABILITY literal RPAREN','effect',5,'p_effect','ppddl_parser.py',360),
  ('literals_lst -> literal literals_lst','literals_lst',2,'p_literals_lst','ppddl_parser.py',369),
  ('literals_lst -> literal','literals_lst',1,'p_literals_lst','ppddl_parser.py',370),
  ('literal -> LPAREN NOT_KEY predicate RPAREN','literal',4,'p_literal','ppddl_parser.py',378),
  ('literal -> predicate','literal',1,'p_literal','ppddl_parser.py',379),
  ('ground_predicates_lst -> ground_predicate ground_predicates_lst','ground_predicates_lst',2,'p_ground_predicates_lst','ppddl_parser.py',387),
  ('ground_predicates_lst -> ground_predicate','ground_predicates_lst',1,'p_ground_predicates_lst','ppddl_parser.py',388),
  ('predicate -> LPAREN NAME variables_lst RPAREN','predicate',4,'p_predicate','ppddl_parser.py',396),
  ('predicate -> LPAREN EQUALS VARIABLE VARIABLE RPAREN','predicate',5,'p_predicate','ppddl_parser.py',397),
  ('predicate -> LPAREN NAME RPAREN','predicate',3,'p_predicate','ppddl_parser.py',398),
  ('ground_predicate -> LPAREN NAME constants_lst RPAREN','ground_predicate',4,'p_ground_predicate','ppddl_parser.py',408),
  ('ground_predicate -> LPAREN NAME RPAREN','ground_predicate',3,'p_ground_predicate','ppddl_parser.py',409),
  ('typed_constants_lst -> constants_lst HYPHEN type typed_constants_lst','typed_constants_lst',4,'p_typed_constants_lst','ppddl_parser.py',417),
  ('typed_constants_lst -> constants_lst HYPHEN type','typed_constants_lst',3,'p_typed_constants_lst','ppddl_parser.py',418),
  ('typed_variables_lst -> variables_lst HYPHEN type typed_variables_lst','typed_variables_lst',4,'p_typed_variables_lst','ppddl_parser.py',426),
  ('typed_variables_lst -> variables_lst HYPHEN type','typed_variables_lst',3,'p_typed_variables_lst','ppddl_parser.py',427),
  ('constants_lst -> constant constants_lst','constants_lst',2,'p_constants_lst','ppddl_parser.py',435),
  ('constants_lst -> constant','constants_lst',1,'p_constants_lst','ppddl_parser.py',436),
  ('variables_lst -> VARIABLE variables_lst','variables_lst',2,'p_variables_lst','ppddl_parser.py',444),
  ('variables_lst -> VARIABLE','variables_lst',1,'p_variables_lst','ppddl_parser.py',445),
  ('names_lst -> NAME names_lst','names_lst',2,'p_names_lst','ppddl_parser.py',453),
  ('names_lst -> NAME','names_lst',1,'p_names_lst','ppddl_parser.py',454),
  ('type -> NAME','type',1,'p_type','ppddl_parser.py',464),
  ('constant -> NAME','constant',1,'p_constant','ppddl_parser.py',469),
]