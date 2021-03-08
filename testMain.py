from ppddl_parser import PDDLParser


#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\navigation_inst_mdp\Domain.ppddl'
#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\navigation_inst_mdp\Problem.ppddl'

#directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\rover\p01.pddl'
#directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\rover\Domain.pddl'

directory1 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\blocksworld\domain.pddl'
directory2 = r'C:\Users\alber\Documents\ISAE-MAE\Research project\MyAlgorithms\learning2parse\ppddl\blocksworld\p01.pddl'


#MyDomain = PDDLParser.parse(directory1)
#MyProblem = PDDLParser.parse(directory2)

listilla = [(0.8, "hola"), (0.2, "adios")]

for prob,cosa in listilla:
    print(prob)
    print(cosa)

    
    
    