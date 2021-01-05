# Parsing-Solving-PlaningProblems
This is the third step of my work related to MDPs and some famous solvers.

In previous repositories I took a simple 'Maze-like' problem, I coded both a declarative and a generative description of the problem and I proposed some solvers to find the solution. Some of them use the declarative version whereas the most complex algorithms use the generative one. However, in this new repository I am trying to go even further. My intentions are:
    1) Collect some famous problems to test a brandnew algorithm and compare the results with other state-of-the-art solvers.
    For some reason, I have decided to use PPDDL language to describe the problems... Honestly, I do not know why because RDDL is the most updated descriptor language.
    In fact, ICAPS and the International Probabilistic Planning Competition is not providing the PPDDL description since no competitor asked for it.
    2) Develop a parser to read and compile .ppddl files. The final code shall understand the description of the problems and create a generative model of the problem.
    In other words, a kind of black box that can be used by the solvers.
    IMPORTANT: I must thank Thiagopbueno and his repositories for providing the skeleton of the parser.
    3) Implement several solvers and do some tests.
