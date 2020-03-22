import mlrose
import numpy as np

coords_list = [(2, 2), (4, 4), (6, 5), (7, 9)]
alpha_list = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
problem_fit = mlrose.TSPOpt(length = 4, coords = coords_list,
                            maximize=False)


best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 0)

print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness)


