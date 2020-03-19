import mlrose
import numpy as np

coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

problem_fit = mlrose.TSPOpt(length = 8, coords = coords_list,
                            maximize=False)


best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)

print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness)