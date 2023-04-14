import numpy as np
import pygad
from functools import cache

import Operators.PrewittOperator as PrewittOperator
import Operators.SobelOperator as SobelOperator


class ImageContrastEnhancement:
    def __init__(self,
                 image: np.ndarray,
                 operator="Prewitt",
                 sol_per_pop=10,
                 num_generations=100,
                 parent_selection_type="rws",
                 crossover_probability=0.8,
                 crossover_type="two_points",
                 mutation_type="random",
                 mutation_probability=0.1,
                 allow_duplicate_genes=False,
                 random_seed=123):
        self.best_chromosome = None
        self.fitness_func = self._choose_fitness_function(operator)
        self.image = image
        self.levels = np.sort(np.unique(image))
        self.num_genes = len(self.levels) - 2

        self.ga_instance = pygad.GA(
            fitness_func=self.fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=self.num_genes,
            num_parents_mating=3,
            num_generations=num_generations,
            parent_selection_type=parent_selection_type,
            crossover_probability=crossover_probability,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_probability=mutation_probability,
            gene_space=range(1, 255),
            allow_duplicate_genes=allow_duplicate_genes,
            random_seed=random_seed
        )

    def _choose_fitness_function(self, operator: str):
        operator = operator.lower()
        if operator == "prewitt":
            return self.prewitt_fitness_function
        elif operator == "sobel":
            return self.sobel_fitness_function
        else:
            raise Exception()



    def prewitt_fitness_function(self, ga_instance,
                                 solution, solution_idx):
        solution.sort()
        chromosome = tuple(solution)
        image_generated_by_chromosome = self._image_generated_by_chromosome(chromosome)
        return PrewittOperator.calculate_sum(image_generated_by_chromosome)

    def sobel_fitness_function(self, ga_instance,
                               solution, solution_idx):
        solution.sort()
        chromosome = tuple(solution)
        image_generated_by_chromosome = self._image_generated_by_chromosome(chromosome)
        return SobelOperator.calculate_sum(image_generated_by_chromosome)


    def image_generated_by_chromosome(self, solution=None):
        if solution is None:
            if self.best_chromosome is None:
                raise Exception()
            solution = self.best_chromosome
        chromosome = tuple(solution)    # we are mapping to tuple to use @cache
        return self._image_generated_by_chromosome(chromosome)

    @cache
    def _image_generated_by_chromosome(self, chromosome: tuple):    
        mapper = {key: value for key, value in zip(self.levels[1:-1], chromosome)}

        mapper[self.levels[0]] = 0
        mapper[self.levels[-1]] = 255

        result_image = np.ndarray(self.image.shape)
        for k in mapper:
            result_image[self.image == k] = mapper[k]
        return result_image.astype(np.uint8)

    def run(self):
        self.ga_instance.run()
        self.best_chromosome = self.ga_instance.best_solution()[0]
