"""
Evolutionary Algorithm for Linear Ordering Problem.
"""
import random
import numpy as np
from typing import List, Tuple, Callable
import copy


class Individual:
    
    def __init__(self, permutation, fitness=None):
        self.permutation = permutation
        self.fitness = fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness})"


class EvolutionaryAlgorithm:
    
    def __init__(
        self,
        problem_instance,
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.9,
        tournament_size=3,
        elitism=True,
        crossover_op=None,
        mutation_op=None
    ):
        self.problem = problem_instance
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        
        self.population = []
        self.best_individual = None
        self.evaluations = 0
        self.convergence_data = []
    
    def _create_random_individual(self):
        perm = list(range(self.problem.n))
        random.shuffle(perm)
        return Individual(perm)
    
    def _evaluate(self, individual):
        if individual.fitness is None:
            individual.fitness = self.problem.evaluate(individual.permutation)
            self.evaluations += 1
        return individual.fitness
    
    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = self._create_random_individual()
            self._evaluate(ind)
            self.population.append(ind)
        
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        self.convergence_data.append({
            'evaluations': self.evaluations,
            'best_fitness': self.best_individual.fitness,
            'avg_fitness': np.mean([ind.fitness for ind in self.population])
        })
    
    def tournament_selection(self):
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def evolve_generation(self):
        new_population = []
        
        if self.elitism:
            best = max(self.population, key=lambda x: x.fitness)
            new_population.append(copy.deepcopy(best))
        
        while len(new_population) < self.pop_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            if random.random() < self.crossover_rate and self.crossover_op:
                child1_perm, child2_perm = self.crossover_op(
                    parent1.permutation, parent2.permutation
                )
                child1 = Individual(child1_perm)
                child2 = Individual(child2_perm)
            else:
                child1 = Individual(copy.copy(parent1.permutation))
                child2 = Individual(copy.copy(parent2.permutation))
            
            if self.mutation_op:
                child1.permutation = self.mutation_op(child1.permutation, self.mutation_rate)
                child2.permutation = self.mutation_op(child2.permutation, self.mutation_rate)
            
            self._evaluate(child1)
            self._evaluate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.pop_size:
                new_population.append(child2)
        
        self.population = new_population[:self.pop_size]
        
        current_best = max(self.population, key=lambda x: x.fitness)
        if current_best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(current_best)
        
        self.convergence_data.append({
            'evaluations': self.evaluations,
            'best_fitness': self.best_individual.fitness,
            'avg_fitness': np.mean([ind.fitness for ind in self.population])
        })
    
    def run(self, max_evaluations=10000, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.evaluations = 0
        self.convergence_data = []
        
        self.initialize_population()
        
        while self.evaluations < max_evaluations:
            self.evolve_generation()
        
        return {
            'best_solution': self.best_individual.permutation,
            'best_fitness': self.best_individual.fitness,
            'evaluations': self.evaluations,
            'convergence': self.convergence_data
        }
