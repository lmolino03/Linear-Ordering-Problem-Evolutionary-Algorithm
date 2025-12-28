"""
Genetic operators for permutation representation.
"""
import random
import numpy as np


class PMXCrossover:
    
    @staticmethod
    def crossover(parent1, parent2):
        parent1 = list(parent1)
        parent2 = list(parent2)
        n = len(parent1)
        
        point1 = random.randint(0, n - 2)
        point2 = random.randint(point1 + 1, n)
        
        child1 = PMXCrossover._pmx_single(parent1, parent2, point1, point2)
        child2 = PMXCrossover._pmx_single(parent2, parent1, point1, point2)
        
        return child1, child2
    
    @staticmethod
    def _pmx_single(parent1, parent2, point1, point2):
        n = len(parent1)
        child = [-1] * n
        
        child[point1:point2] = parent1[point1:point2]
        
        for i in range(point1, point2):
            if parent2[i] not in child[point1:point2]:
                pos = i
                while point1 <= pos < point2:
                    pos = parent2.index(parent1[pos])
                child[pos] = parent2[i]
        
        for i in range(n):
            if child[i] == -1:
                child[i] = parent2[i]
        
        return child


class SwapMutation:
    
    @staticmethod
    def mutate(individual, mutation_rate=0.1):
        individual = list(individual)
        
        if random.random() < mutation_rate:
            n = len(individual)
            pos1 = random.randint(0, n - 1)
            pos2 = random.randint(0, n - 1)
            
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        
        return individual


class InsertMutation:
    
    @staticmethod
    def mutate(individual, mutation_rate=0.1):
        individual = list(individual)
        
        if random.random() < mutation_rate:
            n = len(individual)
            pos1 = random.randint(0, n - 1)
            pos2 = random.randint(0, n - 1)
            
            element = individual.pop(pos1)
            individual.insert(pos2, element)
        
        return individual


class InversionMutation:
    
    @staticmethod
    def mutate(individual, mutation_rate=0.1):
        individual = list(individual)
        
        if random.random() < mutation_rate:
            n = len(individual)
            pos1 = random.randint(0, n - 2)
            pos2 = random.randint(pos1 + 1, n)
            
            individual[pos1:pos2] = reversed(individual[pos1:pos2])
        
        return individual
