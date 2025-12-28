"""
Linear Ordering Problem instance loader and objective function.
"""
import numpy as np


class LOPInstance:
    
    def __init__(self, matrix, name=""):
        self.matrix = np.array(matrix)
        self.n = len(matrix)
        self.name = name
        
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square")
    
    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        n = int(lines[1].strip())
        
        values = []
        for line in lines[2:]:
            line = line.strip()
            if line:
                nums = [int(x) for x in line.split()]
                values.extend(nums)
        
        if len(values) != n * n:
            raise ValueError(f"Expected {n*n} values, got {len(values)}")
        
        matrix = np.array(values).reshape(n, n)
        
        import os
        name = os.path.basename(filepath).replace('.mat', '')
        
        return cls(matrix, name)
    
    def evaluate(self, permutation):
        permutation = np.array(permutation)
        
        if len(permutation) != self.n:
            raise ValueError(f"Permutation length {len(permutation)} != matrix size {self.n}")
        
        if set(permutation) != set(range(self.n)):
            raise ValueError("Invalid permutation: must contain all indices 0 to n-1")
        
        total = 0
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                row_idx = permutation[i]
                col_idx = permutation[j]
                total += self.matrix[row_idx, col_idx]
        
        return total
    
    def __str__(self):
        return f"LOPInstance(name={self.name}, size={self.n}x{self.n})"
    
    def __repr__(self):
        return self.__str__()
