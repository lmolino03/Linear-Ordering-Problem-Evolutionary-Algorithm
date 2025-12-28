"""Experiment runner for LOP evolutionary algorithm."""
import os
import json
import csv
import time
from datetime import datetime
import numpy as np

from core.lop_problem import LOPInstance
from core.permutation_operators import PMXCrossover, SwapMutation
from core.evolutionary_algorithm import EvolutionaryAlgorithm


class ExperimentRunner:
    
    def __init__(self, config, instances_path, results_path):
        self.config = config
        self.instances_path = instances_path
        self.results_path = results_path
        self.results = []
        
        os.makedirs(results_path, exist_ok=True)
    
    def load_instances(self):
        instances = {}
        for instance_file in self.config.instances:
            filepath = os.path.join(self.instances_path, instance_file)
            try:
                instance = LOPInstance.load_from_file(filepath)
                instances[instance.name] = instance
                print(f"Loaded {instance.name}: {instance.n}x{instance.n} matrix")
            except Exception as e:
                print(f"Error loading {instance_file}: {e}")
        return instances
    
    def run_single_experiment(self, instance, pop_size, run_id, seed):
        ea = EvolutionaryAlgorithm(
            problem_instance=instance,
            population_size=pop_size,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            tournament_size=self.config.tournament_size,
            elitism=self.config.elitism,
            crossover_op=PMXCrossover.crossover,
            mutation_op=SwapMutation.mutate
        )
        
        start_time = time.time()
        result = ea.run(max_evaluations=self.config.max_evaluations, seed=seed)
        elapsed_time = time.time() - start_time
        
        return {
            'instance': instance.name,
            'pop_size': pop_size,
            'run_id': run_id,
            'seed': seed,
            'best_fitness': result['best_fitness'],
            'best_solution': result['best_solution'],
            'evaluations': result['evaluations'],
            'time_seconds': elapsed_time,
            'convergence': result['convergence']
        }
    
    def run_all_experiments(self):
        print("=" * 80)
        print("LINEAR ORDERING PROBLEM - EVOLUTIONARY ALGORITHM EXPERIMENTS")
        print("=" * 80)
        print(f"Population sizes: {self.config.population_sizes}")
        print(f"Runs per configuration: {self.config.num_runs}")
        print(f"Max evaluations: {self.config.max_evaluations}")
        print("=" * 80)
        
        instances = self.load_instances()
        print(f"\nLoaded {len(instances)} instances\n")
        
        total_experiments = len(instances) * len(self.config.population_sizes) * self.config.num_runs
        experiment_count = 0
        
        for instance_name, instance in instances.items():
            print(f"\n{'='*80}")
            print(f"Instance: {instance_name} (size: {instance.n}x{instance.n})")
            print(f"{'='*80}")
            
            for pop_size in self.config.population_sizes:
                print(f"\n  Population size: {pop_size}")
                
                for run_id in range(self.config.num_runs):
                    seed = self.config.random_seed + run_id
                    
                    result = self.run_single_experiment(instance, pop_size, run_id, seed)
                    self.results.append(result)
                    
                    experiment_count += 1
                    
                    if (run_id + 1) % 10 == 0:
                        print(f"    Run {run_id + 1}/{self.config.num_runs} - "
                              f"Best: {result['best_fitness']:.0f} - "
                              f"Progress: {experiment_count}/{total_experiments}")
                
                config_results = [r for r in self.results 
                                if r['instance'] == instance_name and r['pop_size'] == pop_size]
                fitnesses = [r['best_fitness'] for r in config_results]
                
                print(f"    Statistics - Mean: {np.mean(fitnesses):.2f}, "
                      f"Std: {np.std(fitnesses):.2f}, "
                      f"Best: {np.max(fitnesses):.0f}, "
                      f"Worst: {np.min(fitnesses):.0f}")
        
        print(f"\n{'='*80}")
        print(f"All experiments completed! Total: {experiment_count}")
        print(f"{'='*80}\n")
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_file = os.path.join(self.results_path, f"detailed_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            results_serializable = convert_numpy_types(self.results)
            json.dump(results_serializable, f, indent=2)
        print(f"Detailed results saved to: {json_file}")
        
        csv_file = os.path.join(self.results_path, f"summary_results_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Instance', 'PopSize', 'Mean', 'Std', 'Best', 'Worst', 
                'Median', 'Known_Optimum', 'Gap_%'
            ])
            
            for instance_name in sorted(set(r['instance'] for r in self.results)):
                for pop_size in self.config.population_sizes:
                    config_results = [r for r in self.results 
                                    if r['instance'] == instance_name and r['pop_size'] == pop_size]
                    
                    if not config_results:
                        continue
                    
                    fitnesses = [r['best_fitness'] for r in config_results]
                    
                    mean_val = np.mean(fitnesses)
                    std_val = np.std(fitnesses)
                    best_val = np.max(fitnesses)
                    worst_val = np.min(fitnesses)
                    median_val = np.median(fitnesses)
                    
                    known_opt = self.config.known_optima.get(instance_name, None)
                    gap_percent = ((known_opt - mean_val) / known_opt * 100) if known_opt else None
                    
                    writer.writerow([
                        instance_name, pop_size, 
                        f"{mean_val:.2f}", f"{std_val:.2f}",
                        f"{best_val:.0f}", f"{worst_val:.0f}",
                        f"{median_val:.0f}",
                        known_opt if known_opt else "N/A",
                        f"{gap_percent:.2f}" if gap_percent is not None else "N/A"
                    ])
        
        print(f"Summary results saved to: {csv_file}")
        
        return json_file, csv_file
