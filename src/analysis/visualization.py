"""
Visualization of experimental results.
Creates plots and charts for the scientific report.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

# Set style for professional publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Configure matplotlib for professional appearance
plt.rcParams.update({
    # Font settings - sans-serif for professional scientific publications
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    
    # Figure settings
    'figure.dpi': 300,  # High resolution for publication
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Line and marker settings
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 1.0,
    
    # Grid settings
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    
    # Legend settings
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'gray',
    'legend.fancybox': True,
})


class ResultsVisualizer:
    """Creates visualizations for experimental results."""
    
    def __init__(self, results_file):
        """
        Initialize visualizer.
        
        Args:
            results_file: Path to JSON results file
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.df = pd.DataFrame(self.results)
    
    def plot_boxplots_by_instance(self, output_dir):
        """Create box plots for each instance showing population size effects."""
        instances = sorted(self.df['instance'].unique())
        
        # Create subplots
        n_instances = len(instances)
        n_cols = 2
        n_rows = (n_instances + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten() if n_instances > 1 else [axes]
        
        for idx, instance in enumerate(instances):
            ax = axes[idx]
            
            # Filter data
            data = self.df[self.df['instance'] == instance]
            
            # Create box plot
            data.boxplot(column='best_fitness', by='pop_size', ax=ax)
            
            ax.set_title(f'{instance}')
            ax.set_xlabel('Population Size')
            ax.set_ylabel('Best Fitness')
            ax.get_figure().suptitle('')  # Remove auto title
            
            # Add known optimum line
            cfg = get_config()
            known_opt = cfg.known_optima.get(instance)
            if known_opt:
                ax.axhline(y=known_opt, color='r', linestyle='--', 
                          label=f'Known Optimum: {known_opt}', linewidth=1.5)
                ax.legend()
        
        # Hide extra subplots
        for idx in range(n_instances, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'boxplots_by_instance.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_mean_comparison(self, output_dir):
        """Bar plot comparing mean fitness across population sizes."""
        # Compute means
        summary = self.df.groupby(['instance', 'pop_size'])['best_fitness'].agg(['mean', 'std']).reset_index()
        
        instances = sorted(summary['instance'].unique())
        
        # Create subplots
        n_instances = len(instances)
        n_cols = 2
        n_rows = (n_instances + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten() if n_instances > 1 else [axes]
        
        for idx, instance in enumerate(instances):
            ax = axes[idx]
            
            data = summary[summary['instance'] == instance]
            
            # Bar plot
            x = np.arange(len(data))
            ax.bar(x, data['mean'], yerr=data['std'], capsize=5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(data['pop_size'])
            ax.set_xlabel('Population Size')
            ax.set_ylabel('Mean Best Fitness')
            ax.set_title(f'{instance}')
            
            # Add known optimum line
            cfg = get_config()
            known_opt = cfg.known_optima.get(instance)
            if known_opt:
                ax.axhline(y=known_opt, color='r', linestyle='--', 
                          label=f'Optimum: {known_opt}', linewidth=1.5)
                ax.legend()
        
        # Hide extra subplots
        for idx in range(n_instances, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'mean_comparison.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_convergence(self, output_dir, instance_name=None):
        """Plot convergence curves."""
        if instance_name is None:
            # Pick first instance
            instance_name = sorted(self.df['instance'].unique())[0]
        
        # Filter for specific instance
        data = self.df[self.df['instance'] == instance_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Best fitness over evaluations
        ax = axes[0]
        for pop_size in sorted(data['pop_size'].unique()):
            pop_data = data[data['pop_size'] == pop_size]
            
            # Extract convergence data from first run
            first_run = pop_data.iloc[0]
            convergence = first_run['convergence']
            
            evals = [c['evaluations'] for c in convergence]
            best_fitness = [c['best_fitness'] for c in convergence]
            
            ax.plot(evals, best_fitness, label=f'Pop={pop_size}', alpha=0.7)
        
        ax.set_xlabel('Evaluations')
        ax.set_ylabel('Best Fitness')
        ax.set_title(f'Convergence - {instance_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Average fitness over evaluations
        ax = axes[1]
        for pop_size in sorted(data['pop_size'].unique()):
            pop_data = data[data['pop_size'] == pop_size]
            
            # Average convergence across runs
            all_convergence = []
            for _, row in pop_data.iterrows():
                convergence = row['convergence']
                avg_fitness = [c['avg_fitness'] for c in convergence]
                all_convergence.append(avg_fitness)
            
            # Compute mean
            mean_convergence = np.mean(all_convergence, axis=0)
            evals = [c['evaluations'] for c in pop_data.iloc[0]['convergence']]
            
            ax.plot(evals, mean_convergence, label=f'Pop={pop_size}', alpha=0.7)
        
        ax.set_xlabel('Evaluations')
        ax.set_ylabel('Average Fitness')
        ax.set_title(f'Average Population Fitness - {instance_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f'convergence_{instance_name}.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_gap_from_optimum(self, output_dir):
        """Plot percentage gap from known optimum."""
        # Compute gaps
        cfg = get_config()
        gaps = []
        for instance in self.df['instance'].unique():
            known_opt = cfg.known_optima.get(instance)
            if not known_opt:
                continue
            
            for pop_size in sorted(self.df['pop_size'].unique()):
                data = self.df[
                    (self.df['instance'] == instance) & 
                    (self.df['pop_size'] == pop_size)
                ]['best_fitness']
                
                mean_fitness = data.mean()
                gap_percent = (known_opt - mean_fitness) / known_opt * 100
                
                gaps.append({
                    'instance': instance,
                    'pop_size': pop_size,
                    'gap_percent': gap_percent
                })
        
        gap_df = pd.DataFrame(gaps)
        
        # Create heatmap
        pivot = gap_df.pivot(index='instance', columns='pop_size', values='gap_percent')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Gap from Optimum (%)'})
        plt.title('Percentage Gap from Known Optimum')
        plt.xlabel('Population Size')
        plt.ylabel('Instance')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gap_heatmap.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def create_all_visualizations(self, output_dir):
        """Create all visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        print("\n1. Box plots by instance...")
        self.plot_boxplots_by_instance(output_dir)
        
        print("\n2. Mean comparison bar plots...")
        self.plot_mean_comparison(output_dir)
        
        print("\n3. Convergence plots...")
        # Create convergence for a few representative instances
        instances = sorted(self.df['instance'].unique())
        for instance in instances[:3]:  # Plot first 3 instances
            self.plot_convergence(output_dir, instance)
        
        print("\n4. Gap from optimum heatmap...")
        self.plot_gap_from_optimum(output_dir)
        
        print("\n" + "=" * 80)
        print(f"All visualizations saved to: {output_dir}")
        print("=" * 80)
