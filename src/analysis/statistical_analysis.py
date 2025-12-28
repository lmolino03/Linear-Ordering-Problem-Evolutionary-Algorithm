"""
Statistical analysis of experimental results.
Includes hypothesis testing and significance analysis.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config


class StatisticalAnalyzer:
    """Performs statistical analysis on experimental results."""
    
    def __init__(self, results_file):
        """
        Initialize analyzer.
        
        Args:
            results_file: Path to JSON results file
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.results)
    
    def compute_summary_statistics(self):
        """Compute summary statistics for all configurations."""
        summary = []
        
        for instance in self.df['instance'].unique():
            for pop_size in sorted(self.df['pop_size'].unique()):
                data = self.df[
                    (self.df['instance'] == instance) & 
                    (self.df['pop_size'] == pop_size)
                ]['best_fitness']
                
                cfg = get_config()
                known_opt = cfg.known_optima.get(instance, None)
                gap = ((known_opt - data.mean()) / known_opt * 100) if known_opt else None
                
                summary.append({
                    'instance': instance,
                    'pop_size': pop_size,
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median(),
                    'known_optimum': known_opt,
                    'gap_percent': gap
                })
        
        return pd.DataFrame(summary)
    
    def kruskal_wallis_test(self, instance):
        """
        Perform Kruskal-Wallis H-test for an instance.
        Tests if population sizes have significantly different results.
        
        Args:
            instance: Instance name
            
        Returns:
            Dictionary with test results
        """
        groups = []
        for pop_size in sorted(self.df['pop_size'].unique()):
            data = self.df[
                (self.df['instance'] == instance) & 
                (self.df['pop_size'] == pop_size)
            ]['best_fitness'].values
            groups.append(data)
        
        statistic, p_value = stats.kruskal(*groups)
        
        return {
            'instance': instance,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def mann_whitney_pairwise(self, instance, alpha=0.05):
        """
        Perform pairwise Mann-Whitney U tests with Bonferroni correction.
        
        Args:
            instance: Instance name
            alpha: Significance level
            
        Returns:
            DataFrame with pairwise comparison results
        """
        pop_sizes = sorted(self.df['pop_size'].unique())
        n_comparisons = len(pop_sizes) * (len(pop_sizes) - 1) // 2
        corrected_alpha = alpha / n_comparisons  # Bonferroni correction
        
        results = []
        
        for i, pop1 in enumerate(pop_sizes):
            for pop2 in pop_sizes[i+1:]:
                data1 = self.df[
                    (self.df['instance'] == instance) & 
                    (self.df['pop_size'] == pop1)
                ]['best_fitness'].values
                
                data2 = self.df[
                    (self.df['instance'] == instance) & 
                    (self.df['pop_size'] == pop2)
                ]['best_fitness'].values
                
                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                results.append({
                    'pop_size_1': pop1,
                    'pop_size_2': pop2,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < corrected_alpha,
                    'mean_1': data1.mean(),
                    'mean_2': data2.mean(),
                    'better': pop1 if data1.mean() > data2.mean() else pop2
                })
        
        return pd.DataFrame(results)
    
    def analyze_all(self, output_dir):
        """
        Perform all statistical analyses and save results.
        
        Args:
            output_dir: Directory to save analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Summary statistics
        print("\nComputing summary statistics...")
        summary = self.compute_summary_statistics()
        summary_file = os.path.join(output_dir, "summary_statistics.csv")
        summary.to_csv(summary_file, index=False)
        print(f"Saved to: {summary_file}")
        
        # Print summary table
        print("\nSummary Statistics:")
        print(summary.to_string(index=False))
        
        # Kruskal-Wallis tests
        print("\n" + "=" * 80)
        print("Kruskal-Wallis Tests (testing if population sizes differ)")
        print("=" * 80)
        
        kw_results = []
        for instance in self.df['instance'].unique():
            result = self.kruskal_wallis_test(instance)
            kw_results.append(result)
            
            sig_marker = "***" if result['significant'] else "ns"
            print(f"\n{instance}:")
            print(f"  H-statistic: {result['statistic']:.4f}")
            print(f"  p-value: {result['p_value']:.6f} {sig_marker}")
        
        kw_df = pd.DataFrame(kw_results)
        kw_file = os.path.join(output_dir, "kruskal_wallis_tests.csv")
        kw_df.to_csv(kw_file, index=False)
        print(f"\nKruskal-Wallis results saved to: {kw_file}")
        
        # Pairwise comparisons
        print("\n" + "=" * 80)
        print("Pairwise Mann-Whitney U Tests (with Bonferroni correction)")
        print("=" * 80)
        
        all_pairwise = []
        for instance in self.df['instance'].unique():
            print(f"\n{instance}:")
            pairwise = self.mann_whitney_pairwise(instance)
            pairwise['instance'] = instance
            all_pairwise.append(pairwise)
            
            # Print significant differences
            sig = pairwise[pairwise['significant']]
            if len(sig) > 0:
                for _, row in sig.iterrows():
                    print(f"  {row['pop_size_1']} vs {row['pop_size_2']}: "
                          f"p={row['p_value']:.6f} *** "
                          f"(Better: {row['better']})")
            else:
                print("  No significant differences found.")
        
        if all_pairwise:
            pairwise_df = pd.concat(all_pairwise, ignore_index=True)
            pairwise_file = os.path.join(output_dir, "pairwise_comparisons.csv")
            pairwise_df.to_csv(pairwise_file, index=False)
            print(f"\nPairwise comparisons saved to: {pairwise_file}")
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)


