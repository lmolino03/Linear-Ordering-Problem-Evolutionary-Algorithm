"""
Analysis runner for statistical analysis and visualization.
"""
import os
import glob

from analysis import StatisticalAnalyzer, ResultsVisualizer


class AnalysisRunner:
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
    
    def run(self):
        print("\n" + "=" * 80)
        print("RUNNING AUTOMATIC ANALYSIS")
        print("=" * 80)
        
        json_files = glob.glob(os.path.join(self.results_dir, "detailed_results_*.json"))
        if not json_files:
            print("Error: No results files found!")
            return
        
        results_file = max(json_files, key=os.path.getmtime)
        print(f"\nAnalyzing results from: {os.path.basename(results_file)}\n")
        
        print("1. Running statistical analysis...")
        try:
            analyzer = StatisticalAnalyzer(results_file)
            analysis_dir = os.path.join(self.results_dir, "analysis")
            analyzer.analyze_all(analysis_dir)
            print("   Statistical analysis completed")
        except Exception as e:
            print(f"   Error in statistical analysis: {e}")
        
        print("\n2. Generating visualizations...")
        try:
            visualizer = ResultsVisualizer(results_file)
            figures_dir = os.path.join(self.results_dir, "figures")
            visualizer.create_all_visualizations(figures_dir)
            print("   Visualizations completed")
        except Exception as e:
            print(f"   Error in visualization: {e}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved in:")
        print(f"  - Analysis: {os.path.join(self.results_dir, 'analysis')}")
        print(f"  - Figures:  {os.path.join(self.results_dir, 'figures')}")
        print()
