"""
Builder for LOP experiments configuration.
"""
import os
import argparse

from .runners import ExperimentRunner, AnalysisRunner


class Builder:
    
    def __init__(self, config):
        self.config = config
        self.args = None
        self._parse_arguments()
    
    def _parse_arguments(self):
        parser = argparse.ArgumentParser(
            description='Run LOP Evolutionary Algorithm experiments',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python src/main.py                    # Run experiments only
  python src/main.py --analyze          # Run experiments and auto-analyze
            """
        )
        
        parser.add_argument('--analyze', action='store_true',
                          help='Automatically run analysis after experiments complete')
        
        self.args = parser.parse_args()
    
    def _setup_paths(self):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(script_dir)
        instances_path = os.path.join(project_root, self.config.instances_path)
        results_path = os.path.join(project_root, self.config.results_path)
        return instances_path, results_path
    
    def build(self):
        instances_path, results_path = self._setup_paths()
        
        runner = ExperimentRunner(self.config, instances_path, results_path)
        analyzer = AnalysisRunner(results_path) if self.args.analyze else None
        
        return runner, analyzer