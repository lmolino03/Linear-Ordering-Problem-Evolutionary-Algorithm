"""
Launcher for experiments and analysis orchestration.
"""


class Launcher:
    
    def __init__(self, runner, analyzer=None):
        self.runner = runner
        self.analyzer = analyzer
    
    def run(self):
        self.runner.run_all_experiments()
        self.runner.save_results()
        
        if self.analyzer:
            self.analyzer.run()
