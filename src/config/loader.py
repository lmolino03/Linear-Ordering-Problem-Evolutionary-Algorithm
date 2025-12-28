"""
Configuration loader for LOP Evolutionary Algorithm experiments
"""
import yaml
import os
from pathlib import Path


class Config:
    """Configuration class to load and access YAML configuration"""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to config.yaml. If None, uses default path.
        """
        if config_path is None:
            # Default path relative to this file
            config_dir = Path(__file__).parent
            config_path = config_dir / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self):
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # Dataset configuration
    @property
    def instances_path(self):
        return self._config['dataset']['instances_path']
    
    @property
    def results_path(self):
        return self._config['dataset']['results_path']
    
    # EA Parameters
    @property
    def population_sizes(self):
        return self._config['ea_parameters']['population_sizes']
    
    @property
    def max_evaluations(self):
        return self._config['ea_parameters']['max_evaluations']
    
    @property
    def num_runs(self):
        return self._config['ea_parameters']['num_runs']
    
    @property
    def random_seed(self):
        return self._config['ea_parameters']['random_seed']
    
    # Genetic operators parameters
    @property
    def mutation_rate(self):
        return self._config['genetic_operators']['mutation_rate']
    
    @property
    def crossover_rate(self):
        return self._config['genetic_operators']['crossover_rate']
    
    @property
    def tournament_size(self):
        return self._config['genetic_operators']['tournament_size']
    
    @property
    def elitism(self):
        return self._config['genetic_operators']['elitism']
    
    # Instance files
    @property
    def instances(self):
        return self._config['instances']
    
    # Known optimal values
    @property
    def known_optima(self):
        return self._config['known_optima']


# Global configuration instance
_config_instance = None

def get_config(config_path=None):
    """
    Get global configuration instance
    
    Args:
        config_path: Optional custom path to config.yaml
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


# For backward compatibility, provide direct access to config values
def load_config(config_path=None):
    """
    Load configuration and return config object
    
    Args:
        config_path: Optional path to config.yaml
        
    Returns:
        Config instance
    """
    return get_config(config_path)
