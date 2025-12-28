"""LOP Evolutionary Algorithm - Main Entry Point"""
from config import get_config
from pipeline import Builder, Launcher

if __name__ == "__main__":
    config = get_config()
    builder = Builder(config)
    components = builder.build()
    launcher = Launcher(*components)
    launcher.run()
