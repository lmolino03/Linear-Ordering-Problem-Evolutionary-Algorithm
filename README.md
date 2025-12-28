# Linear Ordering Problem - Evolutionary Algorithm

Implementation of an Evolutionary Algorithm for solving the Linear Ordering Problem using permutation-based genetic operators.

## Project Structure

```
.
├── src/
│   ├── main.py              # Entry point
│   ├── pipeline/            # Execution orchestration
│   │   ├── builder.py
│   │   ├── launcher.py
│   │   └── runners/
│   ├── core/                # Algorithm implementation
│   │   ├── evolutionary_algorithm.py
│   │   ├── lop_problem.py
│   │   └── permutation_operators.py
│   ├── analysis/            # Statistical analysis and visualization
│   └── config/              # Configuration management
├── dataset/                 # Problem instances (.mat format)
└── requirements.txt
```

## Requirements

- Python 3.8+
- Dependencies: NumPy, Pandas, Matplotlib, Seaborn, SciPy, PyYAML

Install via:
```bash
pip install -r requirements.txt
```

## Usage

Run experiments:
```bash
python src/main.py
```

Run with automatic analysis:
```bash
python src/main.py --analyze
```

## Configuration

Modify `src/config/config.yaml` to adjust:
- Population sizes
- Number of evaluations
- Number of independent runs
- Genetic operator parameters
- Random seed for reproducibility

## Algorithm

**Representation:** Permutation encoding  
**Crossover:** PMX (Partially Mapped Crossover)  
**Mutation:** Swap mutation  
**Selection:** Tournament selection  
**Replacement:** Generational with elitism  

## Dataset Format

Problem instances must be in `.mat` format:
- Line 1: Description (optional)
- Line 2: Matrix dimension n
- Remaining lines: Matrix coefficients (space-separated, row-wise)

Place instances in the `dataset/` directory and reference them in `config.yaml`.

## Results

Experimental results are saved in the `results/` directory:
- `detailed_results_*.json` - Complete experimental data
- `summary_results_*.csv` - Statistical summary
- `analysis/` - Statistical tests
- `figures/` - Convergence plots and visualizations

## Statistical Analysis

The implementation includes:
- Kruskal-Wallis H-test for population size comparison
- Pairwise Mann-Whitney U tests with Bonferroni correction
- Convergence analysis
- Gap analysis from known optimal solutions

## License

MIT License. See LICENSE file for details.
