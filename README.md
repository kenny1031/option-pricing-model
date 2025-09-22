# option-pricing-model

```
option_pricing_project/
├── data/                         # Datasets (synthetic + optional market data)
│   ├── synthetic_qmc.csv         # Generated training/validation dataset
│   ├── results_mc_qmc.csv        # Benchmark convergence/runtime results
│   └── market_data.csv           # (Optional) Real option chain data from Yahoo/CBOE
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_qmc_data_generation.ipynb
│   ├── 02_finite_difference.ipynb
│   ├── 03_ml_training.ipynb
│   └── 04_tableau_export.ipynb
│
├── src/                          # Python source code
│   ├── __init__.py
│   ├── monte_carlo.py            # MC & QMC simulation functions
│   ├── finite_difference.py      # Explicit, implicit, CN schemes
│   ├── black_scholes.py          # Closed-form formulas
│   ├── dataset.py                # Data generation + CSV export
│   ├── surrogate_model.py        # PyTorch model + training loop
│   ├── evaluation.py             # Convergence, runtime, error plots
│   └── utils.py                  # Helper functions
│
├── results/                      # Generated plots & reports
│   ├── convergence_plot.png
│   ├── efficiency_comparison.png
│   └── tableau_ready.csv         # Export file for Tableau dashboards
│
├── requirements.txt              # Dependencies
├── README.md                     # Project overview, setup & usage
└── main.py                       # Entry point (run full pipeline end-to-end)
```