# Option Pricing with Numerical Methods and Machine Learning

This project implements and compares several approaches for pricing European call options under the Black–Scholes framework:

- **Analytical solution** (closed-form Black–Scholes formula)
- **Monte Carlo (MC)** simulation
- **Quasi–Monte Carlo (QMC)** with Sobol sequences
- **Finite Difference (FD)** PDE solvers:
  - Explicit
  - Implicit
  - Crank–Nicolson
- **Machine Learning Surrogate Model** (PyTorch neural network trained on QMC-generated data)

Results are exported to Tableau-ready CSVs for interactive dashboards and efficiency visualisation.

---

## 📚 Mathematical Background

### Black–Scholes PDE
The price $V(S,t)$ of a European call satisfies:
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0,
$$
with terminal condition $V(S,T) = \max(S-K,0)$.

### Analytical Solution
Closed-form formula for a European call:
$$
C(S_0,K,T,r,\sigma) = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2),
$$
where
$$
d_{1,2} = \frac{\ln(S_0/K) + (r \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}.
$$

### Monte Carlo (MC) & Quasi-Monte Carlo (QMC)
- **MC** simulates asset paths under risk-neutral dynamics:
$$
S_T = S_0 \exp\Big((r - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\Big), \quad Z \sim \mathcal{N}(0,1).
$$
- **QMC** replaces pseudo-random $Z$ with low-discrepancy Sobol points, achieving faster convergence.

### Finite Difference (FD) Methods
After log-transformation ($x = \ln(S/K)$), the PDE is discretised:
- **Explicit (Forward Euler):** conditionally stable, $\Delta t \leq c \Delta x^2$.
- **Implicit (Backward Euler):** unconditionally stable, first-order accurate in time.
- **Crank–Nicolson (CN):** unconditionally stable, second-order accurate in time.

### Surrogate Model
- Neural network trained on QMC-generated datasets $(S_0, K, T, r, \sigma) \mapsto \text{price}$.
- Architecture: 3 hidden layers (128–128–64, ReLU).
- Both inputs $X$ and outputs $y$ are standardised during training.
- Predictions are inverse-transformed for reporting.

---

## 📂 Project Structure

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
---

## ⚙️ Installation

```bash
git clone <your-repo>
cd option_pricing_project
pip install -r requirements.txt
```
Dependencies include:
* `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
* `torch`, `scikit-learn`
* `finance`
* `tabpy`