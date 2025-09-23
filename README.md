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

---
## 📂 Project Structure

```
option_pricing_project/
├── data/                         # Dataset
│   └── synthetic_qmc.csv         # Generated training/validation dataset
│
├── src/                          # Python source code
│   ├── monte_carlo.py            # MC & QMC simulation functions
│   ├── finite_difference.py      # Explicit, implicit, CN schemes
│   ├── black_scholes.py          # Closed-form formulas
│   ├── dataset.py                # Data generation + CSV export
│   ├── surrogate_model.py        # PyTorch model + training loop
│   └── evaluation.py             # Convergence, runtime, error plots
│
├── results/                      # Generated plots & reports
│   ├── convergence_plot.png
│   ├── efficiency_comparison.png
│   ├── fd_convergence.png
│   └── evaluation_results.csv  
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
* `numpy`, `scipy`, `pandas`, `matplotlib`
* `torch`, `scikit-learn`
---

## ▶️ Usage

Run the full pipeline end-to-end:

```bash
python3 main.py
```
This will:
1. Generate QMC training data and save to `data/synthetic_qmc.csv`.
2. Train the surrogate neural network.
3. Run MC, QMC, FD, and surrogate pricing comparisons.
4. Save benchmark plots and results into `results/`.

You can also run individual components:
* **QMC dataset generation**
```bash
python3 src/dataset.py
```
* **Finite difference test**
```bash
python3 src/finite_difference.py
```

* **Surrogate model training**
```bash
python3 src/surrogate_model.py
```

## 📊 Outputs
All results are stored in the `results/` folder:
* `convergence_plot.png` - MC vs QMC convergence (error vs samples).
* `efficiency_comparison.png` - Runtime vs accuracy (all methods).
* `fd_convergence.png` - Benchmark convergence/runtime data.
* `evaluation_results.csv` - Benchmark convergence/runtime data.

## Results
### Monte Carlo vs Quasi-Monte Carlo
* **MC**: Converges at the theoretical rate $O(1/\sqrt{N})$.
* **QMC (Sobol)**: Consistently more accurate at the same sample size; variance reduced substantially.
* *Observation:* For small $N$, QMC outperforms MC by more than an order of magnitude in error.

### Finite Difference Methods
* **Explicit**: Stable only under CFL condition; error decreases slowly, consistent with first-order accuracy.
* **Implicit**: Unconditionally stable, also first-order accurate; produced values very close to analytic Black-Scholes at moderate grids.
* **Crank-Nicolson**: Achieved second-order convergence as grid resolution increased; ultimately the most accurate FD scheme.
* *Observation:* At coarse grids implicit sometimes matched Black-Scholes better (due to error cancellation), but CN dominated as resolution refined.

### Machine Learning Surrogate
* Neural Network surrogate trained on QMC dataset generalised well.
* Achieved very low relative error (<0.5%) compared to Black-Scholes.
* **Key advantage**: Once trained, inference time is effectively instantaneous, making it suitable for large-scale or real-time pricing.

### Efficiency Comparison
* **MC**: Slowest convergence, high variance.
* **QMC**: Faster convergence, competitive runtime.
* **FD**: Much faster than simulation, especially CN at fine grids.
* **ML Surrogate**: Initial training cost, but then outperforms all other methods in runtime.
* *Efficiency Frontier:* Surrogate + CN form the most practical approaches depending on whether training is pre-computed.
