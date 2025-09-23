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
The project benchmarks analytical, simulation, PDE, and ML approaches for European call option pricing.

### Monte Carlo vs Quasi–Monte Carlo
* MC: Converges at the theoretical $O(1/\sqrt{N})$ rate; suffers from high variance.
* QMC (Sobol): Substantially more accurate than MC at the same sample size, reducing variance by more than an order of magnitude for small $N$.
* Key finding: QMC consistently delivered the highest accuracy among stochastic simulation methods.

### Finite Difference (FD) Methods
* Explicit: Conditionally stable; slow convergence; consistent with first-order accuracy.
* Implicit: Unconditionally stable, with results often close to Black–Scholes at moderate grids.
* Crank–Nicolson (CN): Achieved second-order convergence and was the most accurate PDE scheme at refined grids.
* Observation: At coarse grids, implicit sometimes matched Black–Scholes better due to error cancellation, but CN dominated at higher resolution.

### Machine Learning Surrogate
* Neural network surrogate trained on QMC-generated data, achieving <0.5% relative error compared to Black–Scholes.
* Strength: Once trained, inference is effectively instantaneous, making it suitable for real-time or large-scale option pricing.
* Trade-off: Training required more time than simulation, but runtime efficiency surpassed all other methods once deployed.

### Efficiency Comparison
* MC: Slowest convergence and highest variance.
* QMC: Faster convergence with competitive runtimes.
* FD: More efficient than simulations, especially Crank–Nicolson on fine grids.
* ML Surrogate: Highest upfront training cost, but inference is fastest in production.
* Efficiency Frontier: A combination of Surrogate + CN offers the best practical trade-off depending on whether training is precomputed.
