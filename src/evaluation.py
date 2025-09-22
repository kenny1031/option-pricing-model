import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from .black_scholes import call_price
from .monte_carlo import mc_option_price, qmc_option_price
from .finite_difference import finite_difference_call
from .surrogate_model import train_surrogate

def relative_error(true, estimated):
    return np.abs(true - estimated) / np.abs(true)

def evaluate_methods(S0=100, K=100, T=1.0, r=0.05, sigma=0.2):
    results = []

    # True analytic price (Black-Scholes)
    true_price = call_price(S0, K, T, r, sigma)

    # MC vs QMC convergence
    N_vals = [100, 500, 1000, 5000, 10000, 20000]
    for N in N_vals:
        # MC
        start = time.time()
        mc_price = mc_option_price(S0, K, T, r, sigma, N, option="call")
        mc_time = time.time() - start

        # QMC
        start = time.time()
        qmc_price = qmc_option_price(S0, K, T, r, sigma, N, option="call")
        qmc_time = time.time() - start

        results.append({
            "N": N,
            "method": "MC",
            "price": mc_price,
            "rel_error": relative_error(true_price, mc_price),
            "time": mc_time
        })
        results.append({
            "N": N,
            "method": "QMC",
            "price": qmc_price,
            "rel_error": relative_error(true_price, qmc_price),
            "time": qmc_time
        })

    # Finite Difference
    grid_sizes = [50, 100, 200, 400]
    schemes = ["explicit", "implicit", "CN"]
    for scheme in schemes:
        for m in grid_sizes:
            start = time.time()
            fd_price = finite_difference_call(S0, K, T, r, sigma,
                                              M=m, N=m, scheme=scheme)
            fd_time = time.time() - start
            results.append({
                "N": m,
                "method": scheme,
                "price": fd_price,
                "rel_error": relative_error(true_price, fd_price),
                "time": fd_time
            })

    # Surrogate Model
    nn_start = time.time()
    model, x_scaler, y_scaler = train_surrogate(n_epochs=100)
    x = np.array([[S0, K, T, r, sigma]])
    x_scaled = x_scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(x_tensor).numpy()
        pred = y_scaler.inverse_transform(pred_scaled)[0, 0]

    nn_end = time.time()
    results.append({
        "N": None,
        "method": "ML surrogate",
        "price": pred,
        "rel_error": relative_error(true_price, pred),
        "time": nn_end - nn_start
    })
    df = pd.DataFrame(results)
    return df, true_price

# Visualisation of results
def plot_mc(df):
    # Convergence plot of MC vs QMC
    subset = df[df["method"].isin(["MC", "QMC"])]
    plt.figure()
    for method in ["MC", "QMC"]:
        sub = subset[subset["method"] == method]
        plt.loglog(sub["N"], sub["rel_error"], marker="o", label=method)
    plt.xlabel("$N$ (samples)")
    plt.ylabel("Relative Error")
    plt.title("Convergence of MC vs QMC")
    plt.legend()
    plt.show()

def plot_fd(df):
    fd_subset = df[df["method"].isin(["explicit", "implicit", "CN"])]
    # Convergence plot: error vs grid size
    plt.figure()
    for scheme in ["explicit", "implicit", "CN"]:
        sub = fd_subset[fd_subset["method"] == scheme]
        plt.loglog(sub["N"], sub["rel_error"], marker="o", label=scheme)
    plt.xlabel("Grid size ($M=N$)")
    plt.ylabel("Relative Error")
    plt.title("Convergence of Finite Difference Scheme")
    plt.legend()
    plt.show()

def plot_efficiency(df):
    methods = ["MC", "QMC", "explicit", "implicit", "CN", "ML surrogate"]
    plt.figure()
    for method in methods:
        sub = df[df["method"] == method]
        plt.scatter(sub["time"], sub["rel_error"], label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Runtime (s, log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.title("Efficiency Comparison of all methods")
    plt.legend()
    plt.show()

def export_results(df, true_price, out_path="results/tableau_ready.csv"):
    """
    Export evaluation results to CSV for Tableau dashboard.
    Includes method, N/grad size, runtime, relative error, and price
    :param df:
    :param true_price:
    :param out_path:
    :return:
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Add reference true price column
    df["true_price"] = true_price

    # Save to CSV
    df.to_csv(out_path, index=False)
    print(f"[INFO] Results exported to {out_path}")

def export_surrogate_surface(model, x_scaler, y_scaler,
                             S0_range=(50,150), K_range=(50,150),
                             T=1.0, r=0.05, sigma=0.2,
                             step=5, out_path="results/surrogate_surface.csv"):
    """
    Generate a surrogate pricing surface over (S0, K) grid with fixed T, r, sigma.
    Export to CSV for Tableau heatmap/surface plot.
    """
    S0_vals = np.arange(S0_range[0], S0_range[1]+step, step)
    K_vals  = np.arange(K_range[0], K_range[1]+step, step)

    rows = []
    for S0 in S0_vals:
        for K in K_vals:
            x = np.array([[S0, K, T, r, sigma]])
            x_scaled = x_scaler.transform(x)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

            with torch.no_grad():
                pred_scaled = model(x_tensor).numpy()
                pred = y_scaler.inverse_transform(pred_scaled)[0,0]

            rows.append({"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "price": pred})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Surrogate pricing surface exported to {out_path}")
    return df

if __name__ == "__main__":
    df, true_price = evaluate_methods()
    print(df)

    plot_mc(df)  # MC vs QMC plots
    plot_fd(df)  # FD plots