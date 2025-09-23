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
from typing import Tuple

def relative_error(true: int | float, estimated: int | float) -> float:
    return np.abs(true - estimated) / np.abs(true)


def evaluate_methods(
    S0: float | int=100,
    K: float | int=100,
    T: float=1.0,
    r: float=0.05,
    sigma: float=0.2
) -> Tuple[pd.DataFrame, float]:
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
    grid_sizes = [50, 100, 200, 400, 600]
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
    model, x_scaler, y_scaler = train_surrogate(n_epochs=100, plot=True)
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
def plot_mc(df: pd.DataFrame, save_path: str | None=None) -> None:
    # Convergence plot of MC vs QMC
    subset = df[df["method"].isin(["MC", "QMC"])]
    plt.figure()
    for method in ["MC", "QMC"]:
        sub = subset[subset["method"] == method]
        plt.loglog(sub["N"], sub["rel_error"], label=method)
    plt.xlabel("$N$ (samples)")
    plt.ylabel("Relative Error")
    plt.title("Convergence of MC vs QMC")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to {save_path}")
    else:
        plt.show()


def plot_efficiency(df: pd.DataFrame, save_path: str | None=None) -> None:
    """
    Plot efficiency curve.
    """
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to {save_path}")
    else:
        plt.show()


def test_fd_convergence(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Plot FD convergence from evaluation DataFrame.
    """
    fd_subset = df[df["method"].isin(["explicit", "implicit", "CN"])]

    plt.figure()
    for scheme in ["explicit", "implicit", "CN"]:
        sub = fd_subset[fd_subset["method"] == scheme]
        if sub.empty:
            continue
        M_vals = sub["N"].astype(float)
        errors = sub["rel_error"].astype(float)
        plt.loglog(M_vals, errors, marker="o", label=scheme)

    # Reference lines
    M_ref = np.array(sorted(fd_subset["N"].dropna().unique()))
    if len(M_ref) > 0:
        plt.loglog(M_ref, 1 / M_ref, "--", color="gray", label="$O(1)$")
        plt.loglog(M_ref, 1 / (M_ref ** 2), "--", color="black", label="$O(1/N^2)$")

    plt.xlabel("Grid size $M=N$")
    plt.ylabel("Relative Error")
    plt.title("FD Convergence Study (Error vs Grid Size)")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved FD convergence plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    df, true_price = evaluate_methods()
    print(df)

    plot_mc(df)  # MC vs QMC plots
    test_fd_convergence(df)  # FD plots