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
def plot_mc(df, save_path=None):
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


def plot_efficiency(df, save_path=None):
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


def test_fd_convergence(S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
                        grid_sizes=[50, 100, 200, 400, 600],
                        save_path=None):
    true_price = call_price(S0, K, T, r, sigma)
    results = []

    for scheme in ["explicit", "implicit", "CN"]:
        print(f"\n=== {scheme.upper()} ===")
        prev_err = None
        for M in grid_sizes:
            fd_price = finite_difference_call(S0, K, T, r, sigma, M=M, N=M, scheme=scheme)
            err = relative_error(true_price, fd_price)
            results.append({"scheme": scheme, "M": M, "price": fd_price, "error": err})

            # Convergence rate
            if prev_err is not None:
                rate = np.log(prev_err/err) / np.log(2)
                print(f"M={M:4d}, Price={fd_price:.6f}, Error={err:.3e}, Rate={rate:.2f}")
            else:
                print(f"M={M:4d}, Price={fd_price:.6f}, Error={err:.3e}")
            prev_err = err

    # --- Plot errors with reference slopes ---
    plt.figure()
    for scheme in ["explicit", "implicit", "CN"]:
        sub = [r for r in results if r["scheme"] == scheme]
        M_vals = [r["M"] for r in sub]
        errors = [r["error"] for r in sub]
        plt.loglog(M_vals, errors, label=scheme)

    # Reference lines
    M_ref = np.array(grid_sizes)
    plt.loglog(M_ref, 1/M_ref, marker="*", color="gray", label="$O(1)$")     # first-order
    plt.loglog(M_ref, 1/(M_ref**2), marker="*", color="black", label="$O(1/N^2)$")  # second-order

    plt.xlabel("Grid size $M=N$")
    plt.ylabel("Relative Error")
    plt.title("FD Convergence Study (Error vs Grid Size)")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    df, true_price = evaluate_methods()
    print(df)

    plot_mc(df)  # MC vs QMC plots
    test_fd_convergence()  # FD plots