import numpy as np
import pandas as pd
from scipy.stats import qmc
from .monte_carlo import qmc_option_price


def generate_qmc_dataset(
    n_samples: int=10000,
    N_qmc: int=5000,
    option: str="call",
    seed: int=42
)->pd.DataFrame:
    """
    Generate dataset of option prices using QMC sampling for parameters
    and QMC for pricing.

    Parameters:
    -----------
    n_samples : int
        Number of parameter combinations (rows in dataset).
    Nqmc : int
        Number of Sobol samples for each option price estimation.
    option : str
        "call" or "put".
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    df : pd.DataFrame
        Data Frame with columns [S0, K, T, r, sigma, price].
    """
    sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
    X = sampler.random(n_samples)

    # Parameter ranges (low, high)
    param_lows = [50, 50, 0.1, 0.01, 0.1]
    param_highs = [150, 150, 2.0, 0.1, 0.5]

    # Scale Sobol points to parameter ranges (all at once)
    params = qmc.scale(X, param_lows, param_highs)

    # Extract parameters
    S0, K, T, r, sigma = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]

    # Compute option prices using QMC
    prices = []
    for i in range(n_samples):
        price = qmc_option_price(
            S0[i], K[i], T[i], r[i], sigma[i],
            N=N_qmc,
            option=option
        )
        prices.append(price)

    # Build DataFrame
    df = pd.DataFrame({
        "S0": S0,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "price": prices
    })
    return df

if __name__ == "__main__":
    # Generate and save dataset
    df = generate_qmc_dataset(n_samples=5000, N_qmc=2000, option="call")
    df.to_csv("../data/synthetic_qmc.csv", index=False)
    print("Dataset saved to ../data/synthetic_qmc.csv")
    print(df.head())