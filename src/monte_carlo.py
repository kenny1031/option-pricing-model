import numpy as np
from scipy.stats import norm, qmc
from .black_scholes import call_price, put_price

def mc_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 100000,
    option: str = "call"
) -> float:
    """
    Monte Carlo estimator for European option pricing.
    """
    # Generate standard normals
    Z = np.random.normal(0, 1, N)

    # Simulate terminal stock prices
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Compute payoff
    if option == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # Discounted expectation
    return np.exp(-r * T) * np.mean(payoff)


def qmc_option_price(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    N: int = 100000,
    option: str = "call"
) -> float:
    """
    Quasi-Monte Carlo estimator using Sobol sequences.
    """
    # Generate Sobol samples in 1D
    sampler = qmc.Sobol(d=1, scramble=True)
    U = sampler.random(N).flatten()

    # Transform to standard normals
    Z = norm.ppf(U)

    # Simulate terminal stock prices
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    # Compute payoff
    if option == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # Discounted expectation
    return np.exp(-r * T) * np.mean(payoff)


if __name__ == "__main__":
    # Example parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    N = 100000

    # Closed form
    bs_call = call_price(S0, K, T, r, sigma)

    # MC & QMC estimates
    mc_call = mc_option_price(S0, K, T, r, sigma, N, option="call")
    qmc_call = qmc_option_price(S0, K, T, r, sigma, N, option="call")

    print(f"Black–Scholes Call (Analytic): {bs_call:.6f}")
    print(f"Monte Carlo Call (N={N}):      {mc_call:.6f}")
    print(f"Quasi-MC Call (N={N}):        {qmc_call:.6f}")