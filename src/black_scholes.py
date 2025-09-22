import numpy as np
from scipy.stats import norm

def d1(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d1 in the Black-Scholes formula.
    """
    return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d2 in the Black-Scholes formula.
    """
    return d1(S0, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price of a European Call option using the Black-Scholes formula.
    """
    d_1 = d1(S0, K, T, r, sigma)
    d_2 = d_1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)

def put_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price of a European Put option using the Black-Scholes formula.
    """
    d_1 = d1(S0, K, T, r, sigma)
    d_2 = d_1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d_2) - S0 * norm.cdf(-d_1)

if __name__ == "__main__":
    # Example usage
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    print("European Call Price:", call_price(S0, K, T, r, sigma))
    print("European Put Price:", put_price(S0, K, T, r, sigma))