import numpy as np
from .black_scholes import call_price, put_price

def finite_difference_call(
    S0: int | float,
    K: int | float,
    T: int | float,
    r: float,
    sigma: float,
    M: int=200,
    N: int=200,
    scheme: str="CN"
) -> float:
    """
    Finite Difference (Explicit, Implicit, Crank–Nicolson) for European Call
    with log-transformation of Black–Scholes PDE.

    Parameters:
    -----------
    S0 : float   - initial stock price
    K  : float   - strike
    T  : float   - maturity
    r  : float   - risk-free rate
    sigma : float - volatility
    M : int      - number of space steps
    N : int      - number of time steps
    scheme : str - "explicit", "implicit", or "CN"
    """
    # Grid in log-price space
    S_max = 4 * K
    x_min, x_max = np.log(0.01), np.log(S_max / K)
    dx = (x_max - x_min) / M
    if scheme == "explicit":
        dt = 0.9 * dx ** 2 / (sigma ** 2)  # CFL condition
        N = int(T / dt)
    else:
        dt = T / N
    x = np.linspace(x_min, x_max, M + 1)
    S = K * np.exp(x)

    # Payoff at maturity (scaled by K)
    u = np.maximum(S - K, 0) / K

    # Coefficients
    alpha = 0.5 * sigma ** 2 / dx ** 2
    beta = (r - 0.5 * sigma ** 2) / (2 * dx)
    gamma = -r

    # Construct tridiagonal A matrix
    A = np.zeros((M - 1, M - 1))
    for j in range(M - 1):
        A[j, j] = -2 * alpha + gamma
        if j > 0:
            A[j, j - 1] = alpha - beta
        if j < M - 2:
            A[j, j + 1] = alpha + beta

    I = np.eye(M - 1)

    # Time-stepping
    for n in range(N):
        t = T - n * dt
        # Boundary conditions
        u[0] = 0  # at S=0
        u[-1] = (S_max / K) - np.exp(-r * t)  # scaled by K
        if scheme == "explicit":
            u[1:M] = u[1:M] + dt * (A @ u[1:M])
        elif scheme == "implicit":
            u[1:M] = np.linalg.solve(I - dt * A, u[1:M])
        elif scheme == "CN":
            u[1:M] = np.linalg.solve(I - 0.5 * dt * A, (I + 0.5 * dt * A) @ u[1:M])

    # Interpolation to get price at S0
    return np.interp(S0, S, K * u)



if __name__ == "__main__":
    # Example usage
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    black_scholes = call_price(S0, K, T, r, sigma)
    price_CN = finite_difference_call(S0, K, T, r, sigma, scheme="CN")
    price_explicit = finite_difference_call(S0, K, T, r, sigma, scheme="explicit")
    price_implicit = finite_difference_call(S0, K, T, r, sigma, scheme="implicit")
    print("Black-Scholes:", black_scholes)
    print("FD Explicit:", price_explicit)
    print("FD Implicit:", price_implicit)
    print("FD CN:", price_CN)
