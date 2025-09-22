import pandas as pd
from src.dataset import generate_qmc_dataset
from src.surrogate_model import train_surrogate
from src.evaluation import (
    evaluate_methods,
    plot_mc,
    plot_fd,
    plot_efficiency,
    export_results,
    export_surrogate_surface
)

def main():
    print("=== Step 1: Generate synthetic dataset with QMC ===")
    df_data = generate_qmc_dataset(n_samples=5000, N_qmc=2000, option="call")
    df_data.to_csv("data/synthetic_qmc.csv", index=False)
    print("Dataset saved to data/synthetic_qmc.csv")

    print("\n=== Step 2: Train ML Surrogate Model ===")
    model, x_scaler, y_scaler = train_surrogate(csv_path="data/synthetic_qmc.csv",
                                    n_epochs=100)
    print("Surrogate model trained successfully.")

    print("\n=== Step 3: Evaluate Methods (MC, QMC, FD, ML) ===")
    df_results, true_price = evaluate_methods(S0=200, K=100, T=1.0,
                                              r=0.05, sigma=0.6)
    # Export to CSV for Tableau
    export_results(df_results, true_price)

    # Export surrogate pricing surface (for Tableau beatmaps)
    export_surrogate_surface(model, x_scaler, y_scaler, T=0.1, r=0.05, sigma=0.2)
    # Plot comparisons
    plot_mc(df_results)
    plot_fd(df_results)
    plot_efficiency(df_results)

if __name__ == "__main__":
    main()