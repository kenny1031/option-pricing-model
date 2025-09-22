import os
from src.evaluation import evaluate_methods, plot_mc, plot_efficiency, test_fd_convergence

def main():
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Step 1: Evaluate methods
    df, true_price = evaluate_methods()
    df.to_csv("results/evaluation_results.csv", index=False)
    print("[INFO] Results saved to results/evaluation_results.csv")

    # Steo 2: Plots
    # MC vs QMC
    plot_mc(df, save_path="results/convergence_plot.png")

    # Efficiency plot
    plot_efficiency(df, save_path="results/efficiency_comparison.png")

    # FD Convergence
    test_fd_convergence(save_path="results/fd_convergence.png")

if __name__ == "__main__":
    main()