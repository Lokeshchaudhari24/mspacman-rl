import numpy as np
import matplotlib.pyplot as plt
import os

def smooth(scores, window=10):
    return np.convolve(scores, np.ones(window)/window, mode='valid')

def compare():
    experiments = {
        "Baseline DDQN":     "../results/exp1_baseline_scores.npy",
        "Life Penalty":      "../results/exp4_both_scores.npy",
        "Stuck Fix":         "../results/exp6_final_scores.npy",
    }

    plt.figure(figsize=(12, 6))

    for label, path in experiments.items():
        if os.path.exists(path):
            scores = np.load(path)
            plt.plot(smooth(scores), label=label, linewidth=2)
        else:
            print(f"Skipping {label} — file not found: {path}")

    plt.title("Experiment Comparison — Ms. Pac-Man RL", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Score (smoothed over 10 episodes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("../results/plots", exist_ok=True)
    plt.savefig("../results/plots/comparison.png", dpi=150)
    plt.close()
    print("Comparison plot saved to results/plots/comparison.png")

if __name__ == "__main__":
    compare()