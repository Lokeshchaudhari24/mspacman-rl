import numpy as np
import matplotlib.pyplot as plt
import os

# ── Experiment definitions ───────────────────────────────────
EXPERIMENTS = [
    {
        'name':  'exp1_baseline',
        'label': 'Baseline DDQN',
        'color': '#378ADD'
    },
    {
        'name':  'exp4_both',
        'label': 'DDQN + Life Penalty',
        'color': '#E24B4A'
    },
]

# ── Smooth rewards ───────────────────────────────────────────
def smooth(rewards, window=20):
    """Rolling average to make the chart easier to read"""
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window)
        smoothed.append(np.mean(rewards[start:i+1]))
    return smoothed

# ── Plot ─────────────────────────────────────────────────────
def compare():
    plt.figure(figsize=(12, 6))

    for exp in EXPERIMENTS:
        path = f'../results/plots/{exp["name"]}_rewards.npy'

        if not os.path.exists(path):
            print(f'Warning: {path} not found — skipping {exp["label"]}')
            continue

        rewards  = np.load(path)
        smoothed = smooth(rewards)

        # Raw rewards (faint)
        plt.plot(rewards,  alpha=0.15, color=exp['color'])

        # Smoothed rewards (bold)
        plt.plot(smoothed, alpha=0.9,  color=exp['color'],
                 linewidth=2.5, label=exp['label'])

    plt.xlabel('Episode',  fontsize=12)
    plt.ylabel('Score',    fontsize=12)
    plt.title('Ms. Pac-Man — DDQN Experiment Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out = '../results/plots/comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Comparison plot saved to {out}')

if __name__ == '__main__':
    compare()