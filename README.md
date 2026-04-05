# Ms. Pac-Man Reinforcement Learning Agent 🎮

A Deep Reinforcement Learning agent that learns to play Ms. Pac-Man
using Double DQN (DDQN) with reward shaping.

## Project Structure
mspacman-rl/
├── src/
│   ├── agent.py       # DQN network, replay buffer, DDQN agent
│   ├── train.py       # Training loop with reward shaping
│   ├── evaluate.py    # Watch the trained agent play
│   └── compare.py     # Plot experiment comparison
├── results/
│   ├── models/        # Saved model weights (.pth)
│   └── plots/         # Training curves + comparison chart
├── requirements.txt
└── README.md

## Setup
```bash
# Clone the repo
git clone https://github.com/Lokeshchaudhari24/mspacman-rl.git
cd mspacman-rl

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download Atari ROMs
python -c "from AutoROM import AutoROM; AutoROM(accept_license=True, source_file=None, install_dir=None, quiet=False)"
```

## Training
```bash
cd src

# Experiment 1 — Baseline DDQN
python train.py --episodes 300 --no-shaping --name exp1_baseline

# Experiment 2 — DDQN with life penalty
python train.py --episodes 300 --life-penalty -500 --name exp4_both
```

## Watch the Agent Play
```bash
cd src
python evaluate.py --model ../results/models/exp4_both_best.pth --episodes 3
```

## Compare Experiments
```bash
cd src
python compare.py
```

## Results

| Experiment | Description | Avg Score |
|---|---|---|
| Baseline DDQN | No reward shaping | ~800 |
| DDQN + Life Penalty | Penalizes ghost deaths | ~1500 |

## Key Concepts

- **DDQN** — Double DQN reduces overestimation of Q-values
- **Replay Buffer** — Agent learns from past experiences
- **Reward Shaping** — Extra signals to teach better behavior
- **Frame Stacking** — Agent sees 4 frames at once to understand motion

## References

- [DeepMind DQN Paper](https://www.nature.com/articles/nature14236)
- [Gymnasium Atari](https://gymnasium.farama.org/environments/atari/)