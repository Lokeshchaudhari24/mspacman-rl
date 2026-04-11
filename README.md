# Ms. Pac-Man Reinforcement Learning 🎮

A Deep Q-Network (DQN) agent trained to play Ms. Pac-Man using PyTorch and Gymnasium.

## Project Structure

mspacman-rl/
├── src/
│   ├── agent.py        # DDQN agent, replay buffer, neural network
│   ├── train.py        # Training loop with reward shaping
│   ├── evaluate.py     # Visual demo of trained agent
│   └── compare.py      # Comparison plot of experiments
├── results/
│   ├── models/         # Saved model checkpoints
│   └── plots/          # Training curves and comparison plots
├── requirements.txt
└── README.md

## Setup

```bash
git clone https://github.com/Lokeshchaudhari24/mspacman-rl.git
cd mspacman-rl
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Watch the Agent Play

```bash
cd src
python evaluate.py --model ../results/models/exp6_final_best.pth --episodes 3
```

## Train from Scratch

```bash
cd src

# Baseline DDQN
python train.py --episodes 300 --no-shaping --name exp1_baseline

# Improved DDQN with life penalty
python train.py --episodes 300 --life-penalty -500 --name exp4_both
```

## Compare Experiments

```bash
cd src
python compare.py
```

## Experiments

| Experiment | Description | Avg Score |
|---|---|---|
| exp1_baseline | Baseline DDQN, no reward shaping | ~582 |
| exp4_both | DDQN + life penalty -500 | ~647 |
| exp6_final | Best tuned version | Best results |

## Key Concepts

- **DDQN** — Two networks, one picks action, one judges it — reduces overestimation
- **Replay Buffer** — Stores 50,000 past experiences for stable learning
- **Reward Shaping** — -500 penalty for dying teaches ghost avoidance
- **Frame Stacking** — 4 frames stacked so agent understands motion
- **Epsilon Decay** — Starts random (1.0), becomes smart (0.1) over time

## Tech Stack

- Python 3.13
- PyTorch — Neural network
- Gymnasium — Atari environment
- Matplotlib — Training plots
- Git + GitHub — Version control

## Results

Training curve and comparison plots are saved in `results/plots/`.
Demo video available as `Recording.mp4`.

## References

- [DeepMind DQN Paper](https://www.nature.com/articles/nature14236)
- [Gymnasium Atari Docs](https://gymnasium.farama.org/environments/atari/)