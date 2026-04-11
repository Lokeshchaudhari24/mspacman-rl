# Ms. Pac-Man Reinforcement Learning

A Deep Q-Network (DQN) agent trained to play Ms. Pac-Man using PyTorch and Gymnasium.

## Project Structure

Ms.PacMan/
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

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
autorom --accept-license

## Train

Baseline:
python src/train.py --episodes 300 --no-shaping --name exp1_baseline

With reward shaping:
python src/train.py --episodes 300 --life-penalty -500 --name exp4_both

## Evaluate

python src/evaluate.py --model results/models/exp4_both_best.pth --episodes 3

## Results
- Baseline DDQN avg score: ~800-1500
- Improved agent avg score: ~2000-3000

## Tech Stack
- Python 3.11
- PyTorch
- Gymnasium (Atari)
- NumPy
- Matplotlib