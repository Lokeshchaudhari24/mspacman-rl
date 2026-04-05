import gymnasium as gym
import numpy as np
import torch
import argparse
import time
from collections import deque
from agent import DDQNAgent

# ── Argument Parser ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',    type=str, required=True,
                    help='Path to saved model .pth file')
parser.add_argument('--episodes', type=int, default=3,
                    help='Number of episodes to watch')
args = parser.parse_args()

# ── Frame Preprocessing ──────────────────────────────────────
def preprocess(obs, frame_stack):
    """Same preprocessing as training"""
    if obs.ndim == 3:
        gray = np.mean(obs, axis=2).astype(np.uint8)
        gray = gray[::2, ::2]
        frame_stack.append(gray)
    return np.array(frame_stack)

# ── Evaluate ─────────────────────────────────────────────────
def evaluate():
    # Create environment with rendering so we can watch
    env = gym.make('ALE/MsPacman-v5', render_mode='human')
    n_actions  = env.action_space.n
    input_shape = (4, 105, 80)

    # Load trained agent
    agent = DDQNAgent(input_shape, n_actions)
    agent.load(args.model)
    agent.epsilon = 0.05   # almost no random moves — use what it learned

    print(f"\nWatching agent play for {args.episodes} episodes...")
    print("Close the game window to stop.\n")

    scores = []

    for episode in range(1, args.episodes + 1):
        obs, info  = env.reset()
        frame_stack = deque(maxlen=4)

        # Initialize frame stack
        gray = np.mean(obs, axis=2).astype(np.uint8)[::2, ::2]
        for _ in range(4):
            frame_stack.append(gray)

        state        = np.array(frame_stack)
        total_reward = 0

        while True:
            # Small delay so we can watch properly
            time.sleep(0.02)

            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done   = terminated or truncated

            next_state   = preprocess(obs, frame_stack)
            total_reward += reward
            state        = next_state

            if done:
                break

        scores.append(total_reward)
        print(f"Episode {episode} — Score: {total_reward:.0f}")

    print(f"\nAverage score: {np.mean(scores):.0f}")
    print(f"Best score:    {np.max(scores):.0f}")
    env.close()

if __name__ == '__main__':
    evaluate()