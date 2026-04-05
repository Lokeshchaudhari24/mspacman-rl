import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from collections import deque
from agent import DDQNAgent

# ── Argument Parser ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--episodes',     type=int,   default=300)
parser.add_argument('--name',         type=str,   default='experiment')
parser.add_argument('--no-shaping',   action='store_true')
parser.add_argument('--life-penalty', type=float, default=0.0)
args = parser.parse_args()

# ── Paths ────────────────────────────────────────────────────
os.makedirs('../results/models', exist_ok=True)
os.makedirs('../results/plots',  exist_ok=True)

# ── Reward Shaping ───────────────────────────────────────────
def shape_reward(reward, info, prev_info, done, args):
    """
    Make the reward signal smarter.
    The default reward is just points scored.
    We add extra signals to teach the agent better behavior.
    """
    shaped = reward

    if not args.no_shaping:
        # Penalize losing a life — teaches ghost avoidance
        if args.life_penalty != 0.0:
            prev_lives = prev_info.get('lives', 3)
            curr_lives = info.get('lives', 3)
            if curr_lives < prev_lives:
                shaped += args.life_penalty

    return shaped

# ── Frame Preprocessing ──────────────────────────────────────
def preprocess(obs, frame_stack):
    """Convert game screen to grayscale and stack frames"""
    if obs.ndim == 3:
        # Convert RGB to grayscale
        gray = np.mean(obs, axis=2).astype(np.uint8)
        gray = gray[::2, ::2]   # downsample to 105x80
        frame_stack.append(gray)
    return np.array(frame_stack)

# ── Main Training Loop ───────────────────────────────────────
def train():
    # Create the Ms. Pac-Man environment
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    n_actions = env.action_space.n
    print(f"Actions available: {n_actions}")

    # Frame stack — agent sees 4 frames at once (like motion blur)
    frame_stack = deque(maxlen=4)
    input_shape = (4, 105, 80)

    agent = DDQNAgent(input_shape, n_actions)

    # Tracking
    all_rewards   = []
    avg_rewards   = []
    best_avg      = -float('inf')
    recent        = deque(maxlen=20)

    print(f"\nStarting training: {args.name}")
    print(f"Episodes: {args.episodes}")
    print(f"Reward shaping: {'OFF' if args.no_shaping else 'ON'}")
    print(f"Life penalty: {args.life_penalty}")
    print("-" * 50)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        prev_info = info.copy()

        # Initialize frame stack with first frame
        gray = np.mean(obs, axis=2).astype(np.uint8)[::2, ::2]
        for _ in range(4):
            frame_stack.append(gray)

        state       = np.array(frame_stack)
        total_reward = 0
        total_loss   = 0
        steps        = 0

        while True:
            action          = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done            = terminated or truncated

            # Shape the reward
            shaped_reward   = shape_reward(reward, info, prev_info, done, args)

            # Preprocess next frame
            next_state      = preprocess(obs, frame_stack)

            # Store in memory
            agent.memory.push(state, action, shaped_reward, next_state, done)

            # Learn
            loss            = agent.learn()
            total_loss     += loss
            total_reward   += reward   # track real reward (not shaped)
            prev_info       = info.copy()
            state           = next_state
            steps          += 1

            if done:
                break

        # Tracking
        all_rewards.append(total_reward)
        recent.append(total_reward)
        avg = np.mean(recent)
        avg_rewards.append(avg)

        # Save best model
        if avg > best_avg:
            best_avg = avg
            agent.save(f'../results/models/{args.name}_best.pth')

        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Avg(20): {avg:7.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps}")

    # Save final model
    agent.save(f'../results/models/{args.name}_final.pth')

    # Save training curve plot
    plt.figure(figsize=(12, 5))
    plt.plot(all_rewards, alpha=0.4, label='Episode reward')
    plt.plot(avg_rewards, linewidth=2, label='Avg reward (20 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Training curve — {args.name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../results/plots/{args.name}_curve.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to results/plots/{args.name}_curve.png")

    # Save raw rewards
    np.save(f'../results/plots/{args.name}_rewards.npy', np.array(all_rewards))

    env.close()
    print("Training complete!")

if __name__ == '__main__':
    train()