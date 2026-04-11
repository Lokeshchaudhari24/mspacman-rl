import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt
from agent import DDQNAgent

def preprocess(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.array(obs, dtype=np.float32)
    if len(obs.shape) == 3:
        obs = np.mean(obs, axis=2)
    obs = obs[::2, ::2]
    obs = obs / 255.0
    return np.expand_dims(obs, axis=0)

def shape_reward(reward, info, prev_info, life_penalty,
                 prev_obs, curr_obs, stuck_counter):
    shaped = reward

    # Punish dying
    if life_penalty != 0:
        prev_lives = prev_info.get('lives', 3) if prev_info else 3
        curr_lives = info.get('lives', 3)
        if curr_lives < prev_lives:
            shaped += life_penalty

    # Detect stuck by comparing frames
    if prev_obs is not None:
        diff = np.mean(np.abs(curr_obs - prev_obs))
        if diff < 0.001:  # frames almost identical = stuck
            stuck_counter[0] += 1
            if stuck_counter[0] > 10:  # stuck for 10+ frames
                shaped -= 10           # punish hard
        else:
            stuck_counter[0] = 0       # reset if moving

    return shaped

def train(episodes, no_shaping, life_penalty, name):
    env = gym.make("ALE/MsPacman-v5")
    obs, _ = env.reset()
    obs = preprocess(obs)
    input_shape = obs.shape
    n_actions = env.action_space.n

    agent = DDQNAgent(input_shape, n_actions)

    os.makedirs("../results/models", exist_ok=True)
    os.makedirs("../results/plots", exist_ok=True)

    scores = []
    best_score = -np.inf

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        obs = preprocess(obs)
        total_reward = 0
        prev_info = None
        prev_obs = None
        stuck_counter = [0]
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = preprocess(next_obs)

            if not no_shaping:
                reward = shape_reward(reward, info, prev_info,
                                      life_penalty, prev_obs,
                                      next_obs, stuck_counter)

            agent.memory.push(obs, action, reward, next_obs, done)
            agent.train()

            prev_obs = obs
            obs = next_obs
            prev_info = info
            total_reward += reward

        scores.append(total_reward)

        if total_reward > best_score:
            best_score = total_reward
            torch.save(agent.policy_net.state_dict(),
                       f"../results/models/{name}_best.pth")

        if ep % 10 == 0:
            avg = np.mean(scores[-10:])
            print(f"Episode {ep}/{episodes} | Avg Score: {avg:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Best: {best_score:.1f}")

    np.save(f"../results/{name}_scores.npy", np.array(scores))

    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.4, label='Score')
    plt.plot(np.convolve(scores, np.ones(10)/10, mode='valid'), label='Avg (10 ep)')
    plt.title(f"Training Curve — {name}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"../results/plots/{name}_curve.png")
    plt.close()
    print(f"Done! Results saved for {name}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",     type=int,   default=300)
    parser.add_argument("--no-shaping",   action="store_true")
    parser.add_argument("--life-penalty", type=float, default=-500)
    parser.add_argument("--name",         type=str,   default="experiment")
    args = parser.parse_args()

    train(args.episodes, args.no_shaping, args.life_penalty, args.name)