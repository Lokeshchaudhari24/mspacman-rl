import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import numpy as np
import torch
import argparse
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

def evaluate(model_path, episodes):
    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    obs, _ = env.reset()
    obs = preprocess(obs)
    input_shape = obs.shape
    n_actions = env.action_space.n

    agent = DDQNAgent(input_shape, n_actions)
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device)
    )
    agent.policy_net.eval()
    agent.epsilon = 0.0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        obs = preprocess(obs)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = preprocess(next_obs)
            total_reward += reward

        print(f"Episode {ep} | Score: {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    evaluate(args.model, args.episodes)