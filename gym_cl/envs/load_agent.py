import os
import sys
import gym
import gym_real
from stable_baselines import PPO2

if __name__ == "__main__":
    env_name = str(sys.argv[1])
    file_name = str(sys.argv[2])

    if file_name[:3] == "mod":
        model_name = file_name
    else:
        dirpath = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "models")
        log_dir = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "tmp")
        model_name = os.path.join(dirpath, file_name)

    env = gym.make(env_name)
    model = PPO2.load(model_name)

    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
