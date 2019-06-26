import sys
import os
import gym
import gym_real
import numpy as np
import datetime
import imageio
import time
import shutil
import subprocess
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import xml.etree.ElementTree as ET


n_counter = 0
worker_best_mean_reward = 0
old_counter = 0
model_name = ''
gif_dir = ''
env_name = 'Real-v0'
worker_total_steps = 50000
if worker_total_steps >= 1000000:
    n_gifs = 5
else:
    n_gifs = 2
log_incs = np.round(
    (worker_total_steps / n_gifs) * 60 / 60000)
models_tmp_dir = ''
total_gif_time = 0
log_dir = ''
w_model = 0
worker_steps = 0
episode = 0
n_step = 0


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, model_name, plt_dir, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    m_name_csv = model_name + ".csv"
    old_file_name = os.path.join(log_folder, "monitor.csv")
    new_file_name = os.path.join(log_folder, m_name_csv)
    save_name = os.path.join(plt_dir, model_name)

    x, y = ts2xy(load_results(log_folder), 'timesteps')
    shutil.copy(old_file_name, new_file_name)
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    print('Saving plot at:', save_name)
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".eps")
    print("plots saved...")
    # plt.show()


class CLEnv(gym.Env):

    def __init__(self):
        global gif_dir, env_name, models_tmp_dir, log_dir
        self.save_path = ''

        self.xml_path = os.path.join(
            gym_real.__path__[0], "envs/assets/real.xml")
        self.w_models_dir = os.path.join(self.save_path, "models/")
        self.w_models_tmp_dir = os.path.join(self.save_path, "models_tmp/")
        models_tmp_dir = self.w_models_tmp_dir
        self.log_dir = os.path.join(self.save_path, "tmp")
        log_dir = self.log_dir
        self.gif_dir = os.path.join(self.save_path, "tmp_gif/")
        gif_dir = self.gif_dir
        self.plt_dir = os.path.join(self.save_path, "plot")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.w_models_dir, exist_ok=True)
        os.makedirs(self.w_models_tmp_dir, exist_ok=True)
        os.makedirs(self.plt_dir, exist_ok=True)

        self.step_n = 0
        self.worker_n = 0
        self.episode = 0

        self.action_high = -0.1
        self.action_low = -1.0

        self.alter_leg(self.action_high)
        self.n_cpu = 8
        self.w_model = self.worker_maintainer(init=True)
        self.initial_obs = self.get_state(self.w_model)
        self.ob_len = len(self.initial_obs)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.ob_len,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=-0.1,
                                shape=(1,), dtype=np.float32)

    def step(self, action):
        global worker_best_mean_reward, worker_total_steps, model_name, w_model, worker_steps, episode, n_step

        self.step_n += 1
        n_step = self.step_n
        self.alter_leg(action[0])
        self.w_model = self.worker_maintainer()
        w_model = self.w_model
        self.w_model.learn(total_timesteps=worker_total_steps,
                           callback=self.callback)
        self.w_model_name = self.epi_dir + '_' + "Worker_" + \
            str(self.step_n) + '_' + str(action[0]) + "_" + \
            str(worker_total_steps) + "_" + self.stamp
        model_name = self.w_model_name
        self.w_model.save(self.w_model_loc)

        plot_results(self.log_dir, self.w_model_name, self.plt_dir)

        observation = self.get_state(self.w_model)
        reward = 1 / (worker_steps)
        if worker_best_mean_reward > 300:
            done = True
            self.episode += 1
            episode = self.episode
        else:
            done = False

        info = {}

        del self.w_model

        return observation, reward, done, info

    def render(self, mode='human'):
        while watch_agent == "y" or "Y":
            subprocess.Popen(
                '''export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so; python load_agent.py '%s' '%s' ''' % (self.env_name, self.w_model_name), shell=True)
            watch_agent = input("Do you want to watch your sick gaits? (Y/n):")

    def reset(self):
        self.step_n = 0
        self.worker_n = 0
        self.episode = 0
        self.w_model = self.worker_maintainer(init=True)
        observation = self.get_state(self.w_model)
        return observation

    def flatten_policy(self, model_params):
        params = dict(model_params)
        list_of_params = []
        for key, value in params.items():
            list_of_params.append(value)
        array_of_params = np.asarray(list_of_params)
        flattened_params = array_of_params.flatten()
        flat_lis = []
        for i in range(len(flattened_params)):
            flat_lis.append(flattened_params[i].flatten())
        return np.concatenate(flat_lis)

    def get_state(self, model):
        observation = self.flatten_policy(model.get_parameters())
        return observation

    def alter_leg(self, leg_length):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        for geom in root.findall("worldbody/body/body/body/body/geom"):
            geom.set("fromto", "0 0 0 0 0 " + str(leg_length))

        for pos in root.findall("worldbody/body/[@name='torso']"):
            pos.set("pos", "0 0 " + str(abs(leg_length) + 0.7))

        tree.write(self.xml_path)

    def worker_maintainer(self, init=False):
        global model_name
        self.stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        self.epi_dir = 'Episode' + str(self.episode)
        self.w_env = gym.make(env_name)
        self.w_env = Monitor(
            self.w_env, self.log_dir, allow_early_resets=True)
        self.w_env = SubprocVecEnv(
            [lambda: self.w_env for i in range(self.n_cpu)])
        if init:
            self.w_model_name = self.epi_dir + '_' + "Worker_" + \
                str(self.step_n) + "_" + \
                str(worker_total_steps) + "_" + self.stamp
            self.w_model_loc = os.path.join(
                self.w_models_dir, self.w_model_name)
            model = PPO2(MlpPolicy, self.w_env, verbose=0)
            model.save(self.w_model_loc)
        else:
            model = PPO2.load(self.w_model_loc, env=self.w_env)
            self.w_model_loc = os.path.join(
                self.w_models_dir, self.w_model_name)

        model_name = self.w_model_name
        print(model_name)
        return model

    def callback(self, _locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_counter, worker_best_mean_reward, old_counter, model_name, gif_dir, env_name, log_incs, models_tmp_dir, total_gif_time, log_dir, w_model, worker_steps, worker_total_steps, n_step, episode
        # Print stats every 1000 calls

        if abs(n_counter - old_counter) >= log_incs:
            gif_start = time.time()
            old_counter = n_counter
            # Evaluate policy performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                worker_steps = x[-1]
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    worker_best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > worker_best_mean_reward:
                    worker_best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(models_tmp_dir + 'best_model.pkl')

            stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            epi_dir = 'Episode' + str(episode)
            model_name = epi_dir + '_' + "Worker_" + \
                str(n_step) + "_" + str(worker_total_steps) + "_" + stamp
            save_str = gif_dir + model_name + '.gif'
            images = []

            env_gif = gym.make(env_name)
            obs = env_gif.reset()
            img = env_gif.sim.render(
                width=200, height=200, camera_name="isometric_view")
            for _ in range(5000):
                action, _ = w_model.predict(obs)
                obs, _, _, _ = env_gif.step(action)
                img = env_gif.sim.render(
                    width=200, height=200, camera_name="isometric_view")
                images.append(np.flipud(img))

            print("creating gif...")
            print("saving gif at:", save_str)
            imageio.mimsave(save_str, [np.array(img)
                                       for i, img in enumerate(images) if i % 2 == 0], fps=29)
            print("gif created...")
            gif_end = time.time()
            total_gif_time += gif_end - gif_start
        n_counter += 1

        return True
