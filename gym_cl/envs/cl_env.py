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

best_mean_reward, n_steps = -np.inf, 0
best_mean_reward = 0
gif_dir = ''
models_tmp_dir = ''
log_dir = ''
mean_reward = 0


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
    y = moving_average(y, window=10)
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
        global models_tmp_dir, log_dir
        self.env_name = 'Real-v0'
        self.total_timesteps = 100000

        self.save_path = ''

        self.xml_path = os.path.join(
            gym_real.__path__[0], "envs/assets/real.xml")
        self.models_dir = os.path.join(self.save_path, "models/")
        self.models_tmp_dir = os.path.join(self.save_path, "models_tmp/")
        models_tmp_dir = self.models_tmp_dir
        self.log_dir = os.path.join(self.save_path, "tmp")
        log_dir = self.log_dir
        self.gif_dir = os.path.join(self.save_path, "tmp_gif/")
        self.plt_dir = os.path.join(self.save_path, "plot")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.models_tmp_dir, exist_ok=True)
        os.makedirs(self.plt_dir, exist_ok=True)

        self.total_gif_time = 0

        self.step_n = 0
        self.worker_n = 0
        self.episode = 0

        self.action_high = -0.1
        self.action_low = -1.0

        self.alter_leg(self.action_high)
        self.n_cpu = 20
        model, model_name, self.prev_model_loc, env = self.worker_maintainer(init=True)
        model.save(self.prev_model_loc)
        self.initial_obs = self.get_state(model)
        self.ob_len = len(self.initial_obs)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.ob_len,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=-0.1,
                                shape=(1,), dtype=np.float32)

        env.close()
        del env, model

    def step(self, action):
        global best_mean_reward, mean_reward

        self.step_n += 1
        self.alter_leg(action[0])
        model, model_name, model_loc, env = self.worker_maintainer(action=action[0])
        model.learn(total_timesteps=self.total_timesteps,
                    callback=self.callback)
        model.save(model_loc)
        self.prev_model_loc = model_loc

        plot_results(self.log_dir, model_name, self.plt_dir)

        step_count = self.make_me_a_gif(model, model_name)

        observation = self.get_state(model)
        reward = 1 / step_count
        print('STEP COUNT:', step_count)
        print('PROF REWARD', reward)
        print('MEAN REWARD', mean_reward)
        if mean_reward > 300:
            done = True
            self.episode += 1
        else:
            done = False

        info = {}

        env.close()
        del env, model

        return observation, reward, done, info

    def render(self, mode='human'):
        while watch_agent == "y" or "Y":
            subprocess.Popen(
                '''export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so; python load_agent.py '%s' '%s' ''' % (self.env_name, model_name), shell=True)
            watch_agent = input("Do you want to watch your sick gaits? (Y/n):")

    def reset(self):
        self.step_n = 0
        self.worker_n = 0
        model, _, model_loc, env = self.worker_maintainer(init=True)
        if self.episode != 0:
            self.prev_model_loc = model_loc
            model.save(model_loc)
        observation = self.get_state(model)
        print('I GOT CALLED HOLY FUCK')

        env.close()
        del model, env
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

    def worker_maintainer(self, init=False, action=None, prev_model_loc=None):
        global model_name
        stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        epi_dir = 'Episode' + str(self.episode)
        env = gym.make(self.env_name)
        env = Monitor(
            env, self.log_dir, allow_early_resets=True)
        env = SubprocVecEnv(
            [lambda: env for i in range(self.n_cpu)])
        if init:
            model_name = epi_dir + '_' + "Worker_" + \
                str(self.step_n) + "_" + \
                str(self.total_timesteps) + "_" + stamp
            model = PPO2(MlpPolicy, env, verbose=1)
        else:
            model_name = epi_dir + '_' + "Worker_" + \
                str(self.step_n) + '_' + str(action) + "_" + \
                str(self.total_timesteps) + "_" + stamp
            model = PPO2.load(self.prev_model_loc, env=env)

        model_loc = os.path.join(
            self.models_dir, model_name)

        return model, model_name, model_loc, env

    def callback(self, _locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward, models_tmp_dir, log_dir, mean_reward
        # Print stats every 1000 calls
        if (n_steps + 1) % 39 == 0:
            # Evaluate policy performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(models_tmp_dir + 'best_model.pkl')

        n_steps += 1

        return True

    def make_me_a_gif(self, model, model_name):
        gif_start = time.time()
        save_str = self.gif_dir + model_name + '.gif'
        images = []
        temp_env = gym.make(self.env_name)
        obs = temp_env.reset()
        done = False
        step = 0
        img = temp_env.sim.render(
            width=100, height=100, camera_name="isometric_view")
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = temp_env.step(action)
            img = temp_env.sim.render(
                width=100, height=100, camera_name="isometric_view")
            images.append(np.flipud(img))
            step += 1

        temp_env.close()
        del temp_env
        del model

        print("creating gif...")
        print("saving gif at:", save_str)
        imageio.mimsave(save_str, [np.array(img)
                                   for i, img in enumerate(images) if i % 2 == 0], fps=29)
        print("gif created...")
        gif_end = time.time()
        self.total_gif_time += gif_end - gif_start

        return step
