import sys
import os
import gym
import gym_real
import numpy as np
import datetime
import random
import imageio
import time
import shutil
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    def __init__(self, n_cpu, worker_total_timesteps, reset_timesteps, save_path, prof_name):
        global models_tmp_dir, log_dir
        self.env_name = 'Real-v0'
        self.n_cpu = n_cpu
        self.worker_total_timesteps = worker_total_timesteps
        self.reset_timesteps = reset_timesteps
        self.save_path = save_path
        self.prof_name = prof_name

        self.all_x = []
        self.all_y = []
        self.all_z = []
        self.vert_x = []
        self.divider = []
        self.prog_x = []
        self.prog_y = []
        self.delta = 3
        self.px = []
        self.py = []

        self.steps_accum = 0

        self.xml_path = os.path.join(
            gym_real.__path__[0], "envs/assets/real.xml")
        self.models_dir = os.path.join(self.save_path, "models/")
        self.models_tmp_dir = os.path.join(self.save_path, "models_tmp/")
        models_tmp_dir = self.models_tmp_dir
        self.log_dir = os.path.join(self.save_path, "tmp")
        self.prof_log_dir = os.path.join(self.save_path, "prof/tmp")
        log_dir = self.log_dir
        self.gif_dir = os.path.join(self.save_path, "tmp_gif/")
        self.plt_dir = os.path.join(self.save_path, "plot")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.models_tmp_dir, exist_ok=True)
        os.makedirs(self.plt_dir, exist_ok=True)
        stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        self.legs_plt = os.path.join(
            self.plt_dir, 'legs_' + stamp + "_" + str(self.worker_total_timesteps))
        self.prog_plt = os.path.join(
            self.plt_dir, 'prog_' + stamp + "_" + str(self.worker_total_timesteps))
        self.prof_plt = os.path.join(
            self.plt_dir, 'prof_' + stamp + "_" + str(self.worker_total_timesteps))

        self.total_gif_time = 0

        self.counter = 0
        self.step_n = 0
        self.worker_n = 0
        self.episode = 0
        self.re_prof = []

        # self.action_high = -0.1
        # self.action_low = -1.0

        # self.alter_leg(self.action_high)

        # model, model_name, self.prev_model_loc, env = self.worker_maintainer(
        #     init=True)
        # # model.save(self.prev_model_loc)
        # self.initial_obs = self.get_state(model)
        # self.ob_len = len(self.initial_obs)
        # self.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(self.ob_len,), dtype=np.float32)
        # self.action_space = Box(low=self.tall_leg, high=self.short_leg,
        #                         shape=(1,), dtype=np.float32)

        self.short_leg = -0.1
        self.tall_leg = -1.0
        self.alter_leg(3)
        self.leg_length = self.short_leg
        self.old_length = self.leg_length
        self.leg_change = 0.2
        self.discrete_actions = [0, 1, 2]
        # self.discrete_actions = [1, 2]
        self.action_space = Discrete(len(self.discrete_actions))
        self.observation_space = Box(low=np.array([self.tall_leg, -np.inf, self.tall_leg, -np.inf, self.tall_leg, -np.inf, self.tall_leg, -np.inf, self.tall_leg, -np.inf, self.tall_leg, -np.inf, self.tall_leg, -np.inf, self.tall_leg, -np.inf, self.tall_leg, -np.inf, -
                                                   1.0, -np.inf]), high=np.array([self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf, self.short_leg, np.inf]))
        model, model_name, self.prev_model_loc, env = self.worker_maintainer(
            init=True)
        self.initial_obs = []
        self.prev_reward = 0
        for i in range(11):
            action = random.choice(self.discrete_actions)
            self.alter_leg(action)
            temp_env = gym.make(self.env_name)
            model.learn(total_timesteps=self.worker_total_timesteps)
            w_obs = temp_env.reset()
            w_done = False
            w_reward = 0, 0
            while not w_done:
                w_action, _ = model.predict(w_obs)
                w_obs, w_reward, w_done, _ = temp_env.step(w_action)
            if i != 0:
                self.initial_obs.append(self.leg_length)
                self.initial_obs.append(w_reward - self.prev_reward)
            self.prev_reward = w_reward
            temp_env.close()
            del temp_env
        print('INIT OBSERVATION', self.initial_obs, len(self.initial_obs))

        env.close()
        self.alter_leg(3)
        self.prev_obs = self.initial_obs
        model.save(self.prev_model_loc)
        del env, model

    def step(self, action):
        global best_mean_reward, mean_reward

        self.step_n += 1
        if self.step_n == self.reset_timesteps:
            self.alter_leg(4)
        else:
            self.alter_leg(action)
        w_done = False
        prev_w_reward = 0
        step_meter = 0

        while not w_done:
            model, model_name, model_loc, env = self.worker_maintainer()
            model.learn(total_timesteps=self.worker_total_timesteps,
                        callback=self.callback)
            self.steps_accum += self.worker_total_timesteps
            model.save(model_loc)
            self.prev_model_loc = model_loc
            env.close()
            del model, env

            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            y = moving_average(y, window=50)
            x = x[len(x) - len(y):]
            for i in x:
                if self.counter != 0:
                    self.prog_x.append(i + self.vert_x[-1])
                    appended_val = x[-1] + self.vert_x[-1]
                else:
                    self.prog_x.append(i)
                    appended_val = x[-1]

            self.vert_x.append(appended_val)
            for i in y:
                self.prog_y.append(i)
            os.remove(os.path.join(log_dir, "monitor.csv"))

            z = [self.leg_length for i in range(len(x))]

            self.all_x.append(x)
            self.all_y.append(y)
            self.all_z.append(z)

            self.plot_legs(self.all_z, self.all_x, self.all_y)

            # w_reward = self.make_me_a_gif(model, model_name)
            w_reward = np.mean(y[-50:])

            if abs(w_reward - prev_w_reward) < self.delta:
                w_done = True
                self.divider.append(self.vert_x[-1])

            prev_w_reward = w_reward
            step_meter += 1
            print('IN THIS STEP I RAN', step_meter)
            print('MY LEG LENGTH IS', self.leg_length)
            self.counter += 1
        self.plot_prog(self.prog_x, self.prog_y, self.divider)

        observation = self.prev_obs[2:]
        observation.append(self.leg_length)
        observation.append(w_reward - self.prev_reward)
        self.prev_obs = observation
        print('OBSERVATION', observation, len(observation))
        # reward = mean_reward * abs(self.leg_length)
        reward = 1 / self.steps_accum
        self.re_prof.append(reward)
        # plot_prof(self.re_prof, self.plt_dir)
        print('STEP COUNT:', w_reward)
        print('PROF REWARD', reward)
        print('MEAN REWARD', mean_reward)
        if self.step_n == self.reset_timesteps:
            done = True
            self.episode += 1
        else:
            done = False

        info = {}

        # px, py = ts2xy(load_results(self.prof_log_dir), 'timesteps')
        # px.append(self.n_)
        self.py.append(reward)
        self.plot_prof(self.py)
        # if px and py:
        # 	py = moving_average(py, window=10)
        # 	px = px[len(px) - len(py):]
        # 	self.plot_prof(px, py)

        return observation, reward, done, info

    def plot_legs(self, x, y, z):
        fig = plt.figure('legs' + str(self.worker_total_timesteps))
        ax = plt.axes(projection="3d")
        for i in range(len(x)):
            ax.plot3D(x[i], y[i], z[i])
        # for i in vert_x:
        # 	plt.axvline(x=i, linestyle='--', color='#ccc5c6', label='leg increment')
        # plt.xlabel('Number of Timesteps')
        # plt.ylabel('Rewards')
        # plt.zlabel('Leg lengths')
        plt.title("Legs" + " Smoothed")
        plt.savefig(self.legs_plt + ".png")
        plt.savefig(self.legs_plt + ".eps")
        print("plots saved...")

    def plot_prog(self, x, y, vert):
        fig = plt.figure('progress' + str(self.worker_total_timesteps))
        plt.plot(x, y)
        # for i in vert:
        #     plt.axvline(x=i, linestyle='--', color='#ccc5c6',
        #                 label='leg increment')
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title('Progress' + " Smoothed")
        plt.savefig(self.prog_plt + ".png")
        plt.savefig(self.prog_plt + ".eps")
        print("plots saved...")

    def plot_prof(self, y):
        fig = plt.figure('professor')
        # plt.plot(x, y)
        plt.plot(y)
        # for i in vert:
        #     plt.axvline(x=i, linestyle='--', color='#ccc5c6',
        #                 label='leg increment')
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title('Professor' + " Smoothed")
        plt.savefig(self.prof_plt + ".png")
        plt.savefig(self.prof_plt + ".eps")
        print("plots saved...")

    def render(self, mode='human'):
        while watch_agent == "y" or "Y":
            subprocess.Popen(
                '''export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so; python load_agent.py '%s' '%s' ''' % (self.env_name, model_name), shell=True)
            watch_agent = input("Do you want to watch your sick gaits? (Y/n):")

    def reset(self):
        self.step_n = 0
        self.worker_n = 0
        self.steps_accum = 0
        model, _, model_loc, env = self.worker_maintainer(init=True)
        if self.episode != 0:
            self.prev_model_loc = model_loc
            model.save(model_loc)
        observation = self.initial_obs
        self.alter_leg(3)

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

    def alter_leg(self, action):
        if action == 0:
            self.old_length = self.leg_length
            if self.leg_length + self.leg_change < self.short_leg:
                self.leg_length += self.leg_change
        elif action == 1 and self.leg_length != self.tall_leg:
            self.old_length = self.leg_length
            if self.leg_length - self.leg_change > self.tall_leg:
                self.leg_length -= self.leg_change
        elif action == 3:
            self.leg_length = self.short_leg
            self.old_length = self.leg_length
        elif action == 4:
            self.leg_length = self.tall_leg
            self.old_length = self.leg_length
        print('LEGS', self.old_length, action, self.leg_length)
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        for geom in root.findall("worldbody/body/body/body/body/geom"):
            geom.set("fromto", "0 0 0 0 0 " + str(self.leg_length))

        for pos in root.findall("worldbody/body/[@name='torso']"):
            pos.set("pos", "0 0 " + str(abs(self.leg_length) + 0.7))

        tree.write(self.xml_path)

    def worker_maintainer(self, init=False, prev_model_loc=None):
        global model_name
        stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        epi_dir = 'Episode' + str(self.episode)
        env = gym.make(self.env_name)
        env = Monitor(
            env, self.log_dir, allow_early_resets=True)
        env = SubprocVecEnv(
            [lambda: env for i in range(self.n_cpu)])
        model_name = epi_dir + '_' + "Worker_" + \
            str(self.step_n) + '_' + "{:.2f}".format(self.leg_length) + "_" + \
            str(self.worker_total_timesteps) + "_" + stamp
        if init:
            # model_name = epi_dir + '_' + "Worker_" + \
            #     str(self.step_n) + "_" + \
            #     str(self.worker_total_timesteps) + "_" + stamp
            model = PPO2(MlpPolicy, env, verbose=1)
            # model_name = "trained_init_student"
            # model_loc = os.path.join(
            #     self.models_dir, model_name)
            # model = PPO2.load(model_loc, env=env)
        else:
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
        step, reward = 0, 0
        img = temp_env.sim.render(
            width=100, height=100, camera_name="isometric_view")
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = temp_env.step(action)
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

        return reward
