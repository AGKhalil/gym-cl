import os
import sys
import time
import subprocess
import shutil
import gym
import gym_real
import numpy as np
import matplotlib.pyplot as plt
import datetime
import imageio
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from collections import OrderedDict
from gym.spaces.utils import flatten
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import xml.etree.ElementTree as ET

# env_name = 'Real-v0'
# env = gym.make(env_name)
n_cpu = 8
print(dir(MlpPolicy))
print(MlpPolicy.proba_distribution)
env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
print('************************************')
print(model.get_parameter_list())
print('************************************')
print(model.get_parameters())
print('************************************')
print(len(model.get_parameters()))
print(dir(model.get_parameters()))
print(type(model.get_parameters()))
print('************************************')
print(model.get_parameters().keys())
params = dict(model.get_parameters())
# print(params)
list_of_params = []
for key, value in params.items():
	print(key, type(value), np.shape(value))
	list_of_params.append(value)
array_of_params = np.asarray(list_of_params)
flattened_params = array_of_params.flatten()
len_lis = 0
flat_lis = []
for i in range(len(flattened_params)):
	print(flattened_params[i].shape)
	len_lis += len(flattened_params[i].flatten())
	flat_lis.append(flattened_params[i].flatten())
row_lis = np.concatenate(flat_lis)
print(len_lis)
print(np.concatenate(flat_lis))
print(len(row_lis))

models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fuck', "models/")
print(models_dir)

# xml_path = os.path.join(gym_real.__path__[0], "envs/assets/real.xml")
# leg_length = -3.0

# tree = ET.parse(xml_path)
# root = tree.getroot()
# for geom in root.findall("worldbody/body/body/body/body/geom"):
# 	geom.set("fromto", "0 0 0 0 0 " + str(leg_length))
# 	print(geom.get("fromto"))

# for pos in root.findall("worldbody/body/[@name='torso']"):
# 	pos.set("pos", "0 0 " + str(abs(leg_length) + 0.7))
# 	print(pos.get('pos'))

tree.write(xml_path)
