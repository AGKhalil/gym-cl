from gym.envs.registration import register

register(
         id='CurriculumLearning-v0',
         entry_point='gym_cl.envs:CLEnv',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )