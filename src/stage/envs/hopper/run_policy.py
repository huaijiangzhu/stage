import sys
import numpy as np
import yaml
import json
import tensorflow as tf
import gym

from baselines.ddpg.models import Actor


class ActorWrapper():

    def __init__(self, exp_folder, observation_shape, action_shape):
        tf.reset_default_graph()
        self.sess = tf.Session()

        conf_file = exp_folder + 'conf.yaml'
        with open(conf_file, 'r') as f:
            conf = yaml.load(f)

        self.obs = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        observation_range=(-5., 5.)
        normalized_obs0 = tf.clip_by_value(self.obs, observation_range[0], observation_range[1])
        self.actor = Actor(action_shape[0])(normalized_obs0, reuse=False)
        #self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.sess, exp_folder + 'latest_graph')


    def act(self, obs):
        action = self.sess.run(self.actor, feed_dict={self.obs: [obs]})
        return action[0]

def run_episode(env, policy):
    state = env.reset()
    episode_reward = 0.0
    done = False

    i = 0

    while not done:
        #i += 1
        #if i < 10:
        state, reward, done, _ = env.step(policy.act(state))
        #else:
        #        state, reward, done, _ = env.step(np.array([0.0, 0.0]))
        episode_reward += reward

    return episode_reward


if __name__ == '__main__':
    exp_folder = sys.argv[1]

    exp_config = yaml.load(open(exp_folder + 'conf.yaml'))
    env_params = exp_config['env_params'][0]['env_specific_params']
    env_params['output_dir'] = ''
    env_params['visualize'] = True

    if not 'max_episode_steps' in exp_config['env_params'][0]:
        max_episode_duration = exp_config['env_params'][0]['max_episode_duration']
        sim_timestep = exp_config['env_params'][0]['env_specific_params']['sim_timestep']
        cont_timestep_mult = exp_config['env_params'][0]['env_specific_params']['cont_timestep_mult']

        max_episode_steps = int(max_episode_duration / (cont_timestep_mult * sim_timestep))
        exp_config['env_params'][0]['max_episode_steps'] = max_episode_steps

    env_params['exp_name'] = exp_folder + 'run_env'
    env_id = 'RoboschoolReacher3Link-v0'
    gym.envs.register(
        id=env_id,
        entry_point=exp_config['env_params'][0]['entry_point'],
        max_episode_steps=exp_config['env_params'][0]['max_episode_steps'],
        kwargs=env_params
        )
    env = gym.make(env_id)

    policy = ActorWrapper(exp_folder, env.observation_space.shape, env.action_space.shape)

    while True:
        run_episode(env, policy)
