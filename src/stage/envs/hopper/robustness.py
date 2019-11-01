import sys
import yaml
import json
import gym

from robot_envs.hopper.run_policy import ActorWrapper, run_episode
from utils.generate_experiments import process


def make_default_env(exp_folder):
    exp_config = yaml.load(open(exp_folder + 'conf.yaml'))

    env_params = exp_config['env_params'][0]['env_specific_params']
    env_params['output_dir'] = ''
    env_params['exp_name'] = exp_folder + 'default_env'

    if not 'max_episode_steps' in exp_config['env_params'][0]:
        max_episode_duration = exp_config['env_params'][0]['max_episode_duration']
        sim_timestep = exp_config['env_params'][0]['env_specific_params']['sim_timestep']
        cont_timestep_mult = exp_config['env_params'][0]['env_specific_params']['cont_timestep_mult']

        max_episode_steps = int(max_episode_duration / (cont_timestep_mult * sim_timestep))
        exp_config['env_params'][0]['max_episode_steps'] = max_episode_steps

    env_id = 'DefaultEnv-v0'
    gym.envs.register(
        id=env_id,
        entry_point=exp_config['env_params'][0]['entry_point'],
        max_episode_steps=exp_config['env_params'][0]['max_episode_steps'],
        kwargs=env_params
        )
    env = gym.make(env_id)
    return env


if __name__ == '__main__':
    exp_folder = sys.argv[1]
    script_conf_file = sys.argv[2]

    with open(script_conf_file, 'r') as f:
        conf = f.read()

    default_env = make_default_env(exp_folder)
    policy = ActorWrapper(exp_folder, default_env.observation_space.shape, default_env.action_space.shape)

    robustness_results = []

    processed = process(conf)
    env_counter = 0
    for p in processed:
        script_conf = yaml.load(p)
        exp_config = yaml.load(open(exp_folder + 'conf.yaml'))

        env_params = exp_config['env_params'][0]['env_specific_params']
        env_params['output_dir'] = ''
        env_params['exp_name'] = exp_folder + 'robustness_env'

        if not 'max_episode_steps' in exp_config['env_params'][0]:
            max_episode_duration = exp_config['env_params'][0]['max_episode_duration']
            sim_timestep = exp_config['env_params'][0]['env_specific_params']['sim_timestep']
            cont_timestep_mult = exp_config['env_params'][0]['env_specific_params']['cont_timestep_mult']

            max_episode_steps = int(max_episode_duration / (cont_timestep_mult * sim_timestep))
            exp_config['env_params'][0]['max_episode_steps'] = max_episode_steps

        for key, value in script_conf['env_specific_params'].items():
            env_params[key] = value

        env_id = 'Env' + str(env_counter) + '-v0'
        env_counter += 1
        gym.envs.register(
            id=env_id,
            entry_point=exp_config['env_params'][0]['entry_point'],
            max_episode_steps=exp_config['env_params'][0]['max_episode_steps'],
            kwargs=env_params
            )
        env = gym.make(env_id)

        current_result = script_conf['env_specific_params'].copy()
        current_result['episode_rewards'] = []
        num_iters = script_conf['num_iters']
        for iter in range(num_iters):
            episode_reward = run_episode(env, policy)
            current_result['episode_rewards'].append(episode_reward)
        robustness_results.append(current_result)

    with open(exp_folder + 'robustness_results.json', 'w') as f:
        json.dump(robustness_results, f)
