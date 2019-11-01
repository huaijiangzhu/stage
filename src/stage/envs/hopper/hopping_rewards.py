import numpy as np


def exp_rew(x, x_range, y_range, curve, flipped=False):

    def f(x):
        return np.exp(curve * x)

    def g(x):
        return (f(x) - f(0)) / (f(1) - f(0))

    if flipped:
        return y_range[1] - g((x_range[1] - x) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0])
    else:
        return y_range[0] + g((x - x_range[0]) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0])


def select_rew(robot, ground_reward, air_reward, selection_criteria):
    if selection_criteria is None:
        return air_reward
    elif selection_criteria == 'height':
        if robot.get_base_height() < robot.get_upright_height():
            return ground_reward
        else:
            return air_reward
    else:
        assert selection_criteria == 'force'
        if robot.get_endeff_force()[2] == 0.0:
            return ground_reward
        else:
            return air_reward


def base_rew(robot, rew_params):
    height = robot.get_base_height()

    ground_reward = exp_rew(height, *rew_params['ground_reward_params'])
    air_reward = exp_rew(height, *rew_params['air_reward_params'])

    return select_rew(robot, ground_reward, air_reward, rew_params['selection_criteria'])


class HoppingReward():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        if self.params['type'] == 'max':
            self.max_value = None

    def get_reward(self):
        base_value = base_rew(self.robot, self.params)

        if self.params['type'] == 'integral':
            return base_value
        else:
            print(self.max_value)
            if self.max_value is None:
                self.max_value = base_value
                return 0.0
            else:
                if base_value > self.max_value:
                    diff = base_value - self.max_value
                    self.max_value = base_value
                    return diff
                else:
                    return 0.0




class BaseAccPenalty():

    def __init__(self, robot, params, initial_delay_in_timesteps=5):
        self.robot = robot
        self.params = params
        self.initial_delay_in_timesteps = initial_delay_in_timesteps

    def reset(self):
        self.prev_vel = None
        self.timestep = 0
        self.abs_acc = 0.0

    def get_reward_internal(self):
        _, joint_vel = self.robot.get_obs_joint_state()
        curr_vel = joint_vel[0]

        reward = 0.0
        if self.timestep > self.initial_delay_in_timesteps:
            self.abs_acc = np.absolute(curr_vel - self.prev_vel) / self.robot.sim_timestep
            if self.abs_acc > self.params['acc_limit']:
                reward = -1.0 * self.params['k'] * self.abs_acc

        self.timestep += 1
        self.prev_vel = curr_vel

        return reward

    def get_reward(self):
        return 0.0
