import time
import numpy as np
import gym
from gym.spaces import Box

from stage.controllers.torque_controller import TorqueController
from stage.controllers.position_gain_controller import PositionGainController
from stage.envs.hopper.hopping_rewards import HoppingReward, BaseAccPenalty
from stage.envs.hopper.impact_penalty import ImpactPenalty


class Robot(gym.Env):

    def __init__(self, controller_params, reward_specs, exp_name=None, log_file=None, full_log=True,
                 observable=[], calc_rew_at_sim_timestep=False, output_dir=None,
                 hopper_height_error=0.0, **kwargs):
        self.p, self.robot_id, self.surface_id, self.obs_joint_ids, self.cont_joint_ids = self.init_simulation(**kwargs)
        self.num_obs_joints = len(self.obs_joint_ids)
        self.num_cont_joints = len(self.cont_joint_ids)
        self.cont_joint_type = self.get_cont_joint_type()
        self.max_torque = np.zeros(self.num_cont_joints)
        self.joint_limits = np.zeros((self.num_cont_joints, 2))
        for i in range(self.num_cont_joints):
            joint_info = self.p.getJointInfo(self.robot_id, self.cont_joint_ids[i])
            self.max_torque[i] = joint_info[10]
            if self.cont_joint_type[i] == 'limited':
                self.joint_limits[i][0] = joint_info[8]
                self.joint_limits[i][1] = joint_info[9]
            else:
                self.joint_limits[i][0] = - np.pi
                self.joint_limits[i][1] = np.pi

        self.init_controller(controller_params)

        self.init_reward(reward_specs)

        self.observable = observable
        self.calc_rew_at_sim_timestep = calc_rew_at_sim_timestep
        self.hopper_height_error = hopper_height_error

        self.action_space = self.controller.get_control_space()
        obs_dim = len(self.get_state())
        high = np.inf * np.ones([obs_dim])
        self.observation_space = Box(-high, high)



    def init_controller(self, controller_params):
        controller_type = controller_params['type']

        if controller_type == 'torque':
            self.controller = TorqueController(self, grav_comp=False)
        elif controller_type == 'torque_gc':
            self.controller = TorqueController(self, grav_comp=True)
        elif controller_type == 'position_gain':
            self.controller = PositionGainController(robot=self, params=controller_params)
        else:
            assert False, 'Unknown controller type: ' + controller_type

    def init_reward(self, reward_params):
        self.reward_parts = {}
        for reward_type, reward_spec in reward_params.items():
            if reward_type == 'hopping_reward':
                self.reward_parts[reward_type] = HoppingReward(self, reward_spec)
            if reward_type == 'impact_penalty':
                self.reward_parts[reward_type] = ImpactPenalty(self, reward_spec)
            if reward_type == 'base_acc_penalty':
                self.reward_parts[reward_type] = BaseAccPenalty(self, reward_spec)

    def get_obs_joint_state(self):
        joint_pos = np.zeros(self.num_obs_joints)
        joint_vel = np.zeros(self.num_obs_joints)
        for i in range(self.num_obs_joints):
            joint_pos[i], joint_vel[i], _, _ = self.p.getJointState(self.robot_id, self.obs_joint_ids[i])
        return joint_pos, joint_vel

    def get_cont_joint_state(self):
        joint_pos = np.zeros(self.num_cont_joints)
        joint_vel = np.zeros(self.num_cont_joints)
        for i in range(self.num_cont_joints):
            joint_pos[i], joint_vel[i], _, _ = self.p.getJointState(self.robot_id, self.cont_joint_ids[i])
        return joint_pos, joint_vel

    def get_total_force(self, links):
        contacts = self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id)
        total_force = np.zeros(3)
        for contact in contacts:
            if contact[4] in links:
                contact_normal = np.array(contact[7])
                normal_force = contact[9]
                total_force += normal_force * contact_normal
        return total_force

    def get_total_ground_force(self):
        contacts = self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id)
        total_force = np.zeros(3)
        for contact in contacts:
            contact_normal = np.array(contact[7])
            normal_force = contact[9]
            total_force += normal_force * contact_normal
        return total_force

    def get_endeff_force(self):
        return self.get_total_force([self.get_endeff_link_id()])

    def get_state(self):
        state = []

        joint_pos, joint_vel = self.get_obs_joint_state()
        joint_pos[0] += self.hopper_height_error
        state += joint_pos.tolist()
        state += joint_vel.tolist()

        if 'endeff_force' in self.observable:
            state += self.get_endeff_force().tolist()

        return np.array(state)


    def _reset(self):

        if self.random_lateral_friction:
            self.lateral_friction = np.random.uniform(self.lateral_friction_range[0], self.lateral_friction_range[1])
            for i in range(self.p.getNumJoints(self.robot_id)):
                self.p.changeDynamics(self.robot_id, i, lateralFriction=self.lateral_friction)
            self.p.changeDynamics(self.surface_id, -1, lateralFriction=self.lateral_friction)

        if self.random_contact_stiffness:
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])

            if self.contact_damping_multiplier is not None:
                contact_damping = self.contact_damping_multiplier * 2.0 * np.sqrt(self.contact_stiffness)
            else:
                if self.contact_damping is None:
                    contact_damping = 2.0 * np.sqrt(self.contact_stiffness)
                else:
                    contact_damping = self.contact_damping

            self.p.changeDynamics(self.surface_id, -1, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)

        self.controller.reset()

        for reward_part in self.reward_parts.values():
            reward_part.reset()

        self.set_initial_configuration()

        state = self.get_state()

        return state

    def _step(self, action):

        sum_sub_rewards = 0.0
        base_acc_penalty = 0.0
        for i in range(self.cont_timestep_mult):
            self.controller.act(action)
            if self.calc_rew_at_sim_timestep:
                sum_sub_rewards += sum([reward_part.get_reward() for reward_part in self.reward_parts.values()])

            if 'base_acc_penalty' in self.reward_parts:
                base_acc_penalty += self.reward_parts['base_acc_penalty'].get_reward_internal()
        state = self.get_state()
        if self.calc_rew_at_sim_timestep:
            reward = sum_sub_rewards
        else:
            reward = sum([reward_part.get_reward() for reward_part in self.reward_parts.values()]) + base_acc_penalty

        return state, reward, False, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass

    def init_torque_control(self):
        for joint_id in self.obs_joint_ids:
            # As per PyBullet manual, this has to be done to be able to do
            # torque control later
            self.p.setJointMotorControl2(self.robot_id, joint_id,
                controlMode=self.p.VELOCITY_CONTROL, force=0)

    def torque_control(self, des_torque):
        des_torque = np.clip(des_torque, -self.max_torque, self.max_torque)
        #print(des_torque)

        for i in range(self.num_cont_joints):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=self.cont_joint_ids[i],
                controlMode=self.p.TORQUE_CONTROL,
                force=des_torque[i])

        self.p.stepSimulation()
