import os
import time
import numpy as np

import pybullet as p
from stage.envs.hopper.robot import Robot

class Hopper(Robot, object):

    def __init__(self, **kwargs):
        super(Hopper, self).__init__(**kwargs)

    def init_simulation(self, floor_height=0.3, 
                        visualize=True, sim_timestep=0.001, 
                        cont_timestep_mult=16, lateral_friction=1.0, 
                        joint_damping=0.0, contact_stiffness=10000.0, 
                        contact_damping=200.0, contact_damping_multiplier=None):

        self.visualize = visualize
        if self.visualize:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        cubeStartPos = [0, 0, 0]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])        
        planeId = p.loadURDF(dir_path + '/urdf/plane_with_restitution.urdf')
        self.robotId = p.loadURDF(dir_path + '/urdf/teststand.urdf', 
                                  cubeStartPos, cubeStartOrientation, 
                                  flags=p.URDF_USE_INERTIA_FROM_FILE)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotId)

        if isinstance(lateral_friction, list):
            self.random_lateral_friction = True
            self.lateral_friction_range = lateral_friction
            self.lateral_friction = np.random.uniform(self.lateral_friction_range[0], self.lateral_friction_range[1])
        else:
            self.random_lateral_friction = False
            self.lateral_friction = lateral_friction

        if isinstance(contact_stiffness, list):
            self.random_contact_stiffness = True
            self.contact_stiffness_range = contact_stiffness
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])
        else:
            self.random_contact_stiffness = False
            self.contact_stiffness = contact_stiffness

        useRealTimeSimulation = False

        # Query all the joints.
        num_joints = p.getNumJoints(self.robotId)
        print("Number of joints={}".format(num_joints))

        for ji in range(num_joints):
            p.changeDynamics(self.robotId, ji, 
                             linearDamping=.04, angularDamping=0.04, 
                             restitution=0.0, lateralFriction=self.lateral_friction, 
                             maxJointVelocity=1000)
            p.changeDynamics(self.robotId, ji, jointDamping=joint_damping)

        p.changeDynamics(planeId, -1, lateralFriction=self.lateral_friction)

        self.contact_damping = contact_damping
        self.contact_damping_multiplier = contact_damping_multiplier

        if self.random_contact_stiffness:
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])

        if self.contact_damping_multiplier is not None:
            contact_damping = self.contact_damping_multiplier * 2.0 * np.sqrt(self.contact_stiffness)
        else:
            if self.contact_damping is None:
                contact_damping = 2.0 * np.sqrt(self.contact_stiffness)
            else:
                contact_damping = self.contact_damping
        p.changeDynamics(planeId, -1, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)

        p.setGravity(0.0, 0.0, -9.81)
        #p.setPhysicsEngineParameter(1e-3, numSubSteps=1)

        self.sim_timestep = sim_timestep
        self.cont_timestep_mult = cont_timestep_mult
        self.dt = self.cont_timestep_mult * self.sim_timestep
        p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

        print(p.getPhysicsEngineParameters())

        return p, self.robotId, planeId, [1, 2, 3], [2, 3]

    def set_initial_configuration(self):
        #joint_pos = np.random.uniform(0.0, 2.0 * np.pi, 2)
        #joint_pos = [0, 0]
        #joint_pos = [-0.25, np.pi / 2.0, np.pi]
        #joint_pos = [0.0, 0.0, 0.0]
        #joint_pos = [0.05, np.pi / 2.0, np.pi]

        # Fully on the ground position
        joint_pos = [0.0499923 , 1.41840898, 3.29190477]

        for i in range(self.num_obs_joints):
            self.p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.obs_joint_ids[i],
                targetValue=joint_pos[i],
                targetVelocity=0.0)

    def get_endeff_link_id(self):
        return 4

    def get_base_link_id(self):
        return 0

    def get_cont_joint_type(self):
        return ['circular', 'circular']

    def get_default_pd_params(self):
        #return  np.array([1.0, 1.0]), np.array([0.1, 0.1])
        #return np.array([1.0, 1.0]), np.array([0.1, 0.1])
        return np.array([1.0, 0.1]), np.array([0.1, 0.01])

    def get_base_height(self):
        return self.get_obs_joint_state()[0][0]

    def get_upright_height(self):
        return 0.345

    def inv_dyn(self, des_acc):
        return np.zeros(des_acc.shape)


if __name__ == '__main__':
    #hopper = Hopper(controller_params={'type': 'torque'}, reward_specs={})
    #hopper = Hopper(controller_params={'type': 'position_gain', 'variant': 'fixed'}, reward_specs={}, visualize=True)
    upright_height = 0.345
    max_height = 1.0
    hopper = Hopper(controller_params={'type': 'position_gain', 'variant': 'fixed'}, \
                    reward_specs={'hopping_reward': {'type': 'max', \
                                                     'selection_criteria': 'height', \
                                                     'ground_reward_params': [[0.0, upright_height], [0.0, 0.2], 0.001],\
                                                     'air_reward_params': [[upright_height, max_height], [0.75, 4.0], 5, True]}}, \
                    visualize=True,
                    observable=['endeff_force'],
                    exp_name='~/Desktop/tmp/')
    hopper._reset()
    a = np.random.uniform(-1.0, 1.0, 2)
    i = 0
    while True:
        i += 1
        #if i % 12 == 0:
        a = np.random.uniform(-1.0, 1.0, 2)
        _, r, _, _ = hopper._step(a)
        #print(r)
        #hopper.p.stepSimulation()
        #print(hopper.get_obs_joint_state())
        #time.sleep(hopper.dt)

    #timesteps = 250
    #goal = np.array([-1.0, 0.0])

    #hopper.controller.base_kp = np.array([0.0, 0.1])
    #hopper.controller.base_kv = np.array([0.0, 0.01])

    #while True:
    #    hopper._reset()

    #    for t in range(timesteps):
    #        hopper._step(goal)
    #        time.sleep(hopper.dt)
