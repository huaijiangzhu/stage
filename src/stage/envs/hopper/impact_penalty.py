class ImpactPenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        pass

    def get_reward(self):
        impact_force = -self.robot.get_total_ground_force()[2]

        if impact_force > self.params['max_impact_force']:
            return -self.params['k'] * impact_force
        else:
            return 0.0
