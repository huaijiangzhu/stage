from gym.envs.registration import register

register(
    id='MBRLReacher3D-v0',
    entry_point='stage.envs.reacher:Reacher3DEnv'
)

register(
    id='MBRLHalfCheetah-v0',
    entry_point='stage.envs.half_cheetah:HalfCheetahEnv'
)

register(
    id='Kuka-v0',
    entry_point='stage.envs.kuka:KukaEnv'
)

register(
    id='KukaPin-v0',
    entry_point='stage.envs.kuka_pin:KukaPinEnv'
)

register(
    id='TwoLink-v0',
    entry_point='stage.envs.twolink:TwoLinkEnv'
)