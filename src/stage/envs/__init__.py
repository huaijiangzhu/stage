from gym.envs.registration import register

register(
    id='Kuka-v0',
    entry_point='stage.envs.kuka:KukaEnv'
)

register(
    id='TwoLink-v0',
    entry_point='stage.envs.twolink:TwoLinkEnv'
)