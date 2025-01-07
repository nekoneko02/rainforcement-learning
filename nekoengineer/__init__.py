from gymnasium.envs.registration import register

register(
    id="nekoengineer/CatPlayground-v0",
    entry_point="nekoengineer.envs:CatPlaygroundEnv",
)
