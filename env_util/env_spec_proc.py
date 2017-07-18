def get_env_spec(args, env_spec_q, **kwargs):
    from env_util.envs import create_env
    kwargs.update({"no_config":True})
    env = create_env(args.env_name, 0, 0, **kwargs)
    env_spec_q.put((env.observation_space_omni, env.observation_space, env.action_space))
    env.close()
    env_spec_q.close()
