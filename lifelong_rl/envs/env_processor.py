import gym

from lifelong_rl.envs.child_policy_env import ChildPolicyEnv
from lifelong_rl.envs.wrappers import NormalizedBoxEnv, NonTerminatingEnv

gym.logger.set_level(40)  # stop annoying Box bound precision error


def make_env(env_name, terminates=True, **kwargs):
    env = None
    base_env = None
    env_infos = dict()

    """
    Episodic reinforcement learning
    """
    if env_name == 'maze':
        from lifelong_rl.envs.environments.maze_env import MazeEnv
        base_env = MazeEnv
        env_infos['mujoco'] = False
    elif env_name == 'half_cheetah':
        from lifelong_rl.envs.environments.half_cheetah import HalfCheetahEnv
        base_env = HalfCheetahEnv
        env_infos['mujoco'] = True
        kwargs.update(
            expose_all_qpos=True,
        )
    elif env_name == 'ant-v3':
        from lifelong_rl.envs.environments.ant_v3 import AntEnv
        base_env = AntEnv
        env_infos['mujoco'] = True
        kwargs.update(
            exclude_current_positions_from_observation=False,
        )
        if 'exclude_contact_forces' not in kwargs:
            kwargs.update(
                exclude_contact_forces=True,
            )
    elif env_name == 'hopper-v3':
        from lifelong_rl.envs.environments.hopper_v3 import HopperEnv
        base_env = HopperEnv
        env_infos['mujoco'] = True
        kwargs.update(
            exclude_current_positions_from_observation=False,
        )
    elif env_name == 'walker2d-v3':
        from lifelong_rl.envs.environments.walker2d_v3 import Walker2dEnv
        base_env = Walker2dEnv
        env_infos['mujoco'] = True
        kwargs.update(
            exclude_current_positions_from_observation=False,
        )
    elif env_name == 'ip':
        from lifelong_rl.envs.environments.inverted_pendulum import InvertedPendulumEnv
        base_env = InvertedPendulumEnv
        env_infos['mujoco'] = True
    elif env_name == 'idp':
        from lifelong_rl.envs.environments.inverted_double_pendulum import InvertedDoublePendulumEnv
        base_env = InvertedDoublePendulumEnv
        env_infos['mujoco'] = True
    elif env_name == 'reacher':
        from lifelong_rl.envs.environments.reacher import ReacherEnv
        base_env = ReacherEnv
        env_infos['mujoco'] = True

    if env is None and base_env is None:
        raise NameError('env_name not recognized')

    if 'cp_info' in kwargs:
        cp_info = kwargs.pop('cp_info')
    else:
        cp_info = None

    if env is None:
        env = base_env(**kwargs)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        env = NormalizedBoxEnv(env)

    if not terminates:
        env = NonTerminatingEnv(env)

    if cp_info is not None:
        env = ChildPolicyEnv(
            env,
            **cp_info,
        )

    return env, env_infos
