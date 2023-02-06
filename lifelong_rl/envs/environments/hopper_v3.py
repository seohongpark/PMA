import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils


DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 2,
    'distance': 3.0,
    'lookat': np.array((0.0, 0.0, 1.15)),
    'elevation': -20.0,
}


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hopper.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-0.2, 0.2),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True,
                 action_repetition=1,
                 ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.action_repetition = action_repetition
        mujoco_env.MujocoEnv.__init__(self, xml_file, 4 * action_repetition)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(
            np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(
            self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def compute_reward(self, ob, next_ob, action=None, task='default'):
        xposbefore = ob[:, 0]
        zposbefore = ob[:, 1]
        xposafter = next_ob[:, 0]
        zposafter = next_ob[:, 1]

        if task in ['default', 'forward']:
            task_reward = (xposafter - xposbefore) / self.dt * self._forward_reward_weight
        elif task == 'hop':
            task_reward = np.maximum(zposafter - zposbefore, 0) / self.dt * self._forward_reward_weight

        z = next_ob[:, 1]
        angle = next_ob[:, 2]
        state = next_ob[:, 2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=1)
        healthy_z = np.logical_and(min_z < z, z < max_z)
        healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)

        is_healthy = np.logical_and(healthy_state, np.logical_and(healthy_z, healthy_angle))
        if self._terminate_when_unhealthy:
            healthy_reward = np.full_like(task_reward, self._healthy_reward)
            done = 1 - is_healthy
        else:
            # healthy_reward = is_healthy * self._healthy_reward
            healthy_reward = np.zeros_like(task_reward)
            done = np.zeros_like(healthy_reward)

        reward = task_reward + healthy_reward

        return reward, done

    def step(self, action, task='default', render=False):
        x_position_before = self.sim.data.qpos[0]
        z_position_before = self.sim.data.qpos[1]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        z_position_after = self.sim.data.qpos[1]
        x_velocity = ((x_position_after - x_position_before) / self.dt)

        ctrl_cost = self.control_cost(action)

        if task in ['default', 'forward']:
            task_reward = self._forward_reward_weight * x_velocity
        elif task == 'hop':
            task_reward = self._forward_reward_weight * np.maximum(z_position_after - z_position_before, 0) / self.dt
        healthy_reward = self.healthy_reward

        if self._terminate_when_unhealthy:
            rewards = task_reward + healthy_reward
        else:
            rewards = task_reward
        # costs = ctrl_cost
        costs = 0.

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'coordinates': np.array([x_position_before, 0.]),
            'next_coordinates': np.array([x_position_after, 0.]),
        }

        if render:
            info['render'] = self.render(mode='rgb_array', width=100, height=100).transpose(2, 0, 1)

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def plot_trajectory(self, trajectory, color, ax):
        trajectory = trajectory.copy()
        # https://stackoverflow.com/a/20474765/2182622
        from matplotlib.collections import LineCollection
        #linewidths = np.linspace(0.5, 1.5, len(trajectory))
        #linewidths = np.linspace(0.1, 0.8, len(trajectory))
        linewidths = np.linspace(0.2, 1.2, len(trajectory))
        points = np.reshape(trajectory, (-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidths, color=color)
        ax.add_collection(lc)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot trajectories onto given ax."""
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        if plot_axis is None:
            plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        if plot_axis is not None:
            ax.axis(plot_axis)
            ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            coords = []
            for _, info_dict in enumerate(trajectory['env_infos']):
                coords.append(info_dict['coordinates'])
            coords.append(info_dict['next_coordinates'])
            coordinates_trajectories.append(np.array(coords))
        for i, traj in enumerate(coordinates_trajectories):
            # Designed to fit in [-5, 5] * [-5, 5] -> roughly --> now multiplied by 20.
            traj[:, 1] = (i - len(coordinates_trajectories) / 2) / 1.25
        return coordinates_trajectories
