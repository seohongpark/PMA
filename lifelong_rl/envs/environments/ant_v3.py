import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 exclude_contact_forces=False):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._exclude_contact_forces = exclude_contact_forces

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

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
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def compute_reward(self, ob, next_ob, action=None, task='default'):
        xposbefore = ob[:, 0]
        yposbefore = ob[:, 1]
        xposafter = next_ob[:, 0]
        yposafter = next_ob[:, 1]

        forward_reward = (xposafter - xposbefore) / self.dt
        sideward_reward = (yposafter - yposbefore) / self.dt

        if task in ['default', 'forward']:
            task_reward = forward_reward
        elif task == 'backward':
            task_reward = -forward_reward
        elif task == 'north':
            task_reward = sideward_reward
        elif task == 'south':
            task_reward = -sideward_reward

        min_z, max_z = self._healthy_z_range
        is_healthy = (min_z <= next_ob[:, 2]) & (next_ob[:, 2] <= max_z)
        if self._terminate_when_unhealthy:
            healthy_reward = np.full_like(forward_reward, self._healthy_reward)
            done = 1 - is_healthy
        else:
            healthy_reward = is_healthy * self._healthy_reward
            done = np.zeros_like(healthy_reward)

        reward = task_reward + healthy_reward

        return reward, done

    def step(self, action, task='default', render=False):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        sideward_reward = y_velocity
        healthy_reward = self.healthy_reward

        if task in ['default', 'forward']:
            task_reward = forward_reward
        elif task == 'backward':
            task_reward = -forward_reward
        elif task == 'north':
            task_reward = sideward_reward
        elif task == 'south':
            task_reward = -sideward_reward

        rewards = task_reward + healthy_reward
        # costs = ctrl_cost + contact_cost
        costs = 0.

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,

            'coordinates': xy_position_before,
            'next_coordinates': xy_position_after,
        }

        if render:
            info['render'] = self.render(mode='rgb_array', width=100, height=100).transpose(2, 0, 1)

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._exclude_contact_forces:
            observations = np.concatenate((position, velocity))
        else:
            observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
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
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot trajectories onto given ax."""
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)
        if plot_axis is not None:
            ax.axis(plot_axis)
            ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            coords = []
            for _, info_dict in enumerate(trajectory['env_infos']):
                coords.append(info_dict['coordinates'])
            coords.append(info_dict['next_coordinates'])
            coordinates_trajectories.append(np.array(coords))
        return coordinates_trajectories

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot multiple trajectories onto ax"""
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)
