import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, terminate_when_unhealthy=True):
        utils.EzPickle.__init__(self)
        self._terminate_when_unhealthy = terminate_when_unhealthy
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum.xml', 5)

    def step(self, action, task='default', render=False):
        ob_before = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        # r = alive_bonus - dist_penalty - vel_penalty
        if task in ['default', 'forward']:
            if self._terminate_when_unhealthy:
                r = 1.0 + (ob[0] - ob_before[0]) / self.dt
                done = bool(y <= 1)
            else:
                r = (y > 1) + (ob[0] - ob_before[0]) / self.dt / 10
                done = False
        elif task == 'stay':
            if self._terminate_when_unhealthy:
                r = 1.0 - ob[0] ** 2
                done = bool(y <= 1)
            else:
                r = (y > 1) - ob[0] ** 2
                done = False
        info = {
            'coordinates': np.array([ob_before[0], 0.]),
            'next_coordinates': np.array([ob[0], 0.]),
        }
        if render:
            info['render'] = self.render(mode='rgb_array', width=100, height=100).transpose(2, 0, 1)

        return ob, r, done, info

    def compute_reward(self, ob, next_ob, action=None, task='default'):
        xposbefore = ob[:, 0]
        xposafter = next_ob[:, 0]

        if task in ['default', 'forward']:
            forward_reward = (xposafter - xposbefore) / self.dt
            if self._terminate_when_unhealthy:
                reward = forward_reward + 1.
                done = (next_ob[:, -1] <= 1)
            else:
                reward = forward_reward / 10 + (next_ob[:, -1] > 1)
                done = np.zeros_like(forward_reward)
        elif task == 'stay':
            stay_reward = -xposafter ** 2
            if self._terminate_when_unhealthy:
                reward = stay_reward + 1.
                done = (next_ob[:, -1] <= 1)
            else:
                reward = stay_reward + (next_ob[:, -1] > 1)
                done = np.zeros_like(stay_reward)

        return reward, done

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10),
            self.sim.data.site_xpos[0][2:3],  # Added for computing done, y coord
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

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
