import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a, task='default', render=False):
        xy_position_before = self.get_body_com("fingertip")[:2].copy()
        self.do_simulation(a, self.frame_skip)
        xy_position_after = self.get_body_com("fingertip")[:2].copy()
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        reward = reward_dist
        ob = self._get_obs()
        done = False
        info = dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,

            coordinates=xy_position_before,
            next_coordinates=xy_position_after,
        )
        if render:
            info['render'] = self.render(mode='rgb_array', width=100, height=100).transpose(2, 0, 1)

        return ob, reward, done, info

    def compute_reward(self, ob, next_ob, action=None, task='default'):
        vec = next_ob[:, -3:]
        reward = -np.linalg.norm(vec, axis=1)

        done = np.zeros_like(reward)

        return reward, done

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

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
