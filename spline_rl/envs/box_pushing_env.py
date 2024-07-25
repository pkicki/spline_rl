import gymnasium as gym
import fancy_gym
import numpy as np
import time
import matplotlib.pyplot as plt


class BoxPushingEnv:
    def __init__(self, horizon=100, gamma=0.99, **kwargs):
        self.env = gym.make('fancy/BoxPushingConstrDensePDFF-v0', render_mode='human')
        #self.env = gym.make('fancy/BoxPushingDense-v0', render_mode='human')
        #self.env = gym.make('fancy/TableTennis4D-v0', render_mode='human')
        self.observation = self.env.reset()
        self.env.render()
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.q = []
        self.q_dot = []

        #dt = self.env.unwrapped.dt if hasattr(self.env.unwrapped, "dt") else 0.1
        #action_space = self._convert_gym_space(self.env.action_space)
        #observation_space = self._convert_gym_space(self.env.observation_space)
        #mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.q.append(observation[:7])
        self.q_dot.append(observation[7:14])
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation, info = self.env.reset()
        self.q = []
        self.q_dot = []
        return observation, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
    
    def reward(self):
        return self.env.reward()
    


if __name__ == "__main__":
    env = BoxPushingEnv()
    qs_d = []
    q_dots_d = []
    q_ddots_d = []

    for i in range(1000):
        action = env.action_space.sample()
        action = np.zeros(21)
        q_base = np.array([0.38706806, 0.17620842, 0.24989142, -2.39914377, -0.07986905,
                           2.56857367, 1.47951693])
        joint_id = 0
        dt = 0.02
        A = 0.5
        T = np.array([1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10])
        T[joint_id] = 3000.
        omega = 1 / (dt * T)
        q_d = q_base + A * np.sin(i * omega)
        q_dot_d = omega * A * np.cos(i * omega)
        q_ddot_d = omega**2 * A * -np.sin(i * omega)

        #T = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        T = 2. * np.ones(7)
        q0 = q_base
        dq0 = np.zeros(7)
        qk = q0 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dq = np.zeros(7)
        a_0 = q0
        a_1 = 3 * a_0 + dq0
        a_3 = qk
        a_2 = 3 * a_3 - dq
        t = i * dt / T
        print(t)
        q_d = a_3 * t**3 + a_2 * t**2 * (1 - t) + a_1 * t * (1 - t)**2 + a_0 * (1 - t)**3
        q_dot_d = 3 * a_3 * t**2 + a_2 * (-3*t**2 + 2*t) + a_1 * (3*t**2 - 4*t + 1) - a_0 * 3 * (1 - t)**2
        q_ddot_d = 6 * a_3 * t + a_2 * (-6*t + 2) + a_1 * (6*t - 4) + a_0 * 6 * (1 - t)
        q_dot_d /= T
        q_ddot_d /= T**2

        #action[joint_id] += np.sin(i * omega)
        #action[joint_id + 7] += omega * np.cos(i * omega)
        #action[joint_id + 14] += omega**2 * -np.sin(i * omega) 
        qs_d.append(q_d)
        q_dots_d.append(q_dot_d)
        q_ddots_d.append(q_ddot_d)
        action = np.concatenate([q_d, q_dot_d, q_ddot_d])
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            qs_d, q_dots_d, q_ddots_d = np.array(qs_d), np.array(q_dots_d), np.array(q_ddots_d)
            q, q_dot = np.array(env.q), np.array(env.q_dot)
            for k in [0, 1]:
                plt.figure()
                plt.subplot(311)
                plt.plot(q[:, k], label='q')
                plt.plot(qs_d[:, k], label='q_desired')
                plt.legend()
                plt.subplot(312)
                plt.plot(q_dot[:, k], label='q_dot')
                plt.plot(q_dots_d[:, k], label='q_dot_desired')
                plt.legend()
                plt.subplot(313)
                plt.plot(q_ddots_d[:, k], label='q_ddot_desired')
                plt.legend()
            plt.show()
            qs_d = []
            q_dots_d = []
            q_ddots_d = []
            a = 0
        env.render()
        time.sleep(1/env.metadata['render_fps'])

        if terminated or truncated:
              observation, info = env.reset()
                  