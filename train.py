## TODO: Train your agent here.
import sys
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from agents.agent import MyAgent
from task import Task

runtime = 5.  # time limit of the episode
init_pose = np.array([0., 0., 1., 0., 0.,
                      0.])  # initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
init_velocities = np.array(
    [0., 0., 0.])  # initial velocity of the quadcopter in (x,y,z) dimensions
init_angle_velocities = np.array(
    [0., 0., 0.])  # initial radians/second for each of the three Euler angles
file_output = 'data.txt'  # file name for saved results

num_episodes = 500
target_pos = np.array([0., 0., 10.])
task = Task(init_pose=init_pose,
            init_velocities=init_velocities,
            init_angle_velocities=init_angle_velocities,
            runtime=runtime,
            target_pos=target_pos)
agent = MyAgent(task)
done = False

labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3',
          'rotor_speed4']
plot_data = {x: [] for x in ['episode', 'total_reward']}
results_array = []
plot_every_n_episodes = 50

for i_episode in range(1, num_episodes + 1):
    results = {x: [] for x in labels}
    episode_reward = 0
    step = 0

    state = agent.reset_episode()  # start a new episode
    while True:
        step += 1
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action=action, reward=reward, next_state=next_state,
                   done=done)
        state = next_state
        episode_reward += reward

        to_write = [task.sim.time] + list(task.sim.pose) + list(
            task.sim.v) + list(task.sim.angular_v) + list(action)

        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])

        if done:
            print("\rEpisode = {:4d} done. step = {}. episode_reward = {}".format(i_episode, step, episode_reward), end="")  # [debug]
            # if i_episode % plot_every_n_episodes == 0:
            #     fig = plt.figure()
            #     ax = fig.gca(projection='3d')
            #     ax.plot(results['x'], results['y'], results['z'], label='trajectory')
            #     ax.legend()
            #     plt.show()
            #
            #     plt.plot(results['time'], results['x'], label='x')
            #     plt.plot(results['time'], results['y'], label='y')
            #     plt.plot(results['time'], results['z'], label='z')
            #     plt.legend()
            #     _ = plt.ylim()
            #     plt.show()

            results_array.append(results)

            # plot_data['episode'].append(i_episode)
            # plot_data['total_reward'].append(agent.total_reward)
            break
    sys.stdout.flush()
