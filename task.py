import numpy as np

from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
            init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities,
                              runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Hover
        self.target_pos = init_pose[:3] if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        max_reward = 1
        min_reward = -1

        ed = (abs(self.sim.pose[:3] - self.target_pos)).sum() # euclidian distance
        avd = (abs(self.sim.angular_v)).sum() # angular v
        vd = (abs(self.sim.v)).sum() # velocity

        reward = 1. - ed/519. - avd / 20. - vd / 6000.
        # reward_z = 10 / abs(self.sim.pose[2] - self.target_pos[2]) - 2
        # loss_z = .2 * abs(self.sim.pose[2] - self.target_pos[2])
        # loss_xy = .01 * (abs(self.sim.pose[:2] - self.target_pos[:2])).sum()
        # reward = 1 - loss_z # - loss_xy
        # reward += 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        reward = np.maximum(np.minimum(reward, max_reward), min_reward)

        # if (abs(self.sim.pose[:3] - self.target_pos)).sum() < 1:
        #     reward = 3

        # if (self.sim.pose[2]) <= 0.1: # penalize falling to the ground
        #     reward = -3 * (self.sim.runtime - self.sim.time)
        #     self.sim.done = True

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(
                rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state


if __name__ == "__main__":
    task = Task(init_pose=np.array([0., 0., 10., 0., 0., 0.]),
                init_velocities=np.array([0., 0., 0.]),
                init_angle_velocities=np.array([0., 0., 0.]),
                runtime=5,
                target_pos=np.array([0., 0., 10.]))
    reward = task.get_reward()
    print(reward)
