import numpy as np

from config import *

class Controller:
    OBS_SIZE = 45

    def __init__(self, dt, ctrl_low, ctrl_high):
        self.dt = dt
        self.ctrl_low = ctrl_low
        self.ctrl_high = ctrl_high

    def reset(self, phase, heading):
        self.target = STAND_POSE.copy()
        self.last_action = np.zeros(len(STAND_POSE))
        self.phase = phase
        self.target_heading = heading
        self.command = np.array([CMD_VX, CMD_VY, CMD_YAW_RATE])   # vx, vy, ω

    def act(self, action):
        action = np.clip(action, -1.0, 1.0)
        prev_action = self.last_action
        self.last_action = action.copy()
        self.target = ACTION_FILTER * self.target + (1 - ACTION_FILTER) * (STAND_POSE + action)

        self.phase = (self.phase + 2 * np.pi * GAIT_FREQ * self.dt) % (2 * np.pi)
        self.target_heading += self.command[2] * self.dt

        return np.clip(self.target, self.ctrl_low, self.ctrl_high), prev_action

    def heading_err(self, yaw):
        return (self.target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

    def observe(self, gravity, gyro, yaw):
        return np.concatenate([
            self.target - STAND_POSE,
            self.last_action,
            gravity,
            gyro,
            [np.sin(self.phase), np.cos(self.phase)],
            [self.heading_err(yaw)],
        ]).astype(np.float32)
