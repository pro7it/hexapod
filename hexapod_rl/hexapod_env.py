import gymnasium as gym
import numpy as np

import mujoco

from config import *

class HexapodEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * FRAME_SKIP

        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.gyro_adr = self.model.sensor("imu_gyro").adr[0]
        self.touch_adrs = [
            self.model.sensor(f"{leg}_touch").adr[0]
            for leg in ["l1", "l2", "l3", "r1", "r2", "r3"]
        ]

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(44,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        self.ctrl_low  = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]

    def _get_obs(self):
        joint_pos = self.data.qpos[7:] - STAND_POSE
        joint_vel = self.data.qvel[6:] * 0.05
        R = self.data.xmat[self.base_id].reshape(3, 3)
        gravity = -R[2, :]
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        phase = [np.sin(self.phase), np.cos(self.phase)]

        return np.concatenate([joint_pos, joint_vel, gravity, gyro, phase]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        noise = self.np_random.uniform(-RESET_NOISE, RESET_NOISE, size=self.model.nu)
        self.data.qpos[2] = INIT_HEIGHT
        self.data.qpos[7:] = STAND_POSE + noise
        self.data.ctrl[:] = STAND_POSE

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_action = np.zeros(self.model.nu)
        self.target = STAND_POSE.copy()

        self.phase = self.np_random.uniform(0, 2 * np.pi)

        self.air_time = np.zeros(6)
        self.last_contact = np.ones(6, dtype=bool)

        self.yaw_rate_avg = 0.0

        obs = self._get_obs()
        return obs, {}

    def reward(self, action, obs):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        velocity_world = self.data.qvel[:3]
        vx, vy, vz = R.T @ velocity_world

        gravity = obs[36:39]
        gyro = obs[39:42]

        self.yaw_rate_avg = YAW_FILTER * self.yaw_rate_avg + (1 - YAW_FILTER) * gyro[2]

        contact = self.data.sensordata[self.touch_adrs] > 0.0
        contact_filt = contact | self.last_contact
        self.last_contact = contact

        first_contact = (self.air_time > 0) & contact_filt
        self.air_time += self.dt
        air = np.minimum(self.air_time, AIR_TIME_MAX)
        self.air_time[contact_filt] = 0.0

        stance_A = np.sin(self.phase) > 0
        expected = np.array([stance_A, not stance_A, stance_A,
                            not stance_A, stance_A, not stance_A])

        terms = {
            "r_vel": np.exp(-((vx - TARGET_VEL) ** 2) / VEL_SIGMA),
            "r_air": np.sum((air - AIR_TIME_MIN) * first_contact),
            "r_yaw": np.exp(-((self.yaw_rate_avg - TARGET_YAW_RATE) ** 2) / YAW_SIGMA),
            "r_gait": 2 * np.mean(contact == expected) - 1,
            "alive": 1.0,
            "pen_yaw": gyro[2] ** 2,
            "pen_tilt": gravity[0] ** 2 + gravity[1] ** 2,
            "pen_rate": np.sum((action - self.prev_action) ** 2),
            "pen_vy": vy ** 2,
            "pen_vz": vz ** 2,
        }

        self.prev_action = action.copy()

        reward = sum(
            REWARD_WEIGHTS[k] * v
            for k, v in terms.items()
        )

        return reward, terms

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        new_target = STAND_POSE + ACTION_SCALE * action
        self.target = ACTION_FILTER * self.target + (1 - ACTION_FILTER) * new_target
        self.data.ctrl[:] = np.clip(self.target, self.ctrl_low, self.ctrl_high)

        for _ in range(FRAME_SKIP): mujoco.mj_step(self.model, self.data)
        self.phase = (self.phase + 2 * np.pi * GAIT_FREQ * self.dt) % (2 * np.pi)

        obs = self._get_obs()

        R = self.data.xmat[self.base_id].reshape(3, 3)
        velocity_local = R.T @ self.data.qvel[:3]

        reward, terms = self.reward(action, obs)

        gravity_z = obs[38]
        terminated = bool(self.data.qpos[2] < MIN_HEIGHT or gravity_z > MIN_UPRIGHT)

        self.step_count += 1
        truncated = self.step_count >= MAX_EPISODE_STEPS

        info = {
            "forward_vel": velocity_local[0],
            "side_vel": velocity_local[1],
            "reward": reward,
            "r_vel": terms["r_vel"],
            "r_air": terms["r_air"],
            "r_yaw": terms["r_yaw"],
            "r_gait": terms["r_gait"],
            "pen_yaw": terms["pen_yaw"],
            "pen_tilt": terms["pen_tilt"],
            "pen_rate": terms["pen_rate"],
            "pen_vy": terms["pen_vy"],
            "pen_vz": terms["pen_vz"],
        }

        if terminated or truncated:
            print(
                f"height={self.data.qpos[2]:.4f}, "
                f"gravity_z={gravity_z:.4f}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )

        return obs, reward, terminated, truncated, info