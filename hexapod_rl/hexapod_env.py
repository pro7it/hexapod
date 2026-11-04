import gymnasium as gym
import numpy as np

import mujoco

from config import *
from controller import Controller

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
        self.foot_site_ids = [
            self.model.site(f"{leg}_foot").id
            for leg in ["l1", "l2", "l3", "r1", "r2", "r3"]
        ]

        g = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        parts = ["femur_col", "tibia_col"]
        self.leg_pairs = [[(g(f"{a}_{pa}"), g(f"{b}_{pb}")) for pa in parts for pb in parts]
                          for a, b in [("l1", "l2"), ("l2", "l3"), ("r1", "r2"), ("r2", "r3")]]
        self._fromto = np.zeros(6)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(Controller.OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        self.controller = Controller(self.dt,
                                     self.model.actuator_ctrlrange[:, 0],
                                     self.model.actuator_ctrlrange[:, 1])

    def _get_obs(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        gravity = -R[2, :]
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        return self.controller.observe(gravity, gyro, self._yaw())

    def _yaw(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        return np.arctan2(R[1, 0], R[0, 0])

    def _feet_x_body(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        rel = self.data.site_xpos[self.foot_site_ids] - self.data.xpos[self.base_id]
        return (rel @ R)[:, 0]

    def _stroke_penalty(self):
        half = np.sin(self.controller.phase) > 0
        if half == self.half:
            return 0.0

        feet_x = self._feet_x_body()
        sA = self.half
        was_stance = np.array([sA, not sA, sA, not sA, sA, not sA])
        s = np.where(was_stance, -1.0, 1.0)
        d = s * (feet_x - self.half_start_x) / STROKE_D

        pen = 0.0 if self.first_half else float(np.mean((d - d.mean()) ** 2))
        self.first_half = False
        self.half = half
        self.half_start_x = feet_x
        return pen

    def _proximity_penalty(self):
        pen = 0.0
        for pairs in self.leg_pairs:
            d = min(mujoco.mj_geomDistance(self.model, self.data, g1, g2, CLEAR_MIN, self._fromto)
                    for g1, g2 in pairs)
            if d < CLEAR_MIN:
                pen += (1 - d / CLEAR_MIN) ** 2
        return pen

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        noise = self.np_random.uniform(-RESET_NOISE, RESET_NOISE, size=self.model.nu)
        self.data.qpos[2] = INIT_HEIGHT
        self.data.qpos[7:] = STAND_POSE + noise
        self.data.ctrl[:] = STAND_POSE

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.controller.reset(phase=self.np_random.uniform(0, 2 * np.pi), heading=self._yaw())
        self.vy_avg = 0.0

        self.air_time = np.zeros(6)

        self.half = np.sin(self.controller.phase) > 0
        self.half_start_x = self._feet_x_body()
        self.first_half = True

        obs = self._get_obs()
        return obs, {}

    def _air_time_reward(self, contact):
        first_contact = (self.air_time > 0) & contact
        air = np.minimum(self.air_time, AIR_TIME_MAX)
        reward = float(np.sum((air - AIR_TIME_MIN) * first_contact))

        self.air_time += self.dt
        self.air_time[contact] = 0.0
        return reward

    def reward(self, action, prev_action, obs):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        vx, vy, vz = R.T @ self.data.qvel[:3]

        gravity = obs[36:39]
        gyro = obs[39:42]

        self.vy_avg = VY_FILTER * self.vy_avg + (1 - VY_FILTER) * vy
        vx_cmd, vy_cmd, yaw_rate_cmd = self.controller.command

        contact = self.data.sensordata[self.touch_adrs] > 0.0
        stance_A = np.sin(self.controller.phase) > 0
        expected = np.array([stance_A, not stance_A, stance_A,
                             not stance_A, stance_A, not stance_A])

        feet_z = self.data.site_xpos[self.foot_site_ids, 2]
        swing = ~expected

        terms = {
            "r_vel": np.clip(vx / vx_cmd, -1.0, 1.0),
            "r_gait": 2 * np.mean(contact == expected) - 1,
            "r_heading": np.exp(-(obs[44] ** 2) / HEADING_SIGMA),
            "r_vy": np.exp(-((self.vy_avg - vy_cmd) ** 2) / VY_SIGMA),
            "r_air": self._air_time_reward(contact),
            "pen_roll": gravity[1] ** 2,
            "pen_pitch": gravity[0] ** 2,
            "pen_stroke": self._stroke_penalty(),
            "pen_angvel": gyro[0] ** 2 + gyro[1] ** 2 + (gyro[2] - yaw_rate_cmd) ** 2,
            "pen_vy": (vy - vy_cmd) ** 2,
            "pen_vz": vz ** 2,
            "pen_rate": np.sum((action - prev_action) ** 2),
            "pen_proximity": self._proximity_penalty(),
            "pen_clearance": float(np.sum(np.clip(1 - feet_z[swing] / FOOT_LIFT, 0, None) ** 2)),
        }

        reward = sum(
            REWARD_WEIGHTS[k] * v
            for k, v in terms.items()
        )

        return reward, terms

    def step(self, action):
        ctrl, prev_action = self.controller.act(action)
        action = self.controller.last_action   # обрізана до [-1, 1]
        self.data.ctrl[:] = ctrl

        for _ in range(FRAME_SKIP): mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        R = self.data.xmat[self.base_id].reshape(3, 3)
        velocity_local = R.T @ self.data.qvel[:3]

        reward, terms = self.reward(action, prev_action, obs)

        gravity_z = obs[38]
        terminated = bool(self.data.qpos[2] < MIN_HEIGHT or gravity_z > MIN_UPRIGHT)

        self.step_count += 1
        truncated = self.step_count >= MAX_EPISODE_STEPS

        info = {
            "forward_vel": velocity_local[0],
            "side_vel": velocity_local[1],
            "heading_err_deg": np.degrees(abs(obs[44])),
            "reward": reward,
            **terms,
        }

        if terminated or truncated:
            print(
                f"height={self.data.qpos[2]:.4f}, "
                f"gravity_z={gravity_z:.4f}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )

        return obs, reward, terminated, truncated, info