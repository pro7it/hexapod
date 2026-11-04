from pathlib import Path
import numpy as np

FRAME_SKIP = 10
MJCF_PATH = Path(__file__).resolve().parent / "model" / "hexapod.xml"

STAND_POSE = np.tile([0.0, 0.26, 0.26], 6)
INIT_HEIGHT = 0.117
RESET_NOISE = 0.02 
MAX_EPISODE_STEPS = 1000
GAIT_FREQ = 1.5
ACTION_FILTER = 0.8

CMD_VX = 0.15
CMD_VY = 0.0
CMD_YAW_RATE = 0.0

# reward
HEADING_SIGMA = 0.0076
VY_FILTER = 0.97
VY_SIGMA = 0.0001

STROKE_D = 0.05

AIR_TIME_MIN = 0.2
AIR_TIME_MAX = 0.5
CLEAR_MIN = 0.02
MIN_HEIGHT = 0.05
MIN_UPRIGHT = -0.7
FOOT_LIFT = 0.02

REWARD_WEIGHTS = {
    "r_vel":      2.0,
    "r_gait":     1.0,
    "r_heading":  1.0,
    "r_vy":       1.0,
    "r_air":      1.0,
    "pen_roll":   -100.0,
    "pen_pitch":  -30.0,
    "pen_stroke": -20.0,
    "pen_angvel": -5.0,
    "pen_vy":     -50.0,
    "pen_vz":     -100.0,
    "pen_rate":   -0.05,
    "pen_proximity": -1.0,
    "pen_clearance": -0.5,
}