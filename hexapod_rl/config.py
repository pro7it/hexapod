from pathlib import Path
import numpy as np

FRAME_SKIP = 10
MJCF_PATH = Path(__file__).resolve().parent / "model" / "hexapod.xml"
STAND_POSE = np.tile([0.0, 0.26, 0.26], 6)
INIT_HEIGHT = 0.117
ACTION_SCALE = 0.5
RESET_NOISE = 0.02 
MIN_HEIGHT = 0.05
MIN_UPRIGHT = -0.7
MAX_EPISODE_STEPS = 1000
GAIT_FREQ = 1.5
ACTION_FILTER = 0.8

# reward
TARGET_YAW_RATE = 0.0
YAW_FILTER = 0.97
YAW_SIGMA = 0.01

V_MAX = 0.15       # м/с, вище цієї швидкості r_vel перестає рости (≈ стеля чесної ходи)
VY_SIGMA = 0.002   # допуск для усередненої бічної швидкості

STROKE_D = 0.05    # м, лише одиниця масштабу для ходу стоп, не ціль

AIR_TIME_MIN = 0.2  # с: коротший відрив стопи — підскок (штраф), довший — крок (бонус)
AIR_TIME_MAX = 0.5  # с: стеля, щоб не було вигідно тримати ногу в повітрі

REWARD_WEIGHTS = {
    "r_vel":      2.0,
    "r_gait":     1.0,
    "r_yaw":      1.0,
    "r_vy":       1.0,
    "r_air":      1.0,
    "pen_roll":   -100.0,
    "pen_pitch":  -30.0,
    "pen_stroke": -20.0,
    "pen_rate":   -0.05,
}