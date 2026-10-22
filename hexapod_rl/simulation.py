from pathlib import Path
import time

import mujoco
import mujoco.viewer

MJCF_PATH = Path(__file__).resolve().parent / "model" / "hexapod.xml"

model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # data.ctrl[:] = 0.5
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model,data)
        viewer.sync()
        slack = model.opt.timestep - (time.time() - step_start)
        if slack > 0: time.sleep(slack)