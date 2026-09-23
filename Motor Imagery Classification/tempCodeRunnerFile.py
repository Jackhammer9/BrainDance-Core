import mujoco
import mujoco.viewer
import os
import time

MODEL_DIR = "Motor Imagery Classification/mujoco_6dof_arm"
XML_PATH = os.path.join(MODEL_DIR, "6dof_arm.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Start at zero
    data.qpos[:] = 0

    while viewer.is_running():

        # ONLY J2 moves
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.0
        data.qpos[3] = 0.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0

        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(0.01)