import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "6dof_arm.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Load the defined CAD home pose: J1=0, J2=+90°, J3=0, J4=0, J5=0, J6=0
data.qpos[:] = model.key_qpos[0]
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
