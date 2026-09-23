import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame
import os


MODEL_DIR = "Motor Imagery Classification/mujoco_6dof_arm"
XML_PATH = os.path.join(MODEL_DIR, "6dof_arm.xml")

MOVE_SPEED = 0.15

IK_GAIN_POSITION = 5.0
IK_GAIN_ORIENTATION = 5.0
DAMPING = 0.05

DT = 0.002


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint4"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint6"),
]

qpos_ids = [
    model.jnt_qposadr[joint_id]
    for joint_id in joint_ids
]

ee_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "link6"
)


data.qpos[:] = 0
mujoco.mj_forward(model, data)

target_pos = data.xpos[ee_body_id].copy()

target_rot = data.xmat[ee_body_id].reshape(3, 3).copy()


pygame.init()

screen = pygame.display.set_mode((500, 180))
pygame.display.set_caption("6-DOF Robot IK Controller")


def rotation_error(target_rotation, current_rotation):

    rotation_difference = (
        target_rotation @ current_rotation.T
    )

    quat = np.zeros(4)

    mujoco.mju_mat2Quat(
        quat,
        rotation_difference.flatten()
    )

    return 2.0 * quat[1:4]


def inverse_kinematics(target_position, target_orientation):

    current_position = data.xpos[ee_body_id].copy()

    current_orientation = (
        data.xmat[ee_body_id]
        .reshape(3, 3)
        .copy()
    )

    position_error = (
        target_position - current_position
    )

    orientation_error = rotation_error(
        target_orientation,
        current_orientation
    )

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacBody(
        model,
        data,
        jacp,
        jacr,
        ee_body_id
    )

    J_position = jacp[:, qpos_ids]
    J_rotation = jacr[:, qpos_ids]

    J = np.vstack(
        (
            J_position,
            J_rotation
        )
    )

    error = np.concatenate(
        (
            position_error * IK_GAIN_POSITION,
            orientation_error * IK_GAIN_ORIENTATION
        )
    )

    JJt = J @ J.T

    damping_matrix = (
        DAMPING ** 2
    ) * np.eye(6)

    qdot = (
        J.T
        @ np.linalg.solve(
            JJt + damping_matrix,
            error
        )
    )

    max_joint_velocity = 2.0

    qdot = np.clip(
        qdot,
        -max_joint_velocity,
        max_joint_velocity
    )

    for i in range(6):

        qpos_id = qpos_ids[i]

        data.qpos[qpos_id] += (
            qdot[i] * DT
        )

        joint_id = joint_ids[i]

        if model.jnt_limited[joint_id]:

            lower = model.jnt_range[joint_id][0]
            upper = model.jnt_range[joint_id][1]

            data.qpos[qpos_id] = np.clip(
                data.qpos[qpos_id],
                lower,
                upper
            )


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    running = True

    while running and viewer.is_running():

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        movement = np.zeros(3)

        if keys[pygame.K_w]:
            movement[0] += MOVE_SPEED

        if keys[pygame.K_s]:
            movement[0] -= MOVE_SPEED

        if keys[pygame.K_a]:
            movement[1] += MOVE_SPEED

        if keys[pygame.K_d]:
            movement[1] -= MOVE_SPEED

        if keys[pygame.K_q]:
            movement[2] += MOVE_SPEED

        if keys[pygame.K_e]:
            movement[2] -= MOVE_SPEED

        target_pos += movement * DT

        inverse_kinematics(
            target_pos,
            target_rot
        )

        mujoco.mj_forward(model, data)

        viewer.sync()

        pygame.display.set_caption(
            "6-DOF IK | "
            f"X={target_pos[0]:.3f} "
            f"Y={target_pos[1]:.3f} "
            f"Z={target_pos[2]:.3f}"
        )

        time.sleep(DT)


pygame.quit()