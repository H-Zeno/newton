
import asyncio
import newton
import os

import numpy as np
import warp as wp

from dotenv import load_dotenv
from dataclasses import dataclass

from newton.examples.cardboard.ik_solver import InverseKinematicsSolver
from newton.examples.cardboard.box_creator import create_box, BoxConfiguration
from newton.examples.cardboard.cardboard_kernels import joint_update_equilibrium_kernel, joint_apply_signed_spring_torque_kernel

from tactile_teleop_sdk import TactileAPI, TactileConfig
load_dotenv()

robot_distance = 0.3  # Distance between the two robots


def warp_transform_to_matrix(tf: wp.transform) -> np.ndarray:
    """Convert warp transform to 4x4 transformation matrix."""
    matrix = np.eye(4, dtype=np.float32)
    pos = wp.transform_get_translation(tf)
    quat = wp.transform_get_rotation(tf)

    # Convert quaternion to rotation matrix
    # quat is (x, y, z, w) in warp
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    matrix[0, 0] = 1 - 2 * y * y - 2 * z * z
    matrix[0, 1] = 2 * x * y - 2 * z * w
    matrix[0, 2] = 2 * x * z + 2 * y * w
    matrix[1, 0] = 2 * x * y + 2 * z * w
    matrix[1, 1] = 1 - 2 * x * x - 2 * z * z
    matrix[1, 2] = 2 * y * z - 2 * x * w
    matrix[2, 0] = 2 * x * z - 2 * y * w
    matrix[2, 1] = 2 * y * z + 2 * x * w
    matrix[2, 2] = 1 - 2 * x * x - 2 * y * y

    matrix[0, 3] = pos[0]
    matrix[1, 3] = pos[1]
    matrix[2, 3] = pos[2]

    return matrix


def matrix_to_warp_transform(matrix: np.ndarray) -> tuple:
    """Convert 4x4 matrix to warp transform (pos, quat)."""
    pos = wp.vec3(matrix[0, 3], matrix[1, 3], matrix[2, 3])

    # Extract rotation matrix
    R = matrix[:3, :3]

    # Convert rotation matrix to quaternion
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    quat = wp.quat(x, y, z, w)
    return pos, quat


class Goal:
    def __init__(self):
        self.initial_transform = None
        self.origin_transform = None
        self.target_transform = None

    def apply_goal_to_arm(
        self,
        goal: dict,
        current_eef_tf: wp.transform,
    ):
        """
        Update arm target based on goal commands.

        Args:
            goal: Dictionary with keys:
                - reset_to_init: bool - reset to initial transform
                - reset_reference: bool - reset reference frame
                - relative_transform: np.ndarray (4x4) - relative transform
                - gripper_closed: bool - gripper state
        """
        # Initialize transforms on first call
        if self.initial_transform is None:
            self.initial_transform = warp_transform_to_matrix(current_eef_tf)
            self.origin_transform = self.initial_transform.copy()
            self.target_transform = self.initial_transform.copy()

        # Handle reset commands
        if goal.reset_to_init:
            self.target_transform = self.initial_transform.copy()
            self.origin_transform = self.initial_transform.copy()
        elif goal.reset_reference:
            # Reset reference to current end-effector pose
            self.origin_transform = warp_transform_to_matrix(current_eef_tf)
        elif goal.relative_transform is not None:
            relative_transform = goal.relative_transform

            # Coordinate transform to local robot frame
            transformation_matrix = np.eye(4, dtype=np.float32)
            transformation_matrix[:3, :3] = self.origin_transform[:3, :3]

            # Transform relative motion to robot's local frame
            relative_transform = np.linalg.inv(transformation_matrix) @ (
                relative_transform @ transformation_matrix
            )

            # Apply relative transform to origin
            self.target_transform = self.origin_transform @ relative_transform

        # Update gripper state
        if goal.gripper_closed is False:
            gripper_closed = False
        else:
            gripper_closed = True

        # Extract position from transformation matrix
        target_pos = self.target_transform[:3, 3]

        # Extract rotation matrix and convert to Euler angles (XYZ convention)
        rotation_matrix = self.target_transform[:3, :3]

        # Convert rotation matrix to Euler angles (XYZ)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        target_ori = np.array([x, y, z])

        return target_pos, target_ori, gripper_closed


@dataclass
class CardboardJointConfiguration:
    default_joint_eq_pos: float = float(wp.radians(7.0))
    target_ke: float = 0.115 #50.0
    target_kd: float = 0.035 #0.7
    friction: float = 0.05
    min_joint_eq_pos: float = float(wp.radians(-52.0))
    max_joint_eq_pos: float = float(wp.radians(52.0))
    min_joint_limit: float = float(wp.radians(-178.0))
    max_joint_limit: float = float(wp.radians(178.0))
    plasticity_angle: float = float(wp.radians(35.0))
    resistance_ke: float = 0.05
    
async def main():
    wp.set_device("cuda")
    
    piper_urdf_path = "/home/zhamers/piper-robot-server/URDF/Piper/piper_description.urdf"

    tactile_config = TactileConfig.from_env()
    api = TactileAPI(tactile_config)
    await api.connect_robot()
    
    # viewer setup
    viewer = newton.viewer.ViewerGL(headless=False)
    
    # Connect VR controls 
    await api.connect_controller(type="parallel_gripper_vr_controller", robot_components=["left", "right"])

    # Connect camera streamer with viewer resolution
    fb_w, fb_h = viewer.renderer.window.get_framebuffer_size()
    await api.connect_camera(camera_name="camera_0", height=fb_h, width=fb_w)
    
    
    
    

    # sim params
    sim_time = 0.0
    frame_dt = 1.0 / 50
    substeps = 4
    sim_dt = frame_dt / substeps

    scene = newton.ModelBuilder()
    scene.add_ground_plane()

    #
    # Cardboard box setup
    #
    joint_cfg = CardboardJointConfiguration()
    box_cfg = BoxConfiguration()
        
    # Create the box to determine joint count
    box_builder = create_box(box_cfg, joint_cfg, key="box", show_visuals=True, show_collision=False)
    num_revolute_joints = box_builder.joint_count - 1

    # Initialize joint equilibrium positions for all revolute joints
    joint_eq_pos_array = wp.array([joint_cfg.default_joint_eq_pos] * num_revolute_joints, dtype=float)

    scene.add_builder(box_builder,
        environment=-1
    )
    
    #
    # Robot setup
    #

    piper1 = newton.ModelBuilder()
    piper1.add_urdf(
        piper_urdf_path,
        xform=wp.transform(
            (0.0, -robot_distance, 0.0),
            wp.quat_identity(),
        ),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )
    piper2 = newton.ModelBuilder()
    piper2.add_urdf(
        piper_urdf_path,
        xform=wp.transform(
            (0.0, robot_distance, 0.0),
            wp.quat_identity(),
        ),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )

    for i, _ in enumerate(piper1.joint_target_ke):
        if i < 6:  # Arm joints (0-5)
            piper1.joint_target_ke[i] = 30.0
            piper1.joint_target_kd[i] = 1.0
            piper1.joint_friction[i] = 0.4
            piper2.joint_target_ke[i] = 30.0
            piper2.joint_target_kd[i] = 1.0
            piper2.joint_friction[i] = 0.4
        else:  # Gripper joints (6-7)
            # Higher stiffness and lower friction for faster gripper response
            piper1.joint_target_ke[i] = 100.0
            piper1.joint_target_kd[i] = 2.0
            piper1.joint_friction[i] = 0.1
            piper2.joint_target_ke[i] = 100.0
            piper2.joint_target_kd[i] = 2.0
            piper2.joint_friction[i] = 0.1

    scene.add_builder(piper1, environment=-1)
    scene.add_builder(piper2, environment=-1)

    # scene.shape_collision_filter_pairs = []


    model = scene.finalize()

    # model.shape_collision_filter_pairs = []

    solver = newton.solvers.SolverMuJoCo(model=model, iterations=20, njmax=256)
    # solver = newton.solvers.SolverXPBD(model=model, iterations=200)
    # solver = newton.solvers.SolverFeatherstone(model=model, update_mass_matrix_interval=1)

    state_0 = model.state()
    state_1 = model.state()

    contacts = model.collide(state_0)

    control = model.control()

    viewer.set_model(model)

    # Set camera position and orientation based on the provided coordinates
    camera_pos = wp.vec3(-0.40, 0.13, 0.72)
    camera_pitch = -33.4
    camera_yaw = -360.4
    viewer.set_camera(camera_pos, camera_pitch, camera_yaw)

    # not required for MuJoCo, but required for other solvers
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Initialize the solver
    ik = InverseKinematicsSolver(use_gui=False)
    ik.load_robot(
        "piper",
        piper_urdf_path,
        ee_link_index=6,
    )

    left_goal_obj = Goal()
    right_goal_obj = Goal()
    # main loop
    while viewer.is_running():
        if not viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                for _ in range(substeps):
                    state_0.clear_forces()

                    # get forces from the viewer and apply to the state
                    viewer.apply_forces(state_0)

                    contacts = model.collide(state_0)
                    
                    # Capture frame from viewer and send to VR headset
                    frame_warp = viewer.get_frame()
                    if frame_warp is not None:
                        frame_numpy = frame_warp.numpy()
                        await api.send_single_frame(frame_numpy)

                    # Get controller goals
                    right_goal = await api.get_controller_goal("right")
                    left_goal = await api.get_controller_goal("left")

                    # Get end-effector transforms in world frame
                    index = model.body_key.index("link6")
                    eef_world_right = wp.transform(*state_0.body_q.numpy()[index])
                    eef_world_left = wp.transform(*state_0.body_q.numpy()[index + 8])

                    # Convert to robot base frame by subtracting the base offset
                    # franka1 is at (0.0, 0.5, 0.0), franka2 is at (0.0, -0.5, 0.0)
                    pos_right = wp.transform_get_translation(eef_world_right)
                    rot_right = wp.transform_get_rotation(eef_world_right)


                    current_eef_tf_right = wp.transform(
                        (pos_right[0], pos_right[1] + robot_distance, pos_right[2]), rot_right
                    )
                    pos_left = wp.transform_get_translation(eef_world_left)
                    rot_left = wp.transform_get_rotation(eef_world_left)
                    current_eef_tf_left = wp.transform(
                        (pos_left[0], pos_left[1] - robot_distance, pos_left[2]), rot_left
                    )

                    target_pos_right, target_ori_right, gripper_closed_right = (
                        right_goal_obj.apply_goal_to_arm(
                            right_goal,
                            current_eef_tf_right,
                        )
                    )
                    target_pos_left, target_ori_left, gripper_closed_left = (
                        left_goal_obj.apply_goal_to_arm(
                            left_goal,
                            current_eef_tf_left,
                        )
                    )
                    joint_angles_right = ik.solve(
                        "piper",
                        target_pos_right,
                        target_ori_euler=target_ori_right,
                    )

                    joint_angles_left = ik.solve(
                        "piper",
                        target_pos_left,
                        target_ori_euler=target_ori_left,
                    )
                    # PyBullet returns all joint angles, so slice to get only first 6 (arm joints)
                    arm_joints_right = joint_angles_right[:6]
                    arm_joints_left = joint_angles_left[:6]

                    # Add gripper joint positions
                    # When closed: joint7=0.035, joint8=-0.035
                    # When open: joint7=0.0, joint8=0.0
                    gripper_pos_right = 0.035 if gripper_closed_right else 0.0
                    gripper_pos_left = 0.035 if gripper_closed_left else 0.0

                    # Combine arm joints (6) + gripper joints (2) for each robot
                    joint_targets_right = arm_joints_right.tolist() + [
                        gripper_pos_right,
                        -gripper_pos_right,
                    ]
                    joint_targets_left = arm_joints_left.tolist() + [
                        gripper_pos_left,
                        -gripper_pos_left,
                    ]

                    # set the joint targets
                    if control.joint_target is not None:
                        control.joint_target.assign(
                            wp.array(
                                [0] * num_revolute_joints + joint_targets_right + joint_targets_left,
                                dtype=wp.float32,
                            )
                        )

                    # Update equilibrium position based on plasticity
                    wp.launch(
                        joint_update_equilibrium_kernel,
                        dim=num_revolute_joints,
                        inputs=[
                            state_0.joint_q,
                            control.joint_target,
                            joint_eq_pos_array,
                            joint_cfg.plasticity_angle,
                        ],
                    )

                    wp.launch(
                        joint_apply_signed_spring_torque_kernel,
                        dim=num_revolute_joints,
                        inputs=[
                            state_0.joint_q,
                            control.joint_f,
                            joint_cfg.resistance_ke
                        ],
                    )

                    solver.step(state_0, state_1, control, contacts, sim_dt)

                    state_0, state_1 = state_1, state_0

        with wp.ScopedTimer("render", active=False):
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.log_contacts(contacts, state_0)
            viewer.end_frame()

        sim_time += frame_dt

    viewer.close()
    # Cleanup tactile API resources
    await api.disconnect_robot()
    # Done
    ik.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
