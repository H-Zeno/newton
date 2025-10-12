
import asyncio

import numpy as np
import warp as wp
from ik_solver import InverseKinematicsSolver
from tactile_teleop_sdk import TactileAPI
from newton.examples.cardboard.box_creator import create_box, BoxConfiguration, CardboardJointConfiguration
from newton.examples.cardboard.cardboard_kernels import joint_update_equilibrium_kernel, joint_apply_signed_spring_torque_kernel

import newton

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


async def main():
    wp.set_device("cuda")

    tactile_api = TactileAPI("tr_-_DmhsI7tlOnj63BrXiLHf3viSJAoq67PNnbM9hv17M")

    await tactile_api.connect_vr_controller()

    # viewer setup
    viewer = newton.viewer.ViewerGL(headless=False)

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
        "/home/fabio/git/questVR_ws/src/Piper_ros/src/piper_description/urdf/piper_description.urdf",
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
        "/home/fabio/git/questVR_ws/src/Piper_ros/src/piper_description/urdf/piper_description.urdf",
        xform=wp.transform(
            (0.0, robot_distance, 0.0),
            wp.quat_identity(),
        ),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )

    for i, _ in enumerate(piper1.joint_target_ke):
        piper1.joint_target_ke[i] = 30.0
        piper1.joint_target_kd[i] = 1.0
        piper1.joint_friction[i] = 0.4
        piper2.joint_target_ke[i] = 30.0
        piper2.joint_target_kd[i] = 1.0
        piper2.joint_friction[i] = 0.4

    scene.add_builder(piper1, environment=-1)
    scene.add_builder(piper2, environment=-1)

    # scene.shape_collision_filter_pairs = []


    model = scene.finalize()

    # model.shape_collision_filter_pairs = []
    print(model)

    solver = newton.solvers.SolverMuJoCo(model=model, iterations=20, njmax=256, contact_stiffness_time_const=sim_dt)
    # solver = newton.solvers.SolverXPBD(model=model, iterations=20)
    # solver = newton.solvers.SolverFeatherstone(model=model, update_mass_matrix_interval=20)

    state_0 = model.state()
    state_1 = model.state()

    contacts = model.collide(state_0)



    control = model.control()

    viewer.set_model(model)

    # not required for MuJoCo, but required for other solvers
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Initialize the solver
    ik = InverseKinematicsSolver(use_gui=False)
    ik.load_robot(
        "piper",
        "/home/fabio/git/questVR_ws/src/Piper_ros/src/piper_description/urdf/piper_description.urdf",
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

                    right_goal = await tactile_api.get_controller_goal("right")
                    left_goal = await tactile_api.get_controller_goal("left")

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
                        # only fr3_joint1–fr3_joint7
                    )
                    joint_angles_left = ik.solve(
                        "piper",
                        target_pos_left,
                        target_ori_euler=target_ori_left,
                        # only fr3_joint1–fr3_joint7
                    )
                    # set the joint targets
                    control.joint_target.assign(
                        wp.array(
                            [0] * num_revolute_joints + joint_angles_right.tolist() + joint_angles_left.tolist(),
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
    # Done
    ik.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
