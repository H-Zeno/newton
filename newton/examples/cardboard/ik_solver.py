import pybullet as p
import pybullet_data
import numpy as np


class InverseKinematicsSolver:
    def __init__(self, use_gui: bool = False):
        """
        Initialize a PyBullet IK solver that can handle multiple robot bodies.
        """
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Maps from body name -> PyBullet body unique ID and metadata
        self.bodies = {}

    def load_robot(self, name: str, urdf_path: str, ee_link_index: int = -1, use_fixed_base=True):
        """
        Load a robot from a URDF and store its metadata.

        Args:
            name (str): Arbitrary name for this robot (e.g., "fr3").
            urdf_path (str): Path to the robot's URDF.
            ee_link_index (int): End-effector link index (default: last link).
            use_fixed_base (bool): Whether the robot should be fixed to the ground.
        """
        body_id = p.loadURDF(urdf_path, useFixedBase=use_fixed_base)
        num_joints = p.getNumJoints(body_id)
        if ee_link_index < 0:
            ee_link_index = num_joints - 1

        # Extract joint limits
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
        for i in range(num_joints):
            info = p.getJointInfo(body_id, i)
            joint_type = info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                ll, ul = info[8], info[9]
                lower_limits.append(ll)
                upper_limits.append(ul)
                joint_ranges.append(ul - ll)
                rest_poses.append(0.5 * (ll + ul) if np.isfinite(ll) and np.isfinite(ul) else 0)
            else:
                lower_limits.append(0)
                upper_limits.append(0)
                joint_ranges.append(0)
                rest_poses.append(0)

        self.bodies[name] = {
            "id": body_id,
            "ee_link": ee_link_index,
            "n_joints": num_joints,
            "lower_limits": lower_limits,
            "upper_limits": upper_limits,
            "joint_ranges": joint_ranges,
            "rest_poses": rest_poses,
        }

        print(f"✅ Loaded '{name}' from {urdf_path} with {num_joints} joints (EE link = {ee_link_index})")

    def solve(self, body_name: str, target_pos, target_ori_euler=None, target_ori_quat=None, joint_subset=None):
        """
        Solve IK for the specified robot body.

        Args:
            body_name (str): Name of the robot (as used in `load_robot`).
            target_pos (list[float]): [x, y, z] desired EE position.
            target_ori_euler (list[float], optional): [roll, pitch, yaw] in radians.
            target_ori_quat (list[float], optional): Quaternion [x, y, z, w].
            joint_subset (list[int], optional): Specific joint indices to include in IK.

        Returns:
            np.ndarray: Joint angles (radians).
        """
        if body_name not in self.bodies:
            raise ValueError(f"Robot '{body_name}' not found. Use load_robot() first.")

        body = self.bodies[body_name]

        if target_ori_quat is None:
            target_ori_quat = (
                p.getQuaternionFromEuler(target_ori_euler)
                if target_ori_euler is not None
                else [0, 0, 0, 1]
            )

        # Default to all movable joints
        n = body["n_joints"]
        if joint_subset is None:
            joint_subset = list(range(n))

        lower = [body["lower_limits"][i] for i in joint_subset]
        upper = [body["upper_limits"][i] for i in joint_subset]
        ranges = [body["joint_ranges"][i] for i in joint_subset]
        rest = [body["rest_poses"][i] for i in joint_subset]

        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=body["id"],
            endEffectorLinkIndex=body["ee_link"],
            targetPosition=target_pos,
            targetOrientation=target_ori_quat,
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest,
            maxNumIterations=500,
            residualThreshold=1e-5,
        )

        return np.array(joint_angles)

    def get_end_effector_pose(self, body_name: str, joint_angles=None):
        """
        Compute FK for a given robot.

        Args:
            body_name (str): Robot name.
            joint_angles (list[float], optional): If provided, sets joint states before computing FK.

        Returns:
            (position, orientation_quat)
        """
        if body_name not in self.bodies:
            raise ValueError(f"Robot '{body_name}' not found.")

        body = self.bodies[body_name]

        if joint_angles is not None:
            for i, angle in enumerate(joint_angles):
                p.resetJointState(body["id"], i, angle)

        link_state = p.getLinkState(body["id"], body["ee_link"])
        ee_pos, ee_ori = link_state[4], link_state[5]
        return np.array(ee_pos), np.array(ee_ori)

    def list_joints(self, body_name: str):
        """List all joints for debugging."""
        if body_name not in self.bodies:
            raise ValueError(f"Robot '{body_name}' not found.")
        body = self.bodies[body_name]
        for i in range(body["n_joints"]):
            name = p.getJointInfo(body["id"], i)[1].decode("utf-8")
            print(f"{i}: {name}")

    def disconnect(self):
        """Disconnect from PyBullet."""
        p.disconnect(self.client)


if __name__ == "__main__":
    import numpy as np
    import newton

    ik = InverseKinematicsSolver(use_gui=False)

    # Load multiple robots (or just one)
    ik.load_robot("piper", "/home/zhamers/piper-robot-server/URDF/Piper/piper_description.urdf", ee_link_index=6)

    # Define a target pose
    target_pos = [0.2, 0.0, 0.5]
    target_ori = [0, np.pi, 0]

    body_id = ik.bodies["piper"]["id"]
    for i in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, i)
        link_name = joint_info[12].decode("utf-8")
        print(f"{i}: {link_name}")

    ik.list_joints("piper")

    # Solve for specific robot
    angles = ik.solve(
        "piper",
        target_pos,
        target_ori_euler=target_ori,
    )
    print("IK joint angles:", np.round(angles, 3))

    # Verify the forward kinematics
    ee_pos, ee_ori = ik.get_end_effector_pose("piper", angles)
    print("Resulting EE position:", np.round(ee_pos, 3))

    ik.disconnect()