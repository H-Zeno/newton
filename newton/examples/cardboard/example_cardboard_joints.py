# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example: Cardboard Joint Simulation
#
# Demonstrates how to set up and simulate cardboard joint mechanics using the
# newton.ModelBuilder() class. This example showcases joint limits, plasticity,
# and spring resistance as found in cardboard-like materials.
#
# Command: python -m newton.examples.cardboard.example_cardboard_joints
#
###########################################################################

import warp as wp

import newton
import newton.examples


@wp.kernel
def apply_signed_spring_torque_kernel(
    joint_q: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    resistance_ke: float,
):
    """Kernel to compute boundary torque based on joint angle limits."""
    tid = wp.tid()
    current_angle = joint_q[tid]
    boundary_torque = 0.0
    if current_angle > 0:
        # Apply clockwise torque (negative) to restore back to plane
        boundary_torque = -resistance_ke * abs(current_angle)
    elif current_angle < 0:
        # Apply anti-clockwise torque (positive) to restore back to plane
        boundary_torque = resistance_ke * abs(current_angle)

    joint_f[tid] = boundary_torque


@wp.kernel
def update_equilibrium_kernel(
    joint_q: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_eq_pos: wp.array(dtype=float),
    plasticity_angle: float,
):
    """Kernel to update equilibrium position based on plasticity."""
    tid = wp.tid()
    current_angle = joint_q[tid]
    current_eq_pos = joint_eq_pos[tid]

    # Calculate difference between current angle and equilibrium
    angle_difference = current_angle - current_eq_pos

    # If difference exceeds plasticity threshold, update equilibrium
    if wp.abs(angle_difference) > plasticity_angle:
        # Move equilibrium toward current position, leaving plasticity_angle gap
        if angle_difference > 0.0:
            joint_eq_pos[tid] = current_angle - plasticity_angle
        else:
            joint_eq_pos[tid] = current_angle + plasticity_angle
 
    joint_target[tid] = joint_eq_pos[tid]


class CardboardJoint:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        builder.add_articulation(key="cardboard_joint")

        self.joint_parameters = {
            "default_joint_eq_pos": float(wp.radians(7.0)),
            "target_ke": 50.0,
            "target_kd": 0.7,
            "min_joint_eq_pos": float(wp.radians(-43.0)),
            "max_joint_eq_pos": float(wp.radians(43.0)),
            "min_joint_limit": float(wp.radians(-178.0)),
            "max_joint_limit": float(wp.radians(178.0)),
            "plasticity_angle": float(wp.radians(23.0)),
            "resistance_ke": 3.5,
        }

        self.joint_eq_pos = self.joint_parameters["default_joint_eq_pos"]
        self.joint_eq_pos_array = wp.array([self.joint_eq_pos], dtype=float)

        # define cardboard plane dimensions
        hx = 0.10
        hy = 0.40
        hz = 0.003

        # create brown cardboard colored box mesh
        cardboard_brown = (0.6, 0.4, 0.2)  # Brown cardboard color

        # create box mesh vertices and indices
        vertices = [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],  # bottom face
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]       # top face
        ]

        indices = [
            0, 1, 2, 0, 2, 3,  # bottom
            4, 7, 6, 4, 6, 5,  # top
            0, 4, 5, 0, 5, 1,  # front
            2, 6, 7, 2, 7, 3,  # back
            0, 3, 7, 0, 7, 4,  # left
            1, 5, 6, 1, 6, 2   # right
        ]

        cardboard_mesh = newton.Mesh(vertices, indices, color=cardboard_brown)

        # create first link
        link_0 = builder.add_body(mass=0.2)
        builder.add_shape_mesh(link_0, mesh=cardboard_mesh)

        # add joints
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=rot),  # rotate pendulum around the z-axis to appear sideways to the viewer
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity(dtype=wp.float32)), # make sure the joint attaches to the end of the rigid body
            mode=newton.JointMode.TARGET_POSITION,
            target=self.joint_eq_pos,
            target_ke=self.joint_parameters["target_ke"],
            target_kd=self.joint_parameters["target_kd"],
            limit_lower=self.joint_parameters["min_joint_limit"],
            limit_upper=self.joint_parameters["max_joint_limit"],
        )

        # add ground plane
        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state() # current state
        self.state_1 = self.model.state() # t + dt state
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # Set camera closer to the cardboard joint object
        # Joint is at (0, 0, 1.0), so position camera at a good viewing angle
        camera_pos = wp.vec3(1.5, 0.0, 1.1)  # Close position with slight offset
        camera_pitch = -10.0  # Look slightly down
        camera_yaw = -175.0   # Angle to view the joint from the side
        self.viewer.set_camera(camera_pos, camera_pitch, camera_yaw)

        # Enable joint visualization to show joint axes
        self.viewer.show_joints = True

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # idea:
            # 1. define an equilibrium position for the cardboard joints
            # 2. use a spring damper system to control the joints (as if there was an actuator inside)

            # Here we implement the controls of the joints!
            self.state_0.clear_forces()
            
            # Update equilibrium position based on plasticity
            wp.launch(
                update_equilibrium_kernel,
                dim=1,
                inputs=[
                    self.state_0.joint_q,
                    self.control.joint_target,
                    self.joint_eq_pos_array,
                    self.joint_parameters["plasticity_angle"],
                ],
            )
            # Apply boundary spring forces using Warp kernel
            if self.control.joint_f is None:
                self.control.joint_f = wp.zeros(1, dtype=float)

            wp.launch(
                apply_signed_spring_torque_kernel,
                dim=1,  # Single joint
                inputs=[
                    self.state_0.joint_q,
                    self.control.joint_f,
                    self.joint_parameters["resistance_ke"]
                ],
            )
            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Update joint coordinates from body positions after XPBD solver step
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0


    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        # rough check that the cardboard joint links are in the correct area
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cardboard joint links in correct area",
            lambda q, qd: abs(q[0]) < 1e-5 and abs(q[1]) < 1.0 and q[2] < 5.0 and q[2] > 0.0,
            [0, 1],
        )

        def check_velocities(_, qd):
            # velocity outside the plane of the cardboard joint should be close to zero
            check = abs(qd[0]) < 1e-4 and abs(qd[6]) < 1e-4
            # velocity in the plane of the pendulum should be reasonable
            check = check and abs(qd[1]) < 10.0 and abs(qd[2]) < 5.0 and abs(qd[3]) < 10.0 and abs(qd[4]) < 10.0
            return check

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cardboard joint links have reasonable velocities",
            check_velocities,
            [0, 1],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        self.viewer.end_frame()

if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = CardboardJoint(viewer)

    newton.examples.run(example, args)
