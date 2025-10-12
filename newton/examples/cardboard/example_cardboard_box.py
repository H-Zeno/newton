from typing import Literal
import newton
import newton.examples
import warp as wp
from newton.examples.cardboard.box_creator import create_box, BoxConfiguration, CardboardJointConfiguration
from newton.examples.cardboard.cardboard_kernels import joint_update_equilibrium_kernel, joint_apply_signed_spring_torque_kernel

wp.set_device("cuda")

class CardboardBox:
    def __init__(self, viewer, solver_type: Literal["mujoco", "xpbd", "featherstone"] = "mujoco"):
        
        self.viewer = viewer
        
        # sim params
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # joint and box config
        self.joint_cfg = CardboardJointConfiguration()
        self.box_cfg = BoxConfiguration()
        
        # Create the box to determine joint count
        box_builder = create_box(self.box_cfg, self.joint_cfg, key="box", show_visuals=True)
        self.num_revolute_joints = box_builder.joint_count
        
        # Initialize joint equilibrium positions for all revolute joints
        self.joint_eq_pos_array = wp.array([self.joint_cfg.default_joint_eq_pos] * self.num_revolute_joints, dtype=float)
        
        self.builder = newton.ModelBuilder()
        self.builder.add_builder(box_builder,
            xform=wp.transform((0.5, 0.0, 0.3), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi * 0))
        )

        self.builder.add_ground_plane()

        self.model = self.builder.finalize()

        if solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(model=self.model, iterations=20, njmax=128)
        elif solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(model=self.model, iterations=20)
        elif solver_type == "featherstone":
            self.solver = newton.solvers.SolverFeatherstone(model=self.model, update_mass_matrix_interval=1, angular_damping=0.1)

        self.state_0 = self.model.state() # current state
        self.state_1 = self.model.state() # t + dt state
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.box_cfg.rigid_contact_margin)

        self.viewer.set_model(self.model)

        # Set camera for better viewing of the cardboard box
        # Box is at (0.5, 0.0, 0.3), so position camera above and at an angle
        camera_pos = wp.vec3(0.91, -0.02, 2.71)  # Above and at an angle
        camera_pitch = -37.7 
        camera_yaw = -178.8  
        self.viewer.set_camera(camera_pos, camera_pitch, camera_yaw)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        
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
            
            self.state_0.clear_forces()
        
            # Update equilibrium position based on plasticity
            wp.launch(
                    joint_update_equilibrium_kernel,
                dim=self.num_revolute_joints,
                inputs=[
                    self.state_0.joint_q,
                    self.control.joint_target,
                    self.joint_eq_pos_array,
                    self.joint_cfg.plasticity_angle,
                ],
            )
            # Apply boundary spring forces using Warp kernel
            if self.control.joint_f is None:
                self.control.joint_f = wp.zeros(self.num_revolute_joints, dtype=float)

            wp.launch(
                joint_apply_signed_spring_torque_kernel,
                dim=self.num_revolute_joints,
                inputs=[
                    self.state_0.joint_q,
                    self.control.joint_f,
                    self.joint_cfg.resistance_ke
                ],
            )
                
            # get forces from the viewer and apply to the state
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.box_cfg.rigid_contact_margin)
            
            # apply kernels for cardboard joints
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
    example = CardboardBox(viewer)

    newton.examples.run(example, args)