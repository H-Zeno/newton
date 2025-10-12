import newton
import warp as wp
from box_creator import create_box, BoxConfiguration

wp.set_device("cuda")

# viewer setup
viewer = newton.viewer.ViewerGL(headless=False)

# sim params
sim_time = 0.0
frame_dt = 1.0 / 50
substeps = 4
sim_dt = frame_dt / substeps


scene = newton.ModelBuilder()

scene.add_builder(
    create_box(BoxConfiguration(joint_target_kd=1.0), key="box", show_collision=False, show_visuals=False),
    xform=wp.transform((0.5, 0.0, 0.3), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi * 0))
)

scene.add_ground_plane()

scene.shape_collision_filter_pairs = []


model = scene.finalize()

solver = newton.solvers.SolverMuJoCo(model=model, iterations=20, njmax=128)
# solver = newton.solvers.SolverXPBD(model=model, iterations=20)
# solver = newton.solvers.SolverFeatherstone(model=model, update_mass_matrix_interval=1, angular_damping=0.1)

state_0 = model.state()
state_1 = model.state()

contacts = model.collide(state_0)

viewer.set_model(model)

# not required for MuJoCo, but required for other solvers
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

# main loop
while viewer.is_running():
    if not viewer.is_paused():
        with wp.ScopedTimer("step", active=False):
            for _ in range(substeps):
                state_0.clear_forces()
                
                # get forces from the viewer and apply to the state
                viewer.apply_forces(state_0)

                contacts = model.collide(state_0)

                # apply kernels for cardboard joints

                solver.step(state_0, state_1, model.control(), contacts, sim_dt)

                state_0, state_1 = state_1, state_0

    with wp.ScopedTimer("render", active=False):
        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    sim_time += frame_dt

viewer.close()