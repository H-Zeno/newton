import warp as wp

@wp.kernel
def joint_apply_signed_spring_torque_kernel(
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
def joint_update_equilibrium_kernel(
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