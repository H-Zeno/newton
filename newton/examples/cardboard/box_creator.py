from dataclasses import dataclass

import warp as wp

import newton


@dataclass
class BoxConfiguration:
    width: float = 0.2
    length: float = 0.3
    height: float = 0.07
    thickness: float = 0.002
    density: float = 690.0

    # contact properties
    contact_offset: float = 0.006
    rigid_contact_margin = 0.02

    # cardboard visual properties
    cardboard_color: tuple[float, float, float] = (0.6, 0.4, 0.2)  # Brown cardboard color


@dataclass
class CardboardJointConfiguration:
    default_joint_eq_pos: float = float(wp.radians(7.0))
    target_ke: float = 11.5 #50.0
    target_kd: float = 0.35 #0.7
    friction: float = 0.0
    min_joint_eq_pos: float = float(wp.radians(-52.0))
    max_joint_eq_pos: float = float(wp.radians(52.0))
    min_joint_limit: float = float(wp.radians(-178.0))
    max_joint_limit: float = float(wp.radians(178.0))
    plasticity_angle: float = float(wp.radians(35.0))
    resistance_ke: float = 3.7


def create_cardboard_box_mesh(hx: float, hy: float, hz: float, color: tuple[float, float, float]) -> newton.Mesh:
    """Create a box mesh with cardboard-like appearance.

    Args:
        hx: Half-width (x-direction)
        hy: Half-height (y-direction)
        hz: Half-depth (z-direction)
        color: RGB color tuple for cardboard appearance

    Returns:
        Newton Mesh object with cardboard appearance
    """
    # Create box mesh vertices
    vertices = [
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],  # bottom face
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]       # top face
    ]

    # Create triangular faces for the box
    indices = [
        0, 1, 2, 0, 2, 3,  # bottom
        4, 7, 6, 4, 6, 5,  # top
        0, 4, 5, 0, 5, 1,  # front
        2, 6, 7, 2, 7, 3,  # back
        0, 3, 7, 0, 7, 4,  # left
        1, 5, 6, 1, 6, 2   # right
    ]

    return newton.Mesh(vertices, indices, color=color)



def create_box(box_cfg: BoxConfiguration, joint_cfg: CardboardJointConfiguration, key: str = "cardboard_box", show_visuals: bool = False, show_collision: bool = False) -> newton.ModelBuilder:
    box = newton.ModelBuilder()

    box.add_articulation(key=key)

    base = box.add_body(
        mass=box_cfg.density * box_cfg.width * box_cfg.length * box_cfg.thickness,
        key="base",
    )

    # Create cardboard mesh for visual appearance
    if show_visuals:
        base_mesh = create_cardboard_box_mesh(
            box_cfg.width / 2,
            box_cfg.length / 2,
            box_cfg.thickness / 2,
            box_cfg.cardboard_color
        )
        box.add_shape_mesh(
            base,
            mesh=base_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                is_visible=True,
                has_particle_collision=False,
                has_shape_collision=False,
            ),
        )
    
    # Add collision shape
    box.add_shape_box(
        base,
        hx=box_cfg.width / 2 - box_cfg.contact_offset,
        hy=box_cfg.length / 2 - box_cfg.contact_offset,
        hz=box_cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_collision,
            has_particle_collision=True,
            has_shape_collision=True,
        ),
    )

    # fix the base
    box.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.2)),
    )

    # flaps
    flap_1 = box.add_body(
        mass=box_cfg.density * box_cfg.width * box_cfg.height * box_cfg.thickness,
        key="flap_1"
    )
    
    # Create cardboard mesh for visual appearance
    if show_visuals:
        flap_1_mesh = create_cardboard_box_mesh(
            box_cfg.width / 2,
            box_cfg.height / 2,
            box_cfg.thickness / 2,
            box_cfg.cardboard_color
        )
        box.add_shape_mesh(
            flap_1,
            mesh=flap_1_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                is_visible=True,
                has_particle_collision=False,
                has_shape_collision=False,
            ),
        )
    
    # Add collision shape
    box.add_shape_box(
        flap_1,
        hx=box_cfg.width / 2 - box_cfg.contact_offset,
        hy=box_cfg.height / 2 - box_cfg.contact_offset,
        hz=box_cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_collision,
            has_particle_collision=True,
            has_shape_collision=True,
        ),
    )
    
    flap_2 = box.add_body(
        mass=box_cfg.density * box_cfg.width * box_cfg.height * box_cfg.thickness,
        key="flap_2"
    )
    
    # Create cardboard mesh for visual appearance
    if show_visuals:
        flap_2_mesh = create_cardboard_box_mesh(
            box_cfg.width / 2,
            box_cfg.height / 2,
            box_cfg.thickness / 2,
            box_cfg.cardboard_color
        )
        box.add_shape_mesh(
            flap_2,
            mesh=flap_2_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                is_visible=True,
                has_particle_collision=False,
                has_shape_collision=False,
            ),
        )
    
    # Add collision shape
    box.add_shape_box(
        flap_2,
        hx=box_cfg.width / 2 - box_cfg.contact_offset,
        hy=box_cfg.height / 2 - box_cfg.contact_offset,
        hz=box_cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_collision,
            has_particle_collision=True,
            has_shape_collision=True,
        ),
    )

    # side flaps
    flap_3 = box.add_body(
        mass=box_cfg.density * box_cfg.length * box_cfg.height * box_cfg.thickness,
        key="flap_3"
    )
    
    # Create cardboard mesh for visual appearance
    if show_visuals:
        flap_3_mesh = create_cardboard_box_mesh(
            box_cfg.height / 2,
            box_cfg.length / 2,
            box_cfg.thickness / 2,
            box_cfg.cardboard_color
        )
        box.add_shape_mesh(
            flap_3,
            mesh=flap_3_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                is_visible=True,
                has_particle_collision=False,
                has_shape_collision=False,
            ),
        )
    
    # Add collision shape
    box.add_shape_box(
        flap_3,
        hx=box_cfg.height / 2 - box_cfg.contact_offset,
        hy=box_cfg.length / 2 - box_cfg.contact_offset,
        hz=box_cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_collision,
            has_particle_collision=True,
            has_shape_collision=True,
        ),
    )
    
    flap_4 = box.add_body(
        mass=box_cfg.density * box_cfg.length * box_cfg.height * box_cfg.thickness,
        key="flap_4"
    )
    
    # Create cardboard mesh for visual appearance
    if show_visuals:
        flap_4_mesh = create_cardboard_box_mesh(
            box_cfg.height / 2,
            box_cfg.length / 2,
            box_cfg.thickness / 2,
            box_cfg.cardboard_color
        )
        box.add_shape_mesh(
            flap_4,
            mesh=flap_4_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                is_visible=True,
                has_particle_collision=False,
                has_shape_collision=False,
            ),
        )
    
    # Add collision shape
    box.add_shape_box(
        flap_4,
        hx=box_cfg.height / 2 - box_cfg.contact_offset,
        hy=box_cfg.length / 2 - box_cfg.contact_offset,
        hz=box_cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_collision,
            has_particle_collision=True,
            has_shape_collision=True,
        ),
    )

    # flap joints
    box.add_joint_revolute(
        parent=base,
        child=flap_1,
        parent_xform=wp.transform(p=wp.vec3(0.0, box_cfg.length / 2, 0.0)),
        child_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.height / 2, 0.0)),
        axis=wp.vec3(1.0, 0.0, 0.0),
        mode=newton.JointMode.TARGET_POSITION,
        target=joint_cfg.default_joint_eq_pos,
        friction=joint_cfg.friction,
        target_ke=joint_cfg.target_ke,
        target_kd=joint_cfg.target_kd,
        limit_lower=joint_cfg.min_joint_limit,
        limit_upper=joint_cfg.max_joint_limit,
    )
    box.add_joint_revolute(
        parent=base,
        child=flap_2,
        parent_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.length / 2, 0.0)),
        child_xform=wp.transform(p=wp.vec3(0.0, box_cfg.height / 2, 0.0)),
        axis=wp.vec3(1.0, 0.0, 0.0),
        mode=newton.JointMode.TARGET_POSITION,
        target=joint_cfg.default_joint_eq_pos,
        friction=joint_cfg.friction,
        target_ke=joint_cfg.target_ke,
        target_kd=joint_cfg.target_kd,
        limit_lower=joint_cfg.min_joint_limit,
        limit_upper=joint_cfg.max_joint_limit,
    )
    box.add_joint_revolute(
        parent=base,
        child=flap_3,
        parent_xform=wp.transform(p=wp.vec3(box_cfg.width / 2, 0.0, 0.0)),
        child_xform=wp.transform(p=wp.vec3(-box_cfg.height / 2, 0.0, 0.0)),
        axis=wp.vec3(0.0, 1.0, 0.0),
        mode=newton.JointMode.TARGET_POSITION,
        target=joint_cfg.default_joint_eq_pos,
        friction=joint_cfg.friction,
        target_ke=joint_cfg.target_ke,
        target_kd=joint_cfg.target_kd,
        limit_lower=joint_cfg.min_joint_limit,
        limit_upper=joint_cfg.max_joint_limit,
    )
    box.add_joint_revolute(
        parent=base,
        child=flap_4,
        parent_xform=wp.transform(p=wp.vec3(-box_cfg.width / 2, 0.0, 0.0)),
        child_xform=wp.transform(p=wp.vec3(box_cfg.height / 2, 0.0, 0.0)),
        axis=wp.vec3(0.0, 1.0, 0.0),
        mode=newton.JointMode.TARGET_POSITION,
        target=joint_cfg.default_joint_eq_pos,
        friction=joint_cfg.friction,
        target_ke=joint_cfg.target_ke,
        target_kd=joint_cfg.target_kd,
        limit_lower=joint_cfg.min_joint_limit,
        limit_upper=joint_cfg.max_joint_limit,
    )

    # # flap ears
    # ear_31 = box.add_body(
    #     mass=box_cfg.density * box_cfg.height * box_cfg.height * box_cfg.thickness,
    #     key="ear_31"
    # )
    
    # # Create cardboard mesh for visual appearance
    # if show_visuals:
    #     ear_31_mesh = create_cardboard_box_mesh(
    #         box_cfg.height / 2,
    #         box_cfg.height / 2,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         ear_31,
    #         mesh=ear_31_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # # Add collision shape
    # box.add_shape_box(
    #     ear_31,
    #     hx=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hy=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hz=box_cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    
    # ear_32 = box.add_body(
    #     mass=box_cfg.density * box_cfg.height * box_cfg.height * box_cfg.thickness,
    #     key="ear_32"
    # )
    
    # # Create cardboard mesh for visual appearance
    # if show_visuals:
    #     ear_32_mesh = create_cardboard_box_mesh(
    #         box_cfg.height / 2,
    #         box_cfg.height / 2,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         ear_32,
    #         mesh=ear_32_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # # Add collision shape
    # box.add_shape_box(
    #     ear_32,
    #     hx=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hy=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hz=box_cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    
    # ear_41 = box.add_body(
    #     mass=box_cfg.density * box_cfg.height * box_cfg.height * box_cfg.thickness,
    #     key="ear_41"
    # )
    
    # # Create cardboard mesh for visual appearance
    # if show_visuals:
    #     ear_41_mesh = create_cardboard_box_mesh(
    #         box_cfg.height / 2,
    #         box_cfg.height / 2,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         ear_41,
    #         mesh=ear_41_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # # Add collision shape
    # box.add_shape_box(
    #     ear_41,
    #     hx=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hy=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hz=box_cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    
    # ear_42 = box.add_body(
    #     mass=box_cfg.density * box_cfg.height * box_cfg.height * box_cfg.thickness,
    #     key="ear_42"
    # )
    
    # # Create cardboard mesh for visual appearance
    # if show_visuals:
    #     ear_42_mesh = create_cardboard_box_mesh(
    #         box_cfg.height / 2,
    #         box_cfg.height / 2,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         ear_42,
    #         mesh=ear_42_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # # Add collision shape
    # box.add_shape_box(
    #     ear_42,
    #     hx=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hy=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hz=box_cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )

    # # flap ears joints
    # box.add_joint_revolute(
    #     parent=flap_3,
    #     child=ear_31,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, box_cfg.length / 2 - box_cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )
    # box.add_joint_revolute(
    #     parent=flap_3,
    #     child=ear_32,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.length / 2 + box_cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, box_cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )
    # box.add_joint_revolute(
    #     parent=flap_4,
    #     child=ear_41,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, box_cfg.length / 2 - box_cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )
    # box.add_joint_revolute(
    #     parent=flap_4,
    #     child=ear_42,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.length / 2 + box_cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, box_cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )

    # # additional flaps
    # flap_11 = box.add_body(
    #     mass=box_cfg.density * box_cfg.width * box_cfg.thickness * box_cfg.thickness,
    #     key="flap_11"
    # )

    # # Create cardboard mesh for visual appearance (thin flap)
    # if show_visuals:
    #     flap_11_mesh = create_cardboard_box_mesh(
    #         box_cfg.width / 2,
    #         box_cfg.thickness,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         flap_11,
    #         mesh=flap_11_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # flap_12 = box.add_body(
    #     mass=box_cfg.density * box_cfg.width * box_cfg.height * box_cfg.thickness,
    #     key="flap_12"
    # )
    
    # # Create cardboard mesh for visual appearance
    # if show_visuals:
    #     flap_12_mesh = create_cardboard_box_mesh(
    #         box_cfg.width / 2,
    #         box_cfg.height / 2,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         flap_12,
    #         mesh=flap_12_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # # Add collision shape
    # box.add_shape_box(
    #     flap_12,
    #     hx=box_cfg.width / 2 - box_cfg.contact_offset,
    #     hy=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hz=box_cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    
    # flap_21 = box.add_body(
    #     mass=box_cfg.density * box_cfg.width * box_cfg.thickness * box_cfg.thickness,
    #     key="flap_21"
    # )
    
    # # Create cardboard mesh for visual appearance (thin flap)
    # if show_visuals:
    #     flap_21_mesh = create_cardboard_box_mesh(
    #         box_cfg.width / 2,
    #         box_cfg.thickness,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         flap_21,
    #         mesh=flap_21_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # flap_22 = box.add_body(
    #     mass=box_cfg.density * box_cfg.width * box_cfg.height * box_cfg.thickness,
    #     key="flap_22"
    # )

    
    # # Create cardboard mesh for visual appearance
    # if show_visuals:
    #     flap_22_mesh = create_cardboard_box_mesh(
    #         box_cfg.width / 2,
    #         box_cfg.height / 2,
    #         box_cfg.thickness / 2,
    #         box_cfg.cardboard_color
    #     )
    #     box.add_shape_mesh(
    #         flap_22,
    #         mesh=flap_22_mesh,
    #         cfg=newton.ModelBuilder.ShapeConfig(
    #             is_visible=True,
    #             has_particle_collision=False,
    #             has_shape_collision=False,
    #         ),
    #     )
    
    # # Add collision shape
    # box.add_shape_box(
    #     flap_22,
    #     hx=box_cfg.width / 2 - box_cfg.contact_offset,
    #     hy=box_cfg.height / 2 - box_cfg.contact_offset,
    #     hz=box_cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )

    # # additional flaps joints
    # box.add_joint_revolute(
    #     parent=flap_1,
    #     child=flap_11,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, box_cfg.height / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.thickness, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )
    # box.add_joint_revolute(
    #     parent=flap_11,
    #     child=flap_12,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, box_cfg.thickness, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )
    # box.add_joint_revolute(
    #     parent=flap_2,
    #     child=flap_21,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.height / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, box_cfg.thickness, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )
    # box.add_joint_revolute(
    #     parent=flap_21,
    #     child=flap_22,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -box_cfg.thickness, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, box_cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     mode=newton.JointMode.TARGET_POSITION,
    #     target=joint_cfg.default_joint_eq_pos,
    #     friction=joint_cfg.friction,
    #     target_ke=joint_cfg.target_ke,
    #     target_kd=joint_cfg.target_kd,
    #     limit_lower=joint_cfg.min_joint_limit,
    #     limit_upper=joint_cfg.max_joint_limit,
    # )

    return box
