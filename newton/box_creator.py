import newton
import warp as wp
from dataclasses import dataclass


@dataclass
class BoxConfiguration:
    width: float = 0.2
    length: float = 0.3
    height: float = 0.07
    thickness: float = 0.002
    density: float = 690.0

    # contact properties
    contact_offset : float = 0.004

    # joint properties
    joint_friction: float = 0.0
    joint_target_ke: float = 0.0
    joint_target_kd: float = 1.0


def create_box(cfg: BoxConfiguration, key: str = "box", show_collision: bool = False, show_visuals: bool = True) -> newton.ModelBuilder:
    box = newton.ModelBuilder()

    box.add_articulation(key=key)

    base = box.add_body(
        mass=cfg.density * cfg.width * cfg.length * cfg.thickness
    )
    base_vis = box.add_shape_box(
        base,
        hx=cfg.width / 2,
        hy=cfg.length / 2,
        hz=cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_visuals,
            has_particle_collision=False,
            has_shape_collision=False,
        ),
    )
    base_col = box.add_shape_box(
        base,
        hx=cfg.width / 2 - cfg.contact_offset,
        hy=cfg.length / 2 - cfg.contact_offset,
        hz=cfg.thickness / 2,
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
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0)),
    )

    # flaps
    flap_1 = box.add_body(
        mass=cfg.density * cfg.width * cfg.height * cfg.thickness,
    )
    flap_1_vis = box.add_shape_box(
        flap_1,
        hx=cfg.width / 2,
        hy=cfg.height / 2,
        hz=cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_visuals,
            has_particle_collision=False,
            has_shape_collision=False,
        ),
    )
    flap_1_col = box.add_shape_box(
        flap_1,
        hx=cfg.width / 2 - cfg.contact_offset,
        hy=cfg.height / 2 - cfg.contact_offset,
        hz=cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_collision,
            has_particle_collision=True,
            has_shape_collision=True,
        ),
    )
    flap_2 = box.add_body(
        mass=cfg.density * cfg.width * cfg.height * cfg.thickness,
    )
    flap_2_vis = box.add_shape_box(
        flap_2,
        hx=cfg.width / 2,
        hy=cfg.height / 2,
        hz=cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_visuals,
            has_particle_collision=False,
            has_shape_collision=False,
        ),
    )
    flap_2_col = box.add_shape_box(
        flap_2,
        hx=cfg.width / 2 - cfg.contact_offset,
        hy=cfg.height / 2 - cfg.contact_offset,
        hz=cfg.thickness / 2,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_visible=show_collision,
            has_particle_collision=True,
            has_shape_collision=True,
        ),
    )
    # flap_3 = box.add_body(
    #     mass=cfg.density * cfg.length * cfg.height * cfg.thickness,
    # )
    # flap_3_vis = box.add_shape_box(
    #     flap_3,
    #     hx=cfg.height / 2,
    #     hy=cfg.length / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # flap_3_col = box.add_shape_box(
    #     flap_3,
    #     hx=cfg.height / 2 - cfg.contact_offset,
    #     hy=cfg.length / 2 - cfg.contact_offset - cfg.thickness / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    # flap_4 = box.add_body(
    #     mass=cfg.density * cfg.length * cfg.height * cfg.thickness,
    # )
    # flap_4_vis = box.add_shape_box(
    #     flap_4,
    #     hx=cfg.height / 2,
    #     hy=cfg.length / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # flap_4_col = box.add_shape_box(
    #     flap_4,
    #     hx=cfg.height / 2 - cfg.contact_offset,
    #     hy=cfg.length / 2 - cfg.contact_offset - cfg.thickness / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )

    # flap joints
    box.add_joint_revolute(
        parent=base,
        child=flap_1,
        parent_xform=wp.transform(p=wp.vec3(0.0, cfg.length / 2, 0.0)),
        child_xform=wp.transform(p=wp.vec3(0.0, -cfg.height / 2, 0.0)),
        axis=wp.vec3(1.0, 0.0, 0.0),
        friction=cfg.joint_friction,
        target_ke=cfg.joint_target_ke,
        target_kd=cfg.joint_target_kd,
    )
    box.add_joint_revolute(
        parent=base,
        child=flap_2,
        parent_xform=wp.transform(p=wp.vec3(0.0, -cfg.length / 2, 0.0)),
        child_xform=wp.transform(p=wp.vec3(0.0, cfg.height / 2, 0.0)),
        axis=wp.vec3(1.0, 0.0, 0.0),
        friction=cfg.joint_friction,
        target_ke=cfg.joint_target_ke,
        target_kd=cfg.joint_target_kd,
    )
    # box.add_joint_revolute(
    #     parent=base,
    #     child=flap_3,
    #     parent_xform=wp.transform(p=wp.vec3(cfg.width / 2, 0.0, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(-cfg.height / 2, 0.0, 0.0)),
    #     axis=wp.vec3(0.0, 1.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )
    # box.add_joint_revolute(
    #     parent=base,
    #     child=flap_4,
    #     parent_xform=wp.transform(p=wp.vec3(-cfg.width / 2, 0.0, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(cfg.height / 2, 0.0, 0.0)),
    #     axis=wp.vec3(0.0, 1.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )

    # # flap ears
    # ear_31 = box.add_body(
    #     mass=cfg.density * cfg.height * cfg.height * cfg.thickness,
    # )
    # ear_31_vis = box.add_shape_box(
    #     ear_31,
    #     hx=cfg.height / 2,
    #     hy=cfg.height / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # ear_31_col = box.add_shape_box(
    #     ear_31,
    #     hx=cfg.height / 2 - cfg.contact_offset,
    #     hy=cfg.height / 2 - cfg.contact_offset,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    # ear_32 = box.add_body(
    #     mass=cfg.density * cfg.height * cfg.height * cfg.thickness,
    # )
    # ear_32_vis = box.add_shape_box(
    #     ear_32,
    #     hx=cfg.height / 2,
    #     hy=cfg.height / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # ear_32_col = box.add_shape_box(
    #     ear_32,
    #     hx=cfg.height / 2 - cfg.contact_offset,
    #     hy=cfg.height / 2 - cfg.contact_offset,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    # ear_41 = box.add_body(
    #     mass=cfg.density * cfg.height * cfg.height * cfg.thickness,
    # )
    # ear_41_vis = box.add_shape_box(
    #     ear_41,
    #     hx=cfg.height / 2,
    #     hy=cfg.height / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=True,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # ear_41_col = box.add_shape_box(
    #     ear_41,
    #     hx=cfg.height / 2 - cfg.contact_offset,
    #     hy=cfg.height / 2 - cfg.contact_offset,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    # ear_42 = box.add_body(
    #     mass=cfg.density * cfg.height * cfg.height * cfg.thickness,
    # )
    # ear_42_vis = box.add_shape_box(
    #     ear_42,
    #     hx=cfg.height / 2,
    #     hy=cfg.height / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=True,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # ear_42_col = box.add_shape_box(
    #     ear_42,
    #     hx=cfg.height / 2 - cfg.contact_offset,
    #     hy=cfg.height / 2 - cfg.contact_offset,
    #     hz=cfg.thickness / 2,
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
    #     parent_xform=wp.transform(p=wp.vec3(0.0, cfg.length / 2 - cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )
    # box.add_joint_revolute(
    #     parent=flap_3,
    #     child=ear_32,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -cfg.length / 2 + cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )
    # box.add_joint_revolute(
    #     parent=flap_4,
    #     child=ear_41,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, cfg.length / 2 - cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )
    # box.add_joint_revolute(
    #     parent=flap_4,
    #     child=ear_42,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -cfg.length / 2 + cfg.thickness / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )

    # # additional flaps
    # flap_11 = box.add_body(
    #     mass=cfg.density * cfg.width * cfg.thickness * cfg.thickness,
    # )
    # flap_11_vis = box.add_shape_box(
    #     flap_11,
    #     hx=cfg.width / 2,
    #     hy=cfg.thickness,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # flap_12 = box.add_body(
    #     mass=cfg.density * cfg.width * cfg.height * cfg.thickness,
    # )
    # flap_12_vis = box.add_shape_box(
    #     flap_12,
    #     hx=cfg.width / 2,
    #     hy=cfg.height / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # flap_12_col = box.add_shape_box(
    #     flap_12,
    #     hx=cfg.width / 2 - cfg.contact_offset,
    #     hy=cfg.height / 2 - cfg.contact_offset,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_collision,
    #         has_particle_collision=True,
    #         has_shape_collision=True,
    #     ),
    # )
    # flap_21 = box.add_body(
    #     mass=cfg.density * cfg.width * cfg.thickness * cfg.thickness,
    # )
    # flap_21_vis = box.add_shape_box(
    #     flap_21,
    #     hx=cfg.width / 2,
    #     hy=cfg.thickness,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # flap_22 = box.add_body(
    #     mass=cfg.density * cfg.width * cfg.height * cfg.thickness,
    # )
    # flap_22_vis = box.add_shape_box(
    #     flap_22,
    #     hx=cfg.width / 2,
    #     hy=cfg.height / 2,
    #     hz=cfg.thickness / 2,
    #     cfg=newton.ModelBuilder.ShapeConfig(
    #         is_visible=show_visuals,
    #         has_particle_collision=False,
    #         has_shape_collision=False,
    #     ),
    # )
    # flap_22_col = box.add_shape_box(
    #     flap_22,
    #     hx=cfg.width / 2 - cfg.contact_offset,
    #     hy=cfg.height / 2 - cfg.contact_offset,
    #     hz=cfg.thickness / 2,
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
    #     parent_xform=wp.transform(p=wp.vec3(0.0, cfg.height / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -cfg.thickness, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )
    # box.add_joint_revolute(
    #     parent=flap_11,
    #     child=flap_12,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, cfg.thickness, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, -cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )
    # box.add_joint_revolute(
    #     parent=flap_2,
    #     child=flap_21,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -cfg.height / 2, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, cfg.thickness, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )
    # box.add_joint_revolute(
    #     parent=flap_21,
    #     child=flap_22,
    #     parent_xform=wp.transform(p=wp.vec3(0.0, -cfg.thickness, 0.0)),
    #     child_xform=wp.transform(p=wp.vec3(0.0, cfg.height / 2, 0.0)),
    #     axis=wp.vec3(1.0, 0.0, 0.0),
    #     friction=cfg.joint_friction,
    #     target_ke=cfg.joint_target_ke,
    #     target_kd=cfg.joint_target_kd,
    # )

    return box