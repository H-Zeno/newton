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
# Example Sim Cloth Hanging
#
# This simulation demonstrates a simple cloth hanging behavior. A planar cloth
# mesh is fixed on one side and hangs under gravity, colliding with the ground.
#
# Command: python -m newton.examples cloth_hanging (--solver [semi_implicit, style3d, xpbd, vbd])
#
###########################################################################

import warp as wp
import numpy as np
from collections import deque
import newton
import newton.examples

class ClothBuilder(newton.ModelBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_aniso_cloth_grid(
        self,
        pos,
        rot,
        vel,
        dim_x: int,
        dim_y: int,
        cell_x: float,
        cell_y: float,
        mass: float,
        *,
        reverse_winding: bool = False,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        # triangle (membrane) parameters (still isotropic per element)
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        # --- anisotropic bending on mesh edges ---
        edge_ke_x: float | None = None,   # bending stiffness for horizontal edges (x-direction)
        edge_ke_y: float | None = None,   # bending stiffness for vertical edges (y-direction)
        edge_ke_diag: float | None = None,  # bending stiffness for interior diagonals
        edge_kd_x: float | None = None,
        edge_kd_y: float | None = None,
        edge_kd_diag: float | None = None,
        # --- optional anisotropic in-plane springs (structural) ---
        add_directional_springs: bool = True,
        spring_ke_x: float | None = None,   # springs along horizontal grid edges
        spring_kd_x: float | None = None,
        spring_ke_y: float | None = None,   # springs along vertical grid edges
        spring_kd_y: float | None = None,
        spring_ke_diag: float | None = None,  # springs along both diagonals of each quad
        spring_kd_diag: float | None = None,
        # --- creases (index in grid coordinates) ---
        crease_x: list[int] | None = None,   # list of column indices (constant x) to weaken (vertical crease lines)
        crease_y: list[int] | None = None,   # list of row indices (constant y) to weaken (horizontal crease lines)
        crease_bend_scale: float = 0.1,      # multiply edge_ke on crease edges by this factor
        crease_stretch_scale: float = 0.25,  # multiply spring_ke crossing the crease by this factor
        particle_radius: float | None = None,
    ):
        """
        Create a planar, rectangular shell/cloth grid with directional bending and
        (optional) directional springs. Also supports 'crease' lines to reduce
        bending and stretch locally for easy folding.

        Grid coordinates:
        - Local x in [0..dim_x], local y in [0..dim_y]
        - Vertex index in row-major order: vid = y*(dim_x+1) + x
        """

        # --- defaults pulled from the builder's existing defaults ---
        tri_ke  = tri_ke  if tri_ke  is not None else self.default_tri_ke
        tri_ka  = tri_ka  if tri_ka  is not None else self.default_tri_ka
        tri_kd  = tri_kd  if tri_kd  is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        # if user doesn't specify directional values, fall back to isotropic defaults
        base_edge_ke = getattr(self, "default_edge_ke", 0.0)
        base_edge_kd = getattr(self, "default_edge_kd", 0.0)
        edge_ke_x    = base_edge_ke if edge_ke_x is None else edge_ke_x
        edge_ke_y    = base_edge_ke if edge_ke_y is None else edge_ke_y
        edge_ke_diag = base_edge_ke if edge_ke_diag is None else edge_ke_diag
        edge_kd_x    = base_edge_kd if edge_kd_x is None else edge_kd_x
        edge_kd_y    = base_edge_kd if edge_kd_y is None else edge_kd_y
        edge_kd_diag = base_edge_kd if edge_kd_diag is None else edge_kd_diag

        base_spring_ke = getattr(self, "default_spring_ke", 0.0)
        base_spring_kd = getattr(self, "default_spring_kd", 0.0)
        spring_ke_x    = base_spring_ke if spring_ke_x is None else spring_ke_x
        spring_kd_x    = base_spring_kd if spring_kd_x is None else spring_kd_x
        spring_ke_y    = base_spring_ke if spring_ke_y is None else spring_ke_y
        spring_kd_y    = base_spring_kd if spring_kd_y is None else spring_kd_y
        spring_ke_diag = base_spring_ke if spring_ke_diag is None else spring_ke_diag
        spring_kd_diag = base_spring_kd if spring_kd_diag is None else spring_kd_diag

        particle_radius = particle_radius if particle_radius is not None else self.default_particle_radius

        crease_x = set(crease_x or [])
        crease_y = set(crease_y or [])

        # --- build vertices & triangles (same layout as add_cloth_grid) ---
        def grid_index(x, y, dim_x):
            return y * dim_x + x

        indices, vertices = [], []
        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):
                vertices.append(wp.vec3(x * cell_x, y * cell_y, 0.0))
                if x > 0 and y > 0:
                    v0 = grid_index(x - 1, y - 1, dim_x + 1)
                    v1 = grid_index(x,     y - 1, dim_x + 1)
                    v2 = grid_index(x,     y,     dim_x + 1)
                    v3 = grid_index(x - 1, y,     dim_x + 1)
                    if reverse_winding:
                        indices.extend([v0, v1, v2])
                        indices.extend([v0, v2, v3])
                    else:
                        indices.extend([v0, v1, v3])
                        indices.extend([v1, v2, v3])

        start_vertex = len(self.particle_q)

        # mass per-area (corrected total mass computation)
        total_mass = mass * (dim_x + 1) * (dim_y + 1)
        total_area = cell_x * cell_y * dim_x * dim_y
        density = total_mass / max(total_area, 1e-12)

        # --- add particles (initially massless; we'll distribute via triangle areas) ---
        num_verts = len(vertices)
        vertices_np = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float32)
        rot_mat_np = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
        verts_3d_np = (vertices_np @ rot_mat_np.T) + np.array([pos[0], pos[1], pos[2]], dtype=np.float32)

        self.add_particles(
            verts_3d_np.tolist(),
            [vel] * num_verts,
            mass=[0.0] * num_verts,
            radius=[particle_radius] * num_verts,
        )

        # --- add triangles with (still) scalar per-triangle parameters ---
        num_tris = int(len(indices) / 3)
        start_tri = len(self.tri_indices)
        inds = start_vertex + np.array(indices, dtype=np.int32)
        inds = inds.reshape(-1, 3)
        areas = self.add_triangles(
            inds[:, 0],
            inds[:, 1],
            inds[:, 2],
            [tri_ke]   * num_tris,
            [tri_ka]   * num_tris,
            [tri_kd]   * num_tris,
            [tri_drag] * num_tris,
            [tri_lift] * num_tris,
        )

        # distribute mass from areal density
        for t in range(num_tris):
            a = areas[t]
            i0, i1, i2 = int(inds[t, 0]), int(inds[t, 1]), int(inds[t, 2])
            m = density * a / 3.0
            self.particle_mass[i0] += m
            self.particle_mass[i1] += m
            self.particle_mass[i2] += m

        end_tri = len(self.tri_indices)

        # --- build edge list with adjacency (for bending constraints) ---
        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)
        edge_indices = np.fromiter(
            (x for e in adj.edges.values() for x in (e.o0, e.o1, e.v0, e.v1)),
            int,
        ).reshape(-1, 4)

        # Precompute each local-vid -> (gx, gy) in grid coords for quick orientation tests
        gx = np.zeros(num_verts, dtype=np.int32)
        gy = np.zeros(num_verts, dtype=np.int32)
        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):
                vid_local = y * (dim_x + 1) + x
                gx[vid_local] = x
                gy[vid_local] = y

        # Helper predicates
        def classify_edge(v0, v1):
            # Work in local grid coords
            v0l, v1l = v0 - start_vertex, v1 - start_vertex
            dx = gx[v1l] - gx[v0l]
            dy = gy[v1l] - gy[v0l]
            if dy == 0 and dx != 0:
                return "h"   # horizontal edge (along +x/-x)
            if dx == 0 and dy != 0:
                return "v"   # vertical edge (along +y/-y)
            return "d"       # diagonal inside a quad

        def on_vertical_crease(v0, v1):
            v0l, v1l = v0 - start_vertex, v1 - start_vertex
            # same column index for both vertices?
            if gx[v0l] != gx[v1l]:
                return False
            return gx[v0l] in crease_x

        def on_horizontal_crease(v0, v1):
            v0l, v1l = v0 - start_vertex, v1 - start_vertex
            if gy[v0l] != gy[v1l]:
                return False
            return gy[v0l] in crease_y

        # Build per-edge bending parameters
        ke_list, kd_list = [], []
        for (o0, o1, v0, v1) in edge_indices:
            kind = classify_edge(v0, v1)
            if kind == "h":
                ke = edge_ke_x
                kd = edge_kd_x
                # horizontal edge lies along a horizontal crease?
                if on_horizontal_crease(v0, v1):
                    ke *= crease_bend_scale
            elif kind == "v":
                ke = edge_ke_y
                kd = edge_kd_y
                # vertical edge lies along a vertical crease?
                if on_vertical_crease(v0, v1):
                    ke *= crease_bend_scale
            else:  # diagonal
                ke = edge_ke_diag
                kd = edge_kd_diag
            ke_list.append(ke)
            kd_list.append(kd)

        self.add_edges(
            edge_indices[:, 0],
            edge_indices[:, 1],
            edge_indices[:, 2],
            edge_indices[:, 3],
            edge_ke=ke_list,
            edge_kd=kd_list,
        )

        # --- optional: add anisotropic structural springs for in-plane orthotropy ---
        if add_directional_springs:
            # Horizontal springs (between (x-1,y) and (x,y))
            for y in range(0, dim_y + 1):
                for x in range(1, dim_x + 1):
                    i = start_vertex + (y * (dim_x + 1) + (x - 1))
                    j = start_vertex + (y * (dim_x + 1) + x)
                    ke, kd = spring_ke_x, spring_kd_x
                    # if this spring *crosses* a vertical crease at column x (i.e., between x-1 and x)
                    if x in crease_x:
                        ke *= crease_stretch_scale
                    self.add_spring(i, j, ke, kd, control=0.0)

            # Vertical springs (between (x,y-1) and (x,y))
            for x in range(0, dim_x + 1):
                for y in range(1, dim_y + 1):
                    i = start_vertex + ((y - 1) * (dim_x + 1) + x)
                    j = start_vertex + (y * (dim_x + 1) + x)
                    ke, kd = spring_ke_y, spring_kd_y
                    # if this spring *crosses* a horizontal crease at row y
                    if y in crease_y:
                        ke *= crease_stretch_scale
                    self.add_spring(i, j, ke, kd, control=0.0)

            # Diagonal springs inside each cell: ((x-1,y-1)->(x,y)) and ((x,y-1)->(x-1,y))
            for y in range(1, dim_y + 1):
                for x in range(1, dim_x + 1):
                    v00 = start_vertex + ((y - 1) * (dim_x + 1) + (x - 1))
                    v10 = start_vertex + ((y - 1) * (dim_x + 1) + x)
                    v01 = start_vertex + (y * (dim_x + 1) + (x - 1))
                    v11 = start_vertex + (y * (dim_x + 1) + x)
                    # diag (\): (x-1,y-1) -> (x,y)
                    self.add_spring(v00, v11, spring_ke_diag, spring_kd_diag, control=0.0)
                    # diag (/): (x,y-1) -> (x-1,y)
                    self.add_spring(v10, v01, spring_ke_diag, spring_kd_diag, control=0.0)

        # --- fix boundary vertices (same semantics as add_cloth_grid) ---
        vertex_id = 0
        for y in range(dim_y + 1):
            for x in range(dim_x + 1):
                particle_mass = mass
                particle_flag = newton.ParticleFlags.ACTIVE
                if (
                    (x == 0 and fix_left)
                    or (x == dim_x and fix_right)
                    or (y == 0 and fix_bottom)
                    or (y == dim_y and fix_top)
                ):
                    particle_flag = particle_flag & ~newton.ParticleFlags.ACTIVE
                    particle_mass = 0.0

                self.particle_flags[start_vertex + vertex_id] = particle_flag
                self.particle_mass[start_vertex + vertex_id] = particle_mass if self.particle_mass[start_vertex + vertex_id] > 0.0 else particle_mass
                vertex_id += 1

    def add_rectangle_tree_cloth(
        self,
        root,
        *,
        pos,
        rot,
        vel,
        cell_x: float,
        cell_y: float,
        mass: float,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        crease_edge_ke_ratio: float = 0.1,
        crease_tri_ke_ratio: float = 0.3,
        particle_radius: float | None = None,
    ):
        """
        Assemble a unified shell from a Rectangle tree. Shared edges are stitched
        (same vertices) and become interior bending edges. Each connection uses the
        Rectangle's angle to set the initial rest fold angle. Bending and triangle
        stiffnesses near the crease are reduced by the given ratios.
        """


        # -------- defaults from builder ----------
        tri_ke  = self.default_tri_ke  if tri_ke  is None else tri_ke
        tri_ka  = self.default_tri_ka  if tri_ka  is None else tri_ka
        tri_kd  = self.default_tri_kd  if tri_kd  is None else tri_kd
        tri_drag = self.default_tri_drag if tri_drag is None else tri_drag
        tri_lift = self.default_tri_lift if tri_lift is None else tri_lift
        edge_ke = self.default_edge_ke if edge_ke is None else edge_ke
        edge_kd = self.default_edge_kd if edge_kd is None else edge_kd
        particle_radius = self.default_particle_radius if particle_radius is None else particle_radius

        # -------- utilities ----------
        def clamp_n(n): return max(1, int(round(n)))
        def rodrigues(axis_unit, angle_rad):
            ax = axis_unit / (np.linalg.norm(axis_unit) + 1e-12)
            x, y, z = ax
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            C = 1.0 - c
            return np.array([[x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
                            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
                            [z*x*C - y*s, z*y*C + x*s, z*z*C + c   ]], dtype=np.float32)

        # Get world frame from rot (consistent with add_cloth_mesh usage)
        R = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
        ex0, ey0, ez0 = R[:, 0], R[:, 1], R[:, 2]
        pos0 = np.array([pos[0], pos[1], pos[2]], dtype=np.float32)

        # -------- pass 1: decide panel grid resolution (nx, ny) and collect graph --------
        # Propagate segment counts so shared edges match exactly.
        seg = {}          # rect -> (nx, ny)
        parent_of = {}    # rect -> (parent_rect, side_from_parent)
        angle_of = {}     # (parent, child) -> angle_degrees

        def nx_for(w): return clamp_n(w)
        def ny_for(h): return clamp_n(h)

        seg[root] = (nx_for(root.x), ny_for(root.y))
        q = deque([root])

        def neighbors(r):
            # (child, side_name, angle_deg_from_parent)
            if r.top:    yield (r.top,    "top",    r.top_angle or 0.0)
            if r.bottom: yield (r.bottom, "bottom", r.bottom_angle or 0.0)
            if r.left:   yield (r.left,   "left",   r.left_angle or 0.0)
            if r.right:  yield (r.right,  "right",  r.right_angle or 0.0)

        while q:
            r = q.popleft()
            rx, ry = seg[r]
            for c, side, ang in neighbors(r):
                # enforce shared segments along the hinge
                if side in ("top", "bottom"):
                    nx_child = rx                    # must match along width
                    ny_child = ny_for(c.y)
                else:
                    nx_child = nx_for(c.x)
                    ny_child = ry                    # must match along height

                if c not in seg:
                    seg[c] = (nx_child, ny_child)
                    parent_of[c] = (r, side)
                    angle_of[(r, c)] = float(ang)
                    q.append(c)
                else:
                    # if already assigned, enforce shared dimension equality
                    cx, cy = seg[c]
                    changed = False
                    if side in ("top", "bottom") and cx != rx:
                        cx = rx; changed = True
                    if side in ("left", "right") and cy != ry:
                        cy = ry; changed = True
                    if changed:
                        seg[c] = (cx, cy)
                        parent_of[c] = (r, side)
                        angle_of[(r, c)] = float(ang)
                        q.append(c)

        # -------- pass 2: compute world frame (origin + axes) per panel based on fold angles --------
        frame_O, frame_ex, frame_ey, frame_ez = {}, {}, {}, {}
        frame_O[root], frame_ex[root], frame_ey[root], frame_ez[root] = pos0, ex0, ey0, ez0

        q = deque([root])
        while q:
            r = q.popleft()
            Or, exr, eyr, ezr = frame_O[r], frame_ex[r], frame_ey[r], frame_ez[r]
            nxr, nyr = seg[r]
            # Child placement: planar origin touching the correct edge, then rotate about hinge axis by angle
            for c, side, ang in neighbors(r):
                if c not in frame_O:
                    cx, cy = seg[c]
                    ang_rad = np.deg2rad(float(ang))
                    # hinge line start (world) and axis
                    if side == "top":
                        hinge_O = Or + eyr * (nyr * cell_y)
                        axis = exr
                        O_planar = hinge_O                                  # child's bottom edge on hinge
                        Rc = rodrigues(axis, ang_rad)
                        Oc = hinge_O + Rc @ (O_planar - hinge_O)            # same as O_planar here
                        exc = exr                                          # axis unchanged
                        eyc = Rc @ eyr
                        ezc = Rc @ ezr
                    elif side == "bottom":
                        hinge_O = Or                                        # y=0
                        axis = exr
                        O_planar = hinge_O - eyr * (cy * cell_y)            # child's top edge on hinge
                        Rc = rodrigues(axis, ang_rad)
                        Oc = hinge_O + Rc @ (O_planar - hinge_O)
                        exc = exr
                        eyc = Rc @ eyr
                        ezc = Rc @ ezr
                    elif side == "left":
                        hinge_O = Or                                        # x=0
                        axis = eyr
                        O_planar = hinge_O - exr * (cx * cell_x)            # child's right edge on hinge
                        Rc = rodrigues(axis, ang_rad)
                        Oc = hinge_O + Rc @ (O_planar - hinge_O)
                        exc = Rc @ exr
                        eyc = eyr
                        ezc = Rc @ ezr
                    elif side == "right":
                        hinge_O = Or + exr * (nxr * cell_x)
                        axis = eyr
                        O_planar = hinge_O                                  # child's left edge on hinge
                        Rc = rodrigues(axis, ang_rad)
                        Oc = hinge_O + Rc @ (O_planar - hinge_O)            # same as O_planar here
                        exc = Rc @ exr
                        eyc = eyr
                        ezc = Rc @ ezr
                    frame_O[c], frame_ex[c], frame_ey[c], frame_ez[c] = Oc, exc, eyc, ezc
                    q.append(c)

        # -------- pass 3: emit vertices (shared on hinges) and triangles ----------
        def grid_index(i, j, nx): return j * (nx + 1) + i

        start_vertex = len(self.particle_q)

        vertices = []              # list of np.array([x,y,z])
        vid_map = {}               # (rect, i, j) -> global local index (before start_vertex offset)
        crease_pairs = set()       # set of (min_vid, max_vid) pairs that lie along a crease
        tris = []                  # (vi, vj, vk) in local indexing
        # we also want to know, for each (rect, side), the list of hinge vertex ids in order:
        hinge_rows = {}            # (rect, 'top'|'bottom'|'left'|'right') -> list of local vertex ids along the hinge (ordered)

        def ensure_vertex(r, i, j):
            key = (r, i, j)
            if key in vid_map:
                return vid_map[key]
            Or, exr, eyr = frame_O[r], frame_ex[r], frame_ey[r]
            p = Or + exr * (i * cell_x) + eyr * (j * cell_y)
            vid = len(vertices)
            vertices.append(p.astype(np.float32))
            vid_map[key] = vid
            return vid

        # Emit root panel vertices & triangles
        rx, ry = seg[root]
        for j in range(ry + 1):
            for i in range(rx + 1):
                ensure_vertex(root, i, j)

        # keep ordered hinge rows for the root (used to stitch with children)
        hinge_rows[(root, "top")]    = [vid_map[(root, i, ry)] for i in range(rx + 1)]
        hinge_rows[(root, "bottom")] = [vid_map[(root, i, 0)]  for i in range(rx + 1)]
        hinge_rows[(root, "left")]   = [vid_map[(root, 0, j)]  for j in range(ry + 1)]
        hinge_rows[(root, "right")]  = [vid_map[(root, rx, j)] for j in range(ry + 1)]

        # Now BFS to add children, sharing hinge vertices
        q = deque([root])
        visited = {root}
        while q:
            r = q.popleft()
            rx, ry = seg[r]
            # triangles for r
            for j in range(1, ry + 1):
                for i in range(1, rx + 1):
                    v0 = vid_map[(r, i - 1, j - 1)]
                    v1 = vid_map[(r, i,     j - 1)]
                    v2 = vid_map[(r, i,     j    )]
                    v3 = vid_map[(r, i - 1, j    )]
                    tris.append((v0, v1, v3))
                    tris.append((v1, v2, v3))
            # neighbors
            for c, side, _ang in neighbors(r):
                if c in visited:
                    continue
                cx, cy = seg[c]
                Or, exr, eyr = frame_O[r], frame_ex[r], frame_ey[r]
                Oc, exc, eyc = frame_O[c], frame_ex[c], frame_ey[c]

                # Determine which edge matches and map child's hinge vertices to parent's
                if side == "top":
                    # parent top row (rx+1 vertices) <-> child bottom row (cx+1)
                    parent_row = hinge_rows[(r, "top")]
                    assert len(parent_row) == cx + 1
                    # place child's entire grid; reuse bottom row ids
                    for i in range(cx + 1):
                        vid_map[(c, i, 0)] = parent_row[i]
                    # others
                    for j in range(1, cy + 1):
                        for i in range(cx + 1):
                            ensure_vertex(c, i, j)
                    # record ordered rows/cols for child
                    hinge_rows[(c, "bottom")] = [vid_map[(c, i, 0)]  for i in range(cx + 1)]
                    hinge_rows[(c, "top")]    = [vid_map[(c, i, cy)] for i in range(cx + 1)]
                    hinge_rows[(c, "left")]   = [vid_map[(c, 0, j)]  for j in range(cy + 1)]
                    hinge_rows[(c, "right")]  = [vid_map[(c, cx, j)] for j in range(cy + 1)]
                    # mark crease edge pairs along the shared row
                    row = hinge_rows[(c, "bottom")]
                    for i in range(1, len(row)):
                        a, b = row[i - 1], row[i]
                        crease_pairs.add((a, b) if a < b else (b, a))

                elif side == "bottom":
                    # parent bottom row <-> child top row
                    parent_row = hinge_rows[(r, "bottom")]
                    assert len(parent_row) == cx + 1
                    for i in range(cx + 1):
                        vid_map[(c, i, cy)] = parent_row[i]
                    for j in range(cy):
                        for i in range(cx + 1):
                            ensure_vertex(c, i, j)
                    hinge_rows[(c, "top")]    = [vid_map[(c, i, cy)] for i in range(cx + 1)]
                    hinge_rows[(c, "bottom")] = [vid_map[(c, i, 0)]  for i in range(cx + 1)]
                    hinge_rows[(c, "left")]   = [vid_map[(c, 0, j)]  for j in range(cy + 1)]
                    hinge_rows[(c, "right")]  = [vid_map[(c, cx, j)] for j in range(cy + 1)]
                    row = hinge_rows[(c, "top")]
                    for i in range(1, len(row)):
                        a, b = row[i - 1], row[i]
                        crease_pairs.add((a, b) if a < b else (b, a))

                elif side == "left":
                    # parent left col <-> child right col
                    parent_col = hinge_rows[(r, "left")]
                    assert len(parent_col) == cy + 1
                    for j in range(cy + 1):
                        vid_map[(c, cx, j)] = parent_col[j]
                    for i in range(cx):
                        for j in range(cy + 1):
                            ensure_vertex(c, i, j)
                    hinge_rows[(c, "right")]  = [vid_map[(c, cx, j)] for j in range(cy + 1)]
                    hinge_rows[(c, "left")]   = [vid_map[(c, 0, j)]  for j in range(cy + 1)]
                    hinge_rows[(c, "top")]    = [vid_map[(c, i, cy)] for i in range(cx + 1)]
                    hinge_rows[(c, "bottom")] = [vid_map[(c, i, 0)]  for i in range(cx + 1)]
                    col = hinge_rows[(c, "right")]
                    for j in range(1, len(col)):
                        a, b = col[j - 1], col[j]
                        crease_pairs.add((a, b) if a < b else (b, a))

                elif side == "right":
                    # parent right col <-> child left col
                    parent_col = hinge_rows[(r, "right")]
                    assert len(parent_col) == cy + 1
                    for j in range(cy + 1):
                        vid_map[(c, 0, j)] = parent_col[j]
                    for i in range(1, cx + 1):
                        for j in range(cy + 1):
                            ensure_vertex(c, i, j)
                    hinge_rows[(c, "left")]   = [vid_map[(c, 0, j)]  for j in range(cy + 1)]
                    hinge_rows[(c, "right")]  = [vid_map[(c, cx, j)] for j in range(cy + 1)]
                    hinge_rows[(c, "top")]    = [vid_map[(c, i, cy)] for i in range(cx + 1)]
                    hinge_rows[(c, "bottom")] = [vid_map[(c, i, 0)]  for i in range(cx + 1)]
                    col = hinge_rows[(c, "left")]
                    for j in range(1, len(col)):
                        a, b = col[j - 1], col[j]
                        crease_pairs.add((a, b) if a < b else (b, a))

                # emit child triangles now
                for j in range(1, cy + 1):
                    for i in range(1, cx + 1):
                        v0 = vid_map[(c, i - 1, j - 1)]
                        v1 = vid_map[(c, i,     j - 1)]
                        v2 = vid_map[(c, i,     j    )]
                        v3 = vid_map[(c, i - 1, j    )]
                        tris.append((v0, v1, v3))
                        tris.append((v1, v2, v3))

                visited.add(c)
                q.append(c)

        # -------- pass 4: add particles, triangles (with per-tri ke), then edges (with per-edge ke) --------
        start_tri = len(self.tri_indices)
        num_verts_local = len(vertices)
        num_tris_local = len(tris)

        # Particles: place them (they're already in world space), zero mass; distribute later by area
        self.add_particles(
            [v.tolist() for v in vertices],
            [vel] * num_verts_local,
            mass=[0.0] * num_verts_local,
            radius=[particle_radius] * num_verts_local,
        )

        # Build triangle index array with global indexing
        inds = np.array(tris, dtype=np.int32) + start_vertex

        # Per-triangle parameters, default
        tri_ke_arr   = np.full(num_tris_local, tri_ke, dtype=np.float32)
        tri_ka_arr   = np.full(num_tris_local, tri_ka, dtype=np.float32)
        tri_kd_arr   = np.full(num_tris_local, tri_kd, dtype=np.float32)
        tri_drag_arr = np.full(num_tris_local, tri_drag, dtype=np.float32)
        tri_lift_arr = np.full(num_tris_local, tri_lift, dtype=np.float32)

        # Build simple adjacency (edge -> list of (tri_id, opposite_vertex_global))
        edge_map = {}  # (minv, maxv) -> [(t, opp), (t, opp)] or [(t, opp)]
        for t, (a, b, c) in enumerate(inds):
            for (v0, v1, opp) in ((a, b, c), (b, c, a), (c, a, b)):
                key = (v0, v1) if v0 < v1 else (v1, v0)
                edge_map.setdefault(key, []).append((t, opp))

        # Triangles touching any crease edge get reduced tri_ke
        for key in crease_pairs:
            gkey = (key[0] + start_vertex, key[1] + start_vertex)  # already added start_vertex above, so DON'T add again
            # Wait: key already stored as full global indices (we used ensure_vertex after start_vertex?). Fix:
            # crease_pairs was built from vid_map (local indices) BEFORE adding start_vertex.
            # But above we added start_vertex when creating 'inds'. So we must look for (global edge) using (local + start_vertex).
            # Build corresponding global key:
            gkey = (key[0] + start_vertex, key[1] + start_vertex)
            lst = edge_map.get(gkey, [])
            for (t, _opp) in lst:
                tri_ke_arr[t] = tri_ke * float(crease_tri_ke_ratio)

        # Add triangles and distribute mass via areas
        areas = self.add_triangles(
            inds[:, 0], inds[:, 1], inds[:, 2],
            tri_ke_arr.tolist(), tri_ka_arr.tolist(), tri_kd_arr.tolist(),
            tri_drag_arr.tolist(), tri_lift_arr.tolist()
        )
        # mass distribution from areal density
        total_area = 0.0
        # area of each Rectangle panel is invariant to folding, so sum in 2D:
        seen = set()
        for r in seg.keys():
            if id(r) not in seen:
                nx, ny = seg[r]
                total_area += float(nx * cell_x) * float(ny * cell_y)
                seen.add(id(r))
        total_mass = mass * num_verts_local
        density = total_mass / max(total_area, 1e-12)

        for t in range(num_tris_local):
            a = areas[t]
            i0, i1, i2 = int(inds[t, 0]), int(inds[t, 1]), int(inds[t, 2])
            m = density * a / 3.0
            self.particle_mass[i0] += m
            self.particle_mass[i1] += m
            self.particle_mass[i2] += m

        end_tri = len(self.tri_indices)

        # Build edge lists for bending constraints, with per-edge stiffness
        o0_list, o1_list, v0_list, v1_list = [], [], [], []
        edge_ke_list, edge_kd_list = [], []

        for (v0, v1), lst in edge_map.items():
            if len(lst) == 2:
                (tA, oppA), (tB, oppB) = lst[0], lst[1]
                o0, o1 = int(oppA), int(oppB)
            else:
                (tA, oppA) = lst[0]
                o0, o1 = int(oppA), -1

            v0_list.append(int(v0))
            v1_list.append(int(v1))
            o0_list.append(o0)
            o1_list.append(o1)

            # crease detection (remember: crease_pairs stored LOCAL ids; convert and compare)
            local_key = (v0 - start_vertex, v1 - start_vertex)
            if local_key[0] > local_key[1]:
                local_key = (local_key[1], local_key[0])
            is_crease = local_key in crease_pairs
            ke = edge_ke * (crease_edge_ke_ratio if is_crease else 1.0)
            kd = edge_kd
            edge_ke_list.append(float(ke))
            edge_kd_list.append(float(kd))

        self.add_edges(
            np.array(o0_list, dtype=np.int32),
            np.array(o1_list, dtype=np.int32),
            np.array(v0_list, dtype=np.int32),
            np.array(v1_list, dtype=np.int32),
            edge_ke=edge_ke_list, edge_kd=edge_kd_list
        )

class Rectangle:
    def __init__(self, x, y):
        self.x = x  # number of segments in x direction (nx)
        self.y = y  # number of segments in y direction (ny)
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.top_angle = None
        self.bottom_angle = None
        self.left_angle = None
        self.right_angle = None
    
    def add_top(self, new_y, angle=90):
        """Add rectangle to top edge (shares edge of length x)"""
        new_rect = Rectangle(self.x, new_y)
        self.top = new_rect
        self.top_angle = angle
        new_rect.bottom = self
        new_rect.bottom_angle = angle
        return new_rect
    
    def add_bottom(self, new_y, angle=90):
        """Add rectangle to bottom edge (shares edge of length x)"""
        new_rect = Rectangle(self.x, new_y)
        self.bottom = new_rect
        self.bottom_angle = angle
        new_rect.top = self
        new_rect.top_angle = angle
        return new_rect
    
    def add_left(self, new_x, angle=90):
        """Add rectangle to left edge (shares edge of length y)"""
        new_rect = Rectangle(new_x, self.y)
        self.left = new_rect
        self.left_angle = angle
        new_rect.right = self
        new_rect.right_angle = angle
        return new_rect
    
    def add_right(self, new_x, angle=90):
        """Add rectangle to right edge (shares edge of length y)"""
        new_rect = Rectangle(new_x, self.y)
        self.right = new_rect
        self.right_angle = angle
        new_rect.left = self
        new_rect.left_angle = angle
        return new_rect

    def __repr__(self):
        return f"Rect({self.x}x{self.y})"

class Example:
    def __init__(
        self,
        viewer,
        solver_type: str = "vbd",
        height=32,
        width=64,
    ):
        # setup simulation parameters first
        self.solver_type = solver_type

        self.sim_height = height
        self.sim_width = width
        self.sim_time = 0.0

        self.fps = 120
        self.frame_dt = 1.0 / self.fps

        self.sim_substeps = 20

        self.iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # if self.solver_type == "style3d":
        #     builder = newton.Style3DModelBuilder()
        # else:
        builder = ClothBuilder()


        builder.add_ground_plane()

        cell_size = 0.5
        tri_ke = 5.0e3
        tri_ka = 5.0e3
        tri_kd = 2
        edge_ke = 4.0e2
        edge_kd = 0.02
        particle_radius = 0.1
        crease_edge_ke_ratio = 0.0001
        crease_tri_ke_ratio = 1

        # builder.add_cloth_grid(**common_params, **solver_params)


        base = Rectangle(10, 5)
        side1 = base.add_right(3, angle=10)  
        # side2 = base.add_left(3, angle=-10)   
        # top = base.add_top(10, angle=10)     
        # bottom = base.add_bottom(10, angle=-10)
        # flap = side1.add_right(2, angle=10) 

        builder.add_aniso_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 4.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=base.x + 3,
            dim_y=base.y,
            cell_x=cell_size,
            cell_y=cell_size,
            # mass=0.1,  # cloth like
            mass=0.1,  # cardboard like
            # membrane
            # tri_ke=3.0e4, tri_ka=1.0e3, tri_kd=5.0e-5, # old values
            # tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e-1, # cloth like
            tri_ke=tri_ke, tri_ka=tri_ka, tri_kd=tri_kd, # cardboard like
            # bending: stiffer across one axis, softer across the other
            # edge_ke_x=2.0e-5, edge_ke_y=6.0e-5, edge_ke_diag=4.0e-5, # old values
            # edge_ke_x=1.0e1, edge_ke_y=1.0e1, edge_ke_diag=1.0e1, # cloth like
            edge_ke_x=edge_ke, edge_ke_y=edge_ke, edge_ke_diag=edge_ke, # cardboard like
            # edge_kd_x=1.0e-3, edge_kd_y=1.0e-3, edge_kd_diag=1.0e-3, # old values
            # edge_kd_x=0, edge_kd_y=0, edge_kd_diag=0, # cloth like
            edge_kd_x=edge_kd, edge_kd_y=edge_kd, edge_kd_diag=edge_kd, # cardboard like
            # in-plane (optional)
            # add_directional_springs=True,
            # spring_ke_x=3.0e4, spring_kd_x=2.0e-4,
            # spring_ke_y=1.5e4, spring_kd_y=2.0e-4,
            # spring_ke_diag=5.0e3, spring_kd_diag=2.0e-4,
            # particle_radius=min(0.60/60, 0.40/40)*0.5,

            crease_x=[10],
            crease_bend_scale=crease_edge_ke_ratio,

            particle_radius=particle_radius,
            # fix_left=True,
        )
        
        # builder.add_rectangle_tree_cloth(
        #     root=base,
        #     pos=wp.vec3(0.0, 0.0, 4.0),
        #     rot=wp.quat_identity(),
        #     vel=wp.vec3(0.0, 0.0, 0.0),
        #     cell_x=cell_size,
        #     cell_y=cell_size,
        #     mass=0.001,
        #     tri_ke=tri_ke,
        #     tri_ka=tri_ka,
        #     tri_kd=tri_kd,
        #     edge_ke=edge_ke,
        #     edge_kd=edge_kd,
        #     particle_radius=particle_radius,
        #     crease_edge_ke_ratio=crease_edge_ke_ratio,
        #     crease_tri_ke_ratio=crease_tri_ke_ratio,
        # )

        box_pos = wp.vec3(base.x * cell_size / 2, base.y * cell_size / 2, 1)
        body_box = builder.add_body(xform=wp.transform(p=box_pos, q=wp.quat_identity()), key="box")
        builder.add_shape_box(body_box, hx=base.x * cell_size / 2, hy=base.y * cell_size / 2, hz=1)



        # Prismatic joint setup (slider)
        mov_pos = wp.vec3((base.x + 2.5) * cell_size, base.y * cell_size / 2, 0)
        parent_body = builder.add_body(
            xform=wp.transform(p=mov_pos, q=wp.quat_identity()),
        )

        # 2. Add your dynamic block (the moving body)
        block_body = builder.add_body(
            xform=wp.transform(p=mov_pos +wp.vec3(0, 0, 1.0), q=wp.quat_identity()),
        )

        # Add shape to the block so you can see it
        builder.add_shape_box(
            body=block_body,
            xform=wp.transform(p=wp.vec3(0, 0, 0), q=wp.quat_identity()),
            hx=cell_size*2, hy=(base.y*cell_size)/2, hz=cell_size,  # half-extents
            cfg=newton.ModelBuilder.ShapeConfig(density=1),
        )

        builder.add_joint_prismatic(
            parent=parent_body,
            child=block_body,
            parent_xform=wp.transform(p=wp.vec3(0, 0, 0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0, 0, 0), q=wp.quat_identity()),
            axis=(0, 0, 1),           # Z-axis
            limit_lower=-2.0,         # optional: min Z position
            limit_upper=2.0           # optional: max Z position
        )

        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0


        # Use XPBD for rigid bodies/joints
        self.rigid_solver = newton.solvers.SolverXPBD(
            model=self.model,
            iterations=self.iterations,
        )

        # Use VBD for cloth particles
        self.cloth_solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
        )
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # Initialize joint positions
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
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Step 1: Update rigid bodies (joints) with XPBD
            if self.model.body_count > 0:
                particle_count = self.model.particle_count
                # Temporarily disable particles for rigid body step
                self.model.particle_count = 0
                
                # Step rigid bodies
                self.contacts = self.model.collide(self.state_0)
                self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                
                # Copy rigid body state
                self.state_0.body_q.assign(self.state_1.body_q)
                self.state_0.body_qd.assign(self.state_1.body_qd)
                
                # Restore particle count
                self.model.particle_count = particle_count

            # Step 2: Update cloth particles with VBD
            self.contacts = self.model.collide(self.state_0)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        if self.solver_type != "style3d":
            # TODO(Style3D): handle ground collisions
            newton.examples.test_particle_state(
                self.state_0,
                "particles are above the ground",
                lambda q, qd: q[2] > 0.0,
            )

        min_x = -float(self.sim_width) * 0.11
        p_lower = wp.vec3(min_x, -4.0, -1.8)
        p_upper = wp.vec3(0.1, 7.0, 4.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    # Add solver-specific arguments
    parser.add_argument(
        "--solver",
        help="Type of solver",
        type=str,
        choices=["semi_implicit", "style3d", "xpbd", "vbd"],
        default="vbd",
    )
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(
        viewer=viewer,
        solver_type=args.solver,
        height=args.height,
        width=args.width,
    )

    newton.examples.run(example, args)

    # Example usage:
# Example usage:
# base = Rectangle(10, 5)
# side1 = base.add_right(3, angle=180)  # flat
# side2 = base.add_left(3, angle=180)   # flat
# top = base.add_top(10, angle=180)     # flat
# bottom = base.add_bottom(10, angle=90)  # 90 degree fold
# flap = side1.add_right(2, angle=135)  # 135 degree fold

# base.visualize()
