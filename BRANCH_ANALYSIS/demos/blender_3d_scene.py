#!/usr/bin/env python3
"""FloPyArcade 3D Blender Scene — renders a cyberpunk groundwater simulation to MP4.

Run headless:
    blender --background --python blender_3d_scene.py
"""

import math
import numpy as np
import bpy

# ─── Configuration ───────────────────────────────────────────────────────────
NROW, NCOL = 12, 16
EXTENT_X, EXTENT_Y = 1000.0, 800.0
Z_TOP, Z_BOT = 50.0, 0.0
SEED = 42
FPS = 30
TOTAL_FRAMES = 90  # 3 seconds
RES_X, RES_Y = 1280, 720
OUTPUT_PATH = "/a0/usr/projects/flopyarcade_roadmapworker2/BRANCH_ANALYSIS/demos/flopyarcade_3d.mp4"

# ─── Clear scene ─────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100

# ─── Synthetic head field ────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
x = np.linspace(0, 1, NCOL)
y = np.linspace(0, 1, NROW)
X, Y = np.meshgrid(x, y)
heads = 25 + 15 * X + 8 * np.sin(4 * Y) + 5 * np.cos(3 * X * Y) + rng.normal(0, 2, (NROW, NCOL))
head_min, head_max = heads.min(), heads.max()

# ─── Colormap (terrain) ──────────────────────────────────────────────────────
def terrain_color(t):
    """Deep blue → green → yellow → brown → white."""
    stops = [
        (0.0,  0.02, 0.08, 0.35),
        (0.2,  0.0,  0.3,  0.6),
        (0.4,  0.1,  0.6,  0.4),
        (0.55, 0.2,  0.8,  0.3),
        (0.7,  0.7,  0.8,  0.2),
        (0.85, 0.85, 0.55, 0.2),
        (1.0,  1.0,  1.0,  1.0),
    ]
    for i in range(len(stops) - 1):
        t0, r0, g0, b0 = stops[i]
        t1, r1, g1, b1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / max(t1 - t0, 1e-9)
            return (r0 + (r1-r0)*f, g0 + (g1-g0)*f, b0 + (b1-b0)*f, 1.0)
    return stops[-1][1:]

# ─── Build terrain mesh ──────────────────────────────────────────────────────
vertices = []
colors = []
for i in range(NROW):
    for j in range(NCOL):
        px = -EXTENT_X/2 + EXTENT_X * j / (NCOL - 1)
        pz = -EXTENT_Y/2 + EXTENT_Y * i / (NROW - 1)
        # Scale head to world Y
        t = (heads[i, j] - head_min) / (head_max - head_min)
        py = Z_BOT + t * (Z_TOP - Z_BOT)
        vertices.append((px, py, pz))
        c = terrain_color(t)
        colors.extend(c)

edges = []
faces = []
for i in range(NROW - 1):
    for j in range(NCOL - 1):
        a = i * NCOL + j
        b = a + 1
        c = a + NCOL
        d = c + 1
        faces.append((a, b, d, c))
        edges.extend([(a, b), (a, c), (b, d), (c, d)])

mesh = bpy.data.meshes.new("Terrain")
mesh.from_pydata(vertices, edges, faces)
mesh.validate()
mesh.update()

# Add color attribute
color_attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
for idx, (r, g, b, a) in enumerate([colors[i*4:i*4+4] for i in range(NROW*NCOL)]):
    color_attr.data[idx].color = (r, g, b, 1.0)

terrain_obj = bpy.data.objects.new("Terrain", mesh)
scene.collection.objects.link(terrain_obj)

# Material with vertex colors
terrain_mat = bpy.data.materials.new("TerrainMat")
terrain_mat.use_nodes = True
nodes = terrain_mat.node_tree.nodes
nodes.clear()
out_node = nodes.new('ShaderNodeOutputMaterial')
bsdf = nodes.new('ShaderNodeBsdfPrincipled')
bsdf.inputs['Roughness'].default_value = 0.7
bsdf.inputs['Metallic'].default_value = 0.1
vc = nodes.new('ShaderNodeVertexColor')
vc.layer_name = "Col"
links = terrain_mat.node_tree.links
links.new(vc.outputs['Color'], bsdf.inputs['Base Color'])
links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
terrain_obj.data.materials.append(terrain_mat)

# ─── Wells (neon cylinders) ──────────────────────────────────────────────────
def make_well(px, pz, is_pump, name):
    """Create a glowing cylinder at (px, pz)."""
    # Find local head at nearest grid point
    gi = int((pz + EXTENT_Y/2) / EXTENT_Y * (NROW - 1))
    gj = int((px + EXTENT_X/2) / EXTENT_X * (NCOL - 1))
    gi = min(max(gi, 0), NROW-1)
    gj = min(max(gj, 0), NCOL-1)
    t = (heads[gi, gj] - head_min) / (head_max - head_min)
    base_y = Z_BOT + t * (Z_TOP - Z_BOT)
    
    height = 120.0
    radius = 15.0
    
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=height,
                                        location=(px, base_y + height/2, pz))
    well = bpy.context.active_object
    well.name = f"Well_{name}"
    
    # Emission material
    mat = bpy.data.materials.new(f"Mat_{name}")
    mat.use_nodes = True
    mn = mat.node_tree.nodes
    mn.clear()
    out = mn.new('ShaderNodeOutputMaterial')
    emis = mn.new('ShaderNodeEmission')
    if is_pump:
        emis.inputs['Color'].default_value = (1.0, 0.1, 0.5, 1.0)  # pink
        emis.inputs['Strength'].default_value = 8.0
    else:
        emis.inputs['Color'].default_value = (0.0, 1.0, 0.6, 1.0)  # green
        emis.inputs['Strength'].default_value = 8.0
    mat.node_tree.links.new(emis.outputs['Emission'], out.inputs['Surface'])
    well.data.materials.append(mat)
    
    # Glow ring at base
    bpy.ops.mesh.primitive_torus_add(major_radius=radius*2, minor_radius=3,
                                     location=(px, base_y, pz))
    ring = bpy.context.active_object
    ring.name = f"Ring_{name}"
    ring_mat = bpy.data.materials.new(f"RingMat_{name}")
    ring_mat.use_nodes = True
    rn = ring_mat.node_tree.nodes
    rn.clear()
    rout = rn.new('ShaderNodeOutputMaterial')
    remis = rn.new('ShaderNodeEmission')
    if is_pump:
        remis.inputs['Color'].default_value = (1.0, 0.1, 0.5, 1.0)
    else:
        remis.inputs['Color'].default_value = (0.0, 1.0, 0.6, 1.0)
    remis.inputs['Strength'].default_value = 12.0
    ring_mat.node_tree.links.new(remis.outputs['Emission'], rout.inputs['Surface'])
    ring.data.materials.append(ring_mat)
    
    return well

make_well(200, 300, True, "P1")
make_well(700, -200, False, "W1")
make_well(-300, -100, True, "P2")

# ─── Trajectory (glowing curve tube) ─────────────────────────────────────────
traj_points = []
rng2 = np.random.default_rng(7)
tx, tz = -400.0, 350.0
for step in range(40):
    gi = int((tz + EXTENT_Y/2) / EXTENT_Y * (NROW - 1))
    gj = int((tx + EXTENT_X/2) / EXTENT_X * (NCOL - 1))
    gi = min(max(gi, 0), NROW-1)
    gj = min(max(gj, 0), NCOL-1)
    t = (heads[gi, gj] - head_min) / (head_max - head_min)
    ty = Z_BOT + t * (Z_TOP - Z_BOT) + 20.0  # slightly above surface
    traj_points.append((tx, ty, tz))
    tx += rng2.uniform(20, 50)
    tz += rng2.uniform(-40, 40)
    tx = min(tx, EXTENT_X/2 - 50)
    tz = max(min(tz, EXTENT_Y/2 - 50), -EXTENT_Y/2 + 50)

# Create curve
curve_data = bpy.data.curves.new("TrajCurve", 'CURVE')
curve_data.dimensions = '3D'
spline = curve_data.splines.new('NURBS')
spline.points.add(len(traj_points) - 1)
for i, (px, py, pz) in enumerate(traj_points):
    spline.points[i].co = (px, py, pz, 1.0)
spline.use_endpoint_u = True

curve_obj = bpy.data.objects.new("Trajectory", curve_data)
scene.collection.objects.link(curve_obj)

# Bevel for tube
curve_data.bevel_depth = 5.0
curve_data.bevel_resolution = 4

# Emission material for trajectory
traj_mat = bpy.data.materials.new("TrajMat")
traj_mat.use_nodes = True
tn = traj_mat.node_tree.nodes
tn.clear()
tout = tn.new('ShaderNodeOutputMaterial')
temis = tn.new('ShaderNodeEmission')
temis.inputs['Color'].default_value = (0.0, 1.0, 1.0, 1.0)  # cyan
temis.inputs['Strength'].default_value = 15.0
traj_mat.node_tree.links.new(temis.outputs['Emission'], tout.inputs['Surface'])
curve_obj.data.materials.append(traj_mat)

# ─── Start / End markers ─────────────────────────────────────────────────────
def make_marker(pos, color, name):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=18, location=pos)
    marker = bpy.context.active_object
    marker.name = name
    mat = bpy.data.materials.new(f"Marker_{name}")
    mat.use_nodes = True
    mn = mat.node_tree.nodes
    mn.clear()
    out = mn.new('ShaderNodeOutputMaterial')
    em = mn.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = color
    em.inputs['Strength'].default_value = 20.0
    mat.node_tree.links.new(em.outputs['Emission'], out.inputs['Surface'])
    marker.data.materials.append(mat)
    return marker

make_marker(traj_points[0], (0.0, 1.0, 0.0, 1.0), "Start")
make_marker(traj_points[-1], (1.0, 0.0, 0.0, 1.0), "End")

# ─── Grid helper (subtle) ────────────────────────────────────────────────────
bpy.ops.mesh.primitive_grid_add(x_subdivisions=NCOL, y_subdivisions=NROW,
                                 size=1, location=(0, Z_BOT - 5, 0))
grid = bpy.context.active_object
grid.name = "GridHelper"
grid.scale = (EXTENT_X, 1, EXTENT_Y)
grid_mat = bpy.data.materials.new("GridMat")
grid_mat.use_nodes = True
gn = grid_mat.node_tree.nodes
gn.clear()
gout = gn.new('ShaderNodeOutputMaterial')
gemis = gn.new('ShaderNodeEmission')
gemis.inputs['Color'].default_value = (0.0, 0.3, 0.4, 1.0)
gemis.inputs['Strength'].default_value = 0.3
grid_mat.node_tree.links.new(gemis.outputs['Emission'], gout.inputs['Surface'])
grid.data.materials.append(grid_mat)

# ─── Lighting ────────────────────────────────────────────────────────────────
# Key light (warm)
bpy.ops.object.light_add(type='POINT', location=(300, 200, 200))
key_light = bpy.context.active_object
key_light.data.energy = 50000
key_light.data.color = (1.0, 0.9, 0.8)
key_light.data.shadow_soft_size = 50

# Fill light (cool cyan)
bpy.ops.object.light_add(type='POINT', location=(-400, 100, -300))
fill_light = bpy.context.active_object
fill_light.data.energy = 20000
fill_light.data.color = (0.2, 0.6, 1.0)

# Rim light
bpy.ops.object.light_add(type='POINT', location=(0, 300, -500))
rim_light = bpy.context.active_object
rim_light.data.energy = 15000
rim_light.data.color = (0.5, 0.0, 1.0)

# ─── World (dark background) ─────────────────────────────────────────────────
world = scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wn.clear()
world_out = wn.new('ShaderNodeOutputWorld')
world_bg = wn.new('ShaderNodeBackground')
world_bg.inputs['Color'].default_value = (0.01, 0.01, 0.02, 1.0)
world_bg.inputs['Strength'].default_value = 1.0
world.node_tree.links.new(world_bg.outputs['Background'], world_out.inputs['Surface'])

# ─── Camera + Orbit ──────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, 150, 800))
camera = bpy.context.active_object
camera.name = "Camera"
camera.data.lens = 45
scene.camera = camera

# Camera target (empty at center)
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 25, 0))
target = bpy.context.active_object
target.name = "CamTarget"

# Track-to constraint
cam_constraint = camera.constraints.new('TRACK_TO')
cam_constraint.target = target
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

# Orbit keyframes (full 360° over TOTAL_FRAMES)
if camera.animation_data:
    camera.animation_data.action = None
else:
    camera.animation_data_create()

radius = 700.0
height_offset = 150.0
for frame in range(1, TOTAL_FRAMES + 1):
    angle = 2 * math.pi * (frame - 1) / TOTAL_FRAMES
    cx = radius * math.cos(angle)
    cz = radius * math.sin(angle)
    cy = height_offset + 50 * math.sin(angle * 0.5)  # slight vertical bob
    camera.location = (cx, cy, cz)
    camera.keyframe_insert('location', frame=frame)

# ─── Render settings (PNG frames → ffmpeg encode) ────────────────────────────
FRAMES_DIR = "/a0/usr/projects/flopyarcade_roadmapworker2/BRANCH_ANALYSIS/demos/frames"
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = FRAMES_DIR + "/frame_"
import os
os.makedirs(FRAMES_DIR, exist_ok=True)

# EEVEE settings (Blender 5.0 compatible)
try:
    scene.eevee.taa_render_samples = 32
    scene.eevee.taa_samples = 16
except Exception:
    pass
try:
    scene.eevee.use_bloom = True
    scene.eevee.bloom_intensity = 0.3
    scene.eevee.bloom_threshold = 1.0
    scene.eevee.bloom_radius = 4.0
except Exception:
    pass

# ─── Render ──────────────────────────────────────────────────────────────────
print(f"Rendering {TOTAL_FRAMES} frames at {RES_X}x{RES_Y} → {OUTPUT_PATH}")
print(f"FPS={FPS}, Duration={TOTAL_FRAMES/FPS:.1f}s")

bpy.ops.render.render(animation=True)

print(f"\n✅ Done! Output: {OUTPUT_PATH}")
