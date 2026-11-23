#!/usr/bin/env python3
"""FloPyArcade Full-Episode 3D Animation — Blender headless render.

Simulates a complete gameplay episode:
- Head field evolves over time (water flows from injection to extraction)
- Agent moves step-by-step along a path
- Trajectory visible as glowing tube
- Wells pulse with activity
- Camera does 1.5 slow orbits

Render:
    blender --background --python blender_full_episode.py
Then encode:
    ffmpeg -framerate 30 -i frames_ep/ep_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 episode.mp4
"""

import math
import os
import numpy as np
import bpy

# ─── Configuration ───────────────────────────────────────────────────────────
NROW, NCOL = 12, 16
EXTENT_X, EXTENT_Y = 1000.0, 800.0
Z_TOP, Z_BOT = 60.0, 0.0
SEED = 42
FPS = 30
TOTAL_FRAMES = 300  # 10 seconds
RES_X, RES_Y = 1280, 720
ORBITS = 1.5
OUTPUT_DIR = "/a0/usr/projects/flopyarcade_roadmapworker2/BRANCH_ANALYSIS/demos/frames_ep"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ─── Well positions (world coords) ──────────────────────────────────────────
WELL_INJ = (700.0, -250.0)
WELL_EX1 = (-350.0, 200.0)
WELL_EX2 = (100.0, 350.0)

# ─── Agent path (world coords) ──────────────────────────────────────────────
rng_path = np.random.default_rng(123)
agent_steps = 40
agent_path = []
ax, az = -400.0, 300.0
for i in range(agent_steps):
    agent_path.append((ax, az))
    ax += rng_path.uniform(15, 40)
    az += rng_path.uniform(-30, 30)
    ax = min(ax, EXTENT_X/2 - 60)
    az = max(min(az, EXTENT_Y/2 - 60), -EXTENT_Y/2 + 60)

# ─── Head field base (static parts) ─────────────────────────────────────────
x = np.linspace(0, 1, NCOL)
y = np.linspace(0, 1, NROW)
X, Y = np.meshgrid(x, y)
base_head = 20.0 + 10.0 * X + 5.0 * np.sin(3 * Y)
rng_noise = np.random.default_rng(SEED)
noise_field = rng_noise.normal(0, 1, (NROW, NCOL))

# Precompute well distance fields
well_dists = []
for wx, wz in [WELL_INJ, WELL_EX1, WELL_EX2]:
    nx = (wx + EXTENT_X/2) / EXTENT_X
    nz = (wz + EXTENT_Y/2) / EXTENT_Y
    dx = X - nx
    dy = Y - nz
    dist = np.sqrt(dx*dx + dy*dy) + 0.05
    well_dists.append(np.exp(-dist*dist*20))

def head_at_frame(frame):
    """Compute head field at given frame."""
    progress = frame / max(TOTAL_FRAMES - 1, 1)
    h = base_head + 2.0 * progress * X
    h += 4.0 * np.sin(4 * Y + 2.0 * progress) * (0.5 + 0.5 * progress)
    well_strength = 8.0 * min(progress * 2.0, 1.0)
    h += well_strength * well_dists[0]  # injection (positive)
    h -= well_strength * well_dists[1]  # extraction 1
    h -= well_strength * well_dists[2]  # extraction 2
    h += 1.5 * noise_field * (0.3 + 0.7 * progress)
    return h

# ─── Colormap ────────────────────────────────────────────────────────────────
def terrain_color(t):
    t = max(0.0, min(1.0, t))
    stops = [
        (0.0,  0.01, 0.05, 0.25),
        (0.15, 0.0,  0.2,  0.5),
        (0.3,  0.0,  0.5,  0.5),
        (0.45, 0.1,  0.7,  0.4),
        (0.6,  0.4,  0.85, 0.3),
        (0.75, 0.8,  0.85, 0.2),
        (0.88, 0.95, 0.6,  0.2),
        (1.0,  1.0,  1.0,  1.0),
    ]
    for i in range(len(stops) - 1):
        t0, r0, g0, b0 = stops[i]
        t1, r1, g1, b1 = stops[i+1]
        if t <= t1:
            f = (t - t0) / max(t1 - t0, 1e-9)
            return (r0+(r1-r0)*f, g0+(g1-g0)*f, b0+(b1-b0)*f)
    return stops[-1][1:]

# ─── Build terrain mesh ──────────────────────────────────────────────────────
NVERT = NROW * NCOL
vertices_base = []
for i in range(NROW):
    for j in range(NCOL):
        px = -EXTENT_X/2 + EXTENT_X * j / (NCOL - 1)
        pz = -EXTENT_Y/2 + EXTENT_Y * i / (NROW - 1)
        vertices_base.append((px, pz))

edges = []
faces = []
for i in range(NROW - 1):
    for j in range(NCOL - 1):
        a = i * NCOL + j
        b = a + 1
        c = a + NCOL
        d = c + 1
        faces.append((a, b, d, c))
        edges.extend([(a,b),(a,c),(b,d),(c,d)])

mesh = bpy.data.meshes.new("Terrain")
mesh.from_pydata([(px, Z_BOT, pz) for px, pz in vertices_base], edges, faces)
mesh.validate()
mesh.update()

color_attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")

terrain_obj = bpy.data.objects.new("Terrain", mesh)
scene.collection.objects.link(terrain_obj)

terrain_mat = bpy.data.materials.new("TerrainMat")
terrain_mat.use_nodes = True
tn = terrain_mat.node_tree.nodes
tn.clear()
out_node = tn.new('ShaderNodeOutputMaterial')
bsdf = tn.new('ShaderNodeBsdfPrincipled')
bsdf.inputs['Roughness'].default_value = 0.6
bsdf.inputs['Metallic'].default_value = 0.05
vc = tn.new('ShaderNodeVertexColor')
vc.layer_name = "Col"
tl = terrain_mat.node_tree.links
tl.new(vc.outputs['Color'], bsdf.inputs['Base Color'])
tl.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
terrain_obj.data.materials.append(terrain_mat)

# ─── Frame handler: update terrain at render time ───────────────────────────
def update_terrain(scene, depsgraph):
    frame = int(bpy.context.scene.frame_current)
    heads = head_at_frame(frame)
    hmin, hmax = float(heads.min()), float(heads.max())
    hrange = max(hmax - hmin, 1e-9)
    for idx in range(NVERT):
        gi = idx // NCOL
        gj = idx % NCOL
        t = (float(heads[gi, gj]) - hmin) / hrange
        py = Z_BOT + t * (Z_TOP - Z_BOT)
        mesh.vertices[idx].co = (vertices_base[idx][0], py, vertices_base[idx][1])
        c = terrain_color(t)
        color_attr.data[idx].color = (c[0], c[1], c[2], 1.0)
    mesh.update()

bpy.app.handlers.frame_change_post.append(update_terrain)

# ─── Wells ──────────────────────────────────────────────────────────────────
def make_well(px, pz, is_pump, name):
    height = 140.0
    radius = 18.0
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=height,
                                        location=(px, Z_BOT + height/2, pz))
    well = bpy.context.active_object
    well.name = f"Well_{name}"
    mat = bpy.data.materials.new(f"Mat_{name}")
    mat.use_nodes = True
    mn = mat.node_tree.nodes
    mn.clear()
    out = mn.new('ShaderNodeOutputMaterial')
    emis = mn.new('ShaderNodeEmission')
    base_color = (1.0, 0.1, 0.5) if is_pump else (0.0, 1.0, 0.5)
    emis.inputs['Color'].default_value = (*base_color, 1.0)
    emis.inputs['Strength'].default_value = 6.0
    mat.node_tree.links.new(emis.outputs['Emission'], out.inputs['Surface'])
    well.data.materials.append(mat)
    # Glow ring
    bpy.ops.mesh.primitive_torus_add(major_radius=radius*2.5, minor_radius=4,
                                     location=(px, Z_BOT, pz))
    ring = bpy.context.active_object
    ring.name = f"Ring_{name}"
    ring_mat = bpy.data.materials.new(f"RingMat_{name}")
    ring_mat.use_nodes = True
    rn = ring_mat.node_tree.nodes
    rn.clear()
    rout = rn.new('ShaderNodeOutputMaterial')
    remis = rn.new('ShaderNodeEmission')
    remis.inputs['Color'].default_value = (*base_color, 1.0)
    remis.inputs['Strength'].default_value = 10.0
    ring_mat.node_tree.links.new(remis.outputs['Emission'], rout.inputs['Surface'])
    ring.data.materials.append(ring_mat)
    return well

make_well(*WELL_INJ, False, "INJ")
make_well(*WELL_EX1, True, "EX1")
make_well(*WELL_EX2, True, "EX2")

# ─── Agent (glowing sphere) ─────────────────────────────────────────────────
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=22,
    location=(agent_path[0][0], Z_TOP + 30, agent_path[0][1]))
agent = bpy.context.active_object
agent.name = "Agent"
agent_mat = bpy.data.materials.new("AgentMat")
agent_mat.use_nodes = True
an = agent_mat.node_tree.nodes
an.clear()
aout = an.new('ShaderNodeOutputMaterial')
aemis = an.new('ShaderNodeEmission')
aemis.inputs['Color'].default_value = (1.0, 0.9, 0.0, 1.0)
aemis.inputs['Strength'].default_value = 20.0
agent_mat.node_tree.links.new(aemis.outputs['Emission'], aout.inputs['Surface'])
agent.data.materials.append(agent_mat)

# Agent position via frame handler
AGENT_START = 15
AGENT_END = TOTAL_FRAMES - 15
def update_agent(scene, depsgraph):
    frame = int(bpy.context.scene.frame_current)
    if frame < AGENT_START or frame > AGENT_END:
        return
    progress = (frame - AGENT_START) / (AGENT_END - AGENT_START)
    step_float = progress * (len(agent_path) - 1)
    step_idx = min(int(step_float), len(agent_path) - 2)
    frac = step_float - step_idx
    px = agent_path[step_idx][0] + frac * (agent_path[step_idx+1][0] - agent_path[step_idx][0])
    pz = agent_path[step_idx][1] + frac * (agent_path[step_idx+1][1] - agent_path[step_idx][1])
    # Approximate y from base head (fast lookup)
    gi = min(max(int((pz + EXTENT_Y/2) / EXTENT_Y * (NROW-1)), 0), NROW-1)
    gj = min(max(int((px + EXTENT_X/2) / EXTENT_X * (NCOL-1)), 0), NCOL-1)
    h_mid = head_at_frame(TOTAL_FRAMES // 2)
    hmin, hmax = float(h_mid.min()), float(h_mid.max())
    t = (float(h_mid[gi,gj]) - hmin) / max(hmax-hmin, 1e-9)
    py = Z_BOT + t * (Z_TOP - Z_BOT) + 35.0
    agent.location = (px, py, pz)

bpy.app.handlers.frame_change_post.append(update_agent)

# ─── Trajectory curve ───────────────────────────────────────────────────────
curve_data = bpy.data.curves.new("TrajCurve", 'CURVE')
curve_data.dimensions = '3D'
spline = curve_data.splines.new('NURBS')
h_mid = head_at_frame(TOTAL_FRAMES // 2)
mhmin, mhmax = float(h_mid.min()), float(h_mid.max())
traj_3d = []
for px, pz in agent_path:
    gi = min(max(int((pz + EXTENT_Y/2) / EXTENT_Y * (NROW-1)), 0), NROW-1)
    gj = min(max(int((px + EXTENT_X/2) / EXTENT_X * (NCOL-1)), 0), NCOL-1)
    t = (float(h_mid[gi,gj]) - mhmin) / max(mhmax-mhmin, 1e-9)
    py = Z_BOT + t * (Z_TOP - Z_BOT) + 25.0
    traj_3d.append((px, py, pz))
spline.points.add(len(traj_3d) - 1)
for i, (px, py, pz) in enumerate(traj_3d):
    spline.points[i].co = (px, py, pz, 1.0)
spline.use_endpoint_u = True
curve_data.bevel_depth = 4.0
curve_data.bevel_resolution = 3
curve_obj = bpy.data.objects.new("Trajectory", curve_data)
scene.collection.objects.link(curve_obj)
traj_mat = bpy.data.materials.new("TrajMat")
traj_mat.use_nodes = True
tn2 = traj_mat.node_tree.nodes
tn2.clear()
tout = tn2.new('ShaderNodeOutputMaterial')
temis = tn2.new('ShaderNodeEmission')
temis.inputs['Color'].default_value = (0.0, 1.0, 1.0, 1.0)
temis.inputs['Strength'].default_value = 8.0
traj_mat.node_tree.links.new(temis.outputs['Emission'], tout.inputs['Surface'])
curve_obj.data.materials.append(traj_mat)

# ─── Grid helper ─────────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=16, size=1, location=(0, Z_BOT - 8, 0))
grid = bpy.context.active_object
grid.name = "GridHelper"
grid.scale = (EXTENT_X, 1, EXTENT_Y)
grid_mat = bpy.data.materials.new("GridMat")
grid_mat.use_nodes = True
gn = grid_mat.node_tree.nodes
gn.clear()
gout = gn.new('ShaderNodeOutputMaterial')
gemis = gn.new('ShaderNodeEmission')
gemis.inputs['Color'].default_value = (0.0, 0.2, 0.3, 1.0)
gemis.inputs['Strength'].default_value = 0.4
grid_mat.node_tree.links.new(gemis.outputs['Emission'], gout.inputs['Surface'])
grid.data.materials.append(grid_mat)

# ─── Lighting ────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type='POINT', location=(400, 250, 300))
key_light = bpy.context.active_object
key_light.data.energy = 60000
key_light.data.color = (1.0, 0.92, 0.82)
key_light.data.shadow_soft_size = 60

bpy.ops.object.light_add(type='POINT', location=(-500, 120, -400))
fill_light = bpy.context.active_object
fill_light.data.energy = 25000
fill_light.data.color = (0.15, 0.5, 1.0)

bpy.ops.object.light_add(type='POINT', location=(0, 350, -600))
rim_light = bpy.context.active_object
rim_light.data.energy = 20000
rim_light.data.color = (0.6, 0.0, 1.0)

# ─── World (dark) ────────────────────────────────────────────────────────────
world = scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wn.clear()
world_out = wn.new('ShaderNodeOutputWorld')
world_bg = wn.new('ShaderNodeBackground')
world_bg.inputs['Color'].default_value = (0.005, 0.005, 0.015, 1.0)
world_bg.inputs['Strength'].default_value = 1.0
world.node_tree.links.new(world_bg.outputs['Background'], world_out.inputs['Surface'])

# ─── Camera + Orbit (keyframes — only 300, manageable) ──────────────────────
bpy.ops.object.camera_add(location=(0, 200, 800))
camera = bpy.context.active_object
camera.name = "Camera"
camera.data.lens = 40
scene.camera = camera

bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 30, 0))
target = bpy.context.active_object
target.name = "CamTarget"

cam_constraint = camera.constraints.new('TRACK_TO')
cam_constraint.target = target
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

if camera.animation_data:
    camera.animation_data.action = None
else:
    camera.animation_data_create()

radius = 750.0
base_height = 180.0
for frame in range(1, TOTAL_FRAMES + 1):
    angle = 2 * math.pi * ORBITS * (frame - 1) / TOTAL_FRAMES
    cx = radius * math.cos(angle)
    cz = radius * math.sin(angle)
    cy = base_height + 40.0 * math.sin(angle * 0.5)
    camera.location = (cx, cy, cz)
    camera.keyframe_insert('location', frame=frame)

# ─── Render settings ─────────────────────────────────────────────────────────
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = OUTPUT_DIR + "/ep_"

try:
    scene.eevee.taa_render_samples = 16
    scene.eevee.taa_samples = 8
except Exception:
    pass

# ─── Render ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FloPyArcade Full-Episode Animation")
print(f"  Frames: {TOTAL_FRAMES} ({TOTAL_FRAMES/FPS:.0f}s @ {FPS}fps)")
print(f"  Resolution: {RES_X}x{RES_Y}")
print(f"  Orbits: {ORBITS}")
print(f"  Agent steps: {len(agent_path)}")
print(f"  Wells: 1 INJ + 2 EX")
print(f"  Output: {OUTPUT_DIR}/ep_*.png")
print(f"{'='*60}\n")

bpy.ops.render.render(animation=True)

print(f"\n✅ Render complete! {TOTAL_FRAMES} frames in {OUTPUT_DIR}/")
print(f"Encode: ffmpeg -framerate {FPS} -i {OUTPUT_DIR}/ep_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 episode_full.mp4")
