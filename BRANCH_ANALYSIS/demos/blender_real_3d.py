#!/usr/bin/env python3
"""FloPyArcade 3D visualization — real simulation data, matplotlib-matching colors.

Coordinate system (from env_base.py):
- extentX = extentY = 100.0, minX = minY = 0.0
- nRow=20, nCol=25
- dRow = extentX/nCol = 4.0 (cell width in x)
- dCol = extentY/nRow = 5.0 (cell depth in y)
- Well at (71.96, 55.92) — env internal coords
- In plot: well_y_plot = extentY - well_y_env = 100 - 55.92 = 44.08
"""
import bpy
import numpy as np
import os
import json
from mathutils import Vector as BVec

# ─── Load real simulation data ───────────────────────────────────────────────
HEADS = np.load('/tmp/sim_heads.npy')          # shape (10, 20, 25)
CMAP = np.load('/tmp/terrain_cmap_3d.npy')     # shape (256, 3) RGB 0-1
WELL_PATH = np.load('/tmp/sim_well_path.npy')  # shape (10, 3)
with open('/tmp/sim_info.json') as f:
    INFO = json.load(f)

NROW, NCOL = 20, 25
NFRAMES_REAL = HEADS.shape[0]

# Environment coordinates (from env_base.py)
EXTENT = 100.0
WELL_X = WELL_PATH[0][0]    # 71.96
WELL_Y = WELL_PATH[0][1]    # 55.92
WELL_Q = INFO['initial_Q']  # -1766

# Cell dimensions
D_ROW = EXTENT / NCOL  # 4.0 (x-direction cell width)
D_COL = EXTENT / NROW  # 5.0 (y-direction cell depth)

# Head range (use global for consistent animation)
HMIN = float(np.nanmin(HEADS))
HMAX = float(np.nanmax(HEADS))
HRANGE = max(HMAX - HMIN, 1e-9)

# Z scaling: map head range to 0..5 Blender units
Z_SCALE = 5.0

# Animation
TOTAL_FRAMES = 120
FPS = 30

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames_real')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Clear scene ─────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
              bpy.data.cameras, bpy.data.curves, bpy.data.worlds):
    for item in list(block):
        if item.users == 0:
            block.remove(item)

# ─── Scene settings ──────────────────────────────────────────────────────────
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = FPS

try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except Exception:
    scene.render.engine = 'BLENDER_EEVEE'

# ─── World: white background ─────────────────────────────────────────────────
world = bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
bg = nodes.new('ShaderNodeBackground')
bg.inputs[0].default_value = (0.95, 0.95, 0.97, 1.0)
bg.inputs[1].default_value = 1.0
out_node = nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs[0], out_node.inputs['Surface'])

# ─── Lighting: bright, even ──────────────────────────────────────────────────
key_data = bpy.data.lights.new('Key', 'SUN')
key_data.energy = 4.0
key_data.color = (1.0, 0.99, 0.97)
key_obj = bpy.data.objects.new('KeyLight', key_data)
scene.collection.objects.link(key_obj)
key_obj.rotation_euler = (0.7, 0.1, 0.4)

fill_data = bpy.data.lights.new('Fill', 'SUN')
fill_data.energy = 2.0
fill_data.color = (0.97, 0.98, 1.0)
fill_obj = bpy.data.objects.new('FillLight', fill_data)
scene.collection.objects.link(fill_obj)
fill_obj.rotation_euler = (0.5, -0.5, -0.7)

# ─── Camera ──────────────────────────────────────────────────────────────────
cam_data = bpy.data.cameras.new('Cam')
cam_data.lens = 45
cam_obj = bpy.data.objects.new('Camera', cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

cam_center = BVec((EXTENT/2, EXTENT/2, 2.5))
cam_radius = EXTENT * 1.1
cam_height = EXTENT * 0.55

# ─── Terrain mesh (cell-corner grid: (NCOL+1) x (NROW+1) verts) ─────────────
# In Blender: X = 0..100 (east), Y = 0..100 (north, flipped from env)
# Env row r → y_plot = EXTENT - (r+0.5)*D_COL  (cell center)
# We use cell-center grid for the mesh (20x25 cells → 21x26 corners)

ncx = NCOL + 1  # 26
ncy = NROW + 1  # 21

verts = []
for i in range(ncy):
    for j in range(ncx):
        x = j * D_ROW
        y = i * D_COL
        verts.append((x, y, 0.0))

faces = []
for i in range(NROW):
    for j in range(NCOL):
        v0 = i * ncx + j
        v1 = v0 + 1
        v2 = v0 + ncx + 1
        v3 = v0 + ncx
        faces.append((v0, v1, v2, v3))

mesh = bpy.data.meshes.new('Terrain')
mesh.from_pydata(verts, [], faces)
mesh.update()

terrain_obj = bpy.data.objects.new('Terrain', mesh)
scene.collection.objects.link(terrain_obj)

# ─── Terrain material: vertex colors ─────────────────────────────────────────
terrain_mat = bpy.data.materials.new('TerrainMat')
terrain_mat.use_nodes = True
tnodes = terrain_mat.node_tree.nodes
tnodes.clear()
attr = tnodes.new('ShaderNodeAttribute')
attr.attribute_name = 'Col'
principled = tnodes.new('ShaderNodeBsdfPrincipled')
principled.inputs['Roughness'].default_value = 0.9
principled.inputs['Metallic'].default_value = 0.0
tout = tnodes.new('ShaderNodeOutputMaterial')
terrain_mat.node_tree.links.new(attr.outputs['Color'], principled.inputs['Base Color'])
terrain_mat.node_tree.links.new(principled.outputs['BSDF'], tout.inputs['Surface'])
terrain_obj.data.materials.append(terrain_mat)

# Color attribute (per-vertex = POINT domain)
mesh.color_attributes.new(name='Col', type='FLOAT_COLOR', domain='POINT')

# ─── Well: red cylinder (pumping) ───────────────────────────────────────────
# Place at env coords directly in Blender space
well_sx = WELL_X
well_sy = WELL_Y

bpy.ops.mesh.primitive_cylinder_add(radius=2.0, depth=6.0, location=(well_sx, well_sy, 3))
well_cyl = bpy.context.active_object
well_cyl.name = 'PumpWell'

well_mat = bpy.data.materials.new('WellRed')
well_mat.use_nodes = True
wn = well_mat.node_tree.nodes
wn.clear()
wbsdf = wn.new('ShaderNodeBsdfPrincipled')
wbsdf.inputs['Base Color'].default_value = (0.85, 0.0, 0.0, 1.0)
wbsdf.inputs['Roughness'].default_value = 0.2
wout = wn.new('ShaderNodeOutputMaterial')
well_mat.node_tree.links.new(wbsdf.outputs['BSDF'], wout.inputs['Surface'])
well_cyl.data.materials.append(well_mat)

# Safety zone ring
bpy.ops.mesh.primitive_torus_add(major_radius=8.0, minor_radius=0.5,
                                  location=(well_sx, well_sy, 0.5))
well_ring = bpy.context.active_object
well_ring.name = 'SafetyZone'
well_ring.data.materials.append(well_mat)

# ─── Text labels ─────────────────────────────────────────────────────────────
def make_text(text, location, size=3.0, color=(0.1, 0.1, 0.1, 1.0)):
    bpy.ops.object.text_add(location=location)
    txt = bpy.context.active_object
    txt.data.body = text
    txt.data.size = size
    txt.data.align_x = 'CENTER'
    tmat = bpy.data.materials.new(f'Txt_{id(text) % 10000}')
    tmat.use_nodes = True
    tn = tmat.node_tree.nodes
    tn.clear()
    tbsdf = tn.new('ShaderNodeBsdfPrincipled')
    if len(color) == 3:
        color = (color[0], color[1], color[2], 1.0)
    tbsdf.inputs['Base Color'].default_value = color
    tbsdf.inputs['Roughness'].default_value = 1.0
    tout = tn.new('ShaderNodeOutputMaterial')
    tmat.node_tree.links.new(tbsdf.outputs['BSDF'], tout.inputs['Surface'])
    txt.data.materials.append(tmat)
    return txt

make_text('FloPyArcade - 2s-d Real MODFLOW Simulation', (EXTENT/2, EXTENT+8, 18), size=4.0)
make_text(f'PUMP WELL  Q={WELL_Q:.0f} m\u00b3/d', (well_sx, well_sy-12, 8), size=2.5,
          color=(0.8, 0.0, 0.0, 1.0))
make_text(f'HEAD RANGE: {HMIN:.1f} - {HMAX:.1f} m', (EXTENT/2, -5, 12), size=2.5,
          color=(0.3, 0.3, 0.3, 1.0))
make_text(f'Grid: {NROW} rows x {NCOL} cols  |  Cell: {D_ROW:.1f} x {D_COL:.1f} m',
          (EXTENT/2, -9, 12), size=2.0, color=(0.5, 0.5, 0.5, 1.0))

# ─── Grid lines (subtle) ─────────────────────────────────────────────────────
grid_mat = bpy.data.materials.new('GridLine')
grid_mat.use_nodes = True
gn = grid_mat.node_tree.nodes
gn.clear()
gbsdf = gn.new('ShaderNodeBsdfPrincipled')
gbsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
gbsdf.inputs['Roughness'].default_value = 1.0
gout = gn.new('ShaderNodeOutputMaterial')
grid_mat.node_tree.links.new(gbsdf.outputs['BSDF'], gout.inputs['Surface'])

grid_curve = bpy.data.curves.new('GridCurve', 'CURVE')
grid_curve.dimensions = '3D'
grid_curve.bevel_depth = 0.08

# Lines along X (for each row boundary)
for i in range(NROW + 1):
    y = i * D_COL
    spline = grid_curve.splines.new('POLY')
    spline.points.add(NCOL)
    for j in range(NCOL + 1):
        spline.points[j].co = (j * D_ROW, y, 0.0, 1.0)

# Lines along Y (for each column boundary)
for j in range(NCOL + 1):
    x = j * D_ROW
    spline = grid_curve.splines.new('POLY')
    spline.points.add(NROW)
    for i in range(NROW + 1):
        spline.points[i].co = (x, i * D_COL, 0.0, 1.0)

grid_obj = bpy.data.objects.new('GridOverlay', grid_curve)
scene.collection.objects.link(grid_obj)
grid_obj.data.materials.append(grid_mat)

# ─── Helper: get head at grid corner (i,j) via bilinear from cell centers ───
def head_at_corner(heads, i, j):
    """Bilinear interpolation at corner (i,j) from (NROW,NCOL) cell centers."""
    i0 = max(0, min(i - 1, NROW - 1))
    i1 = max(0, min(i, NROW - 1))
    j0 = max(0, min(j - 1, NCOL - 1))
    j1 = max(0, min(j, NCOL - 1))
    fi = 0.0 if i <= 0 else (1.0 if i >= NROW else (i - 0.5))
    fj = 0.0 if j <= 0 else (1.0 if j >= NCOL else (j - 0.5))
    return (heads[i0, j0] * (1-fi) * (1-fj) +
            heads[i0, j1] * (1-fi) * fj +
            heads[i1, j0] * fi * (1-fj) +
            heads[i1, j1] * fi * fj)

# ─── Frame handlers ──────────────────────────────────────────────────────────
def head_at_frame(frame_num):
    """Linearly interpolate between real head frames."""
    progress = (frame_num - 1) / max(TOTAL_FRAMES - 1, 1)
    real_idx = progress * (NFRAMES_REAL - 1)
    i0 = int(real_idx)
    i1 = min(i0 + 1, NFRAMES_REAL - 1)
    frac = real_idx - i0
    return HEADS[i0] * (1 - frac) + HEADS[i1] * frac

def update_scene(scene, frame):
    """Update terrain Z, colors, well, and grid for current frame."""
    frame = int(bpy.context.scene.frame_current)
    heads = head_at_frame(frame)

    # Update terrain vertices
    mesh_data = terrain_obj.data
    col_attr = mesh_data.color_attributes['Col']

    for vi in range(len(mesh_data.vertices)):
        i = vi // ncx
        j = vi % ncx
        h_val = head_at_corner(heads, i, j)
        z = ((h_val - HMIN) / HRANGE) * Z_SCALE
        mesh_data.vertices[vi].co.z = z
        # Color
        norm = (h_val - HMIN) / HRANGE
        idx = int(max(0, min(255, norm * 255)))
        r, g, b = CMAP[idx]
        col_attr.data[vi].color = (r, g, b, 1.0)

    mesh_data.update()

    # Update well position (sit on terrain)
    well_i = int(WELL_Y / D_COL)  # row index
    well_j = int(WELL_X / D_ROW)  # col index
    well_i = max(0, min(well_i, NROW - 1))
    well_j = max(0, min(well_j, NCOL - 1))
    well_h = heads[well_i, well_j]
    well_z = ((well_h - HMIN) / HRANGE) * Z_SCALE
    well_cyl.location.z = well_z + 3.0
    well_ring.location.z = well_z + 0.5

    # Update grid lines Z
    for spline in grid_obj.data.splines:
        for pt in spline.points:
            x, y, _, _ = pt.co
            j = int(round(x / D_ROW))
            i = int(round(y / D_COL))
            i = max(0, min(i, NROW))
            j = max(0, min(j, NCOL))
            h_val = head_at_corner(heads, i, j)
            z = ((h_val - HMIN) / HRANGE) * Z_SCALE + 0.05
            pt.co = (x, y, z, 1.0)

# Register
bpy.app.handlers.frame_change_pre.append(update_scene)

# ─── Camera orbit (full 360°) ────────────────────────────────────────────────
for frame in range(1, TOTAL_FRAMES + 1):
    angle = 2 * np.pi * (frame - 1) / TOTAL_FRAMES
    cx = cam_center.x + cam_radius * np.cos(angle)
    cy = cam_center.y + cam_radius * np.sin(angle)
    cz = cam_height
    cam_obj.location = (cx, cy, cz)
    direction = cam_center - BVec((cx, cy, cz))
    rot = direction.to_track_quat('-Z', 'Y').to_euler()
    cam_obj.rotation_euler = rot
    cam_obj.keyframe_insert('location', frame=frame)
    cam_obj.keyframe_insert('rotation_euler', frame=frame)

# ─── Render ──────────────────────────────────────────────────────────────────
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = OUTPUT_DIR + "/real_"

try:
    scene.eevee.taa_render_samples = 64
    scene.eevee.taa_samples = 32
except Exception:
    pass

print(f'=== FloPyArcade Real 3D Render ===')
print(f'Frames: {TOTAL_FRAMES} @ {FPS}fps ({TOTAL_FRAMES/FPS:.1f}s)')
print(f'Grid: {NROW}x{NCOL}, extent: {EXTENT}x{EXTENT} m')
print(f'Well: ({WELL_X:.1f}, {WELL_Y:.1f}), Q={WELL_Q:.0f} m³/d')
print(f'Head: {HMIN:.2f} - {HMAX:.2f} m')
print(f'Output: {OUTPUT_DIR}/')

bpy.ops.render.render(animation=True)

print(f'\nDONE! {TOTAL_FRAMES} frames rendered.')
print(f'Encode: ffmpeg -framerate {FPS} -i {OUTPUT_DIR}/real_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 flopyarcade_real_3d.mp4')
