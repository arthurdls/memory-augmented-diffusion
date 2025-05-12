#!/usr/bin/env python3
import os, sys
import numpy as np
import open3d as o3d

# allow imports
repo = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo, "src"))
sys.path.insert(0, os.path.join(repo, "scripts"))

from memory.voxel_memory import VoxelMemory
from gen_scenes import create_block_world

def look_at(eye, target, up=np.array([0,1,0], dtype=np.float32)):
    f = (target - eye); f /= np.linalg.norm(f)
    r = np.cross(up, f);    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=1)

def visualize_minecraft():
    # 1) build & integrate
    mesh = create_block_world(16, 30)
    center = np.array([8,8,8], dtype=np.float32)
    radius = 20.0
    poses = []
    for ang in [0,90,180,270]:
        rad = np.deg2rad(ang)
        eye = center + np.array([radius*np.cos(rad),0,radius*np.sin(rad)], dtype=np.float32)
        R = look_at(eye, center)
        p = np.eye(4, dtype=np.float32)
        p[:3,:3], p[:3,3] = R, eye
        poses.append(p)

    mem = VoxelMemory(dims=(128,128,128), voxel_size=0.25)
    class Intrinsic:
        def __init__(i, fx, fy, cx, cy):
            i.fx, i.fy, i.cx, i.cy = fx, fy, cx, cy
    intr = Intrinsic(256, 256, 128, 128)

    # CPU–raycast like phase1
    import open3d.core as o3c, open3d.t.geometry as tgeom
    def cpu_render(mesh, pose, fx, fy, cx, cy, W=256, H=256):
        tmesh = tgeom.TriangleMesh.from_legacy(mesh)
        scene = tgeom.RaycastingScene(); scene.add_triangles(tmesh)
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        dirs = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u)], axis=-1).reshape(-1,3).astype(np.float32)
        R, t = pose[:3,:3], pose[:3,3]
        ro = np.tile(t, (dirs.shape[0],1)); rd = (R @ dirs.T).T
        rd /= np.linalg.norm(rd, axis=1, keepdims=True)
        rays = o3c.Tensor(np.hstack([ro, rd]).astype(np.float32), dtype=o3c.Dtype.Float32)
        depth = scene.cast_rays(rays)["t_hit"].numpy().reshape(H,W)
        depth[~np.isfinite(depth)] = 0
        gray = np.zeros_like(depth, dtype=np.uint8)
        valid = depth>0
        if np.any(valid):
            gray[valid] = np.clip(depth[valid]/depth[valid].max()*255,0,255).astype(np.uint8)
        return np.stack([gray]*3,axis=-1), depth

    for p in poses:
        rgb, depth = cpu_render(mesh, p, 256,256,128,128)
        mem.update(p, rgb, depth, intr)

    # 2) make one cube per occupied voxel
    idxs = np.argwhere(mem.counts>0)
    vs   = mem.voxel_size
    cols = mem.color_accum[idxs[:,0], idxs[:,1], idxs[:,2]]
    cnts = mem.counts[idxs[:,0], idxs[:,1], idxs[:,2]].reshape(-1,1)
    avg  = (cols/cnts).clip(0,1)

    cubes = []
    for (i,j,k), color in zip(idxs, avg):
        box = o3d.geometry.TriangleMesh.create_box(vs,vs,vs)
        box.translate(np.array([i,j,k],float)*vs)
        box.compute_vertex_normals()
        # assign that color to every vertex
        vc = np.tile(color, (np.asarray(box.vertices).shape[0],1))
        box.vertex_colors = o3d.utility.Vector3dVector(vc)
        cubes.append(box)

    # 3) visualize
    o3d.visualization.draw_geometries(cubes,
        window_name="Minecraft-Style Voxels",
        width=800, height=600)

if __name__=="__main__":
    visualize_minecraft()