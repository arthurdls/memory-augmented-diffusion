#!/usr/bin/env python3
import os, sys
import numpy as np
import open3d as o3d
from open3d.utility import Vector3dVector

# point to your code
repo = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo,"src"))
sys.path.insert(0, os.path.join(repo,"scripts"))

from memory.voxel_memory import VoxelMemory
from gen_scenes import create_block_world

# same look‐at helper from run_phase1.py
def look_at(eye, target, up=np.array([0,1,0],dtype=np.float32)):
    f = (target-eye); f /= np.linalg.norm(f)
    r = np.cross(up, f);    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r,u,f], axis=1)

# same cpu_render
import open3d.core as o3c, open3d.t.geometry as tgeom
def cpu_render(mesh, pose, fx, fy, cx, cy, W=256, H=256):
    tmesh  = tgeom.TriangleMesh.from_legacy(mesh)
    scene  = tgeom.RaycastingScene(); scene.add_triangles(tmesh)

    u, v   = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs_cam = np.stack([
        (u - cx) / fx,
        (v - cy) / fy,
        np.ones_like(u)
    ], axis=-1).reshape(-1,3).astype(np.float32)

    R, t   = pose[:3,:3], pose[:3,3]
    rays_o = np.tile(t, (dirs_cam.shape[0],1))
    rays_d = (R @ dirs_cam.T).T

    # ←—— **HERE** normalize the directions
    norms  = np.linalg.norm(rays_d, axis=1, keepdims=True)
    rays_d = rays_d / norms

    rays   = o3c.Tensor(
        np.hstack([rays_o, rays_d]).astype(np.float32),
        dtype=o3c.Dtype.Float32
    )
    ans    = scene.cast_rays(rays)
    depth  = ans["t_hit"].numpy().reshape(H, W)
    depth[~np.isfinite(depth)] = 0

    # now depth really is in meters, so the rest stays the same
    valid = depth>0
    maxd  = depth[valid].max() if np.any(valid) else 1.0
    gray  = np.zeros_like(depth, dtype=np.uint8)
    gray[valid] = np.clip(depth[valid]/maxd*255, 0, 255).astype(np.uint8)
    return np.stack([gray]*3,axis=-1), depth


def visualize_multi_view():
    mesh = create_block_world(16,30)
    center = np.array([8,8,8],dtype=np.float32)
    radius = 20.0
    poses = []
    for ang in [0,90,180,270]:
        rad = np.deg2rad(ang)
        eye = center + np.array([radius*np.cos(rad),0,radius*np.sin(rad)])
        R = look_at(eye, center)
        pose = np.eye(4,dtype=np.float32)
        pose[:3,:3] = R; pose[:3,3] = eye
        poses.append(pose)

    mem = VoxelMemory(dims=(128,128,128), voxel_size=0.25)
    class Intrinsic:
        def __init__(i,fx,fy,cx,cy): i.fx,i.fy,i.cx,i.cy = fx,fy,cx,cy
    intr = Intrinsic(256,256,128,128)

    for p in poses:
        rgb,depth = cpu_render(mesh,p,256,256,128,128)
        mem.update(p, rgb, depth, intr)

    idxs = np.argwhere(mem.counts>0)
    pts  = (idxs+0.5)*mem.voxel_size
    pcd  = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd],
        window_name="Multi‐View Voxels", width=800, height=600)

if __name__=="__main__":
    visualize_multi_view()
