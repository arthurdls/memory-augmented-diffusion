## run_phase1.py
#!/usr/bin/env python3
import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d.core as o3c, open3d.t.geometry as tgeom

# allow imports
repo = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo, "src"))
sys.path.insert(0, os.path.join(repo, "scripts"))
from memory.voxel_memory import VoxelMemory
from gen_scenes import create_block_world


def look_at(eye, target, up=np.array([0,1,0],dtype=np.float32)):
    f = (target - eye); f /= np.linalg.norm(f)
    r = np.cross(up, f);      r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=1)


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
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)

    rays   = o3c.Tensor(
        np.hstack([rays_o, rays_d]).astype(np.float32),
        dtype=o3c.Dtype.Float32
    )
    ans    = scene.cast_rays(rays)
    depth  = ans["t_hit"].numpy().reshape(H, W)
    depth[~np.isfinite(depth)] = 0

    valid = depth>0
    maxd  = depth[valid].max() if np.any(valid) else 1.0
    gray  = np.zeros_like(depth, dtype=np.uint8)
    gray[valid] = np.clip(depth[valid]/maxd*255, 0, 255).astype(np.uint8)
    return np.stack([gray]*3,axis=-1), depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', nargs=3, type=int, default=[128,128,128],
                        help='voxel grid dimensions Nx Ny Nz')
    parser.add_argument('--voxel_size', type=float, default=0.25,
                        help='size of each voxel in world units')
    parser.add_argument('--max_dist', type=float, default=50.0,
                        help='max ray-march distance')
    parser.add_argument('--views', nargs='+', type=float, default=[0,90,180,270],
                        help='camera view angles (deg)')
    parser.add_argument('--outdir', type=str, default='data/phase1',
                        help='output directory')
    args = parser.parse_args()

    mesh = create_block_world(size=16, num_blocks=30)
    W, H = 256, 256
    fx = fy = W; cx = cy = W/2

    class Intrinsic:
        def __init__(self, fx, fy, cx, cy):
            self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
    intr = Intrinsic(fx, fy, cx, cy)

    center = np.array([8,8,8], dtype=np.float32)
    radius = 20.0
    poses  = []
    for ang in args.views:
        rad = np.deg2rad(ang)
        eye = center + np.array([radius*np.cos(rad), 0, radius*np.sin(rad)])
        R   = look_at(eye, center)
        p   = np.eye(4, dtype=np.float32)
        p[:3,:3] = R; p[:3,3] = eye
        poses.append(p)

    # prepare output folder
    out = args.outdir
    if os.path.isdir(out):
        for f in os.listdir(out): os.remove(os.path.join(out,f))
    os.makedirs(out, exist_ok=True)

    mem   = VoxelMemory(dims=tuple(args.dims), voxel_size=args.voxel_size)
    saved = False

    for i, pose in enumerate(poses, 1):
        rgb, depth = cpu_render(mesh, pose, fx, fy, cx, cy, W, H)
        if not saved:
            np.save(os.path.join(out, 'depth_gt.npy'), depth)
            plt.imsave(os.path.join(out, 'rgb.png'), rgb)
            plt.imsave(os.path.join(out, 'depth.png'), depth, cmap='gray')
            saved = True

        mem.update(pose, rgb, depth, intr)
        cov = mem.counts.sum() / (W*H) * 100
        print(f" View {i}: coverage = {cov:.2f}%")

    hint = mem.query(poses[0], intr, view_size=(H,W), max_dist=args.max_dist)
    np.save(os.path.join(out, 'depth_hint.npy'), hint)
    plt.imsave(os.path.join(out, 'depth_hint.png'), hint, cmap='viridis')

    # save accumulated color for future phases
    np.save(os.path.join(out, 'color_accum.npy'), mem.color_accum)

    filled = np.count_nonzero(mem.counts)
    total  = mem.counts.size
    print(f"[DEBUG] Voxels occupied: {filled}/{total} ({filled/total*100:.2f}%)")
    print(f"Overall voxel coverage: {mem.counts.sum()/(W*H)*100:.2f}%")
    print("Phase 1 outputs written to", out)

if __name__ == "__main__":
    main()