#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import open3d as o3d

# allow imports from your repo
repo = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo, "src"))
sys.path.insert(0, os.path.join(repo, "scripts"))
from memory.voxel_memory import VoxelMemory
from gen_scenes import create_block_world

def look_at(eye, target, up=np.array([0,1,0], dtype=np.float32)):
    f = (target - eye); f /= np.linalg.norm(f)
    r = np.cross(up, f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=1)

def cpu_render(mesh, pose, fx, fy, cx, cy, W=256, H=256):
    import open3d.core as o3c, open3d.t.geometry as tgeom
    tmesh = tgeom.TriangleMesh.from_legacy(mesh)
    scene = tgeom.RaycastingScene(); scene.add_triangles(tmesh)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs_cam = np.stack([
        (u - cx)/fx,
        (v - cy)/fy,
        np.ones_like(u)
    ], axis=-1).reshape(-1,3).astype(np.float32)

    R, t = pose[:3,:3], pose[:3,3]
    rays_o = np.tile(t, (dirs_cam.shape[0],1))
    rays_d = (R @ dirs_cam.T).T
    norms = np.linalg.norm(rays_d, axis=1, keepdims=True)
    rays_d /= norms

    rays = o3c.Tensor(np.hstack([rays_o, rays_d]).astype(np.float32),
                      dtype=o3c.Dtype.Float32)
    depth = scene.cast_rays(rays)["t_hit"].numpy().reshape(H, W)
    depth[~np.isfinite(depth)] = 0

    gray = np.zeros_like(depth, dtype=np.uint8)
    valid = depth > 0
    if np.any(valid):
        gray[valid] = np.clip(depth[valid]/depth[valid].max()*255, 0, 255).astype(np.uint8)
    return np.stack([gray]*3, axis=-1), depth

def main():
    # -- Phase 1 integration --
    mesh = create_block_world(size=16, num_blocks=30)
    W, H = 256, 256
    fx = fy = W; cx = cy = W/2
    class Intrinsic:
        def __init__(self, fx, fy, cx, cy):
            self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
    intr = Intrinsic(fx, fy, cx, cy)

    # four views around center
    center = np.array([8,8,8], dtype=np.float32)
    radius = 20.0
    poses = []
    for ang in [0,90,180,270]:
        rad = np.deg2rad(ang)
        eye = center + np.array([radius*np.cos(rad), 0, radius*np.sin(rad)], dtype=np.float32)
        R   = look_at(eye, center)
        p   = np.eye(4, dtype=np.float32)
        p[:3,:3], p[:3,3] = R, eye
        poses.append(p)

    out = "data/phase1"
    os.makedirs(out, exist_ok=True)

    mem = VoxelMemory(dims=(128,128,128), voxel_size=0.25)
    saved = False
    for i, pose in enumerate(poses, 1):
        rgb, depth = cpu_render(mesh, pose, fx, fy, cx, cy, W, H)
        if not saved:
            np.save(os.path.join(out, "depth_gt.npy"), depth)
            plt.imsave(os.path.join(out, "rgb.png"), rgb)
            plt.imsave(os.path.join(out, "depth.png"), depth, cmap="gray")
            saved = True
        mem.update(pose, rgb, depth, intr)
        print(f" View {i}: coverage = {mem.counts.sum()/(W*H)*100:.2f}%")

    # -- Phase 2: Mesh extraction via Marching Cubes --
    occ = (mem.counts > 0).astype(np.float32)
    vs = mem.voxel_size
    print(f"Running Marching Cubes on {occ.size} voxels…")
    verts, faces, normals, _ = marching_cubes(
        volume=occ,
        level=0.5,
        spacing=(vs, vs, vs)
    )

    # color vertices by sampling the voxel‐memory color_accum / counts
    # convert verts → voxel indices
    ijk = np.floor(verts / vs).astype(int)
    # clamp in‐bounds
    np.clip(ijk, 0, np.array(mem.dims)-1, out=ijk)
    cols_acc = mem.color_accum[ijk[:,0], ijk[:,1], ijk[:,2]]
    counts   = mem.counts[ijk[:,0], ijk[:,1], ijk[:,2]].reshape(-1,1)
    vert_colors = (cols_acc / np.maximum(counts,1)).clip(0,1)

    # build Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh_o3d.vertex_colors  = o3d.utility.Vector3dVector(vert_colors)

    # save & visualize
    pth = os.path.join(out, "mesh_phase2.ply")
    o3d.io.write_triangle_mesh(pth, mesh_o3d)
    print(f"Mesh saved to {pth}")
    o3d.visualization.draw_geometries(
        [mesh_o3d],
        window_name="Phase 2 Mesh",
        width=800, height=600
    )

if __name__=="__main__":
    main()