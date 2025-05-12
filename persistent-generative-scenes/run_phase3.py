#!/usr/bin/env python3
import os, sys
import numpy as np
from skimage.measure import marching_cubes
import open3d as o3d
import open3d.core as o3c, open3d.t.geometry as tgeom

# allow imports
repo = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo, "src"))
sys.path.insert(0, os.path.join(repo, "scripts"))
from memory.voxel_memory import VoxelMemory
from gen_scenes import create_block_world

def look_at(eye, target, up=np.array([0,1,0], dtype=np.float32)):
    f = (target - eye); f /= np.linalg.norm(f)
    r = np.cross(up, f);      r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=1)

def cpu_render(mesh, pose, fx, fy, cx, cy, W=256, H=256):
    tmesh = tgeom.TriangleMesh.from_legacy(mesh)
    scene = tgeom.RaycastingScene()
    scene.add_triangles(tmesh)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs_cam = np.stack([
        (u - cx)/fx,
        (v - cy)/fy,
        np.ones_like(u)
    ], axis=-1).reshape(-1,3).astype(np.float32)

    R, t = pose[:3,:3], pose[:3,3]
    rays_o = np.tile(t, (dirs_cam.shape[0],1))
    rays_d = (R @ dirs_cam.T).T
    rays_d /= np.linalg.norm(rays_d, axis=1, keepdims=True)

    rays = o3c.Tensor(
        np.hstack([rays_o, rays_d]).astype(np.float32),
        dtype=o3c.Dtype.Float32
    )
    ans   = scene.cast_rays(rays)
    depth = ans["t_hit"].numpy().reshape(H, W)
    depth[~np.isfinite(depth)] = 0.0

    gray = np.zeros_like(depth, dtype=np.uint8)
    mask = depth > 0
    if np.any(mask):
        gray[mask] = np.clip(depth[mask]/depth[mask].max()*255, 0, 255).astype(np.uint8)
    rgb = np.stack([gray]*3, axis=-1)
    return rgb, depth

def hallucinate_texture(depth_hint):
    H, W = depth_hint.shape
    gray = np.zeros((H, W), dtype=np.uint8)
    mask = depth_hint > 0
    if np.any(mask):
        gray[mask] = np.clip(depth_hint[mask]/depth_hint[mask].max()*255, 0, 255).astype(np.uint8)
    return np.stack([gray]*3, axis=-1)

def main():
    # -- Phase 1 memory & scene setup --
    mem  = VoxelMemory(dims=(128,128,128), voxel_size=0.25)
    mesh = create_block_world(size=16, num_blocks=30)

    # camera intrinsics
    W = H = 256
    fx = fy = W; cx = cy = W/2
    class Intrinsic:
        def __init__(self, fx, fy, cx, cy):
            self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
    intr = Intrinsic(fx, fy, cx, cy)

    # four circular views
    center = np.array([8,8,8], dtype=np.float32)
    radius = 20.0
    poses  = []
    for ang in [0,90,180,270]:
        rad = np.deg2rad(ang)
        eye = center + np.array([radius*np.cos(rad),0,radius*np.sin(rad)], dtype=np.float32)
        R   = look_at(eye, center)
        P   = np.eye(4, dtype=np.float32)
        P[:3,:3], P[:3,3] = R, eye
        poses.append(P)

    # 1) Fuse real captures
    print("Fusing real captures…")
    for i,P in enumerate(poses,1):
        rgb, depth = cpu_render(mesh, P, fx, fy, cx, cy, W, H)
        mem.update(P, rgb, depth, intr)
        cov = mem.counts.sum()/(W*H)*100
        print(f"  View {i}: coverage = {cov:.1f}%")

    # 2) Hallucinate & fuse generated textures
    print("Hallucinating & fusing generated textures…")
    for i,P in enumerate(poses,1):
        hint    = mem.query(P, intr, view_size=(H,W))
        rgb_gen = hallucinate_texture(hint)
        mem.apply_rgbd(P, rgb_gen, hint, intr)
        cov = mem.counts.sum()/(W*H)*100
        print(f"  Gen {i}: coverage = {cov:.1f}%")

    # 3) Extract colored mesh
    print("Extracting final mesh with Marching Cubes…")
    occ   = (mem.counts>0).astype(np.float32)
    vs    = mem.voxel_size
    verts, faces, normals, _ = marching_cubes(volume=occ, level=0.5, spacing=(vs,vs,vs))

    # per-vertex color
    ijk        = np.floor(verts/vs).astype(int)
    np.clip(ijk, 0, np.array(mem.dims)-1, out=ijk)
    cols_acc   = mem.color_accum[ijk[:,0],ijk[:,1],ijk[:,2]]
    counts     = mem.counts[ijk[:,0],ijk[:,1],ijk[:,2]].reshape(-1,1)
    vert_colors = (cols_acc/np.maximum(counts,1)).clip(0,1)

    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh_o3d.vertex_colors  = o3d.utility.Vector3dVector(vert_colors)

    out = "data/phase3"
    os.makedirs(out, exist_ok=True)
    pth = os.path.join(out, "mesh_phase3.ply")
    o3d.io.write_triangle_mesh(pth, mesh_o3d)
    print(f"Phase 3 mesh saved to {pth}")

    # visualize
    o3d.visualization.draw_geometries([mesh_o3d],
                                      window_name="Phase 3 Mesh",
                                      width=800, height=600)

if __name__=="__main__":
    main()
