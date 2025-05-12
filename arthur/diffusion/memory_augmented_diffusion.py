import os
import argparse
import numpy as np
import h5py
import torch
from PIL import Image
import open3d as o3d
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
from diffusers import StableDiffusionDepth2ImgPipeline

"""
memory_augmented_diffusion.py

Generate a temporally consistent RGB sequence for a Minecraft-like scene by
iteratively conditioning a pretrained Stable Diffusion depth-to-image model
on the current TSDF voxel-grid depth map plus the previous RGB frame.

The generated frames are fused back into the TSDF volume so that scene memory
persists across views. 
"""

# ──────────────────────────────  Helper Classes  ──────────────────────────────
def make_intrinsics(w, h, fx, fy, cx, cy):
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(w, h, fx, fy, cx, cy)
    return intr


class VoxelGrid:
    def __init__(self, voxel=0.02, trunc=0.07):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel,
            sdf_trunc=trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(self, rgb, depth, intrinsic, extrinsic):
        rgb_img   = o3d.geometry.Image((rgb * 255).astype(np.uint8))
        depth_img = o3d.geometry.Image((depth * 1000).astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img, depth_img, depth_scale=1000.0, convert_rgb_to_intensity=False
        )
        self.volume.integrate(rgbd, intrinsic, extrinsic)

    def mesh(self):
        m = self.volume.extract_triangle_mesh()
        m.compute_vertex_normals()
        return m


# ─────────────────────────────  Rendering Utils  ──────────────────────────────
def render_tsdf(tsdf: VoxelGrid, intrinsic, extrinsic, res):
    w, h = res
    mesh = tsdf.mesh()

    renderer = OffscreenRenderer(w, h)
    renderer.scene.set_background([0, 0, 0, 0])

    mat = MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("scene", mesh, mat)
    renderer.setup_camera(intrinsic, extrinsic)

    rgb   = np.asarray(renderer.render_to_image())          # uint8
    depth = np.asarray(renderer.render_to_depth_image(True))  # metres

    depth[np.isinf(depth) | np.isnan(depth)] = 0.0

    if hasattr(renderer, "release"):
        renderer.release()
    else:
        del renderer
    return rgb, depth


def depth_to_tensor(d: np.ndarray, device="cuda", dtype=torch.float16):
    """
    Convert an (H,W) depth NumPy array in metres to a (1,1,H,W) torch tensor.
    """
    tens = torch.from_numpy(d).to(device=device, dtype=dtype)
    return tens


# ─────────────────────────────  Diffusion Wrapper  ────────────────────────────
def load_depth2img_pipeline(dtype=torch.float16, device="cuda"):
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth", torch_dtype=dtype
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    pipe.safety_checker = lambda imgs, **kw: (imgs, False)
    return pipe


# ───────────────────────────────────  Main  ───────────────────────────────────
def main(args):
    os.makedirs(args.output, exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        rgbs       = f["rgb"][()]
        depths     = f["depth"][()]
        intrinsics = f["intrinsics"][()]
        extrinsics = f["extrinsics"][()]

    N         = min(args.frames, len(rgbs))
    seed_N    = min(args.seed, N)
    H, W      = rgbs.shape[1:3]
    res       = (W, H)
    K         = intrinsics[0]
    intr_o3d  = make_intrinsics(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    print(f"[INFO] Bootstrapping TSDF with {seed_N} seed frames …")
    tsdf = VoxelGrid(args.voxel, args.trunc)
    for i in range(seed_N):
        tsdf.integrate(rgbs[i], depths[i], intr_o3d, extrinsics[i])

    prev_rgb_pil = Image.fromarray((rgbs[seed_N - 1] * 255).astype(np.uint8))

    pipe      = load_depth2img_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(args.seed_val)

    for idx in range(seed_N, N):
        print(f"[INFO] Generating frame {idx + 1}/{N} …", end=" ")

        extr          = extrinsics[idx]
        _, depth_mem  = render_tsdf(tsdf, intr_o3d, extr, res)
        depth_tensor  = depth_to_tensor(depth_mem)            # (1,1,H,W) fp16

        result = pipe(
            prompt=args.prompt,
            image=prev_rgb_pil,
            depth_map=depth_tensor,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )
        new_rgb_pil   = result.images[0]

        # Fuse back into TSDF
        tsdf.integrate(
            np.asarray(new_rgb_pil).astype(np.float32) / 255.0,
            depth_mem,
            intr_o3d,
            extr,
        )

        new_rgb_pil.save(os.path.join(args.output, f"frame_{idx:04d}.png"))
        prev_rgb_pil = new_rgb_pil
        torch.cuda.empty_cache()
        print("done")

    # Visualise final mesh
    o3d.visualization.draw_geometries([tsdf.mesh()], window_name="Final TSDF Mesh")


# ───────────────────────────────────  CLI  ────────────────────────────────────
def cli():
    p = argparse.ArgumentParser("Memory-augmented diffusion pipeline")
    p.add_argument("--dataset",  default="data/raw/scene2/dataset.h5")
    p.add_argument("--output",   default="data/generated/scene2")
    p.add_argument("--frames",   type=int, default=100)
    p.add_argument("--seed",     type=int, default=5,
                   help="Frames to pre-integrate before diffusion")
    p.add_argument("--prompt",   default="minecraft voxel world scene")
    p.add_argument("--steps",    type=int, default=30)
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed_val", type=int, default=42)
    p.add_argument("--voxel",    type=float, default=0.02)
    p.add_argument("--trunc",    type=float, default=0.07)
    main(p.parse_args())


if __name__ == "__main__":
    cli()
