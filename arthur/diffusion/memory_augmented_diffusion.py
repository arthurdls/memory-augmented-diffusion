import os
import argparse
import numpy as np
import h5py
import torch
from PIL import Image
import open3d as o3d
from diffusers import StableDiffusionDepth2ImgPipeline
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from open3d.visualization.rendering import OffscreenRenderer
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd())
from memory.voxel_grid import VoxelGrid, make_intrinsics, render_from_tsdf

"""
memory_augmented_diffusion.py

Generate a temporally consistent RGB sequence for a Minecraft-like scene by
iteratively conditioning a pretrained Stable Diffusion depth-to-image model
on the current TSDF voxel-grid depth map plus the previous RGB frame.

The generated frames are fused back into the TSDF volume so that scene memory
persists across views. 
"""


def predict_depth(depth_net, processor, pil_rgb, ref_depth_m, dtype=torch.float16, device="cpu"):
    # --- 1. relative prediction ------------------------------------------
    inputs = processor(images=pil_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        rel = depth_net(**inputs).predicted_depth[0]       # (1,H,W)
    rel = torch.nn.functional.interpolate(
            rel.unsqueeze(0).unsqueeze(0), size=ref_depth_m.shape[-2:], mode="bicubic",
            align_corners=False).squeeze(0)                # match H×W

    # --- 2. scale to metric units ----------------------------------------
    scale = torch.median(ref_depth_m[ref_depth_m>0]) / torch.median(rel)
    abs_m = (rel * scale).clamp_min_(1e-4)

    # --- 3. convert for both SD-Depth and TSDF integration ---------------
    inv   = (1.0 / abs_m).to(dtype=dtype)              # 0-1 disparity for SD
    depth_tsdf = abs_m.cpu().numpy()                   # float32 metres for TSDF
    return inv.unsqueeze(0), depth_tsdf


def depth_to_tensor(d: np.ndarray, device="cuda", dtype=torch.float16):
    """
    Convert an (H,W) depth open3d image in metres to a (1,H,W) torch tensor input for Stable-Diffusion-Depth-V2.
    """
    depth = np.asarray(d)
    finite_mask = np.isfinite(depth)
    far_val = np.percentile(depth[finite_mask], 99)
    depth[~finite_mask] = far_val
    depth_inv = 1.0 / np.maximum(depth, 1e-5)
    near  = np.percentile(depth_inv, 99)          # 1-percentile = “very near”
    far   = np.percentile(depth_inv, 1)         # 99-percentile = “very far”
    depth_norm = np.clip((depth_inv - far) / (near - far), 0, 1).astype("float32")
    tens = torch.from_numpy(depth_norm).unsqueeze(0).to(device=device, dtype=dtype)
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

    model_id = "Intel/zoedepth-nyu-kitti"
    processor  = AutoImageProcessor.from_pretrained(model_id)
    depth_net  = AutoModelForDepthEstimation.from_pretrained(
                    model_id).to("cpu")

    print(f"[INFO] Bootstrapping TSDF with {seed_N} seed frames …")
    tsdf = VoxelGrid(args.voxel, args.trunc)
    for i in range(seed_N):
        tsdf.integrate(rgbs[i], depths[i], intr_o3d, extrinsics[i])

    prev_rgb_pil = Image.fromarray((rgbs[seed_N - 1] * 255).astype(np.uint8))

    pipe      = load_depth2img_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(args.seed_val)

    # mesh = tsdf.extract_mesh()
    # o3d.visualization.draw_geometries([mesh])
    renderer = OffscreenRenderer(*res)

    for idx in range(seed_N, N):
        print(f"[INFO] Generating frame {idx + 1}/{N} …", end=" ")
        torch.cuda.empty_cache()                 # <– early, before allocations
        renderer.scene.clear_geometry()

        extr          = extrinsics[idx]
        _, depth_mem  = render_from_tsdf(tsdf, intr_o3d, extr, res, renderer)
        depth_tensor  = depth_to_tensor(depth_mem)            # (1,H,W) fp16

        # plt.imshow(prev_rgb_pil)
        # plt.show()
        # plt.imshow(depth_mem, cmap='gray')
        # plt.show()

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
        # plt.imshow(new_rgb_pil)
        # plt.show()
        _, depth_abs = predict_depth(depth_net, processor, new_rgb_pil, torch.from_numpy(np.asarray(depth_mem)))
        # plt.imshow(depth_abs.squeeze(), cmap='gray')
        # plt.show()

        # Fuse back into TSDF
        tsdf.integrate(
            np.asarray(new_rgb_pil).astype(np.float32) / 255.0,
            np.asarray(depth_abs.squeeze()),
            intr_o3d,
            extr,
        )

        new_rgb_pil.save(os.path.join(args.output, f"frame_{idx:04d}.png"))
        prev_rgb_pil = new_rgb_pil
        print("done")

    # Visualise final mesh
    mesh = tsdf.extract_mesh()
    o3d.visualization.draw_geometries([mesh])


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
    p.add_argument("--voxel",    type=float, default=0.10)
    p.add_argument("--trunc",    type=float, default=0.30)
    main(p.parse_args())


if __name__ == "__main__":
    cli()
