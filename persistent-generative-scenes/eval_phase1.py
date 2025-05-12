# File: eval_phase1.py

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    outdir = "data/phase1"
    depth_gt   = np.load(os.path.join(outdir, "depth_gt.npy"))
    depth_hint = np.load(os.path.join(outdir, "depth_hint.npy"))

    # Mask where hint > 0
    mask = depth_hint > 0
    covered = mask.sum()
    total   = depth_gt.size

    print(f"Covered pixels: {covered}/{total} ({covered/total*100:.2f}%)")

    if covered == 0:
        print("⚠️  No depth‐hint pixels were generated. "
              "Check your VoxelMemory.update/query logic or the camera pose.")
        return

    # Compute absolute errors only where hint exists
    errors   = np.abs(depth_hint[mask] - depth_gt[mask])
    mean_err = float(errors.mean())
    max_err  = float(errors.max())

    print(f"Mean absolute error: {mean_err:.4f} m")
    print(f"Max  absolute error: {max_err:.4f} m")

    # Save error map
    err_map = np.zeros_like(depth_gt)
    err_map[mask] = errors

    plt.figure(figsize=(6,6))
    plt.title("Depth Hint Absolute Error")
    plt.imshow(err_map, cmap="magma")
    plt.colorbar(label="|hint – gt| (m)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error.png"), dpi=200)
    print(f"Error heatmap saved to {outdir}/error.png")

if __name__ == "__main__":
    main()
