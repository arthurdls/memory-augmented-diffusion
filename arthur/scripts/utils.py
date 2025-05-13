import h5py
import os
import glob
import cv2               # pip install opencv-python
import sys
import shutil
from PIL import Image
sys.path.append(os.getcwd())
import numpy as np

def save_dataset_h5(path, rgbs, depths, poses, intrinsics, extrinsics):
    with h5py.File(path, 'w') as f:
        # chunk along frame axis so you can read one frame at a time
        f.create_dataset('rgb',   data=rgbs,
                         compression='gzip', chunks=(1,)+rgbs.shape[1:])
        f.create_dataset('depth', data=depths,
                         compression='gzip', chunks=(1,)+depths.shape[1:])
        f.create_dataset('pose',      data=poses)
        f.create_dataset('intrinsics',data=intrinsics)
        f.create_dataset('extrinsics',data=extrinsics)


def frames_to_video(
    frames_dir: str,
    out_dir: str,
    fps: int = 30,
    pattern: str = "*.png",
    codec: str = "mp4v"   # "XVID" for .avi, "avc1" for H.264, etc.
) -> None:
    # Collect and sort all matching frames
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not frame_paths:
        raise FileNotFoundError(f"No images matching {pattern} in {frames_dir}")

    # Use the first frame to get frame size
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError(f"Cannot read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    # Prepare the video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(os.path.join(out_dir, "video.mp4"), fourcc, fps, (width, height))

    # Write each frame
    skipped = 0
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            skipped += 1
            continue
        writer.write(img)

    writer.release()
    print(f"Finished: {len(frame_paths) - skipped} frames written to {out_dir}")
    if skipped:
        print(f"Skipped {skipped} unreadable frame(s).")


def h5_to_frames():
    for i in range(1, 10):
        scene = f"scene{i}"
        with h5py.File(f"data/raw/{scene}/dataset.h5", "r") as f:
            rgbs       = f["rgb"][()]

        for idx in range(200):
            rgb = (rgbs[idx] * 255).astype(np.uint8)
            rgb_img = Image.fromarray(rgb, "RGB")
            rgb_img.save(os.path.join(f"data/raw/{scene}", f"frame_{idx:04d}.png"))


def copy_png_image(src_path: str, new_name: str) -> None:
    if not src_path.lower().endswith('.png'):
        raise ValueError("Source file must be a .png image")
    if not new_name.lower().endswith('.png'):
        raise ValueError("New filename must end with .png")

    directory = os.path.dirname(os.path.abspath(src_path))
    dest_path = os.path.join(directory, new_name)

    shutil.copy2(src_path, dest_path)
    print(f"Copied '{src_path}' to '{dest_path}'")

if __name__ == "__main__":
    for idx in range(182, 200):
        original_file = f"data/generated/scene9/frame_{idx-1:04d}.png"
        new_filename = f"frame_{idx:04d}.png"
        copy_png_image(original_file, new_filename)
