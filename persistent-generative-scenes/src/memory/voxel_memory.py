import numpy as np
from tqdm import tqdm

class VoxelMemory:
    def __init__(self, dims=(128,128,128), voxel_size=0.25):
        self.dims        = np.array(dims, dtype=int)
        self.voxel_size  = float(voxel_size)
        self.color_accum = np.zeros((*self.dims,3), dtype=np.float32)
        self.counts      = np.zeros(self.dims,   dtype=np.int32)

    def update(self, pose, rgb, depth, intrinsic):
        """
        Fuse a real RGB‐D observation into the memory.
        """
        self._fuse(pose, rgb, depth, intrinsic)

    def apply_rgbd(self, pose, rgb, depth, intrinsic):
        """
        Alias for update(), but semantically for hallucinated frames.
        """
        self._fuse(pose, rgb, depth, intrinsic)

    def _fuse(self, pose, rgb, depth, intrinsic):
        H, W = depth.shape
        u,v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        z   = depth.astype(np.float32)

        valid = (z>0) & np.isfinite(z)
        if not np.any(valid):
            return

        # flatten & mask
        idx_flat = valid.reshape(-1)
        u_f = u.reshape(-1)[idx_flat]
        v_f = v.reshape(-1)[idx_flat]
        z_f = z.reshape(-1)[idx_flat]
        cols = rgb.reshape(-1,3)[idx_flat].astype(np.float32)/255.0

        # backproject to camera space
        x = (u_f - intrinsic.cx) * z_f / intrinsic.fx
        y = (v_f - intrinsic.cy) * z_f / intrinsic.fy
        pts_cam = np.stack([x, y, z_f], axis=-1)

        # transform to world
        R, t = pose[:3,:3], pose[:3,3]
        pts_w = (R @ pts_cam.T).T + t

        # keep inside volume
        mins, maxs = np.zeros(3), self.dims * self.voxel_size
        inb = np.all((pts_w >= mins) & (pts_w < maxs), axis=1)
        if not np.any(inb):
            return
        pts_w, cols = pts_w[inb], cols[inb]

        # voxel indices
        vox  = np.floor(pts_w / self.voxel_size).astype(int)
        flat = vox[:,0]*self.dims[1]*self.dims[2] + vox[:,1]*self.dims[2] + vox[:,2]

        # accumulate
        acc = self.color_accum.reshape(-1,3)
        cnt = self.counts.reshape(-1)
        np.add.at(acc, flat, cols)
        np.add.at(cnt, flat, 1)
        self.color_accum = acc.reshape(*self.dims,3)
        self.counts      = cnt.reshape(*self.dims)

    def query(self, pose, intrinsic, view_size=(256,256), max_dist=50.0):
        """
        Ray-march each pixel’s ray through the grid, returning a depth-hint H×W.
        Always returns an array, even if empty.
        """
        H, W = view_size
        depth_hint = np.zeros((H, W), dtype=np.float32)

        # pixel→camera rays
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        dirs_cam = np.stack([
            (u - intrinsic.cx) / intrinsic.fx,
            (v - intrinsic.cy) / intrinsic.fy,
            np.ones_like(u)
        ], axis=-1).reshape(-1,3).astype(np.float32)

        # world‐space origins & directions
        R, t = pose[:3,:3], pose[:3,3]
        rays_o = np.tile(t, (W*H,1))
        rays_d = (R @ dirs_cam.T).T
        rays_d /= np.linalg.norm(rays_d, axis=1, keepdims=True)

        step    = self.voxel_size * 0.8
        n_steps = int(np.ceil(max_dist / step))

        for idx in tqdm(range(W*H), desc="Phase 3 ray‐march"):
            origin    = rays_o[idx]
            direction = rays_d[idx]
            for si in range(n_steps):
                p = origin + direction * (si * step)
                vi = (p / self.voxel_size).astype(int)
                if np.any(vi < 0) or np.any(vi >= self.dims):
                    break
                if self.counts[vi[0],vi[1],vi[2]] > 0:
                    y, x = divmod(idx, W)
                    depth_hint[y, x] = np.linalg.norm(p - t)
                    break

        return depth_hint
