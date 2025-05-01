import h5py
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
