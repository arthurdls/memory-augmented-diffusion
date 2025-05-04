import open3d as o3d
import numpy as np
import h5py
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord


def make_intrinsics(width, height, fx, fy, cx, cy, ):
    """Return an Open3D *legacy* PinholeCameraIntrinsic."""
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsic


class VoxelGrid:
    def __init__(self, voxel_length=0.01, sdf_trunc=0.03):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    def integrate(self, rgb, depth, intrinsic, extrinsic):
        rgb_o3d = o3d.geometry.Image((rgb*255).astype(np.uint8))
        depth_o3d = o3d.geometry.Image((depth*1000).astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=int(np.max(depth)), convert_rgb_to_intensity=False)
        self.volume.integrate(rgbd, intrinsic, extrinsic)

    def extract_mesh(self):
        return self.volume.extract_triangle_mesh()


def render_from_tsdf(voxel_grid,
                     intrinsic,
                     extrinsic,
                     resolution = (640, 480)):
    renderer = OffscreenRenderer(*resolution)
    renderer.scene.set_background([0, 0, 0, 0])               
    mat = MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("mesh", voxel_grid.extract_mesh(), mat)           
    renderer.setup_camera(intrinsic, extrinsic)         

    rgb = renderer.render_to_image()
    depth = renderer.render_to_depth_image(True)
    return rgb, depth


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    grid = VoxelGrid(0.02, 0.07)

    with h5py.File('data/raw/scene3/dataset.h5','r') as f:
        rgbs  = f['rgb'][()]
        depths = f['depth'][()]
        intrinsics = f['intrinsics'][()]
        extrinsics = f['extrinsics'][()]

    intr_o3d = None
    for rgb, depth, intr, extr in zip(rgbs, depths, intrinsics, extrinsics):
        height, width = rgb.shape[:2]
        if intr_o3d is None:
            intr_o3d = make_intrinsics(width, height,
                                          intr[0,0], intr[1,1],
                                          intr[0,2], intr[1,2])
        grid.integrate(rgb, depth, intr_o3d, extr)

    mesh = grid.extract_mesh()
    o3d.visualization.draw_geometries([mesh])

    for extr in extrinsics:
        resolution = (width, height)
        depth_range = (0.0001, 30)
        rgb, depth = render_from_tsdf(grid, intr_o3d, extr, resolution)
        # plt.imshow(rgb)
        # plt.show()
        # plt.imshow(depth, cmap='gray')
        # plt.show()
