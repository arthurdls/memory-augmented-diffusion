import open3d as o3d
import numpy as np

def create_block_world(size=16, num_blocks=50):
    """
    Returns a combined TriangleMesh of random 1×1×1 colored cubes
    within a size^3 grid.
    """
    world = o3d.geometry.TriangleMesh()
    for _ in range(num_blocks):
        # random integer grid position
        x, y, z = np.random.randint(0, size, 3)
        # make a unit cube
        cube = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        # give it a random color
        color = np.random.rand(3)
        cube.paint_uniform_color(color)
        # move it to (x,y,z)
        cube.translate((x, y, z))
        cube.compute_vertex_normals()
        # merge into world
        world += cube

    world.compute_vertex_normals()
    return world


def render_scene(mesh, pose, width=256, height=256):
    """
    Offscreen render with Open3D using the mesh's builtin colors.
    pose: 4×4 camera-to-world transform.
    Returns: rgb (H,W,3) uint8, depth (H,W) float32 in meters.
    """
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("world", mesh, material)
    # Camera intrinsics
    fx = fy = width
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    extrinsic = np.linalg.inv(pose)  # world->camera
    renderer.setup_camera(intrinsic, extrinsic)
    # Render
    img = renderer.render_to_image()
    depth = renderer.render_to_depth_image(True)
    rgb = np.asarray(img)
    depth_map = np.asarray(depth) * 1.0  # depth in meters
    return rgb, depth_map