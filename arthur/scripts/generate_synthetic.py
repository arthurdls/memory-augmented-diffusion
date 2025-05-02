import numpy as np
import open3d as o3d
from open3d.cuda.pybind import geometry, utility
import cv2
import os
from utils import save_dataset_h5

def make_scene():
    boxes = []
    for _ in range(5):
        w,h,d = np.random.uniform(.2,0.7,3)
        box = o3d.geometry.TriangleMesh.create_box(w,h,d)
        box.translate(np.random.uniform(-1,1,3))
        color = np.random.rand(3)      
        box.paint_uniform_color(color)
        boxes.append(box)
    scene = boxes[0]
    for b in boxes[1:]:
        scene += b
    return scene


def make_minecraft_scene(
    grid_size=(40,40), block_size=0.5, sea_level=3,
    num_hills=6, num_trees=30
):
    W, D = grid_size

    # 1) height map (clamped ≥0)
    hills = [(np.random.uniform(0,W), np.random.uniform(0,D),
              np.random.uniform(2,6), np.random.uniform(5,15))
             for _ in range(num_hills)]
    hm = np.zeros((W,D),float)
    for x in range(W):
        for z in range(D):
            v = 2*np.sin(x/5)*np.cos(z/5)
            for cx,cz,hh,rr in hills:
                d = np.hypot(x-cx, z-cz)
                if d<rr: v += hh*(1-d/rr)
            hm[x,z] = max(0,v)

    # 2) unit cube template
    cube_v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                       [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    cube_f = np.array([[0,1,2],[0,2,3],
                       [4,5,6],[4,6,7],
                       [0,1,5],[0,5,4],
                       [2,3,7],[2,7,6],
                       [1,2,6],[1,6,5],
                       [3,0,4],[3,4,7]],int)

    verts, tris, vcols = [], [], []

    def add_cube(ix, iy, iz, col):
        base = len(verts)
        vs = cube_v*block_size + np.array([ix,iy,iz])*block_size
        verts.extend(vs.tolist())
        for f in cube_f:
            tris.append((base+f[0], base+f[1], base+f[2]))
        vcols.extend([col]*8)

    # 3) terrain
    for x in range(W):
        for z in range(D):
            h = int(round(hm[x,z]))
            for y in range(h):
                c = [0.545,0.271,0.075] if y<sea_level else [0.4,0.3,0.2]
                add_cube(x,y,z,c)
            top_c = [0.1,0.8,0.1] if h>=sea_level else [0.545,0.271,0.075]
            add_cube(x,h,z,top_c)

    # 4) water
    for x in range(W):
        for z in range(D):
            if hm[x,z]<sea_level:
                add_cube(x,sea_level,z,[0.0,0.5,0.8])

    # 5) trees
    for _ in range(num_trees):
        x,z = np.random.randint(0,W), np.random.randint(0,D)
        gh = int(round(hm[x,z]))
        if gh<sea_level: continue
        th = np.random.randint(4,7)
        for y in range(1,th+1):
            add_cube(x,gh+y,z,[0.4,0.2,0.1])
        cx,cy,cz = x+0.5, gh+th+0.5, z+0.5
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if abs(dx)+abs(dy)+abs(dz)<=2:
                        add_cube(cx+dx-0.5, cy+dy-0.5, cz+dz-0.5,
                                 [0.0,0.6,0.0])

    # 6) upload to GPU mesh
    mesh = geometry.TriangleMesh()
    mesh.vertices      = utility.Vector3dVector(np.array(verts))
    mesh.triangles     = utility.Vector3iVector(np.array(tris))
    mesh.vertex_colors = utility.Vector3dVector(np.array(vcols))
    mesh.compute_vertex_normals()
    return mesh


def render_frames(
    scene,
    n_frames=300,
    width=640,
    height=480,
    scene_folder="data/raw/scene3/",
    min_height=5.0,        
    max_height=20.0,        
    zoom=0.6,
):
    os.makedirs(scene_folder, exist_ok=True)

    # ---- setup viewer & video writer ----
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(scene)
    ctr = vis.get_view_control()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        os.path.join(scene_folder, "video.mp4"),
        fourcc, 30.0, (width, height)
    )

    # buffers for h5
    rgb_frames   = []
    depth_frames = []
    cam_poses    = []
    intrinsics   = []
    extrinsics   = []

    # bounding‐box info
    bbox      = scene.get_axis_aligned_bounding_box()
    center    = bbox.get_center()
    ground_y  = bbox.get_min_bound()[1]

    # choose a radius so we're always outside the scene
    radius = np.max(bbox.get_extent()) * 1.2

    # linspace of heights (world Y)
    heights = np.linspace(center[1] - min_height,
                          center[1] - max_height,
                          n_frames)

    for i in range(n_frames):
        # azimuth angle [0,2π)
        theta = 2 * np.pi * i / n_frames
        # current camera height
        h     = heights[i]

        # spherical → cartesian
        cam_pos = np.array([
            center[0] + radius * np.cos(theta),
            h,
            center[2] + radius * np.sin(theta),
        ])

        # view direction toward center
        front = center - cam_pos
        front /= np.linalg.norm(front)

        # configure camera
        ctr.set_lookat(center.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(zoom)

        # render
        vis.poll_events()
        vis.update_renderer()

        # capture
        rgb   = np.asarray(vis.capture_screen_float_buffer())
        depth = np.asarray(vis.capture_depth_float_buffer())

        rgb_frames.append(rgb)
        depth_frames.append(depth)
        cam_poses.append(cam_pos.tolist())

        params = ctr.convert_to_pinhole_camera_parameters()
        intrinsics.append(params.intrinsic.intrinsic_matrix.copy())
        extrinsics.append(params.extrinsic.copy())

        # write video frame
        frame_bgr = (rgb * 255).astype(np.uint8)[..., ::-1]
        video.write(frame_bgr)

    # ---- cleanup & save ----
    vis.destroy_window()
    video.release()

    save_dataset_h5(
        os.path.join(scene_folder, "dataset.h5"),
        np.stack(rgb_frames),
        np.stack(depth_frames),
        cam_poses,
        intrinsics,
        extrinsics
    )


if __name__ == "__main__":
    scene = make_minecraft_scene(
        grid_size=(10, 10),
        block_size=0.4,
        sea_level=1,
        num_hills=5,
        num_trees=5,
    )
    o3d.visualization.draw_geometries([scene],
        window_name="Preview Scene",
        width=1200, height=1200,
        mesh_show_back_face=True)
    render_frames(scene, n_frames=100) 