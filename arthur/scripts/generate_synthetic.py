import numpy as np
import open3d as o3d
import cv2
from utils import save_dataset_h5

def make_scene():
    boxes = []
    for _ in range(5):
        w,h,d = np.random.uniform(.5,1.5,3)
        box = o3d.geometry.TriangleMesh.create_box(w,h,d)
        box.translate(np.random.uniform(-2,2,3))
        color = np.random.rand(3)      
        box.paint_uniform_color(color)
        boxes.append(box)
    scene = boxes[0]
    for b in boxes[1:]:
        scene += b
    return scene

def render_frames(scene, n_frames=100):
    width=640 
    height=480
    scene_folder = 'data/raw/scene1/'

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(scene)
    ctr = vis.get_view_control()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video = cv2.VideoWriter(scene_folder + 'video.mp4', fourcc, float(fps), (width, height))

    rgb_list = []
    depth_list = []
    pos_list = []
    intrinsics = []
    extrinsics = []
    for i in range(n_frames):
        theta = 2*np.pi * i/n_frames
        cam_pos = [2*np.cos(theta), 0.5, 2*np.sin(theta)]
        ctr.set_lookat([0,0,0])
        ctr.set_up([0,1,0])
        ctr.set_front((np.array([0,0,0]) - cam_pos) / np.linalg.norm(cam_pos))
        ctr.set_zoom(0.5)
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic
        extrinsic = param.extrinsic
        vis.poll_events(); vis.update_renderer()
        rgb = np.asarray(vis.capture_screen_float_buffer()) 
        rgb_scaled = (rgb * 255).astype(np.uint8)
        depth = np.asarray(vis.capture_depth_float_buffer())
        rgb_list.append(rgb)
        depth_list.append(depth)
        pos_list.append(cam_pos)
        intrinsics.append(intrinsic.intrinsic_matrix)
        extrinsics.append(extrinsic)

        # save rgb and depth
        video.write(rgb_scaled)
    vis.destroy_window()
    video.release()
    rgb_list = np.array(rgb_list)
    depth_list = np.array(depth_list)

    save_dataset_h5(scene_folder + 'dataset.h5', rgb_list, depth_list, pos_list, intrinsics, extrinsics)

if __name__ == "__main__":
    scene = make_scene()
    render_frames(scene, 100)