import os
import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# Paths and settings
root = "/home/rishav/Documents/Project"
frame = "000250"
image_path = f"{root}/data_object_image_2/training/image_2/{frame}.png"
lidar_path = f"{root}/data_object_velodyne/training/velodyne/{frame}.bin"
calib_path = f"{root}/data_object_calib/training/calib/{frame}.txt"
right_image_path = f"{root}/data_object_image_3/training/image_3/{frame}.png"
label_path = f"{root}/data_object_label_2/training/label_2/{frame}.txt"
sam_checkpoint = "/home/rishav/Downloads/sam_l.pt"
model_type = "vit_l"

# Load KITTI calibration data
def load_kitti_calib(calib_file_path):
    calib = {}
    with open(calib_file_path, 'r') as f:
        for line in f.readlines():
            if ":" in line:
                key, value = line.strip().split(":", 1)
                calib[key] = np.array([float(x) for x in value.strip().split()])
    return calib['P2'].reshape(3, 4), calib['R0_rect'].reshape(3, 3), calib['Tr_velo_to_cam'].reshape(3, 4)

P2, R0, Tr_velo_to_cam = load_kitti_calib(calib_path)

# Project LiDAR points to image coordinates
def project_lidar_to_image(points, P2, R0, Tr):
    N = points.shape[0]
    points_hom = np.hstack((points, np.ones((N, 1))))
    pts_cam = (Tr @ points_hom.T).T
    pts_cam = (R0 @ pts_cam.T).T
    pts_img = (P2 @ np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1)))).T).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    return pts_img

# Visualize Bird's Eye View

def visualize_bev(pc, colors, scale=1.0, figsize=(10, 10)):
    x = pc[:, 0] * scale
    y = pc[:, 1] * scale
    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=colors, s=0.5)
    plt.xlabel('X (Forward)')
    plt.ylabel('Y (Left/Right)')
    plt.title('Bird\'s Eye View (BEV)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Filter largest cluster using DBSCAN
def filter_largest_cluster(points, eps=0.8, min_samples=5):
    if len(points) < 10:
        return np.array([])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    if len(set(labels)) <= 1:
        return points
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]
    return points[labels == largest_label]

# Load and convert image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run SAM segmentation
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image_rgb)

# Run YOLO to detect cars
yolo_model = YOLO("yolov8n.pt")
yolo_results = yolo_model.predict(image_rgb, verbose=False)[0]
car_boxes = [yolo_results.boxes.xyxy[i].cpu().numpy().astype(int) for i, cls in enumerate(yolo_results.boxes.cls) if int(cls) == 2]

# Match YOLO boxes to SAM masks
car_masks, car_colors = [], []
colors_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]

for box in car_boxes:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for idx, mask in enumerate(masks):
        seg = mask['segmentation'].astype(bool)
        h, w = seg.shape
        if 0 <= cx < w and 0 <= cy < h and seg[cy, cx]:
            car_masks.append(seg.astype(np.uint8))
            car_colors.append(colors_list[len(car_masks) % len(colors_list)])
            break

# Overlay segmentation masks on image
overlay_img = image_rgb.copy()
for i, seg in enumerate(car_masks):
    h, w = seg.shape
    color = np.array(car_colors[i])
    mask_rgb = np.zeros_like(overlay_img)
    mask_rgb[seg == 1] = color
    overlay_img = cv2.addWeighted(overlay_img, 1.0, mask_rgb, 0.5, 0)
    seg = cv2.erode(seg.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)

plt.imshow(overlay_img)
plt.title("Image with YOLO-centered SAM Car Masks")
plt.axis('off')
plt.show()

# Load LiDAR and project to image
pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
proj_pts = project_lidar_to_image(pc, P2, R0, Tr_velo_to_cam)
colors = np.ones_like(pc) * 0.5

# Color LiDAR points based on matched masks
for idx, seg in enumerate(car_masks):
    h, w = seg.shape
    matched_idxs = [i for i, (u, v) in enumerate(proj_pts.astype(int)) if 0 <= u < w and 0 <= v < h and seg[v, u]]
    raw_pts = pc[matched_idxs]
    valid_pts = filter_largest_cluster(raw_pts)
    for pt in valid_pts:
        idx_in_pc = np.where((pc == pt).all(axis=1))[0]
        if idx_in_pc.size > 0:
            colors[idx_in_pc[0]] = np.array(car_colors[idx]) / 255.0

# Compute bounding boxes
bbox_lines = []
for idx, color in enumerate(car_colors):
    raw_pts = pc[(colors == np.array(color) / 255.0).all(axis=1)]
    car_pts = filter_largest_cluster(raw_pts)
    if len(car_pts) == 0:
        continue
    car_pcd = o3d.geometry.PointCloud()
    car_pcd.points = o3d.utility.Vector3dVector(car_pts)
    aabb = car_pcd.get_axis_aligned_bounding_box()
    aabb.color = np.array(color) / 255.0
    bbox_lines.append(aabb)

# Visualize in Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd] + bbox_lines)

# Load ground truth labels
def load_kitti_labels(label_path):
    gt_coords = []
    with open(label_path, 'r') as f:
        for line in f:
            if line.startswith("Car"):
                parts = line.strip().split()
                x, y, z = map(float, parts[11:14])
                gt_coords.append(np.array([x, y, z]))
    return gt_coords

def transform_cam_to_velo(coords_cam, R0, Tr_velo_to_cam):
    R0_h = np.eye(4); R0_h[:3, :3] = R0
    Tr_h = np.eye(4); Tr_h[:3, :] = Tr_velo_to_cam
    P_cam = np.hstack((coords_cam, np.ones((coords_cam.shape[0], 1)))).T
    P_velo = np.linalg.inv(Tr_h @ R0_h) @ P_cam
    return P_velo[:3, :].T

gt_camera_coords = load_kitti_labels(label_path)
gt_lidar_coords = transform_cam_to_velo(np.array(gt_camera_coords), R0, Tr_velo_to_cam)

# Visualize BEV
visualize_bev(pc, colors)

# Estimate distances using stereo
right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
left_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5,
    P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2, disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32)

disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
focal_length = P2[0, 0]
baseline = 0.54

stereo_distances = []
for box in car_boxes:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    disparity = disparity_map[cy, cx]
    if disparity > 0:
        stereo_distances.append((focal_length * baseline) / disparity)
    else:
        stereo_distances.append(None)

# Compute LiDAR-based distances
lidar_distances = []
for color in car_colors:
    pts = pc[(colors == np.array(color) / 255.0).all(axis=1)]
    pts = filter_largest_cluster(pts)
    lidar_distances.append(np.min(np.linalg.norm(pts, axis=1)) if len(pts) > 0 else None)

# Compare distances
for i, d in enumerate(stereo_distances):
    print(f"Car {i}: {d:.2f} m" if d else f"Car {i}: invalid disparity")

print("\n--- Distance Report ---")
for idx, color in enumerate(car_colors):
    car_pts = pc[(colors == np.array(color) / 255.0).all(axis=1)]
    car_pts = filter_largest_cluster(car_pts)
    if car_pts.shape[0] == 0:
        print(f"Car {idx+1}: No valid point cloud data.")
        continue
    calc_dist = np.min(np.linalg.norm(car_pts, axis=1))
    if idx < len(gt_lidar_coords):
        gt_dist = np.linalg.norm(gt_lidar_coords[idx])
        print(f"Car {idx+1}: Calculated = {calc_dist:.2f} m, Ground Truth = {gt_dist:.2f} m")
    else:
        print(f"Car {idx+1}: Calculated = {calc_dist:.2f} m, Ground Truth = Not available")

# Error comparison
for i in range(len(car_boxes)):
    if stereo_distances[i] is not None and lidar_distances[i] is not None:
        error = abs(stereo_distances[i] - lidar_distances[i])
        print(f"Car {i}: Stereo = {stereo_distances[i]:.2f} m, LiDAR = {lidar_distances[i]:.2f} m, Error = {error:.2f} m")
