import os
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ==== CONFIGURATION ====
image_path = "/home/rishav/Documents/Project/data_object_image_2/testing/image_2/000100.png"
point_cloud_path = "/home/rishav/Documents/Project/data_object_velodyne/testing/velodyne/000100.bin"
calib_path = "/home/rishav/Documents/Project/data_object_calib/testing/calib/000100.txt"
model = YOLO("/home/rishav/Downloads/yolo11m-seg.pt")


# ==== FUNCTIONS ====

def read_calib(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(val) for val in lines[2].split()[1:]]).reshape(3, 4)
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam

def load_point_cloud(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def transform_lidar_to_camera(points, Tr, R0):
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = (R0 @ Tr @ points_hom.T).T[:, :3]
    return points_cam

def project_to_image(points_cam, P2):
    points_hom = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))
    proj = (P2 @ points_hom.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def segment_image(img):
    results = model(img)[0]
    car_masks = []
    car_class_ids = [i for i, name in model.names.items() if name == 'car']

    for i, cls in enumerate(results.boxes.cls):
        if int(cls.item()) in car_class_ids and results.masks is not None:
            # Resize mask to match original image dimensions
            mask = results.masks.data[i].cpu().numpy()
            h, w = img.shape[:2]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            car_masks.append(mask_resized)
    return car_masks


def visualize_segmentation_overlay(img, masks):
    overlay = img.copy()
    for m in masks:
        color = np.random.randint(0, 255, 3).tolist()
        # Create a colored mask
        colored_mask = np.zeros_like(overlay)
        colored_mask[m.astype(bool)] = color

        # Blend with original image
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

    cv2.imshow("Segmentation Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def associate_lidar_with_masks(points_2d, masks, img_shape):
    h, w = img_shape[:2]
    associations = -np.ones((points_2d.shape[0],), dtype=int)
    resized_masks = [cv2.resize(mask.astype(np.uint8), (w, h)) for mask in masks]

    for idx, (x, y) in enumerate(points_2d):
        if 0 <= int(x) < w and 0 <= int(y) < h:
            for mask_id, mask in enumerate(resized_masks):
                if mask[int(y), int(x)]:
                    associations[idx] = mask_id
                    break
    return associations

def cluster_car_points(points):
    db = DBSCAN(eps=0.8, min_samples=10).fit(points)
    labels = db.labels_
    largest_cluster = points[labels == np.argmax(np.bincount(labels[labels != -1]))]
    return largest_cluster

def create_bounding_box(points):
    min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ])
    lines = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]
    return corners, lines

def visualize_open3d(points, associations):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros_like(points)

    boxes = []
    for car_id in np.unique(associations):
        if car_id < 0: continue
        car_pts = points[associations == car_id]
        if car_pts.shape[0] < 30: continue
        filtered = cluster_car_points(car_pts)
        color = np.random.rand(3)
        colors[associations == car_id] = color
        if filtered.shape[0] > 0:
            corners, lines = create_bounding_box(filtered)
            box = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(corners),
                lines=o3d.utility.Vector2iVector(lines)
            )
            box.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in lines])
            boxes.append(box)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd] + boxes)


def visualize_bev(points, associations):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)

    # Create consistent color mapping
    unique_ids = np.unique(associations[associations >= 0])
    color_map = {i: np.random.rand(3) for i in unique_ids}

    # Plot background points
    bg_mask = (associations < 0)
    ax.scatter(points[bg_mask, 1], points[bg_mask, 0],  # Note: Y vs X for proper orientation
               c='lightgray', s=1, alpha=0.3, label='Background')

    # Plot each car instance
    for car_id in unique_ids:
        car_mask = (associations == car_id)
        if np.sum(car_mask) < 10:  # Skip small clusters
            continue

        color = color_map[car_id]
        car_pts = points[car_mask]

        # Plot points
        ax.scatter(car_pts[:, 1], car_pts[:, 0], c=[color], s=5, label=f'Car {car_id}')

        # Calculate bounding box
        min_y, min_x = np.min(car_pts[:, 1]), np.min(car_pts[:, 0])
        max_y, max_x = np.max(car_pts[:, 1]), np.max(car_pts[:, 0])

        # Draw rectangle
        rect = plt.Rectangle((min_y, min_x), max_y - min_y, max_x - min_x,
                             linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    # Configure plot
    ax.set_xlabel('Y (Left/Right) →')
    ax.set_ylabel('X (Forward) ↑')
    ax.set_title("Bird's Eye View")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()

# ==== PIPELINE ====
img = cv2.imread(image_path)
P2, R0, Tr = read_calib(calib_path)
raw_points = load_point_cloud(point_cloud_path)
points_cam = transform_lidar_to_camera(raw_points, Tr, R0)
proj_points = project_to_image(points_cam, P2)
masks = segment_image(img)
associations = associate_lidar_with_masks(proj_points, masks, img.shape)

visualize_open3d(points_cam, associations)
visualize_bev(points_cam, associations)

# Optional: Visualize segmentation overlay
overlay = img.copy()
for m in masks:
    color = np.random.randint(0, 255, 3)
    overlay[m.astype(bool)] = color
cv2.imshow("Segmentation", cv2.addWeighted(img, 0.5, overlay, 0.5, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()
