import os
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ==== CONFIGURATION ====
image_path = "/home/rishav/Documents/Project/data_object_image_2/testing/image_2/000560.png"
point_cloud_path = "/home/rishav/Documents/Project/data_object_velodyne/testing/velodyne/000560.bin"
calib_path = "/home/rishav/Documents/Project/data_object_calib/testing/calib/000560.txt"
model = YOLO("/home/rishav/Downloads/yolo11m-seg.pt")


# ==== FUNCTIONS ====
def read_calib(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(val) for val in lines[2].strip().split()[1:]]).reshape(3, 4)
    return P2


def load_point_cloud(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def project_lidar_to_image(pts_3d, P):
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d = (P @ pts_3d_hom.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d


def segment_image(image):
    results = model(image)[0]
    car_class_ids = [i for i, name in model.names.items() if name == 'car']
    masks = []
    for i, cls in enumerate(results.boxes.cls):
        if int(cls.item()) in car_class_ids and results.masks is not None:
            masks.append(results.masks.data[i].cpu().numpy())
    return masks


def associate_points_with_masks(points_img, masks, img_shape):
    h, w = img_shape[:2]
    associations = -np.ones((points_img.shape[0],), dtype=int)
    resized_masks = [cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for mask in masks]
    for idx, (x, y) in enumerate(points_img):
        if 0 <= int(y) < h and 0 <= int(x) < w:
            for i, mask in enumerate(resized_masks):
                if mask[int(y), int(x)]:
                    associations[idx] = i
                    break
    return associations, resized_masks


def draw_segmentation_on_image(image, masks):
    overlay = image.copy()
    h, w = image.shape[:2]
    for i, mask in enumerate(masks):
        color = np.random.randint(0, 255, 3)
        overlay[mask.astype(bool)] = color
    blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    return blended


def get_3d_bbox(points):
    if len(points) == 0:
        return None
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
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
    return corners


def filter_instance_points(points, eps=0.5, min_samples=10):
    if points.shape[0] < min_samples:
        return points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
    labels = clustering.labels_
    if np.all(labels == -1):
        return points
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    return points[labels == largest_cluster]


def visualize_open3d(points, associations, color_map):
    colors = np.zeros_like(points)
    for i, mask_id in enumerate(associations):
        colors[i] = color_map[mask_id] if mask_id >= 0 else [0.5, 0.5, 0.5]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    car_boxes = []
    for instance_id in np.unique(associations):
        if instance_id < 0:
            continue
        instance_points = points[associations == instance_id]
        filtered_points = filter_instance_points(instance_points, eps=1.0, min_samples=3)

        bbox_corners = get_3d_bbox(filtered_points)
        if bbox_corners is not None:
            lines = [
                [0, 1], [0, 2], [1, 3], [2, 3],
                [4, 5], [4, 6], [5, 7], [6, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            box = o3d.geometry.LineSet()
            box.points = o3d.utility.Vector3dVector(bbox_corners)
            box.lines = o3d.utility.Vector2iVector(lines)
            box.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
            car_boxes.append(box)

    o3d.visualization.draw_geometries([pcd] + car_boxes)


def visualize_bev(points, associations, color_map):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    ax.scatter(points[:, 0], points[:, 1], c='gray', s=1, alpha=0.2)

    for instance_id in np.unique(associations):
        if instance_id < 0:
            continue
        instance_points = points[associations == instance_id]
        filtered_points = filter_instance_points(instance_points)
        if filtered_points.shape[0] == 0:
            continue
        color = color_map[instance_id]
        ax.scatter(filtered_points[:, 0], filtered_points[:, 1], c=[color], s=2)
        min_x, min_y = np.min(filtered_points[:, :2], axis=0)
        max_x, max_y = np.max(filtered_points[:, :2], axis=0)
        rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                             linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_title("Filtered Bird's-Eye View (BEV)")
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()


# ==== MAIN PIPELINE ====
img = cv2.imread(image_path)
P2 = read_calib(calib_path)
points = load_point_cloud(point_cloud_path)
points_img = project_lidar_to_image(points, P2)
masks = segment_image(img)
associations, resized_masks = associate_points_with_masks(points_img, masks, img.shape)
seg_overlay = draw_segmentation_on_image(img, resized_masks)

# Generate consistent colors per instance
color_map = {i: np.random.rand(3) for i in np.unique(associations) if i >= 0}

visualize_open3d(points, associations, color_map)
visualize_bev(points, associations, color_map)

cv2.imshow("Car Segmentation Overlay", seg_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
