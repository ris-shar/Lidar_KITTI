import os
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import matplotlib.pyplot as plt

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
    return points[:, :3]  # x, y, z

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

def visualize_open3d(points, associations):
    colors = np.zeros_like(points)
    unique_ids = np.unique(associations)
    mask_colors = [np.random.rand(3) for _ in range(len(unique_ids))]
    for i, mask_id in enumerate(associations):
        colors[i] = mask_colors[mask_id] if mask_id >= 0 else [0.5, 0.5, 0.5]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    car_boxes = []
    for mask_id in np.unique(associations):
        if mask_id < 0:
            continue
        car_points = points[associations == mask_id]
        if car_points.shape[0] > 0:
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(car_points))
            aabb.color = (1, 1, 0)
            car_boxes.append(aabb)

    o3d.visualization.draw_geometries([pcd] + car_boxes)

def visualize_bev(points, associations):
    bev_image = np.zeros((800, 800, 3), dtype=np.uint8)
    scale = 10  # 1 meter = 10 pixels
    offset = (400, 700)
    instance_colors = [np.random.randint(0, 255, 3).tolist() for _ in range(np.max(associations)+1)]

    for i, (x, y) in enumerate(points[:, [0, 1]]):
        u = int(offset[0] + y * scale)
        v = int(offset[1] - x * scale)
        if 0 <= u < 800 and 0 <= v < 800:
            color = instance_colors[associations[i]] if associations[i] >= 0 else [128, 128, 128]
            bev_image[v, u] = color

    for instance_id in np.unique(associations):
        if instance_id < 0:
            continue
        instance_points = points[associations == instance_id][:, [0, 1]]
        x_min, y_min = np.min(instance_points, axis=0)
        x_max, y_max = np.max(instance_points, axis=0)
        pt1 = (int(offset[0] + y_min * scale), int(offset[1] - x_min * scale))
        pt2 = (int(offset[0] + y_max * scale), int(offset[1] - x_max * scale))
        cv2.rectangle(bev_image, pt1, pt2, (0, 255, 255), 2)

    plt.imshow(bev_image)
    plt.title("Bird's-Eye View")
    plt.axis('off')
    plt.show()

# ==== MAIN PIPELINE ====
img = cv2.imread(image_path)
P2 = read_calib(calib_path)
points = load_point_cloud(point_cloud_path)
points_img = project_lidar_to_image(points, P2)
masks = segment_image(img)
associations, resized_masks = associate_points_with_masks(points_img, masks, img.shape)
seg_overlay = draw_segmentation_on_image(img, resized_masks)

# Show image with segmentation
cv2.imshow("Car Segmentation", seg_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3D and BEV visualization
visualize_open3d(points, associations)
visualize_bev(points, associations)
