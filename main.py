import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d
import matplotlib.pyplot as plt


def load_calibration(calib_file):
    """Load calibration data from file."""
    calib = {}
    with open(calib_file) as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                calib[key] = np.array([float(x) for x in value.strip().split()])
    # Reshape matrices
    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    return calib


def load_image(image_path):
    """Load image from file."""
    image = cv2.imread(image_path)
    return image


def segment_image(model, image):
    """Perform instance segmentation on the image."""
    results = model(image)
    return results
def load_point_cloud(bin_path):
    """Load point cloud data from .bin file."""
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud
def project_lidar_to_image(points, calib):
    """Project LiDAR points to image plane."""
    # Convert to homogeneous coordinates
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    # Apply transformations
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect']
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = calib['Tr_velo_to_cam']
    P2 = calib['P2']
    # Compute projection matrix
    proj_matrix = P2 @ R0_rect @ Tr_velo_to_cam
    # Project points
    points_cam = proj_matrix @ points_hom.T
    points_cam = points_cam.T
    # Normalize
    points_cam[:, 0] /= points_cam[:, 2]
    points_cam[:, 1] /= points_cam[:, 2]
    return points_cam[:, :2]
def associate_points_with_masks(points_img, masks):
    """Associate each point with a mask."""
    associations = []
    for i, point in enumerate(points_img):
        x, y = int(point[0]), int(point[1])
        for idx, mask in enumerate(masks):
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if mask[y, x]:
                    associations.append((i, idx))
                    break
    return associations


def visualize_point_cloud(points, associations, masks_colors):
    """Visualize point cloud with colors based on associations."""
    colors = np.zeros((points.shape[0], 3))
    for idx, mask_idx in associations:
        colors[idx] = masks_colors[mask_idx]
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def visualize_bev(points, associations, masks_colors):
    """Visualize bird's eye view."""
    bev_map = np.zeros((800, 800, 3), dtype=np.uint8)
    for idx, mask_idx in associations:
        x, y = points[idx, 0], points[idx, 1]
        x_img = int((x + 40) * 10)
        y_img = int((y + 40) * 10)
        if 0 <= x_img < 800 and 0 <= y_img < 800:
            bev_map[y_img, x_img] = (masks_colors[mask_idx] * 255).astype(np.uint8)
    plt.imshow(bev_map)
    plt.title("Bird's Eye View")
    plt.show()
def main():
    # Paths
    calib_path = "/home/rishav/Documents/Project/data_object_calib/training/calib/000000.txt"
    image_path = "/home/rishav/Documents/Project/data_object_image_2/training/image_2/000000.png"
    bin_path = "/home/rishav/Documents/Project/data_object_velodyne/training/velodyne/000000.bin"
    # Load data
    calib = load_calibration(calib_path)
    image = load_image(image_path)
    point_cloud = load_point_cloud(bin_path)
    # Load model
    model = YOLO("/home/rishav/Downloads/yolo11m-seg.pt")  # Replace with your model path
    # Perform segmentation
    results = segment_image(model, image)
    masks = results[0].masks.data.cpu().numpy()
    # Generate random colors for masks
    masks_colors = np.random.rand(len(masks), 3)
    # Project LiDAR to image
    points_img = project_lidar_to_image(point_cloud, calib)
    # Associate points with masks
    associations = associate_points_with_masks(points_img, masks)
    # Visualize
    visualize_point_cloud(point_cloud, associations, masks_colors)
    visualize_bev(point_cloud, associations, masks_colors)

if __name__ == "__main__":
    main()
