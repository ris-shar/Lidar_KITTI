import cv2, os, numpy as np, open3d as o3d, matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# ========== CONFIG ==========
root = "/home/rishav/Documents/Project"
frame   = "000250"
split   = "training"          # or "training"
image_path  = f"{root}/data_object_image_2/{split}/image_2/{frame}.png"
lidar_path  = f"{root}/data_object_velodyne/{split}/velodyne/{frame}.bin"
calib_path  = f"{root}/data_object_calib/{split}/calib/{frame}.txt"
model = YOLO("/home/rishav/Downloads/yolo11m-seg.pt")   # segmentation model

# ========== CALIBRATION ==========
def read_calib(fn):
    vals = [l.split()[1:] for l in open(fn)]
    P2 = np.array(vals[2], float).reshape(3,4)
    R0 = np.eye(4); R0[:3,:3] = np.array(vals[4], float).reshape(3,3)
    Tr  = np.eye(4); Tr[:3,:] = np.array(vals[5], float).reshape(3,4)
    return P2, R0, Tr

# -------- LiDAR helpers ----------
def load_lidar(path):              # (N,3)
    return np.fromfile(path, np.float32).reshape(-1,4)[:,:3]

def to_cam(points, Tr, R0):        # LiDAR→camera
    p = np.hstack((points, np.ones((len(points),1))))
    return (R0 @ Tr @ p.T).T[:,:3]

def project_cam(points_cam, P2):   # camera→image
    p = np.hstack((points_cam, np.ones((len(points_cam),1))))
    im = (P2 @ p.T).T
    return im[:,:2] / im[:,2:3]

# -------- YOLO segmentation ------
def car_masks(img):
    res = model(img)[0]
    car_id = [i for i,n in model.names.items() if n=='car']
    ms=[]
    for i,cls in enumerate(res.boxes.cls):
        if int(cls)==car_id[0] and res.masks is not None:
            ms.append(res.masks.data[i].cpu().numpy())
    return ms

# -------- associate 3-D points with masks ----
def associate(points_img, masks, shape, points_cam=None):
    h, w = shape[:2]
    masks = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
    assoc = -np.ones(len(points_img), int)

    # Optional: get mask centers in image
    mask_centers = [np.argwhere(m).mean(axis=0)[::-1] for m in masks]  # (u, v) center

    for idx, (u, v) in enumerate(points_img.astype(int)):
        if not (0 <= u < w and 0 <= v < h):
            continue

        for mid, m in enumerate(masks):
            if not m[v, u]:
                continue

            # (Optional) Distance to mask center – skip if far from center
            center_u, center_v = mask_centers[mid]
            if np.hypot(u - center_u, v - center_v) > 50:  # threshold in pixels
                continue

            # (Optional) Depth filtering – remove far points
            if points_cam is not None:
                depth = points_cam[idx][2]  # Z in camera
                if depth > 60 or depth < 1:  # too far or too close
                    continue

            assoc[idx] = mid
            break

    return assoc, masks


# -------- DBSCAN + bbox ----------
import numpy as np
from sklearn.cluster import DBSCAN


def largest_cluster(pts, eps=0.5, min_samples=10):
    """Returns empty array if no cluster found, always returns 2D array"""
    if len(pts) == 0:
        return np.empty((0, 3))  # Return empty 2D array

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_

    # Handle case where all points are noise
    if np.all(labels == -1):
        return np.empty((0, 3))

    # Get largest cluster (excluding noise)
    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        return np.empty((0, 3))

    largest_label = np.argmax(np.bincount(valid_labels))
    cluster_pts = pts[labels == largest_label]

    # Ensure 2D output even for single point
    return cluster_pts.reshape(-1, 3)


def aabb_corners(pts):
    mn,mx=pts.min(0),pts.max(0)
    return np.array([[mn[0],mn[1],mn[2]],
                     [mx[0],mn[1],mn[2]],
                     [mx[0],mx[1],mn[2]],
                     [mn[0],mx[1],mn[2]],
                     [mn[0],mn[1],mx[2]],
                     [mx[0],mn[1],mx[2]],
                     [mx[0],mx[1],mx[2]],
                     [mn[0],mx[1],mx[2]]])

# ========== MAIN ==========
img   = cv2.imread(image_path)
P2,R0,Tr = read_calib(calib_path)
lidar = load_lidar(lidar_path)

# 1) LiDAR→camera for projection
cam   = to_cam(lidar,Tr,R0)
proj2d= project_cam(cam,P2)

masks = car_masks(img)
assoc, _ = associate(proj2d, masks, img.shape, cam)


# -------- build color map ----------
ids = [i for i in np.unique(assoc) if i>=0]
colors = {i:np.random.rand(3) for i in ids}

# -------- Open3D visualization -----
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar))
col = np.tile([0.5,0.5,0.5], (len(lidar),1))
for i in ids:
    col[assoc==i]=colors[i]
pcd.colors=o3d.utility.Vector3dVector(col)

boxes = []
for i in ids:
    pts = lidar[assoc == i]
    if pts.shape[0] == 0:
        continue  # Skip if no points for this object

    pts = largest_cluster(pts)

    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 20:
        continue  # Skip if result isn't a valid Nx3 array

    # Optional: visualize clustered points in matplotlib (only if needed)
    # ax.scatter(pts[:, 1], pts[:, 0], s=3, c=[c])  # only if `pts` is valid

    cor = aabb_corners(pts)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cor),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
    boxes.append(ls)

o3d.visualization.draw_geometries([pcd, *boxes])

def add_coordinate_axes(vis, size=5.0):
    """Add XYZ axes to Open3D visualization"""
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([
        [0,0,0], [size,0,0],  # X (red)
        [0,0,0], [0,size,0],   # Y (green)
        [0,0,0], [0,0,size]    # Z (blue)
    ])
    axes.lines = o3d.utility.Vector2iVector([[0,1],[2,3],[4,5]])
    axes.colors = o3d.utility.Vector3dVector([
        [1,0,0], [0,1,0], [0,0,1]
    ])
    vis.add_geometry(axes)
def visualize_bev(points, associations, colors):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)

    # Transform coordinates for proper BEV:
    # X (forward) → vertical axis (up in plot)
    # Y (left) → horizontal axis (right in plot)
    x = points[:, 0]  # Forward direction
    y = -points[:, 1]  # Left/right direction (negative for proper orientation)

    # Plot background points
    bg_mask = (associations < 0)
    ax.scatter(y[bg_mask], x[bg_mask], s=1, c='lightgray', alpha=0.3, label='Background')

    # Plot each car instance
    for car_id in np.unique(associations):
        if car_id < 0:
            continue

        car_mask = (associations == car_id)
        car_points = points[car_mask]

        if len(car_points) < 10:  # Minimum points threshold
            continue

        # Get cluster points
        cluster = largest_cluster(car_points)
        if len(cluster) == 0:
            continue

        color = colors[car_id]

        # Transform cluster coordinates
        x_cluster = cluster[:, 0]
        y_cluster = -cluster[:, 1]

        # Plot points
        ax.scatter(y_cluster, x_cluster, s=3, c=[color], label=f'Car {car_id}')

        # Calculate bounding box
        min_y, min_x = np.min(y_cluster), np.min(x_cluster)
        max_y, max_x = np.max(y_cluster), np.max(x_cluster)

        # Draw bounding box
        rect = plt.Rectangle(
            (min_y, min_x),
            max_y - min_y,
            max_x - min_x,
            linewidth=1.5,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

    # Configure plot
    ax.set_xlabel('Right ← Y → Left')
    ax.set_ylabel('Forward (X) ↑')
    ax.set_title("Bird's Eye View")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()
# -------- BEV ----------
# plt.figure(figsize=(12, 6))
# ax = plt.subplot(111)
# bg = assoc < 0
# ax.scatter(lidar[bg, 1], lidar[bg, 0], s=1, c='lightgray', alpha=.3)
#
# for i in ids:
#     pts = largest_cluster(lidar[assoc == i])
#     c = colors[i]
#
#     # Skip if no valid points or wrong shape
#     if len(pts) == 0 or pts.ndim != 2 or pts.shape[1] != 3:
#         continue
#
#     ax.scatter(pts[:, 1], pts[:, 0], s=3, c=[c])
#     mn, mx = pts.min(0), pts.max(0)
#     ax.add_patch(plt.Rectangle(
#         (mn[1], mn[0]),
#         mx[1] - mn[1],
#         mx[0] - mn[0],
#         ec=c, fc='none', lw=1.5
#     ))
# -------- Segmentation overlay ----------
overlay = img.copy()
h, w = img.shape[:2]

for i in ids:
    pts = lidar[assoc == i]
    if pts.shape[0] == 0:
        continue  # Skip if no points for this object

    pts = largest_cluster(pts)

    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 20:
        continue  # Skip if result isn't a valid Nx3 array

    # Compute distances from sensor origin to each point
    distances = np.linalg.norm(pts, axis=1)
    min_distance = np.min(distances)
    print(f"Car ID {i}: Closest point at {min_distance:.2f} meters")


# Load ground truth annotations
label_path = f"{root}/data_object_label_2/{split}/label_2/{frame}.txt"
with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if parts[0] != 'Car':
        continue  # Skip non-car objects

    # Extract object location in camera coordinates
    x, y, z = map(float, parts[11:14])
    distance = np.sqrt(x**2 + y**2 + z**2)
    print(f"Ground truth car at {distance:.2f} meters")



for m, c in zip(masks, colors.values()):
    # Resize each mask to the original image size
    m_resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Overlay the mask color onto the image
    overlay[m_resized.astype(bool)] = (np.array(c) * 255).astype(np.uint8)

# Optional: blended visualization
ids = [i for i in np.unique(assoc) if i >= 0]
colors = {i: np.random.rand(3) for i in ids}
#calling the vidulaization
visualize_bev(lidar, assoc, colors)
cv2.imshow("Segmentation Overlay", cv2.addWeighted(img, 0.5, overlay, 0.5, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()

