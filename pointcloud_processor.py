#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from your_package.msg import SegmentationResult
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge


class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('pointcloud_processor')
        self.sub_seg = self.create_subscription(
            SegmentationResult, 'segmentation_results', self.seg_callback, 10)
        self.sub_pc = self.create_subscription(
            PointCloud2, 'lidar/points_raw', self.pc_callback, 10)
        self.publisher = self.create_publisher(
            PointCloud2, 'colored_pointcloud', 10)
        self.bridge = CvBridge()
        self.current_masks = None

    def seg_callback(self, msg):
        self.current_masks = [self.bridge.imgmsg_to_cv2(m, "mono8")
                              for m in msg.masks]

    def pc_callback(self, msg):
        if self.current_masks is None:
            return

        # Convert PointCloud2 to numpy array
        points = self.pc2_to_numpy(msg)

        # Your point cloud processing logic
        colored_pc = self.color_pointcloud(points)

        # Publish colored point cloud
        self.publisher.publish(self.numpy_to_pc2(colored_pc))

        # Optional: Open3D visualization
        self.show_open3d(colored_pc)

    def color_pointcloud(self, points):
        # Your coloring logic here
        pass

    def show_open3d(self, points):
        # Your Open3D visualization code
        pass


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()