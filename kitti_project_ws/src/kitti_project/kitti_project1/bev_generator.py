#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
import numpy as np
import matplotlib.pyplot as plt


class BEVGenerator(Node):
    def __init__(self):
        super().__init__('bev_generator')
        self.subscription = self.create_subscription(
            PointCloud2, 'colored_pointcloud', self.pc_callback, 10)
        self.publisher = self.create_publisher(
            MarkerArray, 'bev_markers', 10)

    def pc_callback(self, msg):
        points = self.pc2_to_numpy(msg)

        # Generate BEV
        self.generate_bev(points)

        # Optional: Publish as RViz markers
        markers = self.create_bev_markers(points)
        self.publisher.publish(markers)

    def generate_bev(self, points):
        plt.figure(figsize=(12, 8))
        # Your BEV plotting code here
        plt.show(block=False)
        plt.pause(0.1)

    def create_bev_markers(self, points):
        # Convert BEV to RViz markers
        pass


def main(args=None):
    rclpy.init(args=args)
    node = BEVGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()