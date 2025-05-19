#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from kitti_project.msg import SegmentationResult  # Custom message


class SemanticSegmenter(Node):
    def __init__(self):
        super().__init__('semantic_segmenter')
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(
            SegmentationResult, 'segmentation_results', 10)
        self.bridge = CvBridge()
        self.model = YOLO("yolov8s-seg.pt")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image)[0]

        # Publish segmentation results
        seg_msg = SegmentationResult()
        seg_msg.header = msg.header
        seg_msg.masks = [self.bridge.cv2_to_imgmsg(mask, "mono8")
                         for mask in self.extract_car_masks(results)]
        self.publisher.publish(seg_msg)

        # Optional: Show segmented image
        self.show_segmentation(cv_image, results)

    def extract_car_masks(self, results):
        car_id = [i for i, name in results.names.items() if name == 'car']
        return [results.masks.data[i].cpu().numpy()
                for i, cls in enumerate(results.boxes.cls)
                if int(cls) in car_id]

    def show_segmentation(self, image, results):
        # Your visualization code here
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmenter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()