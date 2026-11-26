"""
ROS 2 node that turns RoboPet's camera feed into hand-sign commands for fsm_node.

Runs on the car next to the Project_B7 nodes (face_node, body/owner node,
fsm_node). It does not open the camera itself: the RealSense driver already
publishes the colour image, and only one program can hold the camera.

    camera driver --/camera/camera/color/image_raw--> gesture_node --/hand_gesture--> fsm_node

Topics:
    Subscribes: image_topic (sensor_msgs/Image), /camera/camera/color/image_raw by default
    Publishes:  gesture_topic (std_msgs/String), /hand_gesture by default:
                "stop", "come", "spin" or "switch_owner", sent once per new gesture

Parameters:
    image_topic, gesture_topic: as above (defaults match face_node and fsm_node)
    model_path: trained checkpoint (default <repo>/checkpoints/gesture_model.npz)
    every_n: run the detector on every Nth frame to share the Jetson's CPU (default 2)
    threshold, hold: confidence and consecutive frames needed for a command (0.8, 5)

Run on the car (ROS 2 sourced, from any folder):
    python3 scripts/gesture_node.py
    python3 scripts/gesture_node.py --ros-args -p every_n:=3
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from src.inference.fsm_bridge import GESTURE_TO_FSM, FsmCommandFilter  # noqa: E402
from src.inference.pipeline import CONF_THRESHOLD, HOLD_FRAMES, GesturePipeline  # noqa: E402


class GestureNode(Node):
    def __init__(self):
        super().__init__("gesture_node")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("gesture_topic", "/hand_gesture")
        self.declare_parameter("model_path", os.path.join(REPO, "checkpoints", "gesture_model.npz"))
        self.declare_parameter("every_n", 2)
        self.declare_parameter("threshold", CONF_THRESHOLD)
        self.declare_parameter("hold", HOLD_FRAMES)
        get = lambda name: self.get_parameter(name).value  # noqa: E731

        # One pipeline for the whole run so the tracker and hold count carry across frames.
        self.pipeline = GesturePipeline(get("model_path"), get("threshold"), get("hold"))
        if self.pipeline.model_name != "trained model":
            # The finger-count placeholder only knows fist/palm; warn loudly so a
            # missing checkpoint is not mistaken for a working detector.
            self.get_logger().error(f"No checkpoint at {get('model_path')}: using the placeholder")
        self.filter = FsmCommandFilter()
        self.bridge = CvBridge()
        self.every_n = max(1, int(get("every_n")))
        self.frame_count = 0

        self.publisher = self.create_publisher(String, get("gesture_topic"), 10)
        # Sensor QoS matches the camera driver whether it publishes reliable or best-effort.
        self.create_subscription(Image, get("image_topic"), self.on_image, qos_profile_sensor_data)
        self.get_logger().info(
            f"Gesture node started ({self.pipeline.model_name}, every {self.every_n} frames). "
            f"Poses -> actions: {GESTURE_TO_FSM}")

    def on_image(self, msg):
        self.frame_count += 1
        if self.frame_count % self.every_n:
            return
        try:
            # The pipeline expects the same BGR, un-mirrored frame used for training.
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # a bad frame should not stop the node
            self.get_logger().error(f"Failed to convert colour frame: {e}")
            return

        command = self.pipeline.process(frame).command
        action = self.filter.update(command)
        if action is not None:
            self.publisher.publish(String(data=action))
            self.get_logger().info(f"{command} -> '{action}'")


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
