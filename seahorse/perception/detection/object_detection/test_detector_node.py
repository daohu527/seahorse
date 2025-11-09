import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import Detector

from cyber.python.cyber_py3 import cyber
from modules.common_msgs.sensor_msgs.sensor_image_pb2 import Image as PbImage


def process_and_draw_detections(detector, bgr_numpy_image):
    """
    Perform object detection on a single BGR NumPy image and return a PIL image with results drawn.

    Args:
        detector: An instantiated Detector object.
        bgr_numpy_image: A NumPy array image in BGR format.

    Returns:
        A PIL.Image object with detections drawn on it.
    """
    print("Performing object detection inference...")
    detections = detector(bgr_numpy_image)
    print("Inference completed.")

    # Convert BGR NumPy image back to RGB PIL image for drawing
    rgb_numpy_image = bgr_numpy_image[:, :, ::-1].copy()
    pil_image = Image.fromarray(rgb_numpy_image)

    if not detections:
        print("No objects detected.")
        return pil_image

    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    print(f"Detected {len(detections)} objects.")
    for det in detections:
        print(det)
        b = det.bounding_box
        color_rgb = tuple(det.color)

        # Draw bounding box
        box_coords = [b.x1, b.y1, b.x2, b.y2]
        draw.rectangle(box_coords, outline=color_rgb, width=3)

        # Prepare label text
        label = f"{det.label}: {det.score:.2f}"
        text_bbox = draw.textbbox((b.x1, b.y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw text background
        text_bg_coords = [b.x1, b.y1 - text_height - 5, b.x1 + text_width + 4, b.y1]
        draw.rectangle(text_bg_coords, fill=color_rgb)

        # Draw text
        draw.text((b.x1 + 2, b.y1 - text_height - 3), label, fill="white", font=font)

    return pil_image


class YoloCyberNode:
    """
    A Cyber node encapsulating YOLO detection.
    Subscribes to an image topic, performs detection, and publishes results to another topic.
    """

    def __init__(self, model_path, input_channel, output_channel):
        """
        Initialize node, detector, publisher, and subscriber.
        """
        cyber.init()
        self.node = cyber.Node("yolo_detection_node")

        print("Instantiating detector...")
        self.detector = Detector(model_path=model_path)
        print("Detector instantiated successfully.")

        self.output_channel = output_channel
        self.input_channel = input_channel

        # Create publisher for processed images
        self.publisher = self.node.create_writer(self.output_channel, PbImage)

        # Create subscriber and bind callback function
        self.node.create_reader(self.input_channel, PbImage, self.image_callback)

        print(f"Subscribed to image topic: '{self.input_channel}'")
        print(f"Results will be published to topic: '{self.output_channel}'")

    def image_callback(self, image_msg):
        """
        Callback function to handle incoming image messages.
        """
        print("-" * 80)
        print(
            f"Received image message from topic '{self.input_channel}', timestamp: {image_msg.header.timestamp_sec}"
        )

        # 1. Parse Cyber image message (from bytes to NumPy array)
        # Assume 3 channels (RGB)
        # Use np.frombuffer instead of deprecated np.fromstring
        if image_msg.encoding == "rgb8" or image_msg.encoding == "bgr8":
            channel_num = 3
        else:  # 'gray', 'y', etc.
            channel_num = 1

        np_array = np.frombuffer(image_msg.data, dtype=np.uint8)
        image_reshaped = np_array.reshape(
            (image_msg.height, image_msg.width, channel_num)
        )

        # 2. Convert image from RGB to BGR since the detector expects BGR format
        if image_msg.encoding == "rgb8":
            bgr_image = image_reshaped[:, :, ::-1].copy()
        else:  # Assume it's already BGR or grayscale
            bgr_image = image_reshaped

        # 3. Perform detection and draw results
        result_pil_image = process_and_draw_detections(self.detector, bgr_image)

        # 4. Build Cyber image message for publishing
        result_np_array = np.array(result_pil_image)

        output_msg = PbImage()
        output_msg.header.CopyFrom(image_msg.header)  # Keep timestamp and frame_id
        output_msg.frame_id = image_msg.frame_id
        output_msg.measurement_time = image_msg.measurement_time
        output_msg.encoding = "rgb8"
        output_msg.width = result_np_array.shape[1]
        output_msg.height = result_np_array.shape[0]
        output_msg.data = result_np_array.tobytes()
        output_msg.step = output_msg.width * 3  # 3 bytes per pixel (RGB)

        # 5. Publish results
        self.publisher.write(output_msg)
        print(f"Published processed result to topic '{self.output_channel}'.")

    def spin(self):
        """
        Start the node and keep it running, allowing graceful shutdown.
        """
        print("\nCyber node started. Press Ctrl+C to exit.")
        try:
            self.node.spin()
        except KeyboardInterrupt:
            print("User interrupt detected. Shutting down...")
        finally:
            print("Shutting down Cyber node...")
            cyber.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="YOLO TorchScript detection demo (File and Cyber modes)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5s.torchscript",
        help="Path to TorchScript model file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["file", "cyber"],
        required=True,
        help="Run mode: 'file' reads from local file, 'cyber' subscribes via Cyber topic",
    )

    # File mode arguments
    parser.add_argument(
        "--image", type=str, help="[File mode] Path to input image file"
    )

    # Cyber mode arguments
    parser.add_argument(
        "--input_channel",
        type=str,
        default="/apollo/sensor/camera/front_6mm/image",
        help="[Cyber mode] Input image Cyber topic",
    )
    parser.add_argument(
        "--output_channel",
        type=str,
        default="/apollo/perception/yolo_image_detections",
        help="[Cyber mode] Output Cyber topic for images with detections",
    )

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model) and args.mode == "file":
        print(f"Warning: Model file '{args.model}' not found.")

    if args.mode == "file":
        # --- File mode ---
        if not args.image:
            print("Error: '--image' argument is required in 'file' mode.")
            return
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            return

        print("Running in [File mode]...")

        # Instantiate detector
        print("Instantiating detector...")
        yolo_detector = Detector(model_path=args.model)
        print("Detector instantiated successfully.")

        # Read image
        print(f"Reading image '{args.image}'...")
        pil_image = Image.open(args.image).convert("RGB")
        rgb_numpy_image = np.array(pil_image)
        bgr_numpy_image = rgb_numpy_image[:, :, ::-1].copy()

        # Perform detection and drawing
        pil_result_image = process_and_draw_detections(yolo_detector, bgr_numpy_image)

        # Save result image
        base_name, ext = os.path.splitext(os.path.basename(args.image))
        output_filename = f"{base_name}_detected{ext}"
        pil_result_image.save(output_filename)
        print(f"\nResult image saved as '{output_filename}'")

    elif args.mode == "cyber":
        print("Running in [Cyber mode]...")
        node = YoloCyberNode(
            model_path=args.model,
            input_channel=args.input_channel,
            output_channel=args.output_channel,
        )
        node.spin()


if __name__ == "__main__":
    main()
