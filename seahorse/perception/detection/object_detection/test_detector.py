#!/usr/bin/env python

# Copyright 2025 WheelOS. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created Date: 2025-11-06
# Author: daohu527


import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import Detector


def main():
    parser = argparse.ArgumentParser(description="YOLO TorchScript Detection Example")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.torchscript",
        help="Path to the TorchScript model file (default: yolo11n.torchscript)",
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file to detect"
    )
    args = parser.parse_args()

    model_file = args.model
    image_file = args.image

    # 1. Check if the required files exist
    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        return
    if not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' not found.")
        return

    # 2. Instantiate the detector
    print("Instantiating detector...")
    yolo_detector = Detector(model_path=model_file)
    print("Detector instantiated successfully.")

    # 3. Read image and perform inference
    print(f"Reading image '{image_file}'...")
    pil_image = Image.open(image_file).convert("RGB")
    rgb_numpy_image = np.array(pil_image)
    bgr_numpy_image = rgb_numpy_image[:, :, ::-1].copy()

    print("Performing object detection inference...")
    detections = yolo_detector(bgr_numpy_image)
    print("Inference completed.")

    # 4. Print and visualize results
    print("\n--- Detection Results ---")
    if detections:
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        for det in detections:
            print(det)
            b = det.bounding_box
            color_rgb = tuple(det.color)

            # Draw bounding box
            box_coords = [b.x1, b.y1, b.x2, b.y2]
            draw.rectangle(box_coords, outline=color_rgb, width=3)

            label = f"{det.label}: {det.score:.2f}"

            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Draw text background
            text_bg_coords = [b.x1, b.y1 - text_height - 5, b.x1 + text_width + 4, b.y1]
            draw.rectangle(text_bg_coords, fill=color_rgb)

            # Draw text
            draw.text(
                (b.x1 + 2, b.y1 - text_height - 3), label, fill="white", font=font
            )

        # 5. Save result image
        base_name, ext = os.path.splitext(os.path.basename(image_file))
        output_filename = f"{base_name}_detected{ext}"
        pil_image.save(output_filename)
        print(f"\nResult image saved as '{output_filename}'")
    else:
        print("No objects detected.")


if __name__ == "__main__":
    main()
