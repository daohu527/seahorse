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

from .yoloe_detector import YOLOESegDetector
from .class_names import YOLOE_LVIS_CLASSES


def main():
    parser = argparse.ArgumentParser(
        description="YOLOE Segmentation Detection (Final Version)"
    )
    parser.add_argument("--model", type=str, default="yoloe-11s-seg-pf.torchscript")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    # 1. Check files
    if not os.path.exists(args.model) or not os.path.exists(args.image):
        print("Error: Model or image file not found.")
        return

    # 2. Instantiate detector
    print("Instantiating YOLOE detector...")
    try:
        yolo_detector = YOLOESegDetector(
            model_path=args.model, class_names=YOLOE_LVIS_CLASSES, conf_thresh=args.conf
        )
    except Exception as e:
        print(f"Failed to instantiate detector: {e}")
        return
    print(
        f"Detector successfully instantiated with {yolo_detector.num_classes} known classes."
    )

    # 3. Read image and perform inference
    print(f"Reading image '{args.image}'...")
    pil_image = Image.open(args.image).convert("RGB")
    # convert to numpy array in BGR format, since detector expects BGR
    rgb_numpy = np.array(pil_image)
    bgr_numpy = rgb_numpy[:, :, ::-1]  # swap channels from RGB to BGR

    print("Running object detection and segmentation inference...")
    detections = yolo_detector(bgr_numpy)
    print("Inference completed.")

    # 4. Visualize results
    print("\n--- Detection Results ---")
    if detections:
        # go back to RGB for visualization
        viz_numpy = rgb_numpy.copy()
        mask_overlay = viz_numpy.astype(np.float32)
        alpha = 0.4
        for det in detections:
            if det.mask is not None:
                color_bgr = np.array(det.color, dtype=np.uint8)
                color_rgb = color_bgr[::-1]
                bool_mask = det.mask.astype(bool)
                mask_overlay[bool_mask] = (1 - alpha) * mask_overlay[
                    bool_mask
                ] + alpha * color_rgb

        viz_numpy = np.round(mask_overlay).astype(np.uint8)
        pil_image = Image.fromarray(viz_numpy)
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        for det in detections:
            print(f"- Detected: {det.label} (Confidence: {det.score:.2f})")
            b = det.bounding_box
            color_rgb_tuple = tuple(det.color[::-1])
            box_coords = [b.x_min, b.y_min, b.x_max, b.y_max]
            draw.rectangle(box_coords, outline=color_rgb_tuple, width=3)
            label = f"{det.label}: {det.score:.2f}"
            text_bbox = draw.textbbox((b.x_min, b.y_min), label, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            text_pos = (b.x_min, b.y_min - text_height - 2)
            draw.rectangle(
                [text_pos, (text_pos[0] + text_bbox[2], text_pos[1] + text_height)],
                fill=color_rgb_tuple,
            )
            draw.text(text_pos, label, fill="white", font=font)

        # 5. Save result
        output_filename = f"{os.path.splitext(args.image)[0]}_detected.png"
        pil_image.save(output_filename)
        print(f"\nResult image saved as '{output_filename}'")
    else:
        print("No objects detected.")


if __name__ == "__main__":
    main()
