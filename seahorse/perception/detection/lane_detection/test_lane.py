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
import numpy as np
from PIL import Image, ImageDraw

from .spline_detector import SplineLaneDetector

if __name__ == "__main__":
    model_file = "deeplabv3_resnet101_scripted.pt"
    image_file = "image.png"

    # 1. Check if the required file exists.
    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        print(
            "Please run the export script first to convert the .pt file to TorchScript format."
        )
    elif not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' not found.")
        print(
            "Please download the zidane.jpg file: wget https://ultralytics.com/images/zidane.jpg"
        )
    else:
        # 2. Instantiate Detector
        print("Instantiating detector...")
        lane_detector = SplineLaneDetector(model_path=model_file)
        print("Detector instantiated successfully.")

        # 3. Load Image
        print(f"Loading image '{image_file}'...")
        pil_image = Image.open(image_file).convert("RGB")
        # The detector requires a BGR NumPy array
        bgr_numpy_image = np.array(pil_image)[:, :, ::-1]

        # --- 5. Perform lane detection ---
        print("Performing lane detection...")
        lanes = lane_detector(bgr_numpy_image)
        print(
            f"Detection completed. {len(lanes)} lane/road contours extracted successfully!"
        )

        # --- 6. Visualize results ---
        print("Generating final visualization...")
        viz_image = pil_image.copy()
        draw = ImageDraw.Draw(viz_image)

        if lanes:
            for lane_contour in lanes:
                # Convert (N, 2) contour points to format required by ImageDraw.line
                points_list = lane_contour.flatten().tolist()
                draw.line(points_list, fill=(255, 0, 255), width=4)  # Draw in magenta
        else:
            print("Warning: No lanes detected in the image.")

        # --- 7. Save the final result ---
        output_filename = "torchvision_lane_output.png"
        viz_image.save(output_filename)
        print(f"\nLane detection result successfully saved as '{output_filename}'")
        print(
            "Please check the output image — you should now see the road contours detected by DeepLabV3."
        )
