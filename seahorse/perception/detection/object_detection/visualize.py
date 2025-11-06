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


import cv2
import numpy as np
from .contracts import DetectionResults

COLOR_PALETTE = [
    (255, 140, 0),
    (220, 50, 220),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]


def draw_detections(image: np.ndarray, detections: DetectionResults) -> np.ndarray:
    """Draws detection results onto an image."""
    vis_image = image.copy()
    for detection in detections:
        box = detection.bounding_box
        color = COLOR_PALETTE[detection.class_id % len(COLOR_PALETTE)]
        cv2.rectangle(vis_image, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        label = f"{detection.label} {detection.score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            vis_image, (box.x1, box.y1 - h - 5), (box.x1 + w, box.y1), color, cv2.FILLED
        )
        cv2.putText(
            vis_image,
            label,
            (box.x1, box.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
    return vis_image
