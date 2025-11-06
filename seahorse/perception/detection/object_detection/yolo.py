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


import torch
import numpy as np
from ultralytics import YOLO

from .detector import ObjectDetector
from .contracts import BoundingBox, DetectionResult, DetectionResults


class YOLODetector(ObjectDetector):
    """A concrete detector for YOLO models via the 'ultralytics' library."""

    def __init__(self, model_path: str, device: torch.device = torch.device("cuda")):
        self.model_path = model_path
        super().__init__(device)

    def _load_model(self) -> YOLO:
        """Loads a YOLO model from a given path (e.g., 'yolov8n.pt')."""
        print(f"INFO: Loading ultralytics YOLO model '{self.model_path}'...")
        return YOLO(self.model_path)

    def __call__(self, image: np.ndarray, verbose: bool = False) -> DetectionResults:
        """Overrides the base call to use the library's optimized `predict` method."""
        # The ultralytics library handles preprocessing, inference, and NMS internally.
        # It expects BGR numpy arrays.
        results = self.model.predict(image, device=self.device, verbose=verbose)

        processed_results = []
        for res in results:
            boxes = res.boxes
            names = res.names
            for box in boxes:
                (x1, y1, x2, y2) = box.xyxy[0].int().tolist()
                class_id = int(box.cls)
                processed_results.append(
                    DetectionResult(
                        bounding_box=BoundingBox(x1, y1, x2, y2),
                        class_id=class_id,
                        label=names[class_id],
                        score=float(box.conf),
                    )
                )
        return processed_results
