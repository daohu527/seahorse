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


from typing import Dict, Any
import torch

from .torchvision import TorchvisionDetector
from .yolo import YOLODetector
from .detector import ObjectDetector
from .contracts import BoundingBox, DetectionResult, DetectionResults
from .visualize import draw_detections


def build_object_detector(config: Dict[str, Any]) -> ObjectDetector:
    """
    Factory function to build and return an object detector instance.
    It intelligently routes the config to the correct backend detector.
    """
    backend = config.get("backend")
    if not backend:
        raise ValueError(
            "Detector config must specify a 'backend' ('torchvision' or 'yolo')."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"INFO: Building object detector with backend '{backend}' on device '{device}'..."
    )

    if backend == "torchvision":
        return TorchvisionDetector(
            model_name=config["model_name"],
            weights_name=config["weights_name"],
            score_thresh=config.get("score_thresh", 0.7),
            device=device,
        )
    elif backend == "yolo":
        return YOLODetector(model_path=config["model_path"], device=device)
    else:
        raise ValueError(f"Unknown object detector backend: '{backend}'")


__all__ = [
    "build_object_detector",
    "ObjectDetector",
    "DetectionResult",
    "DetectionResults",
    "BoundingBox",
    "draw_detections",
]
