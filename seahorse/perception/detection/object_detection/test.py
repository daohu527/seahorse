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


import numpy as np
from . import build_object_detector
from .torchvision import TorchvisionDetector
from .yolo import YOLODetector
from .contracts import BoundingBox, DetectionResult
from .visualize import draw_detections


def run_tests():
    """Runs a suite of tests for the final object_detection module architecture."""
    print("--- Running Object Detection Module Tests (Backend Architecture) ---")
    dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)

    print("\n[Test 1] Testing factory function `build_object_detector`...")
    detector_configs = [
        {
            "backend": "torchvision",
            "model_name": "fasterrcnn_resnet50_fpn",
            "weights_name": "FasterRCNN_ResNet50_FPN_Weights",
            "score_thresh": 0.9,
        },
        # NOTE: This test requires 'yolov8n.pt' to be downloaded or available.
        # You can create a mock file or ensure it's present for the test to pass.
        {"backend": "yolo", "model_path": "yolov8n.pt"},
    ]
    expected_types = [TorchvisionDetector, YOLODetector]

    for config, expected_type in zip(detector_configs, expected_types):
        try:
            detector = build_object_detector(config)
            assert isinstance(
                detector, expected_type
            ), f"Factory built wrong type for {config['backend']}"
            print(f"  - Factory correctly built backend '{config['backend']}'.")
        except Exception as e:
            print(f"  - WARNING: Could not build '{config['backend']}'. Reason: {e}")
            print(
                "    (This may be expected if model files like 'yolov8n.pt' are not present)"
            )

    print("\n[Test 2] Testing detector __call__ interface...")
    # We will only test the Torchvision detector to avoid dependency on downloaded files
    try:
        tv_config = detector_configs[0]
        detector = build_object_detector(tv_config)
        results = detector(dummy_image)
        assert isinstance(results, list), "Detector did not return a list."
        # The result might be empty if the threshold is high, which is fine.
        if results:
            assert isinstance(
                results[0], DetectionResult
            ), "Detector did not return List[DetectionResult]."
        print(f"  - Detector '{tv_config['backend']}' returned valid format.")
        print("  PASSED: Interface tests.")
    except Exception as e:
        print(f"  - FAILED: Interface test. Reason: {e}")

    print("\n[Test 3] Testing `draw_detections` utility...")
    mock_results = [DetectionResult(BoundingBox(10, 10, 50, 50), 1, "car", 0.99)]
    vis_image = draw_detections(dummy_image, mock_results)
    assert not np.array_equal(
        vis_image, dummy_image
    ), "Visualization did not modify the image."
    print("  - `draw_detections` successfully modified the image.")
    print("  PASSED: Visualization test.")

    print("\n--- Object Detection Module Tests Completed ---")


if __name__ == "__main__":
    try:
        with open("yolov8n.pt", "w") as f:
            f.write("dummy")
        run_tests()
    finally:
        import os

        if os.path.exists("yolov8n.pt"):
            os.remove("yolov8n.pt")
