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


import time
import cv2
from typing import Dict

from io.source import build_source
from detection import build_object_detector
from visualizer import Visualizer


class Runner:
    """The task flow orchestrator is responsible for the entire perception-visualization process."""

    def __init__(self, config: Dict):
        self.cfg = config

        print("INFO: Initializing system components...")
        # Runner responsible for building all components
        self.source = build_source(config["source"])
        self.visualizer = Visualizer(config["visualize"])

        # Detectors are built using factory functions and are entirely configuration-driven.
        self.object_detector = build_object_detector(
            config["models"]["object_detector"]
        )

        print("INFO: System initialized successfully.")

    def run(self):
        """
        This is the correct location of the loop logic in the original object_detection() function.
        """
        prev_time = time.time()

        # 1. Retrieve data from the data source
        for frame in self.source:
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            results = {"fps": fps}

            # 2. Call the model for inference (using __call__).
            detection_results = self.object_detector(frame)
            results["objects"] = detection_results

            # 3. The results are then visualized using a visualizer.
            if self.cfg["visualize"]["enable"]:
                vis_frame = self.visualizer.draw(frame, results)
                cv2.imshow(self.cfg["visualize"]["window_name"], vis_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.cleanup()

    def cleanup(self):
        pass
