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
import torch
import random
import time
from .detector import TrafficLightDetectorBase
from .contracts import BoundingBox, TrafficLightResult, TrafficLightResults


class TrafficLightDetector(TrafficLightDetectorBase):
    """
    A mock detector that simulates traffic light detection for visualization.
    It simulates a single traffic light at a fixed position and cycles its state.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.possible_states = ["green", "green", "yellow", "red", "red"]
        self.last_change_time = time.time()
        self.current_state = "green"
        self.cycle_interval = random.uniform(4, 8)  # Change state every 4-8 seconds

    def _load_model(self) -> torch.nn.Module:
        """No model to load for a mock detector. Returns None."""
        print("INFO: Initializing TrafficLightDetector (no model loaded).")
        return None

    def __call__(self, image: np.ndarray) -> TrafficLightResults:
        """
        Simulates finding one traffic light and cycling its state every few seconds.
        """
        h, w, _ = image.shape

        # Check if it's time to cycle the traffic light state
        if time.time() - self.last_change_time > self.cycle_interval:
            self.current_state = random.choice(self.possible_states)
            self.last_change_time = time.time()
            self.cycle_interval = random.uniform(4, 8)
            print(f"INFO: Mock traffic light changed to '{self.current_state.upper()}'")

        # Define a fixed, plausible position for the mock traffic light
        box = BoundingBox(int(w * 0.75), int(h * 0.05), int(w * 0.8), int(h * 0.25))

        return [
            TrafficLightResult(bounding_box=box, state=self.current_state, score=0.99)
        ]
