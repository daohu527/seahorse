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
import math
import time
from .detector import LaneDetectorBase
from .contracts import LaneResults


class SplineLaneDetector(LaneDetectorBase):
    """
    A mock detector that generates plausible curved lane lines for visualization.
    It simulates perspective by making the lanes converge at a vanishing point.
    """

    def _load_model(self) -> torch.nn.Module:
        """No model to load for a mock detector. Returns None."""
        print("INFO: Initializing SplineLaneDetector (no model loaded).")
        return None

    def __call__(self, image: np.ndarray) -> LaneResults:
        """Generates two curved lane lines based on image dimensions."""
        h, w, _ = image.shape

        # Define y-points from the bottom of the screen up to a horizon line
        horizon_y = h * 0.6
        y_points = np.linspace(h, horizon_y, num=20)

        # --- Define lane dynamics ---
        # Vanishing point is near the center, with a slight wobble for realism
        vanishing_point_x = w / 2 + math.sin(time.time() * 0.5) * 50

        # Define the width of the lane at the bottom of the image
        bottom_lane_width = w * 0.8

        # Use a quadratic function to model the curve of the lanes
        # The further up the screen (closer to horizon), the closer to the vanishing point.
        normalized_y = (y_points - horizon_y) / (
            h - horizon_y
        )  # 0 at horizon, 1 at bottom

        # --- Generate left and right lanes ---
        # The offset from the vanishing point is proportional to the square of the normalized y
        half_width_at_y = (bottom_lane_width / 2) * (normalized_y**2)

        left_x = vanishing_point_x - half_width_at_y
        right_x = vanishing_point_x + half_width_at_y

        # Stack the x and y coordinates to create lists of points
        left_lane = np.stack((left_x, y_points), axis=-1).astype(int)
        right_lane = np.stack((right_x, y_points), axis=-1).astype(int)

        return [left_lane, right_lane]
