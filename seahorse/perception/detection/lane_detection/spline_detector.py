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

from .detector import LaneDetectorBase
from .contracts import LaneResults


class SplineLaneDetector(LaneDetectorBase):
    def __init__(self, model_path: str, device=None):
        """
        :param model_path: The path to the TorchScript model file (.pt).
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._load_model(model_path)

        self.ROAD_CLASS_ID = 0
        self.MIN_LANE_CONTOUR_LENGTH = 50

    def _load_model(self, model_path: str):
        """Load TorchScript models from local storage."""
        print(f"INFO: Loading TorchScript model from: {model_path}")

        model = torch.jit.load(model_path, map_location=self.device)

        model.eval()
        print("INFO: TorchScript model loaded successfully.")
        return model

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepare images for the TorchVision model: convert to tensors and normalize.
        """
        # BGR (NumPy format) -> RGB
        rgb_image = image[:, :, ::-1].copy()

        # 1. Convert the NumPy array (H, W, C) [0, 255] to a PyTorch tensor (C, H, W) [0.0, 1.0].
        tensor = torch.from_numpy(rgb_image.astype(np.float32)).permute(2, 0, 1) / 255.0

        # 2. Normalization was performed using the mean and standard deviation from ImageNet.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        normalized_tensor = (tensor - mean) / std

        # 3. Add batch dimension (C, H, W) -> (N, C, H, W)
        return normalized_tensor.unsqueeze(0)

    def _find_contours_native(self, binary_mask: np.ndarray) -> list[np.ndarray]:
        """
        A native implementation of the Moore-Neighbor Tracing algorithm.
        This function finds the outer boundary of white regions in a binary image.
        """
        contours = []
        # Add 1 pixel of padding to properly handle the outline on the boundary.
        padded_mask = np.pad(
            binary_mask, pad_width=1, mode="constant", constant_values=0
        )

        # Define 8 neighborhood directions in (y, x) format, starting from west and clockwise.
        neighbors = [
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
        ]

        # Scan pixels to find the starting point of the contour
        for r in range(padded_mask.shape[0]):
            for c in range(padded_mask.shape[1]):
                # The starting point is a white pixel, and its left side is a black pixel (make sure it is the outer contour).
                if padded_mask[r, c] == 255 and padded_mask[r, c - 1] == 0:
                    contour = []
                    start_pos = (r, c)
                    current_pos = start_pos
                    # We enter from the black pixel on the left, so the first neighbor in the counter-clockwise direction is the starting point of the search.
                    search_dir_start = (
                        7  # Start searching from the top left (NW) direction
                    )

                    while True:
                        found_next = False
                        # Starting from the initial search direction, iterate through 8 neighbors clockwise.
                        for i in range(8):
                            check_dir = (search_dir_start + i) % 8
                            dr, dc = neighbors[check_dir]
                            next_pos = (current_pos[0] + dr, current_pos[1] + dc)

                            if padded_mask[next_pos] == 255:
                                # Next boundary point found
                                # Update the starting direction for the next search
                                search_dir_start = (
                                    check_dir + 5
                                ) % 8  # Avoid returning immediately
                                current_pos = next_pos
                                found_next = True
                                break

                        if not found_next:
                            # Isolated pixels
                            contour.append(start_pos)
                            break

                        # If you return to the starting point, the outline is closed.
                        if current_pos == start_pos:
                            break
                        else:
                            contour.append(current_pos)

                    if contour:
                        # Remove padding offsets and convert (r, c) to (x, y) format
                        final_contour = np.array(contour)[:, [1, 0]] - 1
                        contours.append(final_contour.astype(np.int32))
        return contours

    def _postprocess(self, model_output: torch.Tensor) -> LaneResults:
        """
        Extract lane line contours from the segmentation results of the model.
        """
        class_predictions = torch.argmax(model_output.squeeze(), dim=0).cpu().numpy()
        road_mask = np.uint8(class_predictions == self.ROAD_CLASS_ID) * 255

        contours = self._find_contours_native(road_mask)

        lane_lines = []
        for contour in contours:
            if len(contour) > self.MIN_LANE_CONTOUR_LENGTH:
                lane_lines.append(contour)

        return lane_lines

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> LaneResults:
        """
        Implement a complete "preprocess -> infer -> postprocess" perception process.
        """
        input_tensor = self._preprocess(image).to(self.device)
        output = self.model(input_tensor)["out"]
        lanes = self._postprocess(output)
        return lanes
