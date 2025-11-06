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
import torchvision.models.detection as tv_models
from torchvision import transforms
from typing import List, Dict

from .detector import ObjectDetector
from .contracts import BoundingBox, DetectionResult, DetectionResults


class TorchvisionDetector(ObjectDetector):
    """A unified detector for all models in torchvision.models.detection."""

    def __init__(
        self,
        model_name: str,
        weights_name: str,
        score_thresh: float = 0.7,
        device: torch.device = torch.device("cuda"),
    ):
        self.model_name = model_name
        self.weights_name = weights_name
        self.score_thresh = score_thresh
        super().__init__(device)  # This calls _load_model()
        self.class_names = self.weights.meta["categories"]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _load_model(self) -> torch.nn.Module:
        """Dynamically loads any model and its weights from torchvision."""
        print(f"INFO: Loading torchvision model '{self.model_name}'...")
        model_class = getattr(tv_models, self.model_name)
        self.weights = getattr(tv_models, self.weights_name).DEFAULT

        model = model_class(weights=self.weights, box_score_thresh=self.score_thresh)
        return model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> DetectionResults:
        """Implements the standard 'preprocess -> infer -> postprocess' pipeline."""
        # 1. Preprocess
        image_rgb = image[:, :, ::-1]
        input_tensor = [self.transform(image_rgb).to(self.device)]

        # 2. Inference
        model_outputs = self.model(input_tensor)

        # 3. Postprocess
        output = model_outputs[0]
        boxes, labels, scores = (
            output["boxes"].cpu(),
            output["labels"].cpu(),
            output["scores"].cpu(),
        )

        return [
            DetectionResult(
                bounding_box=BoundingBox(int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                class_id=int(l),
                label=self.class_names[l],
                score=float(s),
            )
            for b, l, s in zip(boxes, labels, scores)
        ]
