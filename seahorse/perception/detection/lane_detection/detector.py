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


from abc import ABC, abstractmethod
import numpy as np
import torch
from .contracts import LaneResults


class LaneDetectorBase(ABC):
    """Abstract Base Class for all lane detectors."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """Subclasses must implement this to load their specific model."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, image: np.ndarray) -> LaneResults:
        """The main public method. Takes a numpy image and returns standardized results."""
        raise NotImplementedError
