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

from .detector import TrafficLightDetectorBase
from .mock_detector import MockTrafficLightDetector
from .contracts import TrafficLightResult, TrafficLightResults, BoundingBox


def build_traffic_light_detector(config: Dict[str, Any]) -> TrafficLightDetectorBase:
    """Factory function to build and return a traffic light detector instance."""
    backend = config.get("backend", "mock")  # Default to mock if not specified
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"INFO: Building traffic light detector with backend '{backend}' on device '{device}'..."
    )

    if backend == "mock":
        return MockTrafficLightDetector(device=device)
    # When you have a real model, you'll add its backend here:
    # elif backend == "real_classifier":
    #     return RealTrafficLightDetector(...)
    else:
        raise ValueError(f"Unknown traffic light detector backend: '{backend}'")


__all__ = [
    "build_traffic_light_detector",
    "TrafficLightDetectorBase",
    "TrafficLightResult",
    "TrafficLightResults",
    "BoundingBox",
]
