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


from dataclasses import dataclass
from typing import List


# It's good practice to define this here, even if it's identical
# to the one in object_detection, to keep the module self-contained.
@dataclass(frozen=True)
class BoundingBox:
    """Defines an immutable bounding box (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class TrafficLightResult:
    """Defines a standard, immutable traffic light detection result."""

    bounding_box: BoundingBox
    state: str  # e.g., 'red', 'yellow', 'green', 'unknown'
    score: float


TrafficLightResults = List[TrafficLightResult]
