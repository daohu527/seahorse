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


@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class DetectionResult:
    def __init__(self, bounding_box, class_id, label, score, color, mask=None):
        self.bounding_box = bounding_box
        self.class_id = class_id
        self.label = label
        self.score = float(score)
        self.color = color
        self.mask = mask


DetectionResults = List[DetectionResult]
