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


from typing import List
import numpy as np

# A lane line is represented as a numpy array of [x, y] points.
# Using np.ndarray is highly efficient for numerical operations.
LaneLine = np.ndarray  # Shape will be (N, 2) where N is number of points

# A list of all detected lane lines in a frame.
LaneResults = List[LaneLine]
