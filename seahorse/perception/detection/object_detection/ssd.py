#!/usr/bin/env python

# Copyright 2023 daohu527 <daohu527@gmail.com>
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

from object_detection import ObjectDetection

import torch
from torchvision.models.detection import (
    ssd300_vgg16,
    SSD300_VGG16_Weights
)


class SSD(ObjectDetection):
    def __init__(self, weights=SSD300_VGG16_Weights.DEFAULT):
        self.model = ssd300_vgg16(weights=weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
