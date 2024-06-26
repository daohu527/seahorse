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

import torch

from segmentation import Segmentation

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)


class MaskRCNN(Segmentation):
    def __init__(self, weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1):
        self.model = maskrcnn_resnet50_fpn(weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
