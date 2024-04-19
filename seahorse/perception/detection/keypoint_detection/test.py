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

from keypoint_rcnn import KeypointRCNN
from seahorse.perception.visualize.visualize import show

from torchvision.io import read_image
from torchvision.utils import draw_keypoints
from torchvision.transforms.functional import convert_image_dtype


def test_keypoint(img_file):
    keypoint_rcnn = KeypointRCNN()
    img = read_image(img_file)
    print(img.shape)
    outputs = keypoint_rcnn.detect([convert_image_dtype(img)])

    print(outputs)
    kpts = outputs[0]['keypoints']
    scores = outputs[0]['scores']
    detect_threshold = 0.2
    idx = torch.where(scores > detect_threshold)
    keypoints = kpts[idx]
    res = draw_keypoints(img, keypoints, colors="blue", radius=4)
    show(res)


if __name__ == "__main__":
    img_file = "data/FudanPed00001.png"
    test_keypoint(img_file)
