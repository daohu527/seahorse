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

from faster_rcnn import FasterRCNN
from visualize import show

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import convert_image_dtype


if __name__ == "__main__":
    frcnn = FasterRCNN()
    img = read_image('../../data/FudanPed00001.png')
    print(img.shape)
    outputs = frcnn.detect([convert_image_dtype(img)])

    print(outputs)
    results = [draw_bounding_boxes(img, output['boxes']) for output in outputs]
    show(results)
