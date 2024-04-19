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
from ssd import SSD
from visualize import show

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import convert_image_dtype


def test_frcnn(img_file):
    frcnn = FasterRCNN()
    img = read_image(img_file)
    print(img.shape)
    outputs = frcnn.detect([convert_image_dtype(img)])

    print(outputs)
    results = [draw_bounding_boxes(img, output['boxes'], width=4) for output in outputs]
    show(results)


def test_ssd(img_file):
    ssd = SSD()
    img = read_image(img_file)
    print(img.shape)
    outputs = ssd.detect([convert_image_dtype(img)])

    print(outputs)
    results = [draw_bounding_boxes(img, output['boxes'], width=4) for output in outputs]
    show(results)


if __name__ == "__main__":
    img_file = "data/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151610512404.jpg"
    test_frcnn(img_file)
    # test_ssd(img_file)
