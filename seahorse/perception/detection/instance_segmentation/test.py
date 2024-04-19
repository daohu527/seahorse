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


from mask_rcnn import MaskRCNN
from visualize import show

from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import convert_image_dtype


def test_instance_segmentation(img_file):
    mask_rcnn = MaskRCNN()
    img = read_image(img_file)
    print(img.shape)
    outputs = mask_rcnn.detect([convert_image_dtype(img)])

    output = outputs[0]

    proba_threshold = 0.5
    output_bool_masks = output['masks'] > proba_threshold
    print(f"shape = {output_bool_masks.shape}, dtype = {output_bool_masks.dtype}")

    # There's an extra dimension (1) to the masks. We need to remove it
    output_bool_masks = output_bool_masks.squeeze(1)

    show(draw_segmentation_masks(img, output_bool_masks, alpha=0.9))


if __name__ == "__main__":
    img_file = "data/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151610512404.jpg"
    test_instance_segmentation(img_file)
