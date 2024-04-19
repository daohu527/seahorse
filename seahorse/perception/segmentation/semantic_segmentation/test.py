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

from deep_lab_v3 import DeepLabV3
from fcn import FCN
from lraspp import LRASPP
from visualize import show

from PIL import Image
from torchvision.utils import draw_segmentation_masks
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

def test_deep_model(deep_lab_v3, img_file):
    input_image = Image.open(img_file)
    input_image = input_image.convert("RGB")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    outputs = deep_lab_v3.detect(input_batch)
    output = outputs['out']
    # print(output)

    normalized_masks = output.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(deep_lab_v3.weights.meta["categories"])}
    print(class_to_idx)
    mask = normalized_masks[0]
    num_classes = normalized_masks.shape[1]
    class_dim = 0
    all_classes_masks = mask.argmax(class_dim) == torch.arange(num_classes)[:, None, None]

    show(draw_segmentation_masks(pil_to_tensor(input_image), masks=all_classes_masks, alpha=0.7))


if __name__ == "__main__":
    img_file = "data/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151610512404.jpg"
    # deep_lab_v3 = DeepLabV3()
    # test_deep_model(deep_lab_v3, img_file)

    fcn = FCN()
    test_deep_model(fcn, img_file)

    # lraspp = LRASPP()
    # test_deep_model(lraspp, img_file)
