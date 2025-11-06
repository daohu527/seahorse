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


from modules.drivers.proto.sensor_image_pb2 import Image
from pycyber import cyber

from torchvision.transforms.functional import convert_image_dtype

from detection.faster_rcnn import FasterRCNN


class ImageDetection:
    def __init__(self, subscribe_topic, publish_topic, need_visualize=False) -> None:
        self.name = "ImageDetection"
        self.subscribe_topic = subscribe_topic
        self.publish_topic = publish_topic
        self.need_visualize = need_visualize
        self.detector = FasterRCNN()

    def start(self):
        self._node = cyber.Node(self.name)
        self._reader = self._node.create_reader(
            self.subscribe_topic, Image, self.callback
        )
        self._writer = self._node.create_writer(self.publish_topic, Image, 1)
        self._node.spin()

    def callback(self, img):
        self.preprocess(img)
        outputs = self.detector.detect([convert_image_dtype(img)])
        self.postprocess(outputs)
        if self.need_visualize:
            self.visualize()
        self._writer.write(outputs)

    def preprocess(self, img):
        pass

    def postprocess(self, outputs):
        pass

    def visualize(self):
        pass


if __name__ == "__main__":
    cyber.init()
    image_detection = ImageDetection()
    image_detection.start()
    cyber.shutdown()
