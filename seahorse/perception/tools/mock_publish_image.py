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

import time

from modules.drivers.proto.sensor_image_pb2 import Image
from pycyber import cyber

from torchvision.io import read_image


def create_image():
    # img = read_image()
    msg = Image()
    # msg.data = img
    return msg


def publish_image():
    node = cyber.Node("mock_image")
    writer = node.create_writer("mock/image", Image, 5)

    g_count = 1
    while not cyber.is_shutdown():
        msg = create_image()
        writer.write(msg)
        g_count = g_count + 1
        time.sleep(0.1)


if __name__ == '__main__':
    cyber.init()
    publish_image()
    cyber.shutdown()
