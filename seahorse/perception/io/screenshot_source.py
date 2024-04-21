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

from source import Source


class ScreenshotSource(Source):
    def __init__(self, source, imgsz=640):
        # check_requirements('mss')
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None

        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)

        self.imgsz = imgsz
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1

        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else (monitor['top'] + top)
        self.left = monitor['left'] if left is None else (monitor['left'] + left)
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}


    def __iter__(self):
        return self

    def __next__(self):
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '

        self.frame += 1
        return [str(self.screen)], [im0], None, s  # screen, img, vid_cap, string
