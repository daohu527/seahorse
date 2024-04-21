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



class DataLoader():
    def __init__(self, source, filter, associater):
        pass

    def read_unimodal(self, path):
        if file_type is :
            data = read_image(path)
        elif file_type is :
            data = read_video(path)
        elif file_type is :
            data = read_pointcloud(path)
        elif file_type is :
            data = read_bag(path)
        elif file_type is :
            data = read_record(path)

    def read_multimodal(self, path):
        if file_type is :
            read_multi_data(pointclouds, images, )
        elif file_type is :
            read_bag(path)
        elif file_type is :
            read_record(path)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        for path, file_type in source:
          if (filter(file_type)):
            read_unimodal()

