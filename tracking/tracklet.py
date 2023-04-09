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

import itertools
from enum import Enum

from object import Object

class Status(Enum):
  LOST = 1
  HIT = 2


class Tracklet:
  global_id = itertools.count()

  def __init__(self, detection = None) -> None:
    self.id = self.next_id()
    self.last_update_time = None
    self.objects = []

    self.add(detection)

  def is_lost(self):
    pass

  def latest_obj(self):
    pass

  def add(self, detection):
    if detection is None:
      return
    pass

  def predict(self):
    pass

  def update(self):
    pass

  def next_id(self):
    return next(Tracklet.global_id)
