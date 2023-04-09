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

from tracklet import Tracklet
from similarity.similarity import iou
from data_associate.bipartite_graph_match import BipartiteGraphMatch

class Tracker:
  def __init__(self) -> None:
    self.tracklets = dict()
    self.associator = BipartiteGraphMatch()

  def track(self, detections) -> list:
    self.calc_similarity(self.tracklets, detections)
    match_pairs, unmatch_tracklets, unmatch_detections = \
        self.associator.associate(self.tracklets, detections)

    self.del_tracklet(unmatch_tracklets)
    self.add_tracklet(unmatch_detections)
    for tracklet, detection in match_pairs:
      tracklet.add(detection)

  def add_tracklet(self, unmatch_detections):
    for detection in unmatch_detections:
      tracklet = Tracklet(detection)
      self.tracklets.update((tracklet.id, tracklet))

  def del_tracklet(self, unmatch_tracklets):
    for tracklet in unmatch_tracklets:
      if tracklet.is_lost():
        self.tracklets.pop(tracklet.id)

  def calc_similarity(self, src, dst):
    iou(src, dst)
