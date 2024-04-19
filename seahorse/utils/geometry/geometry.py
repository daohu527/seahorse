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


def _orientation(p, q, r) -> int:
    return (q.y - p.y) * (r.x - p.x) - (q.x - p.x) * (r.y - p.y)

def convex_hull(points) -> list:
    n = len(points)
    if n < 3:
        return []
    
    l = 0
    for i in range(n):
        if points[l].x > points[i].x:
            l = i

    p = l 
    hull = []
    while(True):
        hull.append(points[p])
        q = (p + 1) % n
        for r in range(n):
            if _orientation(points[p], points[q], points[r]) < 0:
                q = r
        p = q
        if p == l:
            break
    
    return hull


def is_point_in_polygon(point, polygon):
    pass

def clipping_polygon():
    pass

def rotating_calipers():
    pass
