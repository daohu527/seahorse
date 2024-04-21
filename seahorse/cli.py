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

import sys
import argparse

from perception.detection import object_detection

TASKS = {
    "instance_segmentation",
    "keypoint_detection",
    "object_detection",
    "semantic_segmentation"
}

TASK2FUNC = {
    "instance_segmentation": instance_segmentation,
    "keypoint_detection": keypoint_detection,
    "object_detection": object_detection,
    "semantic_segmentation": semantic_segmentation,
}

def dispatch_task(task, args):
    return TASK2FUNC[task](args)


def main(args=sys.argv):
    parser = argparse.ArgumentParser(
        description="A pure python autonomous driving framework",
        prog="command.py")

    parser.add_argument(
        "-m", "--model", action="store", type=str, required=False,
        default="", help="")
    parser.add_argument(
        "-s", "--source", action="store", type=str, required=False,
        default="", help="")

    # check env
    parser.add_argument(
        "--checks", action="store", type=str, required=False,
        default="", help="")
    # version
    parser.add_argument(
        "-v", "--version", action="store", type=str, required=False,
        default="", help="")
    # config
    parser.add_argument(
        "--settings", action="store", type=str, required=False,
        default="", help="")
    parser.add_argument(
        "--cfg", action="store", type=str, required=False,
        default="", help="")

    # Task type
    task = args[1]
    if task not in TASKS:
        raise "task {} not support!".format(task)

    # parameters
    args = parser.parse_args(args[2:])
    dispatch_task(task, args)
