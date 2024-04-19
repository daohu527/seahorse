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

import seahorse
from seahorse.perception.io import read


def main(args=sys.argv):
    parser = argparse.ArgumentParser(
        description="A pure python autonomous driving framework",
        prog="command.py")

    parser.add_argument(
        "-m", "--model", action="store", type=str, required=False,
        default="", help="")
    parser.add_argument(
        "-w", "--weights", action="store", type=str, required=False,
        default="", help="")
    parser.add_argument(
        "-s", "--source", action="store", type=list, required=False,
        default="", help="")

    task = args[1]
    args = parser.parse_args(args[2:])
    getattr(seahorse, task)(**overrides)
