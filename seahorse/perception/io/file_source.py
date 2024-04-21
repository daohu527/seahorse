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

from pathlib import Path
from enum import Enum


def read_url(url):
    data = requests.get(url, stream=True).raw
    scheme = urlparse(src).scheme
    if scheme in ("http", "https"):
        pass
    elif scheme in ("rtps"):
        pass


class FileSource(Source):
    """Read any type of file. Image, POINTCLOUD, VIDEO, Rosbag, Record

    Args:
        Source (_type_): _description_
    """
    def __init__(self, path, filter):
        # read path from text
        if isinstance(path, str) and Path(path).suffix == '.txt':
            path = Path(path).read_text().splitlines()

        # read files from path
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            p = Path(p)
            if '*' in a:
                # Specify file type, like '*.jpg'
                files.extend(sorted(glob.glob(a, recursive=True)))
            elif p.is_dir():
                # Find all types use '*.*'
                files.extend(sorted(glob.glob(os.path.join(a, '*.*'))))
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(f'{p} does not exist!')

        self.files = files

        # filter and sort
        images = []
        videos = []
        pointclouds = []
        bags = []
        records = []
        for f in files:
            file_type = check_type(f)
            if file_type is FileType.IMAGE:
                images.append(f)
            elif file_type is FileType.VIDEO:
                videos.append(f)
            elif file_type is FileType.POINTCLOUD:
                pointclouds.append(f)
            elif file_type is FileType.ROSBAG:
                bags.append(f)
            elif file_type is FileType.RECORD:
                records.append(f)


    def __iter__(self):
        return self


    def __next__(self):
        for f in self.files:
            yield f, check_type(f)
