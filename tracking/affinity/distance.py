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

import numpy as np


def euclidean(u, v):
  return np.sqrt(squared_euclidean(u, v))

def squared_euclidean(u, v):
  return np.sum(np.square(u - v))

def standardized_euclidian(u, v):
  pass

def manhattan(u, v):
  pass

def canberra(u, v):
  pass

def chebyshev(u, v):
  pass

def minkowski(u, v):
  pass

def cosine(u, v):
  pass

def pearson_correlation(u, v):
  pass

def spearman_correlation(u, v):
  pass

def mahalanobis(u, v):
  pass

def chi_square(u, v):
  pass

def jensen_shannon(u, v):
  pass

def levenshtein(u, v):
  pass

def hamming(u, v):
  pass

def jaccard(u, v):
  pass

def sorensen_dice(u, v):
  pass

def bray_curtis(u, v):
  pass
