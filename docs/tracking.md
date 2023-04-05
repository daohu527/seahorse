## Module name
tracking

## Features
multi-object tracking

## Design
We use tracking by detection in automatic driving. And divide the tracking into the following 4 steps.
- object detection
- feature extraction
- similarity calculation
- data association

Tracker maintains multiple trajectories, and use the current frame object and the object in the trajectory to match, if it matches, add it, otherwise generate a new trajectory to maintain it.

## Testcase
