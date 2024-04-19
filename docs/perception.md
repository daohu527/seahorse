## Module name
perception

## Features


## Design
```
seahorse object_detection model=FasterRCNN source='https://youtu.be/LNwODJXcvt4'
```
- command_line = seahorse + task_type + parameters, task_type can refer to the dataset
- source is multi-source input, include from file(img, video, pcd), url(http, rtmp), bag(rosbag, record), topic
- Visualization is a must, including real-time display and offline saving
- A variety of results must be saved（img-txt, video, record）
- The parameters are completed by association as much as possible

#### Other cmd
```
seahorse help
seahorse checks
seahorse version
seahorse settings
seahorse copy-cfg
seahorse cfg
```

## Testcase
