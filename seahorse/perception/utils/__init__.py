from PIL import Image
from torchvision.io import read_video

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes

class FileType(Enum):
    IMAGE = 1
    POINTCLOUD = 2
    VIDEO = 3
    ROSBAG = 4
    RECORD = 5


def check_type(file_name):
    suffix = Path(file_name).suffix
    if suffix in IMG_FORMATS:
        return FileType.IMAGE
    elif suffix in ('pcd'):
        return FileType.POINTCLOUD
    elif suffix in VID_FORMATS:
        return FileType.VIDEO
    elif suffix in ('bag'):
        return FileType.ROSBAG
    elif suffix in ('record'):
        return FileType.RECORD
    else:
        raise TypeError(f'{suffix} not supported!')

def read_image(img_file):
    input_image = Image.open(img_file)
    return input_image.convert("RGB")

def read_video(video_file):
    frames, _, _ = read_video(video_file, output_format="TCHW")
    return frames

def read_pointcloud(pointcloud_file):
    pass

def read_bag(bag_file):
    pass

def read_record(record_file):
    import cyber_record

