

from os import listdir
from os.path import isfile, join
from custom_types import VideoDataset, Video
import cv2


TRUE_QUERY_SUB_DIR = "true_query"

"""
    Dataset Directory Format:
     dataset dir
        target dir
            targetvideo.mp4
            query
                queryvideo.mp4
        ... repeat for several target videos

"""


def get_dataset(path: str) -> VideoDataset:
    files = [f for f in listdir(path)]

    return [get_video_set(join(path, f)) for f in files]

# Gets the video and test query videos for a given video path


def get_video_set(path: str) -> (Video, [(Video, bool)]):
    # First file found is target video
    targetVideoPath = [f for f in listdir(path)
                       if isfile(join(path, f))][0]
    targetVideo = get_video_frame(join(path, targetVideoPath))

    true_video_query = get_query_video_set(
        join(path, TRUE_QUERY_SUB_DIR), True)

    return (targetVideo, true_video_query)


# Gets the given query videos for the video path
def get_query_video_set(path: str, type: bool) -> [(Video, bool)]:
    videoPaths = [f for f in listdir(path)]
    return [(get_video_frame(join(path, f)), type) for f in videoPaths]


def get_video_frame(path: str) -> Video:
    cap = cv2.VideoCapture(path)
    video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        cap.release() if not ret else video.append(frame)

    return video
