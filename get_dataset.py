

from os import listdir
from os.path import isfile, join, exists
from custom_types import VideoDataset, Video
import cv2
import skvideo.io


TRUE_QUERY_SUB_DIR = "true_query"
FALSE_QUERY_SUB_DIR = "false_query"

"""
    Dataset Directory Format:
     dataset dir
        target dir
            targetvideo.mp4
            query
                queryvideo.mp4
        ... repeat for several target videos

"""

"""
Each dataset can have 5 target videos
each target will have 5 truths and 5 Falses (At least 5 truths)

Real Dataset:
Take videos from VCDB

Fake Dataset:
Take a video then take slices out of that video.


1. An exact query (Q1)
2. A query with a removal operation (Q2)
3. A query with an insertion operation (Q3); and
4. A query with a removal operation and temporal re-
ordering (Q4)
"""


def get_dataset(path: str) -> VideoDataset:
    files = [f for f in listdir(path)]

    return [get_video_set(join(path, f)) for f in files]

# Gets the video and test query videos for a given video path


def get_video_set(path: str) -> (str, [(str, bool)]):
    # First file found is target video
    targetPath = [f for f in listdir(path)
                  if isfile(join(path, f))][0]
    targetPath = join(path, targetPath)

    truePaths = get_query_video_set(
        join(path, TRUE_QUERY_SUB_DIR), True)

    falsePaths = get_query_video_set(
        join(path, FALSE_QUERY_SUB_DIR), False)

    queryPaths = [ff for f in [truePaths, falsePaths] for ff in f]

    return (targetPath, queryPaths)


# Gets the given query videos for the video path
def get_query_video_set(path: str, type: bool) -> [(str, bool)]:
    if not exists(path):
        return []
    videoPaths = [f for f in listdir(path)]
    return [(join(path, f), type) for f in videoPaths]


def get_video_frame(path: str) -> Video:
    print(path)
    return skvideo.io.vread(path, height=852, width=480)
    #print(path)
    #cap = cv2.VideoCapture(path)
    #video = []
    #while(cap.isOpened()):
    #    ret, frame = cap.read()
    #    cap.release() if not ret else video.append(frame)
    #cv2.destroyAllWindows()
    #return video
