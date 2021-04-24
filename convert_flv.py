import os
import os.path
import sys


"""
Uses to convert FLV files to MP4

VCDB has FLV files. FLV files take a LOT longer to import then mp4
"""

path = str(sys.argv[1])

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for f in files:
    fileName = f.split(".")[0]
    inputPath = os.path.join(path, f)
    outputPath = os.path.join(path, f"{fileName}.mp4")
    command = f"ffmpeg -i {inputPath} {outputPath}"
    os.system(command)
