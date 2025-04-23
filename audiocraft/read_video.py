from .data.video import video_read_local, video_read_global
import json

jsonl_path = "/work/users/t/i/tis/VidMuse/egs/V2M20K/eval/data.jsonl"
video_paths = []

with open(jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if "video_path" in data:
            video_paths.append(data["video_path"])

print(f"Loaded {len(video_paths)} video paths.")

local_ft, global_ft = video_read_global('/work/users/t/i/tis/V2Music/preprocessing/data/video/zRr2QTHukzo_h264.mp4')

for idx, video in enumerate(video_paths):
    try:
        #vr = VideoReader(filepath, ctx=cpu(0))
        local_ft, global_ft = video_read_global(video)
        print(f"Successfully read video {idx} at {video}.")
    except Exception as e:
        print(f"Failed to read video at {video}: {e}")
    

