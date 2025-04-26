import decord
from decord import VideoReader
from decord import cpu
import torch
import math
import einops
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F


def adjust_video_duration(video_tensor, duration, target_fps):
    current_duration = video_tensor.shape[1]
    target_duration = duration * target_fps

    if current_duration > target_duration:
        video_tensor = video_tensor[:, :target_duration]
    elif current_duration < target_duration:
        last_frame = video_tensor[:, -1:]
        repeat_times = target_duration - current_duration
        video_tensor = torch.cat((video_tensor, last_frame.repeat(1, repeat_times, 1, 1)), dim=1)

    return video_tensor

def video_read_local(filepath, seek_time=0., duration=-1, target_fps=2):
    print(filepath)
    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    duration = 29
    if duration > 0:
        total_frames_to_read = target_fps * duration
        frame_interval = int(math.ceil(fps / target_fps))
        start_frame = int(seek_time * fps)
        end_frame = start_frame + frame_interval * total_frames_to_read
        frame_ids = list(range(start_frame, min(end_frame, len(vr)), frame_interval))
    else:
        frame_ids = list(range(0, len(vr), int(math.ceil(fps / target_fps))))

    frames = vr.get_batch(frame_ids)
    frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
    
    resize_transform = transforms.Resize((224, 224))
    frames = [resize_transform(frame) for frame in frames]
    video_tensor = torch.stack(frames)
    video_tensor = einops.rearrange(video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]
    video_tensor = adjust_video_duration(video_tensor, duration, target_fps)
    assert video_tensor.shape[1] == duration * target_fps, f"the shape of video_tensor is {video_tensor.shape}"

    return video_tensor


def video_read_global(filepath, seek_time=0., duration=-1, target_fps=2, global_mode='average', global_num_frames=32):
    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frame_count = len(vr)
    #print(f"frame_count: {frame_count}, fps: {fps}")
    duration = 29
    if duration > 0:
        total_frames_to_read = target_fps * duration
        frame_interval = int(math.ceil(fps / target_fps))
        start_frame = int(seek_time * fps)
        end_frame = start_frame + frame_interval * total_frames_to_read
        frame_ids = list(range(start_frame, min(end_frame, frame_count), frame_interval))
    else:
        frame_ids = list(range(0, frame_count, int(math.ceil(fps / target_fps))))

    #print(f"num frame_ids: {len(frame_ids)}")
    local_frames = vr.get_batch(frame_ids)
    local_frames = torch.from_numpy(local_frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
    
    resize_transform = transforms.Resize((224, 224))
    local_frames = [resize_transform(frame) for frame in local_frames]
    local_video_tensor = torch.stack(local_frames)
    local_video_tensor = einops.rearrange(local_video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]
    local_video_tensor = adjust_video_duration(local_video_tensor, duration, target_fps)

    if global_mode=='average':
        global_frame_ids = torch.linspace(0, frame_count - 1, global_num_frames).long()

        global_frames = vr.get_batch(global_frame_ids)
        global_frames = torch.from_numpy(global_frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
        global_frames = [resize_transform(frame) for frame in global_frames]
        global_video_tensor = torch.stack(global_frames)
        global_video_tensor = einops.rearrange(global_video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]

    assert global_video_tensor.shape[1] == global_num_frames, f"the shape of global_video_tensor is {global_video_tensor.shape}"
    return local_video_tensor, global_video_tensor


def video_read_with_flow_slow(
    filepath: str,
    seek_time: float = 0.,
    duration: int = 29,
    target_fps: int = 1,
    flow_hz: int = 4,
):
    """
    Returns:
      video_tensor: [3, target_fps * duration, 224, 224]
      flow_tensor:  [1, duration, 224, 224]

    Note: VideoReader.get_batch().asnumpy() returns frames in RGB order.
    """

    # 1) Open video and get its FPS
    vr  = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps() 
    print(fps)
    print("max num of frame", len(vr))
    
    # 2a) Compute the video-frame indices at target_fps
    T_vid  = target_fps * duration
    vid_frame_pos = np.linspace(
        seek_time * fps,          
        (seek_time + duration) * fps,  
        num=T_vid,
        endpoint=False
    )
    vid_frame_ids = np.clip(np.round(vid_frame_pos).astype(int), 0, len(vr)-1)
    print("num of video frames", len( vid_frame_ids))

    # 2b) Compute the flow-frame indices at flow_hz (need flow_hz*duration + 1 frames)
    T_flow = flow_hz * duration + 1
    flow_frame_pos = np.linspace(
        seek_time * fps,          
        (seek_time + duration) * fps,  
        num=T_flow,
        endpoint=False
    )
    flow_frame_ids = np.clip(np.round(flow_frame_pos).astype(int), 0, len(vr)-1)
    print("num of flow frames", len(flow_frame_ids))

    # 3) Batch-read frames (NumPy RGB)
    np_vid  = vr.get_batch(vid_frame_ids).asnumpy()   # [T_vid, H0, W0, 3]
    np_flow = vr.get_batch(flow_frame_ids).asnumpy()   # [T_flow, H0, W0, 3]
    
    print("Raw Video dim", np_vid.shape)
    print("Raw Flow dim", np_flow.shape)
    
    # 4) Convert video to torch and batch-resize in one go
    vid_t = torch.from_numpy(np_vid).permute(0, 3, 1, 2)  # [T_vid, 3, H, W]
    vid_resized = F.interpolate(vid_t.float(), size=(224, 224), mode='bilinear', align_corners=False)
    video_tensor = vid_resized.permute(1, 0, 2, 3)         # [3, T_vid, 224, 224]

    # 5) Resize flow frames in NumPy (frames already RGB)
    H, W = 224, 224
    flow_resized_np = np.stack([
        cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        for frame in np_flow
    ], axis=0)  # [T_flow, H, W, 3]

    print("Resized Video dim", vid_resized.shape) 
    print("Resized Flow dim", flow_resized_np.shape) 

    # 6) Pre-allocate and compute magnitudes
    N = flow_hz * duration
    mags_arr = np.empty((N, H, W), dtype=np.float32)
    prev_gray = cv2.cvtColor(flow_resized_np[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(flow_frame_ids)):
        cur_gray = cv2.cvtColor(flow_resized_np[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags_arr[i-1] = mag
        prev_gray = cur_gray

    # 7) Group every flow_hz maps into 1-second averages
    avg_flows = einops.reduce(
        mags_arr,
        "(sec hz) h w -> sec h w",
        reduction="mean",
        sec=duration,
        hz=flow_hz
    )  # [duration, 224, 224]

    # 8) Convert to torch and add channel dim [1, duration, 224, 224]
    flow_tensor = torch.from_numpy(avg_flows).unsqueeze(0)

    return video_tensor, flow_tensor



def video_read_with_flow(
    filepath: str,
    seek_time: float = 0.,
    duration: int = 29,
    target_fps: int = 1,
    flow_hz: int = 4,
):
    """
    Returns:
      video_tensor: [3, target_fps * duration, 224, 224]
      flow_tensor:  [1, duration, 224, 224]

    Frame resize is done by libavfilter (ffmpeg) in VideoReader.
    """

    # 1) Open video with on‐the‐fly resize to 224×224
    vr  = VideoReader(filepath, ctx=cpu(0), width=224, height=224)
    fps = vr.get_avg_fps()
    print(f"fps={fps:.2f}, total_frames={len(vr)}")

    # 2a) Compute indices for video frames at target_fps
    T_vid = target_fps * duration
    vid_pos = np.linspace(
        seek_time * fps,
        (seek_time + duration) * fps,
        num=T_vid,
        endpoint=False
    )
    vid_ids = np.clip(np.round(vid_pos).astype(int), 0, len(vr) - 1)
    print("video frames:", len(vid_ids))

    # 2b) Compute indices for optical‐flow frames at flow_hz
    T_flow = flow_hz * duration + 1
    flow_pos = np.linspace(
        seek_time * fps,
        (seek_time + duration) * fps,
        num=T_flow,
        endpoint=False
    )
    flow_ids = np.clip(np.round(flow_pos).astype(int), 0, len(vr) - 1)
    print("flow frames:", len(flow_ids))

    # 3) Batch‐read already‐resized frames (RGB H×W are 224×224)
    np_vid  = vr.get_batch(vid_ids).asnumpy()    # [T_vid, 224, 224, 3]
    np_flow = vr.get_batch(flow_ids).asnumpy()   # [T_flow, 224, 224, 3]

    # 4) Convert video to torch and reorder to [3, T_vid, 224, 224]
    vid_t = torch.from_numpy(np_vid).permute(0, 3, 1, 2)
    video_tensor = vid_t.permute(1, 0, 2, 3)

    # 5) Compute optical‐flow magnitudes on resized frames
    #    (frames already 224×224, in RGB)
    N = flow_hz * duration
    mags = np.empty((N, 224, 224), dtype=np.float32)
    prev_gray = cv2.cvtColor(np_flow[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(flow_ids)):
        cur_gray = cv2.cvtColor(np_flow[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags[i-1] = mag
        prev_gray = cur_gray

    # 6) Average every flow_hz frames into 1‐second maps
    avg_flows = einops.reduce(
        mags,
        "(sec hz) h w -> sec h w",
        reduction="mean",
        sec=duration,
        hz=flow_hz
    )  # [duration, 224, 224]

    # 7) To torch [1, duration, 224, 224]
    flow_tensor = torch.from_numpy(avg_flows).unsqueeze(0)
    print("video_tensor shape", video_tensor.shape)
    print("flow_tensor shape", flow_tensor.shape)
    return video_tensor, flow_tensor