import os

import torch
import torch.nn as nn
from torchvision.transforms import v2
from huggingface_hub import hf_hub_download
from .synchformer import Synchformer, encode_video_with_sync

# fps for syncformer
sync_rate = 24


def get_synchformer(device="cpu"):
    ckpt = hf_hub_download(
        "hkchengrex/MMAudio",
        filename="synchformer_state_dict.pth",
        subfolder="ext_weights",
    )
    model = Synchformer().eval()
    sd = torch.load(ckpt, weights_only=True, map_location=torch.device(device))
    model.load_state_dict(sd)
    return model.eval()


def downsample(source_rate, target_rate, video_frames, video_pts):
    if source_rate % target_rate == 0:
        downsample_factor = source_rate // target_rate
        video_frames = video_frames[::downsample_factor]
        video_pts = video_pts[::downsample_factor]
        return video_frames, video_pts
    i = 0
    downsample_factor = float(source_rate / target_rate)
    step_in_seconds = 1.0 / target_rate
    indices = []
    progress = float(video_pts[0])
    for i in range(len(video_pts)):
        if video_pts[i] >= progress:
            indices.append(i)
            progress += step_in_seconds
    video_frames = video_frames[indices]
    video_pts = video_pts[indices]
    return video_frames, video_pts


_SYNC_SIZE = 224

sync_transform = v2.Compose(
    [
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def process_synchformer_transform(item, frame_rate=24):
    video_frames = item["video_frames"]
    video_pts = item["video_pts"]
    video_rate = item["video_rate"]
    video_frames, video_pts = downsample(
        video_rate, frame_rate, video_frames, video_pts
    )
    video_frames = sync_transform(video_frames)
    item["sync_inputs"] = video_frames
    item["sync_pts"] = video_pts
    return item


class SynchformerProcessor(nn.Module):
    """nn.Module wrapper around Synchformer for video feature extraction."""

    def __init__(self, frame_rate=24):
        super().__init__()
        self.model = get_synchformer()
        self.frame_rate = frame_rate

    def forward(self, images: torch.Tensor, fps: float) -> dict:
        """Extract Synchformer features from raw video frames.

        Args:
            images: Tensor of shape (T, H, W, C).
            fps: Frame rate of the input video.

        Returns:
            Dict with ``synch_out``, ``synch_pts_seconds``, and
            ``sync_hop_size_ms``.
        """
        item = {
            "video_frames": images.permute(0, 3, 1, 2),  # (T, H, W, C) -> (T, C, H, W)
            "video_pts": torch.arange(len(images), dtype=torch.float32) / fps,
            "video_rate": int(fps),
        }
        item = process_synchformer_transform(item, self.frame_rate)

        video_frames = item["sync_inputs"]
        sync_pts = item["sync_pts"]

        out = encode_video_with_sync(
            self.model, video_frames.unsqueeze(0), batch_size=400
        ).squeeze(0)

        return {
            "synch_out": out.unsqueeze(0),
            "synch_pts_seconds": sync_pts.unsqueeze(0),
            "sync_hop_size_ms": 1000 / self.frame_rate,
        }
