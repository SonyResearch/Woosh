# %%
import os
import time

import torch
import torchaudio

from woosh.inference.flowmap_sampler import sample_euler
from woosh.components.base import LoadConfig
from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
from woosh.utils.video import SynchformerProcessor
from woosh.utils.videoio import extract_video_frames, remux_video

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# %%

# Load model
COMPONENT_PATH = "checkpoints/Woosh-DVFlow-8s"
ldm = FlowMapFromPretrained(LoadConfig(path=COMPONENT_PATH))
ldm = ldm.eval().to(device)

# model to extract video features for conditioning
featuresModel = SynchformerProcessor(frame_rate=24).eval().to(device)


# %%

# Prepare inputs
batch_size = 1
noise = torch.randn(batch_size, 128, 801).to(device)
video_path = "samples/video_sample.mp4"
with torch.inference_mode():
    video_frames, video_rate, pts_arr = extract_video_frames(
        video_path,
        start_time=0,
        end_time=8,
    )
    video_frames = video_frames.to(device)
    features = featuresModel(video_frames, video_rate)
    # can be empty text or a description of the video
    description = (
        "Two figures in costumes walk down a basement hallway, their footsteps echoing on the concrete floor."
    )
    print(features["synch_out"].shape)
    cond = ldm.get_cond(
        {
            "audio": None,
            "description": [description] * batch_size,
            "synch_out": features["synch_out"],
        },
        no_dropout=True,
        device=device,
    )
    # torch.cuda.synchronize()
    # Denoise using ldm and transform to audio with autoencoder
    start_time = time.perf_counter()
    x_fake = sample_euler(
        model=ldm,
        noise=noise,
        cond=cond,
        num_steps=4,
        renoise=[0, 0.5, 0.5, 0.3],
        cfg=3,
    )
    audio_fake = ldm.autoencoder.inverse(x_fake)

# Move to CPU, forces sync to get accurate timing
audio_fake = audio_fake.cpu()
end_time = time.perf_counter()
print(f"Generation took {end_time - start_time:.2f} seconds on {device}")


os.makedirs("outputs", exist_ok=True)

for i in range(batch_size):
    max_abs_value = torch.max(torch.abs(audio_fake[i]))
    normalization_factor = max_abs_value if max_abs_value > 1.0 else 1.0
    scaled = audio_fake[i] / normalization_factor
    torchaudio.save(
        f"outputs/Woosh-DVFlow_audio_{i}.wav",
        scaled,
        sample_rate=48000,
    )
    remux_video(
        output_path=f"outputs/Woosh-DVFlow_video_{i}.mp4",
        video_path=video_path,
        audio_input=scaled,
        sample_rate=48000,
        audio_start=0,
        duration_seconds=8,
    )

# %%
