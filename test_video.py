# %%
import os
import time

import torch
import torchaudio

from sfxfm.inference.flowmatching_sampler import flowmatching_integrate
from sfxfm.components.base import LoadConfig
from sfxfm.model.video_kontext import VideoKontext
from sfxfm.utils.video import SynchformerProcessor
from sfxfm.utils.videoio import extract_video_frames, remux_video

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# %%

# Load model
COMPONENT_PATH = "checkpoints/SFXflowV-8s"
ldm = VideoKontext(LoadConfig(path=COMPONENT_PATH))
ldm = ldm.eval().to(device)

# model to extract video features for conditioning
featuresModel = SynchformerProcessor(frame_rate=24).eval().to(device)


# %%

# Prepare inputs
batch_size = 1
noise = torch.randn(batch_size, 128, 801).to(device)
video_path = (
    "/group2/sfxfm/data/ego4d/v2/full_scale/000cd456-ff8d-499b-b0c1-4acead128a8b.mp4"
)
video_path = "/group2/sfxfm/data/foleybench/videos/3.mp4"
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
        "A person shovels snow and ice off a paved surface, making scraping sounds."
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
    torch.cuda.synchronize()
    # Denoise using ldm and transform to audio with autoencoder
    start_time = time.perf_counter()
    x_fake, steps = flowmatching_integrate(
        ldm,
        noise=noise,
        cond=cond,
        cfg=4.5,
        atol=0.003,
        rtol=0.003,
        return_steps=True,
    )
    audio_fake = ldm.autoencoder.inverse(x_fake)
end_time = time.perf_counter()
print(f"Integrating finished in {steps + 1} steps")
print(f"Generation took {end_time - start_time:.2f} seconds on {device}")

# Move to CPU and save outputs
audio_fake = audio_fake.cpu()
os.makedirs("outputs", exist_ok=True)

for i in range(batch_size):
    max_abs_value = torch.max(torch.abs(audio_fake[i]))
    normalization_factor = max_abs_value if max_abs_value > 1.0 else 1.0
    scaled = audio_fake[i] / normalization_factor
    torchaudio.save(
        f"outputs/output_audio_{i}.wav",
        scaled,
        sample_rate=48000,
    )
    remux_video(
        output_path=f"outputs/output_video_{i}.mp4",
        video_path=video_path,
        audio_input=scaled,
        sample_rate=48000,
        audio_start=0,
        video_chunk=video_frames.cpu(),
        duration_seconds=8,
    )

# %%
