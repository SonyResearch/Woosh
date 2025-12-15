import os

import torch
import torchaudio

from sfxfm.inference.meanflow_samplers import sample_euler
from sfxfm.model.meanflow_from_pretrained import MeanFlowFromPretrained
from sfxfm.module.components.base import LoadConfig
import time

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load model
COMPONENT_PATH = "checkpoints/SFXflowmap"
ldm = MeanFlowFromPretrained(LoadConfig(path=COMPONENT_PATH))
ldm = ldm.eval().to(device)

# Prepare inputs
batch_size = 1
noise = torch.randn(batch_size, 128, 501).to(device)
description = "monster growling"
cond = ldm.get_cond(
    {"audio": None, "description": [description] * batch_size},
    no_dropout=True,
    device=device,
)
# cond["cfg"] = 3 * torch.ones((batch_size,), device=noise.device)

start_time = time.time()
with torch.inference_mode():
    # Denoise using ldm and transform to audio with autoencoder
    x_fake = sample_euler(
        model=ldm,
        noise=noise,
        cond=cond,
        num_steps=5,
        renoise=[0, 0.5, 0.5, 0.5, 0.3],
        step_schedule="linear",
        cfg=4.5,
    )
    audio_fake = ldm.autoencoder.inverse(x_fake)
    os.makedirs("outputs", exist_ok=True)
end_time = time.time()
print(f"Generation took {end_time - start_time:.2f} seconds on {device}")

# move to CPU for saving
audio_fake = audio_fake.cpu()
for i in range(batch_size):
    torchaudio.save(
        f"outputs/output_{i}.wav",
        audio_fake[i],
        sample_rate=48000,
    )
