# %%
import os
import time

import torch
import torchaudio

from sfxfm.inference.flowmatching_sampler import flowmatching_integrate
from sfxfm.model.ldm import LatentDiffusionModel
from sfxfm.components.base import LoadConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# %%

# Load model
COMPONENT_PATH = "checkpoints/Woosh-Flow"
ldm = LatentDiffusionModel(LoadConfig(path=COMPONENT_PATH))
ldm = ldm.eval().to(device)

# %%

# Prepare inputs
batch_size = 1
noise = torch.randn(batch_size, 128, 501).to(device)
description = "sportscar engine revving and driving away quickly"
cond = ldm.get_cond(
    {"audio": None, "description": [description] * batch_size},
    no_dropout=True,
    device=device,
)

# Denoise using ldm and transform to audio with autoencoder
start_time = time.perf_counter()
with torch.inference_mode():
    x_fake, steps = flowmatching_integrate(
        ldm,
        noise=noise,
        cond=cond,
        cfg=4.5,
        atol=0.001,
        rtol=0.001,
        return_steps=True,
        device=device,
        dtype=torch.float32 if device=="mps" else torch.float64,
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
        f"outputs/Woosh-Flow_{i}.wav",
        scaled,
        sample_rate=48000,
    )

# %%
