# %%
import os
import time

import torch
import torchaudio

from sfxfm.inference.flowmatching_sampler import flowmatching_integrate
from sfxfm.components.autoencoders import AudioAutoEncoder
from sfxfm.components.base import LoadConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# %%

# Load model
COMPONENT_PATH = "checkpoints/SFXae"
ae = AudioAutoEncoder(LoadConfig(path=COMPONENT_PATH))
ae.load_from_config()
ae.eval().to(device)

# %%

# Prepare inputs
batch_size = 1
audio, fs = torchaudio.load("sample.wav")
audio = audio.unsqueeze(0)
audio = audio.detach().to(device=device)


# Denoise using ldm and transform to audio with autoencoder
start_time = time.perf_counter()

x = ae.forward(audio)
audio_fake = ae.inverse(x).detach()

end_time = time.perf_counter()
print(f"Encoding/decoding took {end_time - start_time:.2f} seconds on {device}")

# Move to CPU and save outputs
audio_fake = audio_fake.cpu()
os.makedirs("outputs", exist_ok=True)
for i in range(batch_size):
    max_abs_value = torch.max(torch.abs(audio_fake[i]))
    normalization_factor = max_abs_value if max_abs_value > 1.0 else 1.0
    scaled = audio_fake[i] / normalization_factor
    torchaudio.save(
        f"outputs/output_ae_{i}.wav",
        scaled,
        sample_rate=48000,
    )

# %%
