from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open

from omegaconf import OmegaConf

from sfxfm.module.audioretrieval_module import AudioRetrievalModel
from sfxfm.utils.loading import lazy_loading


COMPONENT_PATH = Path("checkpoints/SFXclap")


config_path = COMPONENT_PATH / "config.yaml"
config_r_path = COMPONENT_PATH / "config_full_clap.yaml"
weights_path = COMPONENT_PATH / "weights.safetensors"
weights_audio_path = COMPONENT_PATH / "weights_audio.safetensors"

config = OmegaConf.load(config_r_path)

with lazy_loading():
    model = AudioRetrievalModel(**config)

text = [
    "footsteps, muddy, water",
    "footsteps, wooden surface",
    "long, slow growling of creature, monster or animal",
    "ambience, beach, waves, seagulls",
]

audio_paths = [
    "examples/footsteps_on_mud.wav",
    "examples/footsteps_on_wood.wav",
    "examples/long_growling.wav",
    "examples/beach_ambiance.wav",
]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = model.to(device).eval()
state_dict = {}
with safe_open(weights_path, framework="pt") as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

with safe_open(weights_audio_path, framework="pt") as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

model.load_state_dict(state_dict)
results = model({"audio": audio_paths, "text": text}, device=device, use_tensor=True)
scores = results["audio"] @ results["text"].T
scores_shape = scores.shape

scores = scores.detach().cpu().numpy()
with np.printoptions(precision=6, suppress=True):
    print("Scores:")
    print(scores)

# scores_calibrated = scores_calibrated.detach().cpu().numpy()
# with np.printoptions(precision=6, suppress=True):
#     print("Calibrated Scores:")
#     print(scores_calibrated)
