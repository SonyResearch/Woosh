from pathlib import Path
import torch
from safetensors import safe_open
from omegaconf import OmegaConf
from woosh.module.audioretrieval_module import AudioRetrievalModel
from woosh.utils.loading import lazy_loading

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

COMPONENT_PATH = Path("checkpoints/Woosh-CLAP")
config_path = COMPONENT_PATH / "config.yaml"
weights_text_path = COMPONENT_PATH / "weights_text.safetensors"
weights_audio_path = COMPONENT_PATH / "weights_audio.safetensors"

config = OmegaConf.load(config_path)
with lazy_loading():
    model = AudioRetrievalModel(**config)

text = [
    "glass breaking",
    "footsteps, muddy, water",
]

audio_paths = [
    "samples/810333__mokasza__glass-breaking.mp3",
    "samples/810333__mokasza__glass-breaking.mp3",
]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = model.to(device).eval()
state_dict = {}
with safe_open(weights_text_path, framework="pt") as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

with safe_open(weights_audio_path, framework="pt") as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

model.load_state_dict(state_dict)
results = model({"audio": audio_paths, "text": text}, device=device, use_tensor=True)
scores = (results["audio"] * results["text"]).sum(dim=-1)
scores_shape = scores.shape

print(f"Text-audio CLAP scores:")
for t,a,score in zip(text, audio_paths, scores):
    print(f"  '{t}' vs. '{Path(Path(a).name).stem}': {score:.2f}")
