from sfxfm.inference.flowmap_sampler import sample_euler
from sfxfm.model.flowmap_from_pretrained import FlowMapFromPretrained
from sfxfm.components.base import LoadConfig

import gradio as gr
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
# Load model
COMPONENT_PATH = "checkpoints/SFXflowmap"
ldm = FlowMapFromPretrained(LoadConfig(path=COMPONENT_PATH))
ldm = ldm.eval().to(device)

@torch.inference_mode()
def generate(prompts):
    batch_size = len(prompts)
    noise = torch.randn(batch_size, 128, 501).to(device)

    cond = ldm.get_cond(
        {"audio": None, "description": prompts},
        no_dropout=True,
        device=device,
    )
    # Denoise using ldm and transform to audio with autoencoder
    x_fake = sample_euler(
        model=ldm,
        noise=noise,
        cond=cond,
        num_steps=4,
        renoise=[0, 0.5, 0.5, 0.3],
        cfg=4.5,
    )
    audio_fake = ldm.autoencoder.inverse(x_fake).cpu()
    return audio_fake


with gr.Blocks() as demo:
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", scale=3)
        batch_size = gr.Number(value=2, label="Batch size", minimum=1, maximum=4, scale=1)
        run_btn = gr.Button()

    @gr.render(inputs=[prompt, batch_size], triggers=[prompt.submit, run_btn.click])
    def audio_out(text, batch_size):
        if len(text) == 0:
            gr.Markdown("## No Input Provided")
        
        prompts = [text] * batch_size
        audios = generate(prompts)
        print(audios.mean(), audios.min(), audios.max(), audios.dtype)
        audios = (32767 * audios).to(dtype=torch.int16)
        audios = [t.squeeze().numpy() for t in torch.unbind(audios)]
        audio_components = []
        for i, audio in enumerate(audios):
            audio_components.append(gr.Audio((48000, audio), key=f"Audio{i}", label=f"Output #{i}"))

demo.launch(show_error=True)
