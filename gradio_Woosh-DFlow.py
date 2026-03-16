"""Woosh Text-to-Audio Gradio Demo"""

import argparse
import logging
import os
import time

import gradio as gr
import torch

from woosh.components.base import LoadConfig
from woosh.inference.flowmap_sampler import sample_euler
from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
LATENT_CHANNELS = 128
LATENT_FRAMES = 501  # ~5s at 48kHz after autoencoder

# Global model state
ldm = None
device = None


def load_model(checkpoint_path: str):
    global ldm, device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info(f"Loading model from {checkpoint_path} on {device}")
    ldm = FlowMapFromPretrained(LoadConfig(path=checkpoint_path))
    ldm = ldm.eval().to(device)
    log.info("Model loaded.")


@torch.inference_mode()
def generate(
    prompt: str,
    batch_size: int = 1,
    cfg_scale: float = 4.5,
    seed: int = -1,
    solver: str = "dopri5",
    atol: float = 0.001,
    rtol: float = 0.001,
    progress=gr.Progress(),
):
    if ldm is None:
        raise gr.Error("Model not loaded!")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)
    torch.manual_seed(seed)

    noise = torch.randn(batch_size, LATENT_CHANNELS, LATENT_FRAMES).to(device)

    cond = ldm.get_cond(
        {"audio": None, "description": [prompt] * batch_size},
        no_dropout=True,
        device=device,
    )

    progress(0.1, desc="Generating...")
    start_time = time.perf_counter()

    # Denoise using ldm and transform to audio with autoencoder
    steps = 4
    x_fake = sample_euler(
        model=ldm,
        noise=noise,
        cond=cond,
        num_steps=steps,
        renoise=[0, 0.5, 0.5, 0.3],
        cfg=cfg_scale,
    )
    audio_fake = ldm.autoencoder.inverse(x_fake)

    elapsed = time.perf_counter() - start_time
    log.info(f"Generated in {steps} steps, {elapsed:.2f}s on {device}")
    progress(1.0, desc=f"Done in {steps} steps ({elapsed:.1f}s)")

    # Normalize to prevent clipping
    audio_fake = audio_fake.cpu().float()
    peak = audio_fake.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    audio_fake = (audio_fake / peak * 32767).to(torch.int16)

    results = []
    for i in range(batch_size):
        wav = audio_fake[i]  # (1, T)
        results.append((SAMPLE_RATE, wav.squeeze().numpy()))
    return results


def build_ui():
    with gr.Blocks(title="Woosh Text-to-Audio") as demo:
        gr.Markdown("# Woosh \u2014 Text-to-Audio Generation")

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. sportscar engine revving and driving away quickly",
                scale=3,
            )
            run_btn = gr.Button("Generate", variant="primary", scale=1)

        with gr.Row():
            batch_size = gr.Slider(
                minimum=1, maximum=8, step=1, value=2, label="Batch size"
            )
            cfg_scale = gr.Slider(
                minimum=0.0, maximum=15.0, step=0.1, value=4.5, label="CFG scale"
            )
            seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

        with gr.Accordion("Advanced", open=False):
            solver = gr.Dropdown(
                ["dopri5", "dopri8", "bosh3", "adaptive_heun"],
                value="dopri5",
                label="ODE solver",
            )
            atol = gr.Number(value=0.001, label="Absolute tolerance")
            rtol = gr.Number(value=0.001, label="Relative tolerance")

        @gr.render(
            inputs=[prompt, batch_size, cfg_scale, seed, solver, atol, rtol],
            triggers=[prompt.submit, run_btn.click],
        )
        def render_outputs(prompt, batch_size, cfg_scale, seed, solver, atol, rtol):
            batch_size = int(batch_size)
            audios = generate(
                prompt=prompt,
                batch_size=batch_size,
                cfg_scale=cfg_scale,
                seed=int(seed),
                solver=solver,
                atol=float(atol),
                rtol=float(rtol),
            )
            for i, (sr, wav) in enumerate(audios):
                gr.Audio(value=(sr, wav), label=f"Output #{i + 1}")

    return demo


def main():
    parser = argparse.ArgumentParser(description="Woosh Text-to-Audio Gradio Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Woosh-DFlow",
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )
    parser.add_argument(
        "--server-name", default="127.0.0.1", help="Server address to bind"
    )
    parser.add_argument("--server-port", type=int, default=None, help="Server port")
    args = parser.parse_args()

    load_model(args.checkpoint)
    demo = build_ui()
    demo.launch(
        show_error=True,
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
