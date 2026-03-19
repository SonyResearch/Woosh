"""Woosh Text+Video-to-Audio Gradio Demo"""

import argparse
import logging
import os
import tempfile
import time

import gradio as gr
import torch

from api.utils import CLAPCaptionPostprocessTransform
from woosh.components.base import LoadConfig
from woosh.inference.flowmatching_sampler import flowmatching_integrate
from woosh.model.video_kontext import VideoKontext
from woosh.utils.video import SynchformerProcessor
from woosh.utils.videoio import extract_video_frames, remux_video

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
LATENT_CHANNELS = 128
DURATION_SECONDS = 8
LATENT_FRAMES = 801  # ~8s at 48kHz after autoencoder

# Global model state
ldm = None
features_model = None
device = None


def load_model(checkpoint_path: str):
    global ldm, features_model, device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info(f"Loading model from {checkpoint_path} on {device}")
    ldm = VideoKontext(LoadConfig(path=checkpoint_path))
    ldm = ldm.eval().to(device)
    features_model = SynchformerProcessor(frame_rate=24).eval().to(device)
    log.info("Model loaded.")


normalize_transform = CLAPCaptionPostprocessTransform(remove_punctuation=False)


def normalize_text(text):
    # Normalize the text by removing special characters and replacing spaces with underscores
    res = normalize_transform({"captions": [text]})
    text = res["captions"][0]
    print("normalized text:", text)
    return text


@torch.inference_mode()
def generate(
    video_path: str,
    prompt: str,
    negative_prompt: str = "",
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
    if not video_path:
        raise gr.Error("Please upload a video.")

    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)
    torch.manual_seed(seed)

    progress(0.05, desc="Extracting video frames...")
    video_frames, video_rate, pts_arr = extract_video_frames(
        video_path,
        start_time=0,
        end_time=DURATION_SECONDS,
    )
    video_frames = video_frames.to(device)

    progress(0.15, desc="Computing video features...")
    features = features_model(video_frames, video_rate)

    noise = torch.randn(batch_size, LATENT_CHANNELS, LATENT_FRAMES).to(device)

    prompt = normalize_text(prompt) if prompt else ""

    cond = ldm.get_cond(
        {
            "audio": None,
            "description": [prompt] * batch_size,
            "synch_out": features["synch_out"].expand(batch_size, -1, -1),
        },
        no_dropout=True,
        device=device,
    )

    cond_neg = None
    if negative_prompt.strip():
        negative_prompt = normalize_text(negative_prompt)
        cond_neg = ldm.get_cond(
            {
                "audio": None,
                "description": [negative_prompt] * batch_size,
                "synch_out": features["synch_out"].expand(batch_size, -1, -1),
            },
            no_dropout=True,
            device=device,
        )

    progress(0.25, desc="Generating audio...")
    gen_start = time.perf_counter()

    x_fake, steps = flowmatching_integrate(
        ldm,
        noise=noise,
        cond=cond,
        cond_neg=cond_neg,
        cfg=cfg_scale,
        atol=atol,
        rtol=rtol,
        method=solver,
        return_steps=True,
        device=device,
        dtype=torch.float32 if device == "mps" else torch.float64,
    )
    audio_fake = ldm.autoencoder.inverse(x_fake)

    elapsed = time.perf_counter() - gen_start
    log.info(f"Generated in {steps} steps, {elapsed:.2f}s on {device}")
    progress(0.9, desc=f"Done in {steps} steps ({elapsed:.1f}s). Muxing video...")

    # Normalize to prevent clipping
    audio_fake = audio_fake.cpu().float()
    peak = audio_fake.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    audio_fake = audio_fake / peak

    results = []
    for i in range(batch_size):
        wav = audio_fake[i]
        # Create a temporary video file with muxed audio
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        remux_video(
            output_path=tmp.name,
            video_path=video_path,
            audio_input=wav,
            sample_rate=SAMPLE_RATE,
            audio_start=0,
            duration_seconds=DURATION_SECONDS,
        )
        results.append(tmp.name)

    progress(1.0, desc="Done!")
    return results


def build_ui():
    with gr.Blocks(title="Woosh Text+Video-to-Audio") as demo:
        gr.Markdown("# Woosh — Text + Video to Audio Generation")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt (optional)",
                    placeholder="e.g. A person shovels snow, making scraping sounds.",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt (optional)",
                    placeholder="e.g. music, speech",
                    lines=1,
                )
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, value=1, label="Batch size"
                    )
                    cfg_scale = gr.Slider(
                        minimum=0.0,
                        maximum=15.0,
                        step=0.1,
                        value=4.5,
                        label="CFG scale",
                    )
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                run_btn = gr.Button("Generate", variant="primary")

        with gr.Accordion("Advanced", open=False):
            solver = gr.Dropdown(
                ["dopri5", "dopri8", "bosh3", "adaptive_heun"],
                value="dopri5",
                label="ODE solver",
            )
            with gr.Row():
                atol = gr.Number(value=0.001, label="Absolute tolerance")
                rtol = gr.Number(value=0.001, label="Relative tolerance")

        @gr.render(
            inputs=[
                video_input,
                prompt,
                negative_prompt,
                batch_size,
                cfg_scale,
                seed,
                solver,
                atol,
                rtol,
            ],
            triggers=[run_btn.click],
        )
        def render_outputs(
            video_input,
            prompt,
            negative_prompt,
            batch_size,
            cfg_scale,
            seed,
            solver,
            atol,
            rtol,
        ):
            batch_size = int(batch_size)
            videos = generate(
                video_path=video_input,
                prompt=prompt,
                negative_prompt=negative_prompt,
                batch_size=batch_size,
                cfg_scale=cfg_scale,
                seed=int(seed),
                solver=solver,
                atol=float(atol),
                rtol=float(rtol),
            )
            for i, video_path in enumerate(videos):
                gr.Video(value=video_path, label=f"Output #{i + 1}", width=480)

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Woosh Text+Video-to-Audio Gradio Demo"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Woosh-VFlow-8s",
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
