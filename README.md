# Woosh - Sound Effect Generative Models

This repository provides inference code and open weights for the sound effect generative models developed at Sony AI. The
current public release includes four models addressing the text-to-audio (T2A) and video-to-
audio (V2A) tasks:

- **Audio encoder/decoder (Woosh-AE)**: High-quality latent encoder/decoder providing latents
for generative modeling and decoding audio from generated latents.

- **Text conditioning (Woosh-CLAP)**: Multimodal text-audio alignment model providing token la-
tents for diffusion model conditioning.

- **T2A Generation (Woosh-Flow and Woosh-DFlow)**: Original and distilled LDMs generating au-
dio unconditionally or from given a text prompt.

- **V2A Generation (Woosh-VFlow)**: Multimodal LDM generating audio from a video sequence
with optional text prompts.

## Installation

Start by installing [uv](https://docs.astral.sh/uv) first

```bash
 pip install uv
```

and then the Woosh environment, with either:

`cpu` support,

```bash
uv sync --extra cpu
```

or `cuda` support,

```bash
uv sync --extra cuda
```

### Download model weights

Open model weights are available for all Woosh models trained on public datasets. You can download
and unzip the pretrained weights from the [releases](https://github.com/SonyResearch/woosh-sfx/releases)
page, or otherwise using the [github CLI](https://cli.github.com) as

```bash
gh release download v0.1.1
unzip *.zip
```

The checkpoints should be located in folders named `checkpoints/MODEL_NAME`, each containing config and weight files.

### Download media samples

We provide audio samples to be used as inputs to our `test_Woosh-*.py` test scripts. You can download
and unzip the file `samples.zip` from the [releases](https://github.com/SonyResearch/woosh-sfx/releases)
page, or otherwise using the [github CLI](https://cli.github.com) as

```bash
gh release download v0.1.1 -p 'samples.zip'
unzip samples.zip
```

## Usage

An inference test script for every model is provided. Just run any of the following

```python
uv run test_Woosh-AE.py
uv run test_Woosh-Flow.py
uv run test_Woosh-DFlow.py
uv run test_Woosh-VFlow.py
```

and the generated audio/video will be written to `outputs/` as `.wav`/`.mp4` audio/video files.

Check our [tech report](https://arxiv.org/abs/2412.15322) on arxiv.org for a description of all models.

## Citation
For details about model architecture, training and evaluation, please check our tech report
available on [arxiv.org](https://arxiv.org/abs/2412.15322).

```bibtex
@misc{hadjeres2026,
      title={Woosh: A Sound Effects Foundation Model},
      author={Gaetan Hadjeres, Marc Ferras, Khaled Koutini, Benno Weck-Hufnagel, Alexandre Bittar, Thomas Hummel, Zineb Lahrici, Hakim Missoum, Joan Serrà and Yuki Mitsufuji},
      year={2026},
      eprint={2412.15322},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.15322},
}
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
- The inference code in this repository is released under a [MIT](https://choosealicense.com/licenses/mit/) license.
- The open weights, in the [releases](https://github.com/SonyResearch/woosh-sfx/releases) page, are released under the [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license.
