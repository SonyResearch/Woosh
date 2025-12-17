# SFXFM

Public release of the Sound Effect Foundation model by Sony AI.

## Installation

Use [uv](https://docs.astral.sh/uv) to install the SFXFM environment, with either

`cpu` support

```bash
uv sync --extra cpu
```

or `cuda` support

```bash
uv sync --extra cuda
```

### Download model weights

You can download the pretrained model weights from the [releases](https://github.com/SonyResearch/SFXFM/releases) page, or otherwise from the [github CLI](https://cli.github.com) as

```bash
gh release download v0.1.1
unzip SFX\*.zip
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
