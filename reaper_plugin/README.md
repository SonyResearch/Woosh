This is a proof-of-concept sound effect generation plugin that allows direct media drop-in
on the [Reaper DAW](https://www.reaper.fm) timeline. It provides a
way to interact with professional audio tools. The plugin queries our [SFXFM api](../api)
to generate audio, storing the audio output locally, and Reaper is instructed to load it at the
current location in the timeline via the Python [reapy](https://pypi.org/project/python-reapy) package.
Please check our [demo video](https://github.com/SonyResearch/SFXFM/releases/download/v0.1.1/reaper-plugin-demo.mp4)
for an overview of the plugin usage.

The following instructions have been tested on an Apple Silicon Mac. Please follow analogous instructions on Windows or Linux platforms.

# Installing `reapy`
The `reapy` package is already part of the SFXFM `uv` environment. Run

```
uv sync --extra cpu
```

to sync the environment, and

```
uv pip list | grep reapy
```

to double check it has been properly installed.

# Installing Reaper's ReaScript/Python
Controlling Reaper via ReaScript/Python requires a Python install that is to be set up in Reaper.
Use a Python `brew` install and write down the Python version:

```
brew install python
```

To set up Reaper to use ReaScript/Python:

1. Find the dynamic library path and file for the Python install above. Do this either manually,
  or feel lucky running the following `bash` script lines:

```
PYTHON_VERSION=`brew info python | grep -A 1 "Installed" | tail -1 | sed -e "s/.*python@\([0-9]\.[0-9]*\).*/\1/g"`
echo PYTHON_VERSION=$PYTHON_VERSION
echo PYTHON_DYLIB_DIR=`brew info python | grep -A 1 "Installed" | tail -1 | sed -e "s/[ ].*//g"`/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib
echo PYTHON_DYLIB_FILE=libpython${PYTHON_VERSION}.dylib
```
2. Open `Reaper` and go to `Reaper->Setting->ReaScript` menu:

  - Check `Enable Python for use with ReaScript`
  - Set `Custom path to Python dll directory` to PYTHON_DYLIB_DIR above
  - Set `Force ReaScript to use specific Python .dylib` to PYTHON_DYLIB_FILE above

3. Restart Reaper

# Setup Tkinter for the Reaper plugin
Our Reaper plugin uses Tcl/Tk for the UI. To set it up, run:

1. Install TCL/TK

```
brew install tcl-tk
```

2. Find the location of the TCL and TK libraries if necessary

Our Reapy script attempts to find the TCL and TK library folders automatically on the Mac platform only.
If you are on Windows or Linux, or the process fails on Mac you will have to manually find the right
TCL/TK folders for your machine and set them as `TCL_LIBRARY` and `TK_LIBRARY` enviroment variables, inside
the `reapy_script.py`.


# Launch

1. Launch the API server
Get the API server running. This will receive `http` requests from `reapy_script.py`,
generate audio for the request and store the audio on `/tmp`.

```
uv run uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```

2. Launch the Reapy script

This script will build a simple Tkinter UI dialog where users can enter a text prompt. The script
will forward the prompt to the API server, and finally instruct Reaper to place the audio in the timeline

```
uv run python reapy_script.py --ui
```

3. Launch Reaper

Click on the Reaper application on your system.


# Troubleshooting
A way to make sure `reapy` connects properly to the Reaper application is running

```
python -c "import reapy; reapy.configure_reaper()````
```

on the terminal, with Reaper open and properly set up for ReaScript as described above.

There is a known bug on the `configparser` package that `reapy==0.10.0` depends on. If you encounter an
error regarding `UNNAMED_SECTION` you can edit the `configparser.py` file on the current environment
and replace the line

```
if UNNAMED_SECTION in self._sections:
```

by

```
if str(UNNAMED_SECTION) in self._sections:
```
