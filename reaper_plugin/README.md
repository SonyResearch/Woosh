# Installing Reapy
Reapy should already installed after running
> uv sync

Make Reapy aware of your REAPER installation:

Path for REAPER
on macOS
install python with Brew

Enable Reascript in Reaper
Open Reaper Preferences (Settings), under Plug-ins -- Reascript, select Enable Python for use with ReaScript.
The next field is: Custom path to Python dll directory
On Apple Silicon with Python installed using homebrew, you can find the libpython$PYTHON_VERSION.dylib in /opt/homebrew/opt/python@$PYTHON_VERSION/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib, relace $PYTHON_VERSION with the version you intend to use, e.g. 3.14.
In the dylib file, type in the name of the dylib file, e.g libpython3.12.dylib
Restart Reaper
For some reason, providing the dylibs from uv did not work


Open REAPER and then in a terminal:
```python -c "import reapy; reapy.configure_reaper()````


on macos; issue with the 
brew install tcl-tk

and set the correct path (e.g. in uv)
os.environ["TCL_LIBRARY"] = os.path.expanduser(
    "~/.local/share/uv/python/cpython-3.13.0-macos-aarch64-none/lib/tcl8.6"
)
os.environ["TK_LIBRARY"] = os.path.expanduser(
    "~/.local/share/uv/python/cpython-3.13.0-macos-aarch64-none/lib/tk8.6"
)

## Launch
- Run the API
- launching the reaper plugin with
> uv run reaper_plugin.reapy_script.py --ui

