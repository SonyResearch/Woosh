"""
Simple UI that calls the SFXFM API to generate sound files based on text prompts
and inserts them at the current cursor position in REAPER.

launch it with
> python reapy_script.py --ui

make sure to have reapy installed and configured
and that the SFXFM API server is running
"""

import reapy
import httpx
import os
import uuid
import logging
import subprocess
import re
import random
import click
from reapy import reascript_api as RPR

logger = logging.getLogger("sfxfm")
logging.basicConfig(level=logging.INFO)

PORT = 8000
API_URL = f"http://0.0.0.0:{PORT}"
api_url = f"{API_URL}/generate"

def get_tcl_lib_dirs_from_brew():
    """ Nasty way of getting TCL_LIBRARY and TK_LIBRARY paths from `brew info` """
    try:
        bi = subprocess.check_output("brew info tcl-tk", shell=True).decode().split("\n")
    except:
        raise FileNotFoundError(f"finding TCL/TK library folders automatically failed: set the TCL_LIBRARY and TK_LIBRARY environment variables manually in `reapy_script.py`")
    for n, line in enumerate(bi):
        if "Installed" in line:
            # get path from next line
            line = bi[n+1]
            path = re.sub(r"[ ].*", "", line)
            version_major_minor = os.path.basename(path)
            parts = version_major_minor.split(".")
            if len(parts)>2:
                version_major_minor = ".".join(parts[:2])

            tcl_lib_path = os.path.join(path, "lib", f"tcl{version_major_minor}")
            if not os.path.exists(tcl_lib_path):
                raise FileNotFoundError(f"TCL library dir {tcl_lib_path} not found: set the TCL_LIBRARY environment variable manually")
            tk_lib_path = os.path.join(path, "lib", f"tk{version_major_minor}")
            if not os.path.exists(tk_lib_path):
                raise FileNotFoundError(f"TCL library dir {tk_lib_path} not found: set the TCL_LIBRARY environment variable manually")
                
            return tcl_lib_path, tk_lib_path


# RPR_APITest()
def generate(prompt: str) -> str:
    """ Call SFXFM API to generate audio from a prompt """
    headers = {"Accept": "application/json"}

    data = {
        "version": "0.1",
        "token": "string",
        "args": {
            "model": "SFXflowmap",
            "prompt": prompt,
            "cfg": 3.0,
            "sampler": "heun",
            "num_steps": 5,
            "sigma_min": 0.0001,
            "sigma_max": 80,
            "rho": 7,
            "S_churn": 1,
            "S_min": 0,
            "S_noise": 1,
            "guidance_scale": 7.5,
            "noise_scheduler": "karras",
            "seed": random.randint(0, 2**32 - 1),
        },
    }
    response = httpx.post(api_url, json=data, headers=headers, timeout=45.0)

    # Generate a random filename
    filename = f"{uuid.uuid4().hex}.flac"
    save_dir = "/tmp/sfxfm_flac"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    logger.info(
        f"Received response from SFXFM API: {response.status_code}, {len(response.content)} bytes"
    )
    # Save the FLAC file
    with open(filepath, "wb") as f:
        f.write(response.content)
        logger.info(f"Saved FLAC file to {filepath}")

    # Upload the FLAC file to a temporary file hosting service
    # return upload_flac_file(filepath)

    return filepath


def insert_file_at_cursor(file_path: str):
    """Insert a file at the current cursor position in REAPER."""
    if not os.path.isfile(file_path):
        reapy.print(f"File does not exist: {file_path}")
        return
    # cursor_position = RPR.GetCursorPosition()

    project = reapy.Project()  # Current project
    # cursor_position = project.cursor_position

    if project.n_tracks == 0:
        project.add_track(index=0)

    RPR.InsertMedia(file_path, 0)


def ui_main():
    import tkinter as tk
    from tkinter import messagebox

    def on_insert():
        prompt = entry.get()
        if not prompt.strip():
            messagebox.showwarning("Input Error", "Please enter a prompt.")
            return
        try:
            file = generate(prompt)
            reapy.print(f"Generated sound file: {file}")
            insert_file_at_cursor(file)
            # messagebox.showinfo("Success", f"Inserted sound file: {file}")
        except Exception as e:
            # messagebox.showerror("Error", str(e))
            pass

    root = tk.Tk()
    root.title("SFXFM Sound Inserter")
    root.attributes("-topmost", True)
    root.geometry("400x180")

    entry_font = ("Arial", 22)
    label_font = ("Arial", 12)
    button_font = ("Arial", 12)

    tk.Label(root, text="Description:", font=label_font).pack(padx=10, pady=8)
    entry = tk.Entry(root, width=32, font=entry_font)
    entry.pack(padx=10, pady=8)
    entry.bind("<Return>", lambda event: on_insert())
    insert_btn = tk.Button(root, text="Insert", command=on_insert, font=button_font)
    insert_btn.pack(padx=10, pady=10)
    root.mainloop()


def cli(prompt):
    """Run CLI to generate and insert sound file."""

    if os.name == "Darwin":
        # automatically get TCL and TK library path from brew
        tcl_lib_path, tk_lib_path = get_tcl_lib_dirs_from_brew()
    else:
        if "TCL_LIBRARY" not in os.environ:
            # set path manually here
            os.environ["TCL_LIBRARY"] = os.path.expanduser(
                "/opt/homebrew/Cellar/tcl-tk/9.0.3/lib/tcl9.0/"
            )
        if "TK_LIBRARY" not in os.environ:
            # set path manually here
            os.environ["TK_LIBRARY"] = os.path.expanduser(
                "/opt/homebrew/Cellar/tcl-tk/9.0.3/lib/tk9.0/"
            )

    try:
        reapy.print(f"Generating sound for prompt: {prompt}")
        file = generate(prompt)
        if file and os.path.isfile(file):
            reapy.print(f"Generated sound file: {file}")
            insert_file_at_cursor(file)
            click.echo(f"Inserted sound file: {file}")
        else:
            click.echo("Failed to generate or locate sound file.")
    except Exception as e:
        click.echo(f"Error: {e}")


@click.command()
@click.option(
    "--ui",
    is_flag=True,
    help="Launch Tkinter UI for sound generation and insertion.",
)
def entry(ui):
    if ui:
        ui_main()
    else:
        cli("A short, dark, and ominous soundscape with deep bass and eerie textures.")


if __name__ == "__main__":

    entry()
