from pathlib import Path
import yaml

VIDEO_DEMO_DIR = Path("static/video_demo")

# Description shown at the top of the section (plain text or HTML).
SECTION_DESCRIPTION = (
    "We compare our models against baselines across several benchmarks. "
    "<b>OV</b> means the model is conditioned only on video without the caption."
)

# Dataset order and display names.
# Keys are folder names; values are the display titles shown in the HTML.
# Datasets not listed here are appended at the end (sorted) using an
# auto-generated title (underscores → spaces, title-cased).
DATASET_ORDER = {
    "foleybench": "FoleyBench",
    "ogame": "OGameData",
    "vggsound": "VGGSound",
    "vggsound_recap": "VGGSound (recaptioned)",
}


def dataset_title(folder_name: str) -> str:
    """Return the display title for a dataset folder."""
    if folder_name in DATASET_ORDER:
        return DATASET_ORDER[folder_name]
    return folder_name.replace("_", " ").title()


# Collect all datasets (subdirectories containing a manifest.yaml)
_available = {
    d.name: d
    for d in VIDEO_DEMO_DIR.iterdir()
    if d.is_dir() and (d / "manifest.yaml").exists()
}
# Build ordered list: explicit order first, then any extras sorted alphabetically.
_ordered_names = [n for n in DATASET_ORDER if n in _available]
_remaining = sorted(n for n in _available if n not in DATASET_ORDER)
datasets = [_available[n] for n in _ordered_names + _remaining]


# ------------------------------------------------------------------
# HTML output
# ------------------------------------------------------------------

print(f"""
<section class="hero is-small is-light">
  <div class="hero-body">
    <div class="container">
      <h2 class="title is-3 is-centered has-text-centered">Video-to-Audio Generation Demos</h2>
      <p class="has-text-centered" style="margin-bottom: 1.5rem;">{SECTION_DESCRIPTION}</p>
""")

for dataset_dir in datasets:
    dataset_name = dataset_dir.name
    manifest_path = dataset_dir / "manifest.yaml"

    with open(manifest_path, "r") as fp:
        manifest = yaml.safe_load(fp)

    samples = manifest.get("samples", [])

    print(
        f"""      <h3 class="title is-4 is-centered has-text-centered">{dataset_title(dataset_name)}</h3>"""
    )
    print("""      <div class="columns is-centered has-text-centered">
        <div class="column is-full">
""")

    for sample in samples:
        caption = sample.get("caption", "")
        videos = sample.get("videos", {})

        print(f"""          <div class="box" style="margin-bottom: 1.5rem;">""")
        print(f"""            <p style="margin-bottom: 1rem;"><em>{caption}</em></p>""")
        print(
            f"""            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 12px;">"""
        )

        for model_name, video_rel_path in videos.items():
            video_src = f"{VIDEO_DEMO_DIR}/{dataset_name}/{video_rel_path}"
            print(
                f"""              <figure style="display: inline-block; text-align: center;">"""
            )
            print(
                f"""                <figcaption style="font-weight: bold; margin-bottom: 4px;">{model_name}</figcaption>"""
            )
            print(
                f"""                <video controls style="width: 220px;" preload="metadata" class="video-thumb">"""
            )
            print(f"""                  <source src="{video_src}" type="video/mp4">""")
            print(f"""                </video>""")
            print(f"""              </figure>""")

        print(f"""            </div>""")
        print(f"""          </div>""")

    print("""        </div>
      </div>
""")

print("""    </div>
  </div>
</section>
<script>
// Seek every video to 0.1 s after metadata loads so browsers paint a thumbnail.
document.querySelectorAll('video.video-thumb').forEach(function(v) {
  v.addEventListener('loadedmetadata', function() {
    if (v.readyState >= 1) { v.currentTime = 0.1; }
  });
  // In case metadata was already loaded before the listener was attached.
  if (v.readyState >= 1) { v.currentTime = 0.1; }
});
</script>
""")

# python generate-video-section.py > video-section.html
# python -m http.server 8000
