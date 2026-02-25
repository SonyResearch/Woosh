from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import soundfile
import librosa
from librosa.display import specshow
import yaml

def plot_specgrams(sample_name, files, n_fft=1024, hop_size=16, y_axis="linear"):

    filenames = [ item["filename"] for model_name, item in files.items() ]
    sample_names_as_in_file = [ Path(fn).stem.split("-")[0] for fn in filenames ]

    output_dir = Path(Path(filenames[0]).parent)
    xs = [ soundfile.read(item["filename"]) for model_name, item in files.items() ]
    regions = [ item["region"] for model_name, item in files.items() ]
    sample_names = [ sample_name for model_name, item in files.items() ]
    model_names = [ model_name for model_name, item in files.items() ]

    from_time = regions[0][0]
    to_time = regions[0][1] if regions[0][1]>0 else len(xs[0][0])/xs[0][1]
    v_size = 2 * len(xs)
    h_size = int(0.5*(to_time-from_time)*v_size)
    fig, axs = plt.subplots(nrows=len(xs), figsize=(h_size, v_size))
    for n, ((x, fs), region, sample_name_as_in_file, model_name) in enumerate(zip(xs, regions, sample_names_as_in_file, model_names)):


        from_time, to_time = region[0], region[1]
        from_sample = int(from_time * fs)
        to_sample = int(to_time * fs)
        print(f"x={len(x)}, from={from_sample}, to_sample={to_sample}")
        x = x[from_sample:to_sample+1]

        specgram = librosa.amplitude_to_db(np.abs(librosa.stft(x, hop_length=hop_size, n_fft=n_fft)), ref=np.max)
        ax = axs[n]
        specshow(specgram, y_axis=y_axis, sr=fs, hop_length=hop_size, x_axis=None, ax=ax, auto_aspect=True)

        ax.set_title(model_name)
        yticks = np.linspace(0, 20000, 3)
        ylabels = [ f"{y:.0f}Hz" for y in yticks ]
        ax.set_yticks(yticks, labels=ylabels)
        ax.get_xaxis().set_visible(False)

        ax.set_ylabel("")

    # plt.show()
    figure_fn = output_dir / f"{sample_name_as_in_file}-specgrams.png"
    plt.savefig(figure_fn, dpi=300)
    plt.close(fig)

    files["figure"] = figure_fn

# load sample generation config
with open("generate-ae-samples.yaml", 'r') as fp:
    files = yaml.safe_load(fp)

for sample_name in files:
    plot_specgrams(sample_name, files[sample_name], y_axis="linear")

print("""
<section class="hero is-small is-light">
  <div class="hero-body">
    <div class="container">
      <h2 class="title is-3 is-centered has-text-centered">Woosh-AE: Audio Encoder/Decoder</h2>
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
""")

for sample, d in files.items():
    print(f"""          <div><b>{sample}</b> - <a href="{d["figure"]}" target="_blank" >Spectrograms</a></div>\n""")
    del d["figure"]
    for n, (model, item) in enumerate(d.items()):
        # if model == "figure":
        #     continue
        filename = item["filename"]
        print( """          <figure style="display: inline-block;">\n"""
              f"""            <figcaption>{model}</figcaption>\n"""
              f"""            <audio controls src="{filename}" style="width:200px;"></audio>\n"""
               """          </figure>"""
        )
        if n<len(d)-1:
            print("""           &nbsp;""")
        else:
            print("""           <br><br>""")

print("""
        </div>
      </div>
    </div>
  </div>
</section>
""")
