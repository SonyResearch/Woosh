# Generating videos for the demo

Use `notebooks/koutini/video_demo_generate_script.py` in the main repo

## Example usage:

```bash
# Test:
python notebooks/koutini/video_demo_generate_script.py --models medium_44k pbds3-8s-e70 --indices 500,502 --output-dir ./video_demo/ ogame
# Help:
python notebooks/koutini/video_demo_generate_script.py --help

# extracting samples from Datasets

python notebooks/koutini/video_demo_generate_script.py --models medium_44k pbds3-8s-e70  mfv-4step-cfg3  medium_44k_uncond pbds3-8s-e70-uncond mfv-4step-cfg3-uncond --indices 400,500,4600,4683,771,859,1001,2900,241,2971,2964 --output-dir ./video_demo/ foleybench

python notebooks/koutini/video_demo_generate_script.py --models medium_44k pbds3-8s-e70  mfv-4step-cfg3  medium_44k_uncond pbds3-8s-e70-uncond mfv-4step-cfg3-uncond --indices 143,504,521,780,800,1600,1620 --output-dir ./video_demo/ ogame

python notebooks/koutini/video_demo_generate_script.py --models medium_44k pbds3-8s-e70  mfv-4step-cfg3  medium_44k_uncond pbds3-8s-e70-uncond mfv-4step-cfg3-uncond --indices 9105,12109,12111,12113 --output-dir ./video_demo/ vggsound
python notebooks/koutini/video_demo_generate_script.py --models medium_44k_recap pbds3-8s-e70-recap mfv-4step-cfg3-recap --indices 9105,12109,12111,12113 --output-dir ./video_demo/ vggsound_recap

```

then copy the generated demo to the docs static folder in the public repo.

```bash
cp -r video_demo /home/khaled.koutini/workspaces/sfxfm_public/docs/static/
```

 then use docs/generate-video-section.py to generate the video html

```bash
# only few samples for the index page
python generate-video-section.py index > video-index-sample.html
# regenerate the video.html page
python generate-video-section.py > videos.html
# test the server
python -m http.server 8000
```