# Painting → Music

Grades a photo of a painting from 1 to 4 in four categories (Colours, Colour palette, Energy, Brushstrokes), finds the music genre that fits (Classical, Jazz, Ambient, Rock/Metal) and plays a matching track. It uses a CLIP image model adapted to art on WikiArt, together with 38 image measurements.

The full description of the grading, the models and the evaluation is in [GRADING_SYSTEM.md](GRADING_SYSTEM.md).

## Setup

```bash
git clone https://github.com/coffinovicha/temp.git
cd temp
curl -L -o clip_art_adapter.pt https://github.com/coffinovicha/temp/releases/download/v1/clip_art_adapter.pt
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt gradio
```

`clip_art_adapter.pt` (153 MB) is too large for a git repository, so it is attached to the [v1 release](https://github.com/coffinovicha/temp/releases/tag/v1). The CLIP model itself (3.7 GB) downloads automatically the first time the app starts.

## Run

```bash
python3 app.py                     # Windows: py app.py, or double-click "Start app.bat"
```

The app opens in your browser at http://127.0.0.1:7860.

## Not included

The 196 painting photos are not in this repository; they are copyrighted artworks. The app works without them. They are only needed for the example pictures in the app and for retraining with `painting_analyzer.py train`.

## Music

All tracks in `music/` are by Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 4.0 (http://creativecommons.org/licenses/by/4.0/). See `music/CREDITS.txt`.
