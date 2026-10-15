"""
Painting -> Music: web app
==========================

Drop in a photo of a painting: the app shows the painting as it was measured (cropped),
its four grades, how well it matches each music genre, and plays a matching track from
the music/ folder (best genre, closest energy and mood).

    py app.py            # opens in your browser at http://127.0.0.1:7860
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import gradio as gr

import painting_analyzer as pa

ASSETS = pa.HERE / "assets"
LOGO_SVG = ASSETS / "logo.svg"   # palette with one paint dab per genre and a music note
FAVICON = ASSETS / "logo_64.png"
DATASET = pa.HERE / "261013_Dataset_CU_Paintings_JPG-20261014T131814Z-1-001" / "261013_Dataset_CU_Paintings_JPG"
EXAMPLES = [p for p in (DATASET / f"{n}_CU_Paintings.jpg" for n in (1, 15, 38, 52, 76, 113, 150, 196)) if p.exists()]
MOOD_WORDS = {"bright": "bright", "neutral": "balanced", "dark": "dark"}

GENRES = pa.load_genres()
BUNDLE = pa.load_model(pa.DEFAULT_MODEL)
TRACKS = pa.load_music()


def _bar(grade: int) -> str:
    return "■" * grade + "□" * (4 - grade)


def grades_markdown(s: dict[str, int]) -> str:
    rows = [f"| {pa.CATEGORY_TITLES[c]} | {_bar(s[c])} {s[c]}/4 | {pa.SCORE_LABELS[c][s[c]]} |" for c in pa.CATEGORIES]
    return ("### Grades\n| Category | Grade | Meaning |\n|---|---|---|\n" + "\n".join(rows)
            + f"\n\n**Total {sum(s.values())}/16** · colours read as **{MOOD_WORDS[pa.painting_mood(s)]}**")


def track_outputs(state: dict):
    """Audio file and description of the current track in `state`."""
    tracks = pa.pick_music(state["genre"], state["scores"], TRACKS)
    if not tracks:
        return None, f"No music found for **{state['genre']}**. Put mp3/wav files in `music/{state['genre'].replace('/', '_')}/`."
    t = tracks[state["i"] % len(tracks)]
    s = state["scores"]
    energy = "?" if t["Energy"] is None else f"{t['Energy']}/4"
    info = (f"### ♫ \"{t['Title']}\" — {t['Artist']}\n"
            f"{t['Genre']} · energy {energy} · {MOOD_WORDS.get(t['Mood'], t['Mood'])} mood · "
            f"track {state['i'] % len(tracks) + 1} of {len(tracks)}\n\n"
            f"Chosen because the painting has energy {s['energy']}/4 and "
            f"{MOOD_WORDS[pa.painting_mood(s)]} colours.")
    if t.get("License"):
        info += (f"\n\n<sub>\"{t['Title']}\" {t['Artist']} (incompetech.com), licensed under "
                 f"[Creative Commons: By Attribution 4.0](http://creativecommons.org/licenses/by/4.0/)</sub>")
    return str(t["Path"]), info


def analyze(path):
    if not path:
        return None, "", "", None, None, "", None, gr.update()
    r = pa.evaluate(Path(path), GENRES, BUNDLE)
    ranking = r["ranking"]
    (best, best_v), (second, second_v) = ranking[0], ranking[1]
    summary = (f"## {best}\n**{best_v:.0%} match** · second option: {second} ({second_v:.0%})\n\n"
               f"<sub>{'Trained models (CLIP + measurements)' if BUNDLE else 'Fixed rules'}; "
               f"the match % blends the genre table with the AI model.</sub>")
    state = {"scores": r["scores"], "genre": best, "i": 0}
    audio, info = track_outputs(state)
    img = cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB)
    return (img, summary, grades_markdown(r["scores"]), {n: float(v) for n, v in ranking},
            audio, info, state, gr.update(value=best))


def another_track(state):
    if not state:
        return None, "", state
    state = {**state, "i": state["i"] + 1}
    return *track_outputs(state), state


def change_genre(genre, state):
    if not state:
        return None, "", state
    state = {**state, "genre": genre, "i": 0}
    return *track_outputs(state), state


def header_html() -> str:
    logo = ""
    if LOGO_SVG.exists():
        data = base64.b64encode(LOGO_SVG.read_bytes()).decode()
        logo = f'<img src="data:image/svg+xml;base64,{data}" alt="" style="width:72px;height:72px;flex:none">'
    return ('<div style="display:flex;align-items:center;gap:18px">' + logo +
            '<div><h1 style="margin:0">Painting → Music</h1>'
            '<p style="margin:4px 0 0">Add a photo of a painting. The app cuts out the painting, grades it, '
            'finds the music genre that fits, and plays a matching track.</p></div></div>')


def build() -> gr.Blocks:
    with gr.Blocks(title="Painting to Music") as demo:
        gr.HTML(header_html())
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="filepath", label="Photo of a painting", sources=["upload", "clipboard"], height=380)
                go = gr.Button("Analyze", variant="primary")
                if EXAMPLES:
                    gr.Examples([str(p) for p in EXAMPLES], inputs=photo, label="Try a painting from the collection")
            with gr.Column():
                painting = gr.Image(label="The painting as measured", interactive=False, height=380)
                summary = gr.Markdown()
        with gr.Row():
            with gr.Column():
                genre_bars = gr.Label(label="Genre match", num_top_classes=4)
                grades = gr.Markdown()
            with gr.Column():
                audio = gr.Audio(label="Music", type="filepath", autoplay=True, interactive=False)
                info = gr.Markdown()
                with gr.Row():
                    genre_pick = gr.Dropdown([g.name for g in GENRES], label="Play music from genre", interactive=True)
                    another = gr.Button("Another track")
        state = gr.State()

        outputs = [painting, summary, grades, genre_bars, audio, info, state, genre_pick]
        go.click(analyze, photo, outputs)
        photo.change(analyze, photo, outputs)
        another.click(another_track, state, [audio, info, state])
        genre_pick.input(change_genre, [genre_pick, state], [audio, info, state])
    return demo


if __name__ == "__main__":
    if BUNDLE is not None and BUNDLE["clip"]:
        pa._load_clip()  # load CLIP now, so the first painting is fast
    print(f"{len(TRACKS)} music tracks found in {pa.DEFAULT_MUSIC}")
    build().queue().launch(inbrowser=True, allowed_paths=[str(pa.HERE)], theme=gr.themes.Soft(),
                           favicon_path=str(FAVICON) if FAVICON.exists() else None)
