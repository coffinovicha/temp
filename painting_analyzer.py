"""
Painting -> Music Genre Analyzer
================================

Looks at a painting, grades it 1-4 in four categories, and finds the music
genre from genres.csv that fits it best.

Before measuring, the painting is cut out of the photo (wall, frame and mat removed).
Photos the automatic crop gets wrong can be cropped by hand in crops.csv.
Grades come from models trained on labels.csv when painting_model.joblib exists,
otherwise from fixed rules (THRESHOLDS). The trained models use the CLIP image model
(Apple DFN5B ViT-H/14, adapted to art on WikiArt: clip_art_adapter.pt) together with
38 measurements, when torch and open_clip are installed.

Categories (each graded 1-4):
    Colours         1 = muted      2 = natural     3 = bright         4 = vibrant
    Colour palette  1 = pastel     2 = harmonised  3 = deep / warm    4 = high contrast
    Energy          1 = (very) low 2 = medium      3 = high           4 = very high
    Brushstrokes    1 = invisible  2 = subtle      3 = visible        4 = dominant

Commands
--------
  Choose photos in a dialog and see each one with its result in a window:
      py painting_analyzer.py analyze

  Analyze one painting or a whole folder of paintings:
      py painting_analyzer.py analyze my_painting.jpg --show
      py painting_analyzer.py analyze paintings_folder --csv results.csv
      py painting_analyzer.py analyze paintings_folder --cards result_pictures

  Match a description (words or numbers) without an image:
      py painting_analyzer.py match --colors vibrant --palette "high contrast" --energy "very high" --brushwork dominant
      py painting_analyzer.py match --colors 1 --palette 3 --energy 2 --brushwork 2

  Match every row of a table you wrote yourself (same columns as genres.csv):
      py painting_analyzer.py match --table my_paintings.csv

  Train the AI on a labels table (grades and/or genre per painting):
      py painting_analyzer.py train paintings_folder --labels labels.csv

  ...or on paintings sorted into one sub-folder per genre (genre only):
      dataset/Classical/*.jpg   dataset/Jazz/*.jpg   dataset/Ambient/*.jpg   dataset/Rock_Metal/*.jpg
      py painting_analyzer.py train dataset

After training, "analyze" uses the trained grade models and blends the trained
genre model with the genre table. Add --no-model to use only the fixed rules,
and --no-crop to measure the whole photo.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_GENRES = HERE / "genres.csv"
DEFAULT_MODEL = HERE / "painting_model.joblib"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANALYSIS_SIZE = 768  # longest side in pixels, keeps measurements comparable

CATEGORIES = ["colors", "palette", "energy", "brushwork"]
CATEGORY_TITLES = {
    "colors": "Colours",
    "palette": "Colour palette",
    "energy": "Energy",
    "brushwork": "Brushstrokes",
}
SCORE_LABELS = {
    "colors": {1: "muted", 2: "natural", 3: "bright", 4: "vibrant"},
    "palette": {1: "pastel, light", 2: "elegant, harmonised", 3: "deep, warm", 4: "high contrast"},
    "energy": {1: "low", 2: "medium", 3: "high", 4: "very high"},
    "brushwork": {1: "not visible", 2: "subtle", 3: "visible", 4: "dominant"},
}

# Grade boundaries for the rule-based grading (used when no trained model exists).
# Fitted to the 196 hand-labelled paintings in labels.csv (cropped photos).
THRESHOLDS = {
    "colors": (23.0, 44.0, 66.0),        # on the "colorfulness" measure
    "energy": (0.43, 0.62, 0.79),        # on the 0-1 "energy" measure
    "brushwork": (0.014, 0.041, 0.081),  # on the 0-1 "fine texture" measure
    "palette_high_contrast": 0.23,       # brightness spread that counts as high contrast
    "palette_colourful_min": 60.0,       # this colourful...
    "palette_colourful_contrast": 0.14,  # ...with at least this contrast also counts as high contrast
    "palette_pastel_min_value": 0.65,    # average brightness above this can be pastel
    "palette_pastel_max_sat": 0.30,      # ...if the colours are this unsaturated
    "palette_deep_max_value": 0.40,      # average brightness below this counts as deep
    "palette_warm_min": 0.85,            # this share of warm colours...
    "palette_warm_max_value": 0.65,      # ...below this brightness counts as warm
}

# How much a trained model counts when blended with the genre table (0 = table only).
MODEL_WEIGHT = 0.5


# --------------------------------------------------------------------------- #
# Genre table
# --------------------------------------------------------------------------- #

# Each phrase maps to the grade it stands for (None = not applicable).
# Longer phrases are checked first, so "very high" is not also read as "high".
# Typos such as "suibtle" are caught by fuzzy matching.
VOCAB = {
    "colors": {
        "muted": 1, "dull": 1, "desaturated": 1, "grey": 1, "gray": 1,
        "natural": 2, "soft": 2, "earthy": 2,
        "bright": 3, "luminous": 3, "light": 3,
        "vibrant": 4, "vivid": 4, "saturated": 4, "intense": 4,
    },
    "palette": {
        "pastel": 1, "light": 1, "soft": 1,
        "elegant": 2, "harmonised": 2, "harmonized": 2, "harmonious": 2, "balanced": 2,
        "deep colors": 3, "deep colours": 3, "deep": 3, "warm beige": 3, "warm": 3,
        "beige": 3, "dark": 3,
        "high contrast": 4, "contrast": 4, "complementary": 4,
    },
    "energy": {
        "very low": 1, "very high": 4,
        "low": 1, "calm": 1, "medium": 2, "moderate": 2, "high": 3,
    },
    "brushwork": {
        "not applicable": None, "n/a": None, "na": None,
        "none": 1, "invisible": 1, "hidden": 1, "smooth": 1,
        "subtle": 2, "soft": 2,
        "visible": 3, "moderate": 3, "clear": 3,
        "dominant": 4, "strong": 4, "heavy": 4, "expressive": 4,
    },
}

GENRE_COLUMNS = {
    "colors": ("colors", "colours", "color", "colour"),
    "palette": ("color palette", "colour palette", "palette"),
    "energy": ("energy",),
    "brushwork": ("brushwork_visibility", "brushwork_visability", "brushwork visibility",
                  "brushwork", "brushstrokes", "brushstroke"),
}


def parse_value(category: str, text: str) -> set[int] | None:
    """Turn a table cell such as 'medium, high' into the grades it allows: {2, 3}.
    Returns None for 'not applicable' (that category is ignored for the genre)."""
    text = str(text).strip().lower()
    if not text:
        return None
    if re.fullmatch(r"[1-4](\s*[,/&]\s*[1-4])*", text):
        return {int(n) for n in re.findall(r"[1-4]", text)}

    vocab = VOCAB[category]
    found: set[int] = set()
    saw_na = False
    rest = f" {text} "
    for phrase in sorted(vocab, key=len, reverse=True):
        pattern = rf"(?<![a-z]){re.escape(phrase)}(?![a-z])"
        if re.search(pattern, rest):
            if vocab[phrase] is None:
                saw_na = True
            else:
                found.add(vocab[phrase])
            rest = re.sub(pattern, " ", rest)

    if not found and not saw_na:
        single_words = [k for k in vocab if " " not in k and vocab[k] is not None]
        for word in re.findall(r"[a-z]+", rest):
            close = difflib.get_close_matches(word, single_words, n=1, cutoff=0.75)
            if close:
                found.add(vocab[close[0]])
    if not found:
        if saw_na:
            return None
        raise ValueError(f"Can't understand {CATEGORY_TITLES[category]} value: '{text}'")
    return found


def _find_column(header: list[str], category: str) -> str | None:
    lowered = {h.strip().lower(): h for h in header}
    for name in GENRE_COLUMNS[category]:
        if name in lowered:
            return lowered[name]
    return None


@dataclass
class Genre:
    name: str
    allowed: dict[str, set[int] | None] = field(default_factory=dict)


def load_genres(path: Path = DEFAULT_GENRES) -> list[Genre]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c: _find_column(reader.fieldnames or [], c) for c in CATEGORIES}
        missing = [CATEGORY_TITLES[c] for c, col in cols.items() if col is None]
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")
        genres = []
        for row in reader:
            name = (row.get("Genre") or row.get("genre") or "").strip()
            if name:
                genres.append(Genre(name, {c: parse_value(c, row[cols[c]]) for c in CATEGORIES}))
    if not genres:
        raise ValueError(f"No genres found in {path}")
    return genres


def genre_points(scores: dict[str, int], genre: Genre) -> dict[str, float | None]:
    """Points per category for one genre: 1.0 for an exact hit, minus 1/3 per grade
    of distance. None where the category is not applicable for the genre."""
    points = {}
    for c in CATEGORIES:
        allowed = genre.allowed.get(c)
        points[c] = 1.0 - min(abs(scores[c] - a) for a in allowed) / 3.0 if allowed else None
    return points


def match_genres(scores: dict[str, int], genres: list[Genre]) -> list[tuple[str, float]]:
    """Compare 1-4 grades to every genre: the match is the average of the genre's points.
    Returns (genre, 0-1 match), best first."""
    results = []
    for g in genres:
        sims = [p for p in genre_points(scores, g).values() if p is not None]
        results.append((g.name, float(np.mean(sims)) if sims else 0.0))
    return sorted(results, key=lambda r: r[1], reverse=True)


# --------------------------------------------------------------------------- #
# Finding the painting in the photo
# --------------------------------------------------------------------------- #

DEFAULT_CROPS = HERE / "crops.csv"  # hand-made crops for photos the automatic crop gets wrong
AUTO_CROP = True    # cut away wall, frame and mat before measuring (--no-crop turns it off)
DETECT_SIZE = 800   # longest side used while looking for the painting
_manual_crops: dict[str, tuple[float, float, float, float]] | None = None


def manual_crop(path: Path) -> tuple[float, float, float, float] | None:
    """(left, top, right, bottom) as fractions of the photo, if crops.csv has this file."""
    global _manual_crops
    if _manual_crops is None:
        _manual_crops = {}
        if DEFAULT_CROPS.exists():
            with open(DEFAULT_CROPS, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    _manual_crops[row["file"].strip().lower()] = tuple(
                        float(row[k]) for k in ("left", "top", "right", "bottom"))
    return _manual_crops.get(path.name.lower())


def _corners(points) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    p = np.array(points, np.float32).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)


def _rectangles(small: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Rectangles around the centre of the photo (frame, mat and canvas edges),
    as (share of the photo, corners), largest first."""
    h, w = small.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 7, 40, 7)
    v = float(np.median(gray))
    edges = cv2.dilate(cv2.Canny(gray, 0.33 * v, 1.06 * v), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        hull = cv2.convexHull(c)
        if not 0.06 * w * h <= cv2.contourArea(hull) <= 0.97 * w * h:
            continue
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) != 4 or cv2.pointPolygonTest(approx, (w / 2, h / 2), False) < 0:
            continue
        _, _, bw, bh = cv2.boundingRect(approx)
        if cv2.contourArea(approx) / (bw * bh) < 0.85:
            continue
        p = _corners(approx)
        angles = []
        for i in range(4):
            v1, v2 = p[i - 1] - p[i], p[(i + 1) % 4] - p[i]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        if max(abs(a - 90) for a in angles) > 7:  # must be (nearly) a rectangle
            continue
        found.append((cv2.contourArea(approx) / (w * h), p))
    return sorted(found, key=lambda t: -t[0])


def _wall_box(small: np.ndarray) -> np.ndarray | None:
    """For canvases without a frame: the non-wall area around the centre.
    Wall = light, grey and smooth."""
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = cv2.blur(np.abs(gray - cv2.blur(gray, (9, 9))), (9, 9))
    wall = (hsv[..., 1] < 40) & (hsv[..., 2] > 150) & (detail < 4)
    wall = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    art = cv2.morphologyEx(1 - wall, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(art)
    part = labels[h // 2, w // 2]
    if part == 0 or stats[part][4] < 0.1 * w * h:
        return None
    x, y, bw, bh = stats[part][:4]
    return np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float32)


def _trim_frame(canvas: np.ndarray, band: float = 0.16, thresh: float = 40) -> tuple[int, int, int, int]:
    """Find a frame or mat inside the straightened rectangle. It shows up as a straight
    line along the full length of a side, at about the same distance from the edge on
    at least 3 sides. Returns the (top, bottom, left, right) of what lies inside it."""
    gray = cv2.GaussianBlur(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    h, w = gray.shape
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    rows = np.median(gy[:, int(w * .1):int(w * .9)], axis=1)  # line strength per row
    cols = np.median(gx[int(h * .1):int(h * .9), :], axis=0)  # ...and per column
    limit = int(min(h, w) * band)

    def lines(p):
        return [i for i in range(2, limit) if p[i] >= thresh and p[i] == p[max(0, i - 3):i + 4].max()]

    sides = [lines(rows), lines(rows[::-1]), lines(cols), lines(cols[::-1])]  # top bottom left right
    for width in sorted({p for s in sides for p in s}, reverse=True):  # widest frame first
        tol = max(3, 0.15 * width)
        match = [min((p for p in s if abs(p - width) <= tol), key=lambda p: abs(p - width), default=None)
                 for s in sides]
        if sum(m is not None for m in match) >= 3:
            t, b, l, r = (m if m is not None else width for m in match)
            return t, h - b, l, w - r
    return 0, h, 0, w


def find_canvas(img: np.ndarray, box: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Cut the painting out of the photo: drop the wall, frame and mat, and straighten it.
    `box` = (left, top, right, bottom) as fractions of the photo crops by hand instead."""
    h, w = img.shape[:2]
    if box is not None:
        left, top, right, bottom = box
        return img[round(top * h):round(bottom * h), round(left * w):round(right * w)]
    # Shrink in two steps: the slight extra smoothing makes the edge search more reliable.
    k1 = 1024 / max(h, w)
    mid = cv2.resize(img, None, fx=k1, fy=k1, interpolation=cv2.INTER_AREA)
    k2 = DETECT_SIZE / max(mid.shape[:2])
    small = cv2.resize(mid, None, fx=k2, fy=k2, interpolation=cv2.INTER_AREA)
    s = k1 * k2
    rects = _rectangles(small)
    if rects:  # innermost rectangle that is still a good part of the outermost one
        biggest = rects[0][0]
        quad = min((r for r in rects if r[0] >= max(0.12, 0.3 * biggest)), key=lambda r: r[0])[1]
    else:
        quad = _wall_box(small)
        if quad is None:
            sh, sw = small.shape[:2]
            quad = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    quad = quad / s
    tl, tr, br, bl = quad
    cw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    ch = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    m = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32))
    flat = cv2.warpPerspective(img, m, (cw, ch))
    k = 600 / max(cw, ch)
    t, b, l, r = (v / k for v in _trim_frame(cv2.resize(flat, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)))
    dx, dy = 0.015 * (r - l), 0.015 * (b - t)  # small safety margin
    return flat[int(t + dy):int(b - dy), int(l + dx):int(r - dx)]


# --------------------------------------------------------------------------- #
# Image measurements
# --------------------------------------------------------------------------- #

def load_image(path: Path) -> np.ndarray:
    """Read a photo, cut out the painting (see find_canvas) and resize it for measuring."""
    data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if AUTO_CROP:
        img = find_canvas(img, manual_crop(path))
    h, w = img.shape[:2]
    scale = ANALYSIS_SIZE / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)


def measure(img: np.ndarray) -> dict[str, float]:
    """Raw measurements of the painting, mostly on a 0-1 scale."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_deg = hsv[..., 0] * 2.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Colourfulness (Hasler and Suesstrunk, 2003).
    b, g, r = [img[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))

    sat_mean, val_mean = float(sat.mean()), float(val.mean())
    lum_std = float(gray.std())

    # How spread out the hues are (0 = one hue, 1 = all hues), weighted by saturation.
    w = sat * val
    theta = np.deg2rad(hue_deg)
    wsum = float(w.sum()) + 1e-9
    resultant = np.hypot((w * np.cos(theta)).sum(), (w * np.sin(theta)).sum()) / wsum
    hue_spread = float(1.0 - resultant)

    colourful = sat > 0.15
    warm = colourful & ((hue_deg < 70) | (hue_deg > 320))
    warmth = float(warm.sum() / max(int(colourful.sum()), 1))

    # Edges and fine texture (brushstrokes show up as small-scale texture).
    blurred = cv2.GaussianBlur((gray * 255).astype(np.uint8), (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 150)
    edge_density = float((edges > 0).mean())
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    texture = float(np.abs(gray - smooth).mean())

    # Share of strong gradients that run diagonally (feels more dynamic).
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 90.0  # 45 = diagonal
    strong = mag > max(float(np.percentile(mag, 75)), 1e-3)
    diagonal = float((np.abs(ang[strong] - 45.0) < 22.5).mean()) if strong.any() else 0.0

    colour_strength = 0.5 * min(colorfulness / 90.0, 1.0) + 0.5 * min(sat_mean / 0.6, 1.0)
    energy = (0.40 * min(edge_density / 0.18, 1.0)
              + 0.25 * min(lum_std / 0.30, 1.0)
              + 0.20 * min(sat_mean / 0.60, 1.0)
              + 0.15 * diagonal)

    return {
        "saturation": sat_mean,
        "brightness": val_mean,
        "colorfulness": colorfulness,
        "colour_strength": float(colour_strength),
        "contrast": lum_std,
        "hue_spread": hue_spread,
        "warmth": warmth,
        "dark_share": float((val < 0.25).mean()),
        "light_share": float((val > 0.80).mean()),
        "edge_density": edge_density,
        "texture": texture,
        "diagonal": diagonal,
        "energy": float(energy),
        **detail_measurements(img, gray, sat, texture),
    }


def _local_std(x: np.ndarray, size: int = 9) -> np.ndarray:
    m = cv2.blur(x, (size, size))
    return np.sqrt(np.maximum(cv2.blur(x * x, (size, size)) - m * m, 0))


def detail_measurements(img: np.ndarray, gray: np.ndarray, sat: np.ndarray, texture: float) -> dict[str, float]:
    """Extra measurements used only by the trained models (see MODEL_FEATURES)."""
    out = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128

    # Colour
    out["sat_p75"] = float(np.percentile(sat, 75))
    out["sat_p90"] = float(np.percentile(sat, 90))
    chroma = np.hypot(a, b)
    out["chroma_mean"] = float(chroma.mean())
    out["chroma_p90"] = float(np.percentile(chroma, 90))
    out["lightness"] = float(lab[..., 0].mean() / 255)
    colour = sat > 0.2
    out["colour_share"] = float(colour.mean())
    hist = np.histogram(hsv[..., 0][colour], bins=18, range=(0, 180))[0].astype(float)
    p = hist[hist > 0] / max(hist.sum(), 1)
    out["hue_entropy"] = float(-(p * np.log(p)).sum() / np.log(18)) if colour.any() else 0.0
    out["local_chroma_std"] = float(np.median(np.hypot(_local_std(a), _local_std(b))))  # broken colour
    q = (hsv[..., 0] // 15).astype(np.int32) * 4 + np.minimum(hsv[..., 1] // 64, 3)
    q[hsv[..., 1] < 50] = -1
    out["colour_changes"] = float(((np.diff(q, axis=1) != 0).mean() + (np.diff(q, axis=0) != 0).mean()) / 2)

    # Light and dark
    out["lum_range"] = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    lstd = _local_std(gray)
    out["flat_share"] = float((lstd < 0.015).mean())  # smooth, untextured areas
    out["local_std_med"] = float(np.median(lstd))

    # Texture at several scales
    out["tex_s1"] = float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1)).mean())
    out["tex_s4"] = float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 4)).mean())
    b1, b2, b3 = (cv2.GaussianBlur(gray, (0, 0), s) for s in (1.5, 4, 10))
    out["band_mid"] = float(np.abs(b1 - b2).mean())
    out["band_low"] = float(np.abs(b2 - b3).mean())
    out["band_ratio"] = out["band_mid"] / (out["band_low"] + 1e-6)
    big = float(np.abs(cv2.GaussianBlur(gray, (0, 0), 2) - cv2.GaussianBlur(gray, (0, 0), 8)).mean())
    out["tex_ratio"] = texture / (big + 1e-6)  # fine texture compared with large shapes

    # Edges and stroke direction
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    out["grad_mean"] = float(np.hypot(gx, gy).mean())
    g8 = cv2.GaussianBlur((gray * 255).astype(np.uint8), (3, 3), 0)
    out["edges_low"] = float((cv2.Canny(g8, 30, 90) > 0).mean())
    h, w = gray.shape
    small = cv2.resize(g8, (max(1, w // 3), max(1, h // 3)), interpolation=cv2.INTER_AREA)
    out["edges_coarse"] = float((cv2.Canny(cv2.GaussianBlur(small, (3, 3), 0), 50, 120) > 0).mean())
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    for s in (2, 4):  # structure tensor: brushstrokes are locally aligned texture
        jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), s * 2) for v in (sx * sx, sy * sy, sx * sy))
        trace = jxx + jyy
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (trace + 1e-6)
        strong = trace > np.percentile(trace, 50)
        out[f"coherence_s{s}"] = float(coh[strong].mean())
        ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
        hist = np.histogram(ang[strong], bins=12, range=(-np.pi / 2, np.pi / 2), weights=coh[strong])[0]
        p = hist[hist > 0] / max(hist.sum(), 1e-9)
        out[f"orient_entropy_s{s}"] = float(-(p * np.log(p)).sum() / np.log(12))
    return out


def _grade(value: float, cuts: tuple[float, float, float]) -> int:
    return 1 + sum(value >= c for c in cuts)


def grade(m: dict[str, float]) -> dict[str, int]:
    """Turn raw measurements into 1-4 grades with fixed rules (see THRESHOLDS)."""
    t = THRESHOLDS
    if m["contrast"] >= t["palette_high_contrast"] or (
        m["colorfulness"] >= t["palette_colourful_min"] and m["contrast"] >= t["palette_colourful_contrast"]
    ):
        palette = 4
    elif m["brightness"] >= t["palette_pastel_min_value"] and m["saturation"] < t["palette_pastel_max_sat"]:
        palette = 1
    elif m["brightness"] < t["palette_deep_max_value"] or (
        m["warmth"] > t["palette_warm_min"] and m["brightness"] < t["palette_warm_max_value"]
    ):
        palette = 3
    else:
        palette = 2

    return {
        "colors": _grade(m["colorfulness"], t["colors"]),
        "palette": palette,
        "energy": _grade(m["energy"], t["energy"]),
        "brushwork": _grade(m["texture"], t["brushwork"]),
    }


def feature_vector(m: dict[str, float]) -> np.ndarray:
    """Numbers the trainable models learn from: all measurements."""
    return np.array([m[k] for k in MODEL_FEATURES], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Trainable model
# --------------------------------------------------------------------------- #

# Measurements the trained models learn from: the 13 documented ones plus the details.
MODEL_FEATURES = [
    "saturation", "brightness", "colorfulness", "colour_strength", "hue_spread", "warmth",
    "contrast", "dark_share", "light_share", "edge_density", "diagonal", "texture", "energy",
    "sat_p75", "sat_p90", "chroma_mean", "chroma_p90", "lightness", "colour_share", "hue_entropy",
    "local_chroma_std", "colour_changes", "lum_range", "flat_share", "local_std_med",
    "tex_s1", "tex_s4", "band_mid", "band_low", "band_ratio", "tex_ratio",
    "grad_mean", "edges_low", "edges_coarse",
    "coherence_s2", "orient_entropy_s2", "coherence_s4", "orient_entropy_s4",
]
DEFAULT_LABELS = HERE / "labels.csv"
DEFAULT_TRAINING_REPORT = HERE / "training_report.csv"
MODEL_VERSION = 2

# CLIP: a large pre-trained image model (Apple DFN5B, ViT-H/14) that describes each painting
# with 1024 numbers capturing content and style. Needs torch and open_clip_torch.
CLIP_MODEL = ("ViT-H-14-quickgelu", "dfn5b")
# Last 4 vision layers of that CLIP re-trained on 101,750 WikiArt paintings (finetune_clip_art.py).
# Used when the file exists; delete it to go back to the original CLIP (then run 'train' again).
CLIP_ADAPTER = HERE / "clip_art_adapter.pt"
CLIP_WEIGHT = 0.5  # share of the CLIP model in each prediction (the rest: Random Forest)
_clip = None


def clip_name() -> str:
    """Which CLIP is in use; stored with the trained models so they always match."""
    return "/".join(CLIP_MODEL) + ("+" + CLIP_ADAPTER.name if CLIP_ADAPTER.exists() else "")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _new_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0)


def _new_clip_model(y: np.ndarray):
    """Logistic regression on CLIP numbers + measurements; strength chosen by inner cross-validation."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if int(min(np.unique(y, return_counts=True)[1])) >= 8:
        lr = LogisticRegressionCV(Cs=[0.1, 0.3, 1, 3, 10, 30], cv=4, max_iter=5000, class_weight="balanced")
    else:
        lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")
    return make_pipeline(StandardScaler(), lr)


def clip_available() -> bool:
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load_clip():
    global _clip
    if _clip is None:
        import open_clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP {clip_name()} on {device.upper()}...")
        model, _, _ = open_clip.create_model_and_transforms(*CLIP_MODEL)
        if CLIP_ADAPTER.exists():
            adapter = torch.load(CLIP_ADAPTER, map_location="cpu")
            _, unexpected = model.visual.load_state_dict(
                {k: t.float() for k, t in adapter["state"].items()}, strict=False)
            if unexpected:
                raise SystemExit(f"{CLIP_ADAPTER.name} does not fit {CLIP_MODEL[0]}.")
        model = model.to(device).eval()
        if device == "cuda":
            model = model.half()
        size = model.visual.image_size
        _clip = {"model": model, "device": device, "size": size[0] if isinstance(size, tuple) else size,
                 "mean": np.array(model.visual.preprocess_cfg["mean"], np.float32),
                 "std": np.array(model.visual.preprocess_cfg["std"], np.float32)}
    return _clip


def clip_embedding(img: np.ndarray) -> np.ndarray:
    """CLIP numbers for a painting: the whole painting padded to a square, plus up to
    3 square crops along its long side, averaged so nothing is cut off."""
    import torch
    clip = _load_clip()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    side = max(h, w)
    fill = [int(v) for v in np.median(rgb.reshape(-1, 3), axis=0)]
    views = [cv2.copyMakeBorder(rgb, (side - h) // 2, side - h - (side - h) // 2, (side - w) // 2,
                                side - w - (side - w) // 2, cv2.BORDER_CONSTANT, value=fill)]
    short = min(h, w)
    if side / short >= 1.15:
        for p in np.linspace(0, side - short, 3).round().astype(int):
            views.append(rgb[p:p + short, :] if h > w else rgb[:, p:p + short])
    size = clip["size"]
    batch = np.stack([(cv2.resize(v, (size, size), interpolation=cv2.INTER_AREA if v.shape[0] > size
                                  else cv2.INTER_CUBIC).astype(np.float32) / 255 - clip["mean"]) / clip["std"]
                      for v in views]).transpose(0, 3, 1, 2)
    with torch.no_grad():
        x = torch.from_numpy(batch).to(clip["device"])
        f = clip["model"].encode_image(x.half() if clip["device"] == "cuda" else x).float()
    f = f / f.norm(dim=-1, keepdim=True)
    m = f.mean(0)
    return (m / m.norm()).cpu().numpy()


def _fit_task(xm: np.ndarray, xc: np.ndarray | None, y: np.ndarray) -> dict:
    """One prediction task (a category or the genre): a Random Forest on the measurements,
    plus a CLIP model when CLIP numbers are given."""
    task = {"rf": _new_model().fit(xm, y), "clip": None}
    if xc is not None:
        task["clip"] = _new_clip_model(y).fit(np.hstack([xc, xm]), y)
    return task


def _task_proba(task: dict, xm: np.ndarray, xc: np.ndarray | None) -> tuple[list, np.ndarray]:
    """(classes, probabilities): Random Forest and CLIP model averaged (CLIP_WEIGHT)."""
    p = task["rf"].predict_proba(xm)
    if task["clip"] is not None and xc is not None:
        p = (1 - CLIP_WEIGHT) * p + CLIP_WEIGHT * task["clip"].predict_proba(np.hstack([xc, xm]))
    return list(task["rf"].classes_), p


def predict(bundle: dict, m: dict[str, float], e: np.ndarray | None, genres: list[Genre], scores: dict[str, int]):
    """Grades and genre ranking from a trained bundle. `scores` holds the rule grades,
    used for any category the bundle has no model for."""
    xm = feature_vector(m)[None, :]
    xc = None if e is None else e[None, :]
    s = dict(scores)
    for c, task in bundle["tasks"].items():
        if c != "genre":
            classes, p = _task_proba(task, xm, xc)
            s[c] = int(classes[int(np.argmax(p[0]))])
    table = match_genres(s, genres)
    probs = None
    if "genre" in bundle["tasks"]:
        classes, p = _task_proba(bundle["tasks"]["genre"], xm, xc)
        probs = dict(zip(classes, p[0]))
    return s, table, probs


def _training_set_from_folders(data_dir: Path, genres: list[Genre]):
    """Paintings sorted into one sub-folder per genre: genre labels only."""
    by_key = {_norm(g.name): g.name for g in genres}
    items = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = by_key.get(_norm(folder.name))
        if label is None:
            print(f"  skipping folder '{folder.name}' (not a genre in the table)")
            continue
        files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        print(f"  {label}: {len(files)} paintings")
        items += [(p, {"genre": label}) for p in files]
    return items


def _training_set_from_labels(data_dir: Path, labels: Path, genres: list[Genre]):
    """A labels CSV (File, Colours, Colour palette, Energy, Brushstrokes, Genre): any of
    the grade columns and/or the genre column can be filled in."""
    by_key = {_norm(g.name): g.name for g in genres}
    files = {p.name.lower(): p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    items = []
    with open(labels, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c: _find_column(reader.fieldnames or [], c) for c in CATEGORIES}
        for row in reader:
            path = files.get((row.get("File") or "").strip().lower())
            if path is None:
                continue
            target = {}
            for c, col in cols.items():
                if col and row[col].strip():
                    target[c] = _score_from_input(c, row[col])
            genre = by_key.get(_norm(row.get("Genre") or ""))
            if genre:
                target["genre"] = genre
            if target:
                items.append((path, target))
    print(f"  {len(items)} labelled paintings found in {data_dir}")
    return items


def train(data_dir: Path, genres: list[Genre], model_path: Path, labels: Path | None = None,
          report_path: Path = DEFAULT_TRAINING_REPORT, use_clip: bool = True) -> None:
    """Train one model per category (when grades are labelled) and one for the genre.
    Prints cross-validated accuracy and writes, per painting, what the models predict
    when that painting was NOT part of their training (an honest test)."""
    import joblib
    from sklearn.model_selection import KFold, StratifiedKFold

    if not data_dir.is_dir():
        raise SystemExit(f"Training folder not found: {data_dir}")
    if use_clip and not clip_available():
        print("CLIP is not installed (torch, open_clip_torch), training without it.\n")
        use_clip = False
    items = (_training_set_from_labels(data_dir, labels, genres) if labels
             else _training_set_from_folders(data_dir, genres))
    paths, XM, XC, targets = [], [], [], []
    for n, (p, target) in enumerate(items, 1):
        try:
            img = load_image(p)
        except ValueError as e:
            print(f"    {e}")
            continue
        XM.append(feature_vector(measure(img)))
        if use_clip:
            XC.append(clip_embedding(img))
        paths.append(p)
        targets.append(target)
        if n % 25 == 0:
            print(f"  measured {n}/{len(items)}")
    if not paths:
        raise SystemExit("No labelled paintings could be read.")
    XM = np.vstack(XM)
    XC = np.vstack(XC) if use_clip else None
    keys = [k for k in CATEGORIES + ["genre"] if len({t[k] for t in targets if k in t}) >= 2]
    if not keys:
        raise SystemExit("Training needs at least two different labels in some column.")

    # Honest test: 5-fold cross-validation with the same folds for every task, so the
    # whole chain (grades -> genre table + genre model) can be tested as well.
    genre_y = [t.get("genre", "") for t in targets]
    counts = np.unique(genre_y, return_counts=True)[1]
    folds = int(min(5, counts.min())) if "genre" in keys else 5
    splitter = (StratifiedKFold(folds, shuffle=True, random_state=0) if "genre" in keys and folds >= 2
                else KFold(min(5, len(paths)), shuffle=True, random_state=0))
    oof = {k: [None] * len(paths) for k in keys}
    genre_oof = [None] * len(paths)
    for tr, te in splitter.split(XM, genre_y):
        for k in keys:
            idx = [i for i in tr if k in targets[i]]
            y = np.array([targets[i][k] for i in idx])
            if len(set(y)) < 2:
                continue
            task = _fit_task(XM[idx], None if XC is None else XC[idx], y)
            classes, p = _task_proba(task, XM[te], None if XC is None else XC[te])
            for i, row in zip(te, p):
                oof[k][i] = classes[int(np.argmax(row))]
                if k == "genre":
                    genre_oof[i] = dict(zip(classes, row))

    report = {p: {"File": p.name} for p in paths}
    print(f"\nCross-validated accuracy ({'CLIP + measurements' if use_clip else 'measurements only'}):")
    for k in keys:
        title = "Genre" if k == "genre" else CATEGORY_TITLES[k]
        idx = [i for i in range(len(paths)) if k in targets[i] and oof[k][i] is not None]
        y = np.array([targets[i][k] for i in idx])
        pred = np.array([oof[k][i] for i in idx])
        extra = "" if k == "genre" else f", within 1 grade {np.mean(np.abs(pred - y) <= 1):.0%}"
        print(f"  {title:<16} exact {np.mean(pred == y):.0%}{extra}   ({len(y)} paintings)")
        for i in idx:
            report[paths[i]][f"{title} label"] = targets[i][k]
            report[paths[i]][f"{title} predicted"] = oof[k][i]
    if "genre" in keys and all(c in keys for c in CATEGORIES):
        hits = []
        for i in range(len(paths)):
            if genre_oof[i] is None or any(oof[c][i] is None for c in CATEGORIES):
                continue
            table = match_genres({c: int(oof[c][i]) for c in CATEGORIES}, genres)
            blend = {n: (1 - MODEL_WEIGHT) * t + MODEL_WEIGHT * genre_oof[i].get(n, 0.0) for n, t in table}
            final = max(blend, key=blend.get)
            report[paths[i]]["Final genre predicted"] = final
            hits.append(final == targets[i]["genre"])
        print(f"  {'Final genre':<16} exact {np.mean(hits):.0%}   (grades + genre table + genre model, as in 'analyze')")

    bundle = {"version": MODEL_VERSION, "features": MODEL_FEATURES,
              "clip": clip_name() if use_clip else None, "tasks": {}}
    for k in keys:
        idx = [i for i in range(len(paths)) if k in targets[i]]
        bundle["tasks"][k] = _fit_task(XM[idx], None if XC is None else XC[idx],
                                       np.array([targets[i][k] for i in idx]))
    joblib.dump(bundle, model_path)
    print(f"\nModel saved to {model_path}")
    rows = sorted(report.values(), key=lambda r: _natural_key(Path(r["File"])))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Cross-validated predictions per painting saved to {report_path}")


def load_model(model_path: Path):
    if not model_path.exists():
        return None
    import joblib
    bundle = joblib.load(model_path)
    if bundle.get("version") != MODEL_VERSION or bundle.get("features") != MODEL_FEATURES:
        print(f"{model_path.name} was trained with an older version of this program and is ignored. "
              "Run 'train' again to use it.\n")
        return None
    if bundle["clip"] and bundle["clip"] != clip_name():
        print(f"{model_path.name} was trained with CLIP {bundle['clip']}; run 'train' again.\n")
        return None
    if bundle["clip"] and not clip_available():
        print("This model uses CLIP, but torch/open_clip_torch are not installed: "
              "using its Random Forest part only.\n")
        bundle["clip"] = None
    return bundle


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

RAW_FOR = {"colors": "colorfulness", "palette": "contrast", "energy": "energy", "brushwork": "texture"}


def _bar(fraction: float, width: int = 20) -> str:
    n = round(fraction * width)
    return "#" * n + "." * (width - n)


def print_report(title: str, scores: dict[str, int], ranking: list[tuple[str, float]],
                 measurements: dict[str, float] | None = None,
                 table: dict[str, float] | None = None,
                 model_probs: dict[str, float] | None = None) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"{'Category':<16}{'Grade':<8}{'Meaning':<22}{'Measured' if measurements else ''}")
    for c in CATEGORIES:
        raw = f"{measurements[RAW_FOR[c]]:.3f}" if measurements else ""
        print(f"{CATEGORY_TITLES[c]:<16}{scores[c]}/4{'':<5}{SCORE_LABELS[c][scores[c]]:<22}{raw}")
    print(f"{'Total':<16}{sum(scores.values())}/16")

    print("\nGenre comparison")
    for name, sim in ranking:
        extra = ""
        if model_probs is not None and table is not None:
            extra = f"   table {table[name]:.0%}, AI model {model_probs.get(name, 0.0):.0%}"
        print(f"  {name:<14}{sim:>5.0%}  {_bar(sim)}{extra}")
    best, best_sim = ranking[0]
    print(f"\nMatching music genre: {best}  ({best_sim:.0%} match)")
    if len(ranking) > 1:
        print(f"Second option:        {ranking[1][0]}  ({ranking[1][1]:.0%} match)")
    print()


CARD_HEIGHT = 640      # pixel height of the result picture
CARD_MAX_WIDTH = 1000  # very wide paintings are shrunk to fit this
PANEL_WIDTH = 440


def render_card(img: np.ndarray, title: str, scores: dict[str, int],
                ranking: list[tuple[str, float]]) -> np.ndarray:
    """Result picture: the painting on the left, grades and genre match on the right."""
    h, w = img.shape[:2]
    scale = min(CARD_HEIGHT / h, CARD_MAX_WIDTH / w)
    pic = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    pad = CARD_HEIGHT - pic.shape[0]
    pic = cv2.copyMakeBorder(pic, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    panel = np.full((CARD_HEIGHT, PANEL_WIDTH, 3), 255, np.uint8)
    dark, grey, green = (30, 30, 30), (120, 120, 120), (60, 140, 60)

    def text(s, x, y, size=0.6, color=dark, thick=1):
        s = s.encode("ascii", "replace").decode()  # OpenCV fonts only draw ASCII
        cv2.putText(panel, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

    text(title[:32], 20, 40, 0.7, thick=2)
    y = 90
    for c in CATEGORIES:
        text(CATEGORY_TITLES[c], 20, y)
        text(f"{scores[c]}/4", 180, y, thick=2)
        text(SCORE_LABELS[c][scores[c]], 240, y, 0.5, grey)
        y += 34
    text(f"Total {sum(scores.values())}/16", 20, y)
    y += 50
    text("Genre match", 20, y, thick=2)
    y += 32
    for name, sim in ranking:
        text(name, 20, y)
        cv2.rectangle(panel, (160, y - 14), (360, y + 2), (225, 225, 225), -1)
        cv2.rectangle(panel, (160, y - 14), (160 + round(200 * sim), y + 2), green, -1)
        text(f"{sim:.0%}", 370, y)
        y += 32
    best, best_sim = ranking[0]
    text("Matching music genre:", 20, CARD_HEIGHT - 105, color=grey)
    text(f"{best} ({best_sim:.0%})", 20, CARD_HEIGHT - 68, 0.9, green, 2)
    if len(ranking) > 1:
        second, second_sim = ranking[1]
        text(f"2nd option: {second} ({second_sim:.0%})", 20, CARD_HEIGHT - 28)
    return np.hstack([pic, panel])


def save_card(card: np.ndarray, path: Path) -> None:
    cv2.imencode(".jpg", card)[1].tofile(str(path))  # works with non-ASCII Windows paths


def pick_images(parent=None) -> list[Path]:
    """Let the user choose painting files in a normal Windows file dialog."""
    import tkinter as tk
    from tkinter import filedialog
    root = parent or tk.Tk()
    if parent is None:
        root.withdraw()
        root.attributes("-topmost", True)
    patterns = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
    names = filedialog.askopenfilenames(parent=root, title="Choose paintings",
                                        filetypes=[("Images", patterns), ("All files", "*.*")])
    if parent is None:
        root.destroy()
    return sorted((Path(n) for n in names), key=_natural_key)


def show_results(results: list[tuple[Path, dict[str, int], list[tuple[str, float]]]],
                 analyze_more=None) -> None:
    """Window that shows each painting with its result.
    Left/Right arrows (or the buttons) browse, Esc closes."""
    import base64
    import tkinter as tk

    root = tk.Tk()
    root.title("Painting -> Music Genre")
    picture = tk.Label(root, bg="white")
    picture.pack()
    bar = tk.Frame(root)
    bar.pack(fill="x", pady=4)
    counter = tk.Label(bar)
    state = {"i": 0}

    def draw():
        path, scores, ranking = results[state["i"]]
        card = render_card(load_image(path), path.name, scores, ranking)
        png = base64.b64encode(cv2.imencode(".png", card)[1].tobytes())
        photo = tk.PhotoImage(data=png)
        picture.configure(image=photo)
        picture.image = photo  # keep a reference so tkinter doesn't drop it
        counter.configure(text=f"{state['i'] + 1} / {len(results)}   {path.name}")

    def step(delta):
        state["i"] = (state["i"] + delta) % len(results)
        draw()

    def open_more():
        new = analyze_more(pick_images(root))
        if new:
            results.extend(new)
            state["i"] = len(results) - len(new)
            draw()

    tk.Button(bar, text="< Previous", command=lambda: step(-1)).pack(side="left", padx=6)
    tk.Button(bar, text="Next >", command=lambda: step(1)).pack(side="right", padx=6)
    if analyze_more is not None:
        tk.Button(bar, text="Open photos...", command=open_more).pack(side="right", padx=6)
    counter.pack()
    root.bind("<Left>", lambda e: step(-1))
    root.bind("<Right>", lambda e: step(1))
    root.bind("<Escape>", lambda e: root.destroy())
    draw()
    root.mainloop()


# Measurement columns in the results CSV: colour, then light/dark, then structure.
CSV_MEASUREMENTS = ["saturation", "brightness", "colorfulness", "colour_strength", "hue_spread",
                    "warmth", "contrast", "dark_share", "light_share", "edge_density",
                    "diagonal", "texture", "energy"]


def evaluate(path: Path, genres: list[Genre], model_bundle) -> dict:
    """Everything the program finds out about one painting, without printing anything."""
    img = load_image(path)
    m = measure(img)
    s = grade(m)
    table, probs = match_genres(s, genres), None
    if model_bundle is not None:  # trained models replace the fixed rules
        e = clip_embedding(img) if model_bundle["clip"] else None
        s, table, probs = predict(model_bundle, m, e, genres, s)
    ranking = table
    if probs is not None:
        blended = {n: (1 - MODEL_WEIGHT) * t + MODEL_WEIGHT * probs.get(n, 0.0) for n, t in table}
        ranking = sorted(blended.items(), key=lambda r: r[1], reverse=True)
    return {"img": img, "measurements": m, "scores": s, "table": dict(table), "model_probs": probs,
            "ranking": ranking, "music": pick_music(ranking[0][0], s)}


def analyze_image(path: Path, genres: list[Genre], model_bundle):
    """Analyze one painting, print its report and return (CSV row, grades, genre ranking, image)."""
    r = evaluate(path, genres, model_bundle)
    s, ranking, m = r["scores"], r["ranking"], r["measurements"]
    print_report(f"Painting: {path.name}", s, ranking, m, r["table"], r["model_probs"])
    if r["music"]:
        t = r["music"][0]
        print(f"Music: \"{t['Title']}\" by {t['Artist']}  ({t['Genre']}, energy {t['Energy']}, {t['Mood']})\n")
    row = {"File": str(path), **{CATEGORY_TITLES[c]: s[c] for c in CATEGORIES}, "Total": sum(s.values())}
    by_name = {g.name: g for g in genres}
    for option, (name, sim) in enumerate(ranking[:2], 1):
        row[f"Genre {option}"] = name
        row[f"Match {option}"] = f"{sim:.0%}"
        for c, p in genre_points(s, by_name[name]).items():
            row[f"{CATEGORY_TITLES[c]} points {option}"] = "n/a" if p is None else round(p, 2)
    row["Music"] = f"{r['music'][0]['Title']} ({r['music'][0]['Artist']})" if r["music"] else ""
    row.update({k: round(m[k], 1 if k == "colorfulness" else 4) for k in CSV_MEASUREMENTS})
    return row, s, ranking, r["img"]


# --------------------------------------------------------------------------- #
# Music
# --------------------------------------------------------------------------- #

DEFAULT_MUSIC = HERE / "music"  # one sub-folder per genre, plus tracks.csv describing the tracks
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
MOOD_VALUE = {"dark": -1, "neutral": 0, "bright": 1}
_music: list[dict] | None = None


def load_music(folder: Path = DEFAULT_MUSIC) -> list[dict]:
    """Tracks described in music/tracks.csv, plus any other audio files dropped into a
    genre folder (music/Classical, music/Jazz, music/Ambient, music/Rock_Metal)."""
    global _music
    if _music is not None and folder == DEFAULT_MUSIC:
        return _music
    tracks, known = [], set()
    if (folder / "tracks.csv").exists():
        with open(folder / "tracks.csv", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                path = folder / row["File"]
                if path.exists():
                    tracks.append({**row, "Path": path, "Energy": int(row["Energy"])})
                    known.add(path.resolve())
    if folder.is_dir():
        for sub in folder.iterdir():
            if not sub.is_dir():
                continue
            genre = sub.name.replace("_", "/")
            for p in sorted(sub.iterdir()):
                if p.suffix.lower() in AUDIO_EXTS and p.resolve() not in known:
                    tracks.append({"File": f"{sub.name}/{p.name}", "Title": p.stem, "Artist": "unknown",
                                   "Genre": genre, "Energy": None, "Mood": "neutral", "Path": p,
                                   "License": "", "Source": ""})
    if folder == DEFAULT_MUSIC:
        _music = tracks
    return tracks


def painting_mood(scores: dict[str, int]) -> str:
    """Bright, neutral or dark, from the Colours and Colour palette grades."""
    v = {1: 1, 2: 0, 3: -1, 4: 0}[scores["palette"]] + {1: -1, 2: 0, 3: 0.5, 4: 1}[scores["colors"]]
    return "bright" if v >= 1 else "dark" if v <= -1 else "neutral"


def pick_music(genre: str, scores: dict[str, int], tracks: list[dict] | None = None) -> list[dict]:
    """Tracks of the genre, best match first: closest energy to the Energy grade and
    closest mood to the painting's colours."""
    tracks = load_music() if tracks is None else tracks
    mood = MOOD_VALUE[painting_mood(scores)]

    def distance(t):
        e = 1.5 if t["Energy"] is None else abs(t["Energy"] - scores["energy"])
        return e + 0.75 * abs(MOOD_VALUE.get(t["Mood"], 0) - mood)
    return sorted((t for t in tracks if _norm(t["Genre"]) == _norm(genre)), key=lambda t: (distance(t), t["Title"]))


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _natural_key(path: Path) -> list:
    """Sort '2_x.jpg' before '10_x.jpg' (plain sorting would put 10 first)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def _score_from_input(category: str, value: str) -> int:
    allowed = parse_value(category, value)
    if not allowed:
        raise SystemExit(f"{CATEGORY_TITLES[category]} needs a value (1-4 or a word).")
    return round(float(np.mean(sorted(allowed))))


def cmd_analyze(args, genres):
    model = None if args.no_model else load_model(args.model)
    if model is not None:
        kind = f"CLIP {model['clip']} + measurements" if model["clip"] else "measurements only"
        print(f"Using trained AI model: {args.model.name} ({kind})\n")
    target = args.path
    if target is None:  # no path given: choose photos in a dialog and show the results
        files = pick_images()
        if not files:
            raise SystemExit("No paintings chosen.")
        args.show = True
    else:
        if not target.exists():
            raise SystemExit(f"Not found: {target}")
        files = [target] if target.is_file() else sorted(
            (p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS), key=_natural_key)
        if not files:
            raise SystemExit(f"No images found at {target}")

    rows = []

    def run(paths: list[Path]):
        results = []
        for f in paths:
            try:
                row, scores, ranking, img = analyze_image(f, genres, model)
            except ValueError as e:
                print(e)
                continue
            rows.append(row)
            results.append((f, scores, ranking))
            if args.cards:
                args.cards.mkdir(parents=True, exist_ok=True)
                save_card(render_card(img, f.name, scores, ranking),
                          args.cards / f"{f.stem}_result.jpg")
        return results

    results = run(files)
    if args.cards and results:
        print(f"Result pictures saved to {args.cards}")
    if args.show and results:
        show_results(results, run)
    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Results saved to {args.csv}")


def cmd_match(args, genres):
    if args.table:
        with open(args.table, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = {c: _find_column(reader.fieldnames or [], c) for c in CATEGORIES}
            if not all(cols.values()):
                raise SystemExit("The table needs Colors, Color Palette, Energy and Brushwork columns.")
            for i, row in enumerate(reader, 1):
                label = (row.get("Painting") or row.get("Name") or f"Row {i}").strip()
                scores = {c: _score_from_input(c, row[cols[c]]) for c in CATEGORIES}
                print_report(label, scores, match_genres(scores, genres))
        return
    values = {"colors": args.colors, "palette": args.palette,
              "energy": args.energy, "brushwork": args.brushwork}
    missing = [f"--{k}" for k, v in values.items() if v is None]
    if missing:
        raise SystemExit(f"Please give {', '.join(missing)} (or use --table).")
    scores = {c: _score_from_input(c, v) for c, v in values.items()}
    print_report("Described painting", scores, match_genres(scores, genres))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grade a painting 1-4 per category and match it to a music genre.")
    ap.add_argument("--genres", type=Path, default=DEFAULT_GENRES, help="genre table CSV")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze an image or a folder of images")
    a.add_argument("path", type=Path, nargs="?",
                   help="image or folder (leave out to choose photos in a dialog)")
    a.add_argument("--show", action="store_true", help="show each painting with its result in a window")
    a.add_argument("--cards", type=Path, help="save a result picture for each painting to this folder")
    a.add_argument("--csv", type=Path, help="also save results to this CSV")
    a.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    a.add_argument("--no-model", action="store_true", help="use only the genre table")

    m = sub.add_parser("match", help="match grades or words to a genre, no image needed")
    m.add_argument("--colors")
    m.add_argument("--palette")
    m.add_argument("--energy")
    m.add_argument("--brushwork")
    m.add_argument("--table", type=Path, help="CSV with one painting per row")

    t = sub.add_parser("train", help="train the AI on labelled paintings")
    t.add_argument("data", type=Path, help="folder of paintings (one sub-folder per genre, or use --labels)")
    t.add_argument("--labels", type=Path, help="CSV with File, Colours, Colour palette, Energy, Brushstrokes, Genre")
    t.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    t.add_argument("--no-clip", action="store_true", help="train without the CLIP image model")
    t.add_argument("--report", type=Path, default=DEFAULT_TRAINING_REPORT,
                   help="where to save the cross-validated prediction for each painting")

    for p in (a, t):
        p.add_argument("--no-crop", action="store_true", help="measure the whole photo (wall and frame included)")

    args = ap.parse_args(argv)
    global AUTO_CROP
    AUTO_CROP = not getattr(args, "no_crop", False)
    genres = load_genres(args.genres)
    if args.cmd == "analyze":
        cmd_analyze(args, genres)
    elif args.cmd == "match":
        cmd_match(args, genres)
    else:
        print(f"Training on {args.data}")
        train(args.data, genres, args.model, args.labels, args.report, not args.no_clip)


if __name__ == "__main__":
    sys.exit(main())
