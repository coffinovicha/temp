# Painting → Music Genre Analyzer

This program looks at a photo of a painting, grades it from 1 to 4 in four categories, and finds the music genre in `genres.csv` that fits it best.

All numbers in this document come from the dataset of 196 paintings in `261013_Dataset_CU_Paintings_JPG`.

---

## Contents

1. [How to use it](#1-how-to-use-it)
2. [Overview of the process](#2-overview-of-the-process)
3. [Finding the painting in the photo](#3-finding-the-painting-in-the-photo)
4. [All measurements](#4-all-measurements)
5. [Grading system](#5-grading-system)
6. [From grades to genre percentage](#6-from-grades-to-genre-percentage) (and the music)
7. [The labels](#7-the-labels)
8. [How well it works](#8-how-well-it-works)
9. [Results on the dataset](#9-results-on-the-dataset)
10. [Weaknesses and next steps](#10-weaknesses-and-next-steps)
11. [Files in this folder](#11-files-in-this-folder)

---

## 1. How to use it

| What you want | Command |
|---|---|
| **Open the app**: drop in a photo, see the painting, grades and genre, and hear matching music | double-click `Painting to Music` or `Start app.bat`, or `py app.py` (opens http://127.0.0.1:7860 in your browser) |
| Choose photos in a dialog and see the results in a window | `py painting_analyzer.py analyze` |
| Analyze one painting and show it | `py painting_analyzer.py analyze my_painting.jpg --show` |
| Analyze a folder and browse it in a window | `py painting_analyzer.py analyze paintings_folder --show` |
| Save all results to a table | `py painting_analyzer.py analyze paintings_folder --csv results.csv` |
| Save a result picture for each painting | `py painting_analyzer.py analyze paintings_folder --cards result_pictures` |
| Train the AI on a labels table | `py painting_analyzer.py train paintings_folder --labels labels.csv` |
| Train the AI on genre folders | `py painting_analyzer.py train dataset` (one sub-folder per genre) |
| Train without the CLIP image model | add `--no-clip` to `train` |
| Use only the fixed rules, not the AI | add `--no-model` to `analyze` |
| Measure the whole photo, without cropping | add `--no-crop` to `analyze` or `train` |
| Match a description without an image | `py painting_analyzer.py match --colors vibrant --palette "high contrast" --energy "very high" --brushwork dominant` |
| Match grades directly | `py painting_analyzer.py match --colors 1 --palette 3 --energy 2 --brushwork 2` |

**Result window controls:** ← / → or *Previous / Next* to browse, *Open photos...* to add more, **Esc** to close.

**Supported formats:** `.jpg .jpeg .png .bmp .tif .tiff .webp`

**Grades are always listed in this order:** Colours, Colour palette, Energy, Brushstrokes. For example, **3, 4, 3, 4** = bright colours, high-contrast palette, high energy, dominant brushstrokes.

---

## 2. Overview of the process

```
Photo ──► 1. Find the painting ──► 2. Measure (38 numbers) ──► 3. Grade (4 × 1–4) ──► 4. Compare with genres.csv ──► Genre + %
          (remove wall, frame)     + CLIP (1024 numbers)        trained models                + trained genre model
                                                                 or fixed rules
```

| Step | What happens | Section |
|---|---|---|
| 1 | The painting is cut out of the photo: wall, frame and mat are removed, and the canvas is straightened | 3 |
| 2 | 38 numbers are measured: colour, light/dark, texture, edges and stroke direction. The CLIP image model adds 1024 numbers describing content and style | 4, 5.1 |
| 3 | Each category gets a grade from 1 to 4, from **trained models** (default) or **fixed rules** (fallback) | 5 |
| 4 | The grades are compared with each genre in `genres.csv` and blended 50/50 with a **trained genre model** | 6 |

---

## 3. Finding the painting in the photo

Many photos show the wall, the frame, a mat, glass reflections or even neighbouring paintings. Before, those were measured too: a gold frame added edges and texture, and a white wall raised the brightness. Now the painting is cut out first, in the function `find_canvas`.

**Automatic crop:**
1. The photo is shrunk and searched for **rectangles around the centre**: the outlines of the frame, the mat and the canvas. Only shapes with corners within 7° of a right angle count.
2. The **innermost rectangle** that is still a good part of the outermost one is chosen. This is usually the edge of the canvas.
3. If there is no frame, the program looks for the **non-wall area** around the centre. Wall means light, grey and smooth.
4. The rectangle is **straightened**, which corrects photos taken at a slight angle.
5. Any remaining frame or mat inside it is removed: a straight line at the **same distance from the edge on at least 3 sides**.
6. A small 1.5% safety margin is cut off all round.

**Hand crops (`crops.csv`):** for 15 photos the automatic crop was wrong. Examples are paintings behind a wide mat, a canvas with the wall above it, and a photo containing two paintings. These photos are cropped by hand: each row gives the painting's edges as fractions of the photo (0 = left/top, 1 = right/bottom).

```
file,left,top,right,bottom,note
94_CU_Paintings.jpg,0.03,0.23,0.96,0.77,canvas inside the frame
```

All 196 crops were checked by eye.

**Effect:** with the old grading rules, cropping alone raised genre agreement from 31% to 37% (see section 8).

---

## 4. All measurements

The program measures **38 numbers** for every painting. The first **13** are the documented ones; they are saved in `results.csv` and used by the fixed rules. The other **25** are details used only by the trained models. The trained models also get **1024 CLIP numbers** (section 5.1).

### The 13 main measurements

| # | Measurement | What it means | How it is calculated | Dataset (min / median / max) |
|---|---|---|---|---|
| 1 | **saturation** | How strong or pure the colours are (0 = grey, 1 = pure colour) | Average S channel in HSV | 0.04 / 0.28 / 0.72 |
| 2 | **brightness** | How light the painting is (0 = black, 1 = white) | Average V channel in HSV | 0.21 / 0.51 / 0.80 |
| 3 | **colorfulness** | How colourful the image looks to a person | Hasler & Süsstrunk (2003) formula (see 5.2) | 5.5 / 38.9 / 108.0 |
| 4 | **colour_strength** | Combined colour score | 50% colorfulness + 50% saturation | |
| 5 | **hue_spread** | How many different hues are used (0 = one hue, 1 = all around the colour wheel) | Spread of hue angles, weighted by saturation × brightness | |
| 6 | **warmth** | Share of coloured pixels that are warm (red, orange, yellow, pink) | Hue below 70° or above 320°, counting only pixels with saturation > 0.15 | 0.00 / 0.81 / 1.00 |
| 7 | **contrast** | How different the light and dark areas are | Standard deviation of the grey-scale image | 0.06 / 0.19 / 0.31 |
| 8 | **dark_share** | Share of the image that is very dark | Brightness below 0.25 | |
| 9 | **light_share** | Share of the image that is very light | Brightness above 0.80 | |
| 10 | **edge_density** | How busy the image is | Share of pixels marked as edges by the Canny edge detector | 0.01 / 0.10 / 0.33 |
| 11 | **diagonal** | Share of strong lines that run diagonally | Direction of the strongest 25% of gradients (Sobel); diagonal if within 22.5° of 45° | 0.18 / 0.43 / 0.59 |
| 12 | **texture** | Fine surface detail, such as brushstrokes | Average difference between the image and a slightly blurred copy (σ = 2 px) | 0.009 / 0.038 / 0.118 |
| 13 | **energy** | Combined liveliness score | 40% edges + 25% contrast + 20% saturation + 15% diagonal (see 5.2) | 0.23 / 0.55 / 0.83 |

### The 25 detail measurements (trained models only)

| Group | Measurements | What they capture |
|---|---|---|
| Colour | `sat_p75`, `sat_p90`, `chroma_mean`, `chroma_p90`, `lightness`, `colour_share`, `hue_entropy` | How strong the most colourful parts are, how much of the painting is coloured, and how many colour families are used |
| Broken colour | `local_chroma_std`, `colour_changes` | How much the colour changes within small areas, as with visible, layered brushwork |
| Light and dark | `lum_range`, `flat_share`, `local_std_med` | The brightness range, and how much of the painting is smooth and untextured |
| Texture at several scales | `tex_s1`, `tex_s4`, `band_mid`, `band_low`, `band_ratio`, `tex_ratio` | Very fine detail (canvas grain), stroke-sized detail, larger shapes, and how they compare |
| Edges | `grad_mean`, `edges_low`, `edges_coarse` | Overall edge strength, faint edges, and busyness at a coarse scale |
| Stroke direction | `coherence_s2`, `coherence_s4`, `orient_entropy_s2`, `orient_entropy_s4` | Whether the texture runs in one direction locally, as brushstrokes do, and how many directions occur |

---

## 5. Grading system

There are two ways to turn the measurements into grades:

| | **Trained models** (default) | **Fixed rules** (fallback) |
|---|---|---|
| When used | When `painting_model.joblib` exists | With `--no-model`, or before any training |
| How it works | Models learn from the labelled paintings which combinations of the CLIP numbers and the 38 measurements mean which grade | One measurement per category, compared with fixed cut-off points |
| Pros | Much more accurate, especially for energy, brushstrokes and genre (section 8) | Easy to explain and adjust |
| Cons | You can't see a single formula behind a grade; CLIP needs PyTorch | Energy and brushstrokes are often wrong |

### 5.1 Trained models

#### The CLIP image model

**CLIP** is a large neural network pre-trained on billions of images with their captions. It turns an image into **1024 numbers** that capture what it shows and how it looks: "a calm misty landscape", "thick expressive brushstrokes", "an ancient Egyptian papyrus". The pixel measurements can't see this kind of thing.

| | |
|---|---|
| **Model** | Apple **DFN5B-CLIP ViT-H/14** (`ViT-H-14-quickgelu`, `dfn5b` in OpenCLIP), one of the strongest open CLIP models (about 83% zero-shot on ImageNet), pre-trained by Apple on about 5 billion image–caption pairs, and then **adapted to art on WikiArt** (below) |
| **Size** | 3.7 GB; downloaded once to the Hugging Face cache |
| **Speed** | About 0.3 s per painting on the RTX 4050 GPU; also works on the CPU, but slower |
| **Input** | The cropped painting. CLIP only looks at squares, so the program averages the **whole painting padded to a square** with up to **3 square crops** along its long side. This way nothing is cut off, not even on long panoramas |

**CLIP is not trained on your 196 paintings.** With so few, re-training a network with almost a billion parameters would only memorise them. Instead, CLIP's 1024 numbers are used as input for small models that are trained on your paintings (a "linear probe"), which is the standard way to use CLIP with few examples.

#### Adapting CLIP to art (WikiArt)

CLIP *was* adapted to art in general, using a large art collection instead of your paintings:

| | |
|---|---|
| **Data** | WikiArt (Hugging Face `Artificio/WikiArt`): 103,250 paintings, of which 101,750 were used for training and 1,500 held back for checking |
| **Task** | Recognise each painting's **style** (58 styles, e.g. Impressionism, Expressionism, Abstract Expressionism), **genre** (29, e.g. portrait, landscape, abstract) and **artist** (159) |
| **What changed** | Only the **last 4 of CLIP's 32 vision layers** (80M of its 632M parameters). The rest stays frozen, which fits the 6 GB GPU and keeps most of CLIP's knowledge |
| **Not forgetting** | A distillation term keeps the new CLIP numbers close to the original ones; they stayed 97.7% similar |
| **Training** | 1,600 steps of 64 paintings (one pass over WikiArt), learning rate 1e-5, 82 minutes on the RTX 4050 |
| **Result on held-back WikiArt** | style 64%, genre 77%, artist 82% correct |
| **Files** | `finetune_clip_art.py` (the training script), `clip_art_adapter.pt` (the 4 re-trained layers, 160 MB), `clip_art_training_log.txt` |

**Did it help on your paintings?** The same honest test as in section 8 was repeated over 10 different splits of the data:

| | Original CLIP | Art-adapted CLIP | Difference | Splits where art-adapted was better / same / worse |
|---|---|---|---|---|
| Colours | 66.2% | 68.0% | +1.8 | 9 / 0 / 1 |
| Colour palette | 62.4% | 63.3% | +0.9 | 7 / 1 / 2 |
| Energy | 56.9% | 57.3% | +0.4 | 5 / 1 / 4 |
| Brushstrokes | 60.3% | 62.0% | +1.7 | 9 / 1 / 0 |
| Genre model | 70.8% | 72.2% | +1.4 | 8 / 0 / 2 |
| **Final genre** | 68.8% | **70.9%** | **+2.1** | **10 / 0 / 0** |

The gain is small (about 4 more paintings out of 196 get the right genre) but consistent, so the program uses the art-adapted CLIP. Energy did not clearly improve: WikiArt's labels (style, genre, artist) say little about how energetic a painting feels.

To go back to the original CLIP, delete or rename `clip_art_adapter.pt` and run `train` again.

#### What is trained

For each of the 5 tasks (Colours, Colour palette, Energy, Brushstrokes, Genre), two models are trained and their answers averaged 50/50 (`CLIP_WEIGHT`):

| Model | Input | What it is |
|---|---|---|
| **Logistic regression** | 1024 CLIP numbers + 38 measurements | A linear model that gives a probability for each grade or genre. Its strength is chosen automatically by an extra cross-validation inside the training data |
| **Random Forest** | 38 measurements | 500 decision trees, each asking yes/no questions such as "is colorfulness above 44?"; the trees vote |

CLIP is better at energy, brushstrokes and genre. The pixel measurements are better at Colours. Averaging the two gives the best balance (section 8).

#### Training

```
py painting_analyzer.py train paintings_folder --labels labels.csv
```
Training does the following:
1. Crops, measures and runs CLIP on every labelled painting.
2. Tests everything with 5-fold cross-validation and prints the accuracy, including the **final genre** after the whole chain from section 6.
3. Saves the models to `painting_model.joblib`.
4. Writes `training_report.csv`.

Without PyTorch, or with `--no-clip`, it trains the Random Forests only. If you change `labels.csv`, for example with your team's own grades, run the same command again and the models learn the new labels.

### 5.2 Fixed rules

The cut-off points below were **fitted to the 196 labelled paintings**. The old guessed cut-offs gave grade 1 for Energy and Brushstrokes to no painting at all. All values are in `THRESHOLDS` in `painting_analyzer.py`.

Painting **1_CU_Paintings.jpg** (cropped) is the worked example in each category.

#### Colours (1–4): how colourful?

**Measurement: `colorfulness`** (Hasler & Süsstrunk, 2003)
```
rg = R − G                yb = ½(R + G) − B
colorfulness = √(σ_rg² + σ_yb²) + 0.3 × √(μ_rg² + μ_yb²)
```
σ = standard deviation (how much the colour varies) and μ = average (the overall colour cast).

| colorfulness | Grade | Meaning |
|---|---|---|
| below 23 | **1** | muted |
| 23 – 44 | **2** | natural |
| 44 – 66 | **3** | bright |
| 66 and above | **4** | vibrant |

*Change:* the old rule used `colour_strength`, plus a "bright and light" exception. `colorfulness` alone matches the labels better.

**Painting 1:** colorfulness is 28.2, so the grade is **2 (natural)**.

#### Colour palette (1–4): what kind of colour scheme?

The rules are checked in order, and the first that matches decides the grade:

| Order | Rule | Grade | Meaning |
|---|---|---|---|
| 1 | contrast ≥ 0.23, **or** colorfulness ≥ 60 with contrast ≥ 0.14 | **4** | high contrast |
| 2 | brightness ≥ 0.65 **and** saturation < 0.30 | **1** | pastel, light |
| 3 | brightness < 0.40, **or** warmth > 0.85 with brightness < 0.65 | **3** | deep, warm |
| 4 | none of the above | **2** | elegant, harmonised |

*Changes:* the high-contrast cut-off went from 0.26 to 0.23. "Many hues" was replaced by "very colourful". Pastel needs brightness 0.65 instead of 0.62. Deep needs brightness below 0.40 instead of 0.45. Warm needs 85% warm colours instead of 70%, and allows brightness up to 0.65 instead of 0.55.

**Painting 1:** contrast is 0.237, which is at least 0.23, so the grade is **4 (high contrast)**.

#### Energy (1–4): how lively?

**Measurement: `energy`** (formula unchanged)
```
energy = 0.40 × min(edge_density / 0.18, 1)     ← how busy (40%)
       + 0.25 × min(contrast / 0.30, 1)         ← light/dark tension (25%)
       + 0.20 × min(saturation / 0.60, 1)       ← colour intensity (20%)
       + 0.15 × diagonal                        ← movement (15%)
```

| energy | Grade | Meaning |
|---|---|---|
| below 0.43 | **1** | low |
| 0.43 – 0.62 | **2** | medium |
| 0.62 – 0.79 | **3** | high |
| 0.79 and above | **4** | very high |

*Change:* the old cut-offs were 0.25 / 0.42 / 0.60. I also tried re-fitting the formula weights, but that only gained 1 percentage point, so the formula stays the same.

**Painting 1:** energy is 0.525, so the grade is **2 (medium)**.

#### Brushstrokes (1–4): how visible?

**Measurement: `texture`**
```
texture = average |grey image − blurred grey image|     (Gaussian blur, σ = 2 px)
```

| texture | Grade | Meaning |
|---|---|---|
| below 0.014 | **1** | not visible |
| 0.014 – 0.041 | **2** | subtle |
| 0.041 – 0.081 | **3** | visible |
| 0.081 and above | **4** | dominant |

*Change:* the old cut-offs were 0.008 / 0.018 / 0.032. Even re-fitted, this single measurement agrees with the labels only 31–34% of the time. It can't tell brushstrokes apart from papyrus fibres, canvas grain or glass reflections. The trained model, which also uses stroke direction and broken colour, does better (46–52%).

**Painting 1:** texture is 0.047, so the grade is **3 (visible)**.

**Painting 1 with the fixed rules: 2, 4, 2, 3.** My labels for it are 1, 4, 3, 3.

---

## 6. From grades to genre percentage

### Step A: The genre table

`genres.csv` is unchanged. I tested genre profiles learned from the labels instead, and they were not better on paintings held back from fitting (45% with yours, 44–50% with learned ones).

| Genre | Colours | Palette | Energy | Brushstrokes |
|---|---|---|---|---|
| **Classical** | bright → 3 | elegant, harmonised → 2 | very low **and** very high → 1 **or** 4 | not applicable → skipped |
| **Jazz** | muted → 1 | deep colours, warm beige → 3 | medium, high → 2 **or** 3 | subtle → 2 |
| **Ambient** | muted → 1 | pastel, light → 1 | low → 1 | subtle → 2 |
| **Rock/Metal** | vibrant → 4 | high contrast → 4 | very high → 4 | dominant → 4 |

### Step B: Points per category
```
points = 1 − (distance to the closest allowed grade) / 3
```
| Distance | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Points | 1.00 | 0.67 | 0.33 | 0.00 |

### Step C: Table score
```
table score = average of the points (skipped categories are left out)
```

*Tested:* counting "not applicable" as a neutral 0.5 instead of skipping it made results worse (58% → 51%), so skipping stays.

### Step D: Blend with the trained genre model
```
final % = 50% × table score + 50% × genre model confidence        (MODEL_WEIGHT = 0.5)
```
The genre with the highest final % is the **best genre**, and the next one is the **second option**. Without a trained model, the final % is the table score.

**Example: painting 1** (trained grades 1, 4, 3, 3):

| Genre | Table score | AI model | Final |
|---|---|---|---|
| **Jazz** | 83% | 72% | **78%** ← best genre |
| Rock/Metal | 58% | 7% | 33% ← second option |
| Ambient | 50% | 8% | 29% |
| Classical | 44% | 12% | 28% |

**The percentage is a closeness score, not a probability.** It mixes how close the grades are to the genre's row in `genres.csv` with how confident the trained genre model is.

### Step E: Choosing the music

The `music/` folder holds **32 tracks by Kevin MacLeod** (incompetech.com), **8 per genre**, licensed under Creative Commons Attribution 4.0 (CC BY 4.0). They are free to use as long as he is credited, so the app shows the credit with every track, and `music/CREDITS.txt` lists them all. Each track is described in `music/tracks.csv`:

| Column | Meaning |
|---|---|
| Genre | Classical, Jazz, Ambient or Rock/Metal. Kevin MacLeod has no "Ambient" genre, so calm synth/pad pieces from his Contemporary, Electronica, Soundtrack and Unclassifiable genres were used |
| Energy (1–4) | From the tempo (below 85 BPM = 1, below 115 = 2, below 150 = 3, faster = 4), one lower for calm/relaxed pieces and one higher for driving/aggressive/intense ones |
| Mood | **bright**, **neutral** or **dark**, from the composer's mood tags (Bright, Uplifting… vs. Dark, Mysterious, Somber…) |

| Genre | Tracks |
|---|---|
| Classical | Gymnopedie No. 1, Sinfonia Number 5, Canon in D Major, Sheep May Safely Graze, Brandenburg Concerto No. 4, Danse Macabre, Hall of the Mountain King, Egmont Overture Finale |
| Jazz | Cool Vibes, I Knew a Guy, Jazz Brunch, Bass Walker, George Street Shuffle, Faster Does It, Deuces, Rollin at 5 |
| Ambient | Airship Serenity, Silver Blue Light, Eastminster, Ossuary 5 – Rest, Soaring, Floating Cities, Lightless Dawn, Odyssey |
| Rock/Metal | Last Kiss Goodnight, Happy Bee, Motivator, Pulse, Cool Rock, Metalmania, Summon the Rawk, Exhilarate |

**How a track is picked** (function `pick_music`):
1. Only tracks of the painting's **best genre** are considered. In the app you can pick another genre.
2. The painting's **mood** comes from its colour grades (function `painting_mood`): pastel palettes and vibrant colours count towards *bright*, deep/warm palettes and muted colours count towards *dark*.
3. Every track gets a distance = |track energy − painting Energy grade| + 0.75 × |track mood − painting mood|, where dark = −1, neutral = 0 and bright = 1. The closest track plays first, and **Another track** steps through the rest in order.

**Adding your own music:** put mp3/wav/ogg/flac files into `music/Classical`, `music/Jazz`, `music/Ambient` or `music/Rock_Metal`. They are found automatically. Without a row in `tracks.csv` they count as energy unknown and neutral mood; add a row to describe them.

---

## 7. The labels

`labels.csv` holds a "correct answer" for every painting: the four grades and a genre.

| | |
|---|---|
| **Who labelled** | **Claude (AI), by looking at each cropped painting.** These are not ratings by people. |
| **How** | Each painting was judged by eye against the meaning of each grade (section 1) and the descriptions in `genres.csv`, together with the titles and art groups in `Cu_dataset_v04.csv` |
| **Note column** | A short description of each painting, explaining the label |

**Label counts:**

| Grade | Colours | Colour palette | Energy | Brushstrokes |
|---|---|---|---|---|
| 1 | 40 | 19 | 58 | 39 |
| 2 | 65 | 47 | 68 | 61 |
| 3 | 61 | 81 | 50 | 58 |
| 4 | 30 | 49 | 20 | 38 |

| Genre | Classical | Jazz | Ambient | Rock/Metal |
|---|---|---|---|---|
| Paintings | 66 | 65 | 42 | 23 |

**Examples of judgement calls:**
- The Egyptian papyri (113–128) are labelled **Classical**: ancient, ceremonial, ornamental, low energy, no visible brushwork.
- The *Vivaldi's Seasons* series (76–81) is labelled **Classical**, following its title, even though the colours are vibrant.
- The grey Istanbul and seascape paintings (108–112) are labelled **Ambient**: muted, calm, misty.
- Wild gestural abstracts with thick paint (e.g. 34, 58, 145, 196) are labelled **Rock/Metal**. Calmer, warmer gestural abstracts (e.g. 33, 35, 55) are labelled **Jazz**.

**Important:** the labels are one AI rater's judgement, so "accuracy" in section 8 means "agrees with these labels". Labels from several people would be a stronger test. To use them, replace the columns in `labels.csv` and run `train` again.

---

## 8. How well it works

The program was compared with the labels for all 196 paintings. For the trained models this uses **5-fold cross-validation**: the paintings are split into 5 groups, and each group is tested by models trained only on the other 4. Every painting is therefore tested by a model that never saw it, which is an honest estimate for new paintings.

| Version | Colours | Palette | Energy | Brushstrokes | **Genre** |
|---|---|---|---|---|---|
| Old rules, whole photo (before) | 46% | 36% | 24% | 26% | **31%** |
| Old rules, cropped | 55% | 41% | 22% | 24% | **37%** |
| Re-tuned rules, cropped | 68–71% | 58–64% | 43–46% | 31–34% | **38%** |
| Random Forests on the 38 measurements, cropped | 67–72% | 56–60% | 44% | 46–52% | **55–57%** |
| CLIP alone, no training (zero-shot) | 40% | 45% | 35% | 42% | **51%** |
| CLIP + Random Forests, cropped | 66–68% | 61–63% | 56–58% | 60–61% | 69–71% |
| **Art-adapted CLIP + Random Forests, cropped** (current) | **68–69%** | **63–66%** | **54–57%** | **61–62%** | **71–73%** |
| *Always guessing the most common answer* | *33%* | *41%* | *35%* | *31%* | *34%* |

"Genre" is the **final genre** after the whole chain (grades → genre table, blended with the genre model), exactly as `analyze` produces it. Ranges come from different ways of splitting the data. For the re-tuned rules, the lower number is the honest cross-validated one.

*Zero-shot* means CLIP compared each painting with sentences such as "a painting that feels like jazz music", without any training. That this is clearly worse shows that training on the labels matters.

**Within one grade:** the current models are within 1 grade of the label 92–97% of the time.

**Genre by genre** (current models, cross-validated; rows = label, columns = prediction):

| Label ↓ / Predicted → | Classical | Jazz | Ambient | Rock/Metal | Correct |
|---|---|---|---|---|---|
| Classical (66) | **54** | 4 | 7 | 1 | 82% |
| Jazz (65) | 5 | **47** | 4 | 9 | 72% |
| Ambient (42) | 8 | 2 | **31** | 1 | 74% |
| Rock/Metal (23) | 1 | 10 | 1 | **11** | 48% |

Rock/Metal is still the weak spot: with only 23 examples, it's often mistaken for Jazz, the other genre with lively, strong colours. The art-adapted CLIP raised it from 8 to 11 correct.

**Classical bias:**

| | Classical | Jazz | Ambient | Rock/Metal |
|---|---|---|---|---|
| Labels | 66 | 65 | 42 | 23 |
| Before (old rules, whole photo) | **96** | 41 | **6** | 53 |
| Now (art-adapted CLIP + Random Forests, cross-validated) | 68 | 63 | 43 | 22 |

The bias came from the measurements: frames and walls pushed energy and brushstrokes up for almost every painting, and Classical accepts energy 4 and skips brushstrokes.

**Per painting:** `training_report.csv` shows, for each painting, its label and the cross-validated prediction for every category, the genre model, and the final genre.

---

## 9. Results on the dataset

`results.csv` and `result_pictures/` hold the final answers for all 196 paintings, using cropping, CLIP and the trained models.

⚠️ The models were **trained on these same 196 paintings**, so the results match the labels for every painting (196 of 196 genres, and the grade counts below equal the label counts). This shows the models learned the labels; it does **not** mean they are 100% accurate. For new paintings, expect the cross-validated accuracy from section 8, about 71–73% for the genre.

| Grade | Colours | Colour palette | Energy | Brushstrokes |
|---|---|---|---|---|
| 1 | 40 | 19 | 58 | 39 |
| 2 | 65 | 47 | 68 | 61 |
| 3 | 61 | 81 | 50 | 58 |
| 4 | 30 | 49 | 20 | 38 |

| Genre | Best genre | Second option |
|---|---|---|
| Classical | 66 | 68 |
| Jazz | 65 | 72 |
| Ambient | 42 | 23 |
| Rock/Metal | 23 | 33 |

**Columns in `results.csv`:** File, the 4 grades, Total, then for the best genre and the second option: genre, match % and points per category ("n/a" = skipped for that genre), the first music track chosen for it, then the 13 main measurements.

---

## 10. Weaknesses and next steps

1. **The labels come from one AI rater.** The most valuable next step is to have several people label the paintings, or at least 40–80 of them, then retrain and compare.
2. **Energy and brushstrokes are still the hardest categories.** CLIP helped a lot, but judging movement or whether texture is paint, canvas, papyrus or glass glare from a photo stays difficult.
3. **196 paintings is a small training set.** More labelled paintings would help the trained models most.
4. **CLIP could be tried in other versions.** Other strong models, such as SigLIP 2 or a larger DFN CLIP at 378 px, might do a little better. Test them by changing `CLIP_MODEL`, removing or redoing the art adapter, and retraining. The art adaptation could also be trained longer or on more layers; so far it gave a small, consistent gain (section 5.1).
5. **Cropping needs a hand crop now and then.** New photos with unusual framing may need a line in `crops.csv`. Photographing the canvas straight on and filling the frame avoids this.
6. **The genre table is coarse.** Classical and Jazz differ mostly in energy and palette, so they are often confused.

Where to change things in `painting_analyzer.py`:

| What | Where |
|---|---|
| Rule cut-off points | `THRESHOLDS` |
| Rule logic | function `grade` |
| Measurements | functions `measure` and `detail_measurements` |
| Measurements used by the trained models | `MODEL_FEATURES` |
| CLIP model, art adapter and CLIP's share in each prediction | `CLIP_MODEL`, `CLIP_ADAPTER`, `CLIP_WEIGHT` |
| Cropping | function `find_canvas` (hand crops: `crops.csv`) |
| Genre matching | functions `genre_points` and `match_genres` |
| AI model weight in the genre % | `MODEL_WEIGHT` |
| Words understood in `genres.csv` | `VOCAB` |

---

## 11. Files in this folder

| File / folder | What it is |
|---|---|
| `painting_analyzer.py` | The program (command line) |
| `app.py` | The web app: photo in, painting, grades, genre and music out |
| `Start app.bat` | Double-click to start the web app |
| `Painting to Music` (shortcut) | Same as `Start app.bat`, with the app's logo as its icon |
| `assets/` | The app's logo: a painter's palette with one paint dab per genre (gold = Classical, blue = Jazz, teal = Ambient, red = Rock/Metal) and a music note. `logo.svg` (original), `logo_512.png`, `logo_64.png` (browser tab icon), `logo.ico` (shortcut icon) |
| `music/` | 32 Kevin MacLeod tracks (CC BY 4.0) in one folder per genre, `tracks.csv` (genre, energy, mood) and `CREDITS.txt` |
| `genres.csv` | Genre table: what each music genre looks like in the four categories |
| `labels.csv` | The "correct answers": 4 grades and a genre per painting, labelled by Claude by eye |
| `crops.csv` | Hand crops for the 15 photos the automatic crop got wrong |
| `clip_art_adapter.pt` | CLIP's last 4 layers, re-trained on WikiArt (160 MB); loaded on top of the original CLIP |
| `finetune_clip_art.py` | The script that adapted CLIP to art on WikiArt (about 80 minutes on the GPU) |
| `clip_art_training_log.txt` | Log of that training run: loss and WikiArt accuracy every 250 steps |
| `painting_model.joblib` | The trained models: for each of the 4 categories and the genre, a CLIP logistic regression and a Random Forest (21 MB). The CLIP model itself (3.7 GB) stays in the Hugging Face cache |
| `training_report.csv` | Honest test: label vs cross-validated prediction for every painting |
| `results.csv` | Final results for all 196 paintings: grades, best genre, second option, points and measurements |
| `result_pictures/` | One picture per painting: the cropped painting next to its grades and genre match |
| `Cu_dataset_v04.csv` | Painting information: artist, title, year, art group |
| `requirements.txt` | Python packages needed (`py -m pip install -r requirements.txt`); torch and open_clip_torch are needed for CLIP |
| `GRADING_SYSTEM.md` | This document |
| `261013_Dataset_CU_Paintings_JPG-.../` | The 196 painting photos |
