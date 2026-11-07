# Design: Post-Disaster Image Generator (Gen AI Capstone)

Date: 2026-11-08. Owner: Muhammed Shahid S (solo build). Deadline: report upload before 2026-11-15 00:00.

## 1. Intent

Hazard maps say "high risk" but do not show what that means. This project trains a conditional
generative model that takes a real pre-disaster satellite image of an area in Kerala and generates a
plausible post-disaster image at a user-chosen severity, for landslides and floods. The result is a
visual "disaster forecast" for non-expert audiences.

Why it matters to the owner: this is the graded capstone (worth more than the written exam), and
**marks are based on code quality**. The code must be clean, tested and reproducible, not a
notebook hack.

## 2. Goals, non-goals, anti-goals

Goals
- A working pix2pix model trained on real Kerala before/after pairs (Sentinel-2).
- Conditioning on slope, rainfall, soil, hazard type and a severity value from 0 to 1.
- Honest evaluation on a held-out event, compared against a baseline.
- A demo where the user picks an area in Kerala, moves a severity slider and sees the result.
- A clean Python package with tests, lint, a README and git history.
- A final report under 10 pages: title, introduction, methodology, data, results, discussion.

Non-goals
- Diffusion models (future work).
- Roof-level detail or very-high-resolution imagery (Sentinel-2 is 10 m).
- Tsunami or other hazards.
- Video generation (the demo crossfades between severities instead).
- Selecting areas outside Kerala.
- A standalone web app (the demo runs in Colab).

Anti-goals (failure even if it runs)
- Any metric or figure in the report that was not produced by the code.
- Leakage: test tiles taken from the same event used for training.
- Logic that lives only in notebooks, magic numbers spread through the code, or no tests.
- Over-engineering: abstractions and options nobody uses.

## 3. Constraints

- Laptop: i3-10110U, 8 GB RAM, no GPU. It is used only for editing, git and CPU unit tests.
- Compute: Google Colab free tier (T4 GPU), with Kaggle as the backup. Sessions drop after a few
  hours, so training must checkpoint and resume.
- Data access: Google Earth Engine (GEE) needs a free non-commercial account and Cloud project.
  Exports go to Google Drive (free 15 GB). Nothing large is downloaded to the laptop.
- Time: 7 days (Nov 8–14).
- Code hosting: GitHub is optional. Git is local now; a GitHub remote can be added at any time.
  Without GitHub, the package is copied to Drive and Colab installs it from there.

## 4. Approaches considered

| Option | Trade-off | Decision |
|---|---|---|
| A. Sentinel-2 pairs from GEE, Kerala, both hazards, all conditioning layers | Matches the submitted aim; the demo can fetch any Kerala area. Risks: monsoon clouds and few events. | **Chosen** |
| B. xBD (Maxar, high resolution) | Roofs visible and real damage labels, but floods only, no Kerala, and no free high-resolution input for the demo | Rejected: breaks the aim (landslides, Kerala) |
| C. A ready dataset (Sen1Floods11, Landslide4Sense) | No before/after pairs, so it cannot train before-to-after translation | Rejected |
| Diffusion instead of a GAN | More modern, but slower to train and riskier in 7 days on a free GPU | Future work |

## 5. Decided design

### 5.1 Data (GEE, `data/gee_export.py`)

Events (Kerala; final list depends on the day-1 cloud check):

| Event | Hazard | Date |
|---|---|---|
| Kerala floods | flood | Aug 2018 |
| Kavalappara (Malappuram) | landslide | 8 Aug 2019 |
| Puthumala (Wayanad) | landslide | 8 Aug 2019 |
| Pettimudi (Idukki) | landslide | 6 Aug 2020 |
| Kokkayar / Koottickal | landslide | 16 Oct 2021 |
| Mundakkai–Chooralmala (Wayanad) | landslide | 30 Jul 2024 |

- **Imagery:** `COPERNICUS/S2_SR_HARMONIZED`, bands B2, B3, B4 and B8 (blue, green, red, near
  infrared), cloud-masked with the scene classification layer.
  - Before image: median composite from the dry season before the event.
  - After image: median composite from the next dry season.
  - Why: landslide scars last for months, so dry-season composites avoid monsoon cloud and keep
    the comparison season-matched.
- **Slope:** `COPERNICUS/DEM/GLO30`, using `ee.Terrain.slope`.
- **Rainfall:** `UCSB-CHG/CHIRPS/DAILY`, total over the event window (7 days before the event).
- **Soil:** OpenLandMap soil texture class (USDA), top layer.
- **Hazard type:** 0 = flood, 1 = landslide, as one constant channel per tile.
- **Grid:** every layer is resampled to the same 10 m grid and exported as one stacked GeoTIFF per
  event to Drive. The GeoTIFFs are then cut into 256×256 tiles saved as `.npz`.
- **Risk:** floods recede within days, so a dry-season "after" image may show no water. If the
  day-1 check finds no usable flood pair, the fallback is a near-event composite with stricter
  cloud filtering. If that also fails, floods are scoped down to future work and the report says so.

### 5.2 Severity label (`data/severity.py`)

Severity is computed from the real change, so nothing is hand-labelled.
- Landslide: the fraction of tile pixels where NDVI (vegetation index) drops by more than a set
  threshold.
- Flood: the fraction of pixels where NDWI (water index) rises above a set threshold.
- The fraction is scaled to 0–1. The thresholds live in the config.
- The same function also scores generated images (see 5.5).

### 5.3 Dataset (`data/dataset.py`)

- A PyTorch `Dataset` reads the tiles.
- Model input (9 channels): 4 before bands, slope, rain, soil, severity (one constant channel)
  and hazard type. Target: 4 after bands.
- Each channel is normalized with statistics computed on the training split only.
- Augmentation: flips and 90° rotations, applied to input and target together.
- A weighted sampler oversamples tiles with severity above 0.05, because unchanged tiles dominate
  the data.
- Split by event: one landslide event is held out as the test set (the builder chooses the one
  with enough tiles). Validation uses a held-out spatial block (the easternmost 10% of tiles) from
  each training event.

### 5.4 Model (`models/`)

- Generator: pix2pix U-Net for 256×256 inputs (Isola et al. 2017), 9 input and 4 output channels,
  tanh output.
- Discriminator: 70×70 PatchGAN that sees the input and the image (real or generated) together.
- Loss: conditional GAN loss plus 100 × L1 loss (the pix2pix default).
- Optimizer: Adam, learning rate 2e-4, betas (0.5, 0.999), batch size 16.
- About 100 epochs, the final value tuned to the time budget.
- Fixed random seeds. A checkpoint is saved to Drive every epoch, and `--resume` continues from
  it.

### 5.5 Evaluation (`evaluate.py`)

All metrics are computed on the held-out event.
- SSIM and PSNR between the generated and real after images, using torchmetrics.
- FID on RGB, with the known caveat that Inception features suit satellite images poorly.
- **Severity consistency (task-specific metric):** run `severity.py` on generated images and
  correlate the result with the requested severity. This follows the literature's advice to use
  domain-aware checks.
- Baseline: identity ("after = before"). The model has to beat it to be worth anything.
- **Diffusion comparison (approved, syllabus Modules 1 and 3):** run a pretrained Stable Diffusion
  img2img pipeline (Hugging Face `diffusers`, fp16 on a T4) on the same held-out before images.
  Test 3–4 prompt designs, for example "satellite image of the same area after a landslide". Score
  it with the same metrics. The discussion compares the task-trained GAN with the general,
  untrained diffusion model.
- **Human evaluation (approved, Module 2):** 5–10 classmates each label 20 images as real or
  generated. Report the share of generated images labelled "real". The survey form and the
  answers are saved in the repo.
- IS is not used, and the report explains why: its ImageNet classes do not fit satellite images.
- Stretch goal: an ablation without the conditioning layers.
- Outputs: a metrics JSON file and figures (before / real after / generated, and severity sweeps).
  These are the only source of numbers for the report.

### 5.6 Demo (`notebooks/03_demo.ipynb`)

- A geemap map in Colab. The user searches for a place or draws a rectangle, limited to the Kerala
  bounding box.
- The demo calls the same `gee_export` functions to fetch the before image and the conditioning
  layers for that area.
- A severity slider and a hazard toggle drive the model, and the display crossfades the outputs
  at severity 0.1, 0.3, 0.6 and 1.0.
- Severities above the range seen in training are labelled as extrapolation.

### 5.7 Code quality

```
src/disastergen/  config.py  data/{gee_export,severity,dataset}.py
                  models/{generator,discriminator}.py  train.py  evaluate.py
configs/default.yaml   notebooks/01_data 02_train 03_demo   tests/
README.md  pyproject.toml  .gitignore
```
- One typed config (a dataclass loaded from YAML), so the code has no magic numbers.
- Type hints and docstrings on public functions.
- `ruff` for lint.
- `pytest` tests on CPU with small tensors: severity math, tiling, dataset shapes and alignment
  of augmentation, generator and discriminator output shapes, and a short train step whose loss
  is finite.
- Notebooks are thin: install the package, call its functions and show results.
- `logging` instead of print statements.
- README: setup, how to reproduce, results table and sample images.
- Small, meaningful git commits.
- If GitHub is added: one Actions workflow that runs ruff and pytest.

## 6. Build order (no dates; the owner works when free)

1. GEE account and Cloud project; repo skeleton, config and test setup; cloud-free check for each
   event (go / no-go).
2. `gee_export`, tiling and `severity`, with tests; export every event to Drive.
3. Dataset and models, with tests; overfit sanity check on about 20 tiles in Colab.
4. Full training run in Colab, using resume across sessions; write `evaluate.py`.
5. Evaluation, baseline and figures; diffusion comparison; human evaluation; stretch ablation.
6. Demo notebook; README.
7. Report (under 10 pages) from the metrics JSON and figures; submit before 2026-11-15 00:00.

The report's introduction and literature survey can be reused from the topic PDF.

## 7. Left to the builder

- The exact NDVI and NDWI thresholds, tuned by viewing a few tiles and recorded in the config.
- Which event to hold out for testing.
- The final number of epochs.
- The tile stride and overlap.
- The GeoTIFF band order.

## 8. Decisions and open items

- Approved 2026-11-08: the full version (pix2pix, diffusion comparison, human evaluation).
- Code hosting: GitHub, user `muhammedshahidsds25-collab`. Moodle accepts only the report, so
  the report links to the repo. The repo needs a GitHub Actions workflow that runs ruff and pytest.
- There is no presentation. Notebooks are committed with their outputs, and the README shows the
  results, so the grader never needs to run anything.
- Open: whether the report lists both names from the topic submission or only the owner.
- Open: the owner's Earth Engine Cloud project ID, needed before the data step.
