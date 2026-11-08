# kerala-hazard-vision-ai: visualizing post-disaster landslide and flood impact

A conditional GAN (pix2pix) that takes a real **pre-disaster Sentinel-2 image of an area in
Kerala** and generates a plausible **post-disaster image** at a chosen severity. It is
conditioned on terrain slope, rainfall and soil type.

Hazard maps say "high risk"; this project shows what that risk could *look like*, to help
non-experts understand it.

Generative AI capstone (course M5221352, Digital University Kerala).
Author: Muhammed Shahid S.

> Generated images are plausible scenarios for hazard communication, **not forecasts**.

## How it works

```
Google Earth Engine                        kerala_hazard_vision_ai package                    Colab notebooks
─────────────────────                      ───────────────────                    ───────────────
Sentinel-2 before ─┐                       severity.py  NDVI/NDWI change → 0–1     01_data
Sentinel-2 after  ─┼─► 11-band stack ──►   tiling.py    256×256 tiles (.npz)
Copernicus DEM    ─┤   per event           dataset.py   9-channel input, event split
CHIRPS rainfall   ─┤   (GeoTIFF on Drive)  models/      U-Net G + 70×70 PatchGAN D
OpenLandMap soil  ─┘                       train.py     pix2pix loss, resumable    02_train
                                           evaluate.py  SSIM/PSNR/FID, severity    03_evaluate
                                           diffusion_baseline.py  Stable Diffusion
                                           human_eval.py  real-vs-generated survey
                                           demo.py      draw area → severity sweep 04_demo
```

**Model input (9 channels):**
- the before bands B2, B3, B4 and B8
- slope
- 7-day rainfall
- soil texture class
- severity (0–1)
- hazard type (flood = 0, landslide = 1)

**Output:** the 4 after bands.

**Severity labels** are measured, not hand-labelled. Severity is the share of a tile's pixels
whose change matches the hazard:
- landslide: an NDVI (vegetation index) drop
- flood: NDWI (water index) crossing into water

**Events** are listed in [`configs/events.yaml`](configs/events.yaml):
- Kerala floods 2018
- Kavalappara 2019
- Puthumala 2019
- Pettimudi 2020 (held out as the test event)
- Kokkayar 2021
- Wayanad 2024

**Evaluation** uses the held-out event and compares:
- pix2pix
- an identity baseline ("nothing changes")
- zero-shot Stable Diffusion img2img, with 4 prompt designs

It also includes a human real-vs-generated survey.

## Results

All numbers come from [`outputs/metrics.json`](outputs/metrics.json), written by
`notebooks/03_evaluate.ipynb`.

| Method | SSIM ↑ | PSNR ↑ | FID ↓ |
|---|---|---|---|
| pix2pix (ours) | _pending_ | _pending_ | _pending_ |
| Identity baseline | _pending_ | _pending_ | _pending_ |
| Stable Diffusion (best prompt) | _pending_ | _pending_ | _pending_ |

Other results, also pending until the notebooks run:
- severity consistency (Pearson r between requested and measured severity)
- the human-evaluation fooled rate

Figures will appear in [`outputs/figures/`](outputs/figures/):
- `triplets.png`
- `sweep.png`
- `severity_consistency.png`
- `losses.png`
- `demo_*.png`

## Repository layout

| Path | What it holds |
|---|---|
| `src/kerala_hazard_vision_ai/` | All logic, typed and documented |
| `configs/` | Every tunable value (`default.yaml`) and the event list (`events.yaml`) |
| `notebooks/` | Thin Colab drivers, committed with their outputs |
| `tests/` | pytest suite (CPU-only, no network) |
| `docs/superpowers/` | Design spec and implementation plan |

## Running it

**Tests (any laptop, CPU only):**

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # Linux/macOS: .venv/bin/pip
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/ruff check .
.venv/Scripts/pytest
```

CI runs the same lint and tests on every push (`.github/workflows/ci.yml`).

**Full pipeline (Google Colab, T4 GPU):**

You need a free [Earth Engine](https://earthengine.google.com) non-commercial Cloud project.
Open the notebooks in order and set `GEE_PROJECT` where asked:

1. `01_data.ipynb`
   - Checks each event box on a map.
   - Runs the cloud-free go/no-go check.
   - Exports the stacks to Drive and tiles them.
2. `02_train.ipynb`
   - Runs an overfit sanity check, then full training.
   - Re-running it resumes from the last epoch.
3. `03_evaluate.ipynb`
   - Computes the metrics, figures, Stable Diffusion comparison and human-evaluation survey.
4. `04_demo.ipynb`
   - Opens the interactive map: draw an area, pick the hazard and rainfall, then crossfade
     through the severities.

Tiles, checkpoints and outputs live in `MyDrive/kerala_hazard_vision_ai/`. Copy `outputs/` and the
executed notebooks back into the repo to publish the results.

## Limitations

- **Resolution:** at 10 m, individual buildings are not visible. Only area-scale damage is
  visible, such as landslide scars and flooded fields.
- **Few events:** with 6 events, results describe a small sample, and overlapping tiles make
  the effective sample size smaller than the tile count.
- **Floods:** flood water recedes before the next dry season, so flood pairs may show little
  water. See the go/no-go note in `01_data.ipynb`.
- **Soil:** soil texture is a class number treated as a continuous value.
- **Demo rainfall:** the demo uses a constant rainfall scenario instead of a forecast.
- **Area check:** the demo checks the Kerala bounding rectangle, not the state border.
- **Extrapolation:** severities above the training maximum are extrapolation and are labelled
  so in the demo.

## Data credits

- Copernicus Sentinel-2 and Copernicus DEM GLO-30 (ESA / European Union)
- CHIRPS daily rainfall (UCSB Climate Hazards Center)
- OpenLandMap soil texture (EnvirometriX, CC BY-SA 4.0)
- All accessed through Google Earth Engine
- Stable Diffusion v1.5 (CreativeML OpenRAIL-M licence)
- Method: Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*,
  CVPR 2017
