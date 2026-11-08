# kerala-hazard-vision-ai: GAN vs diffusion for visualising landslide and flood impact

Two conditional generative models take a real **pre-disaster Sentinel-2 image of an area in
Kerala** and generate a plausible **post-disaster image** at a chosen severity. Both are
conditioned on terrain slope, rainfall and soil type, and both learn from the same data:

- **pix2pix**, a conditional GAN (Isola et al., 2017)
- a **Palette-style conditional diffusion model** (Saharia et al., 2022)

The project compares them head to head, against a "nothing changes" baseline and a zero-shot
Stable Diffusion model.

Hazard maps say "high risk"; this project shows what that risk could *look like*, to help
non-experts understand it.

Generative AI capstone (course M5221352, Digital University Kerala).
Author: Muhammed Shahid S.

> Generated images are plausible scenarios for hazard communication, **not forecasts**.

## How it works

```
Google Earth Engine            kerala_hazard_vision_ai                         Colab notebooks
───────────────────            ───────────────────────                         ───────────────
Sentinel-2 before ─┐           data/severity.py   NDVI/NDWI change → 0–1       01_data
Sentinel-2 after  ─┼─► 11-band data/tiling.py     256×256 tiles (.npz)
Copernicus DEM    ─┤   stack   data/dataset.py    9-channel input, event split
CHIRPS rainfall   ─┤   per     models/            U-Net + PatchGAN (GAN)
OpenLandMap soil  ─┘   event   models/diffusion.py  conditional DDPM / DDIM
                               train.py           pix2pix training, resumable  02_train_pix2pix
                               train_diffusion.py diffusion training, EMA, AMP 03_train_diffusion
                               evaluate.py        SSIM/PSNR/FID, severity, time 04_evaluate
                               diffusion_baseline.py  zero-shot Stable Diffusion
                               human_eval.py      real vs pix2pix vs diffusion
                               demo.py            draw area → severity sweep   05_demo
```

**Model input (9 channels):**
- the before bands B2, B3, B4 and B8
- slope
- 7-day rainfall
- soil texture class
- severity (0–1)
- hazard type (flood = 0, landslide = 1)

**Output:** the 4 after bands. The diffusion model also receives the noisy after image
(13 channels in) and predicts the noise.

**Severity labels** are measured, not hand-labelled. Severity is the share of a tile's
cloud-free pixels whose change matches the hazard:
- landslide: an NDVI (vegetation index) drop
- flood: NDWI (water index) crossing into water

**Events** are listed in [`configs/events.yaml`](configs/events.yaml). Before and after images
are matching dry-season composites (Dec–Apr) on either side of each event.
- **Kerala:**
  - floods 2018
  - Kavalappara 2019
  - Puthumala 2019
  - Pettimudi 2020 (held out as the test event)
  - Kokkayar 2021
  - Wayanad 2024
- **Extra training data from the Western Ghats:**
  - Taliye 2021
  - Ambeghar 2021
  - Irshalwadi 2023
  - Shiroor 2024

**Evaluation** uses the held-out Kerala event and compares four methods:
- pix2pix
- conditional diffusion
- an identity baseline ("nothing changes")
- zero-shot Stable Diffusion img2img, with 4 prompt designs

It also includes a human survey with real, pix2pix and diffusion images.

## Results

All numbers come from [`outputs/metrics.json`](outputs/metrics.json), written by
`notebooks/04_evaluate.ipynb`.

| Method | SSIM ↑ | PSNR ↑ | FID ↓ | s / tile ↓ |
|---|---|---|---|---|
| pix2pix (GAN) | _pending_ | _pending_ | _pending_ | _pending_ |
| Conditional diffusion | _pending_ | _pending_ | _pending_ | _pending_ |
| Identity baseline | _pending_ | _pending_ | _pending_ | — |
| Stable Diffusion (best prompt) | _pending_ | _pending_ | _pending_ | — |

Other results, also pending until the notebooks run:
- severity consistency (Pearson r between requested and measured severity) for both models
- the human-evaluation fooled rate for each model

Figures will appear in [`outputs/figures/`](outputs/figures/):
- `comparison.png`
- `sweep_*.png`
- `diffusion_diversity.png`
- `severity_consistency.png`
- `losses*.png`
- `demo_*.png`

## Repository layout

| Path | What it holds |
|---|---|
| `src/kerala_hazard_vision_ai/` | All logic, typed and documented |
| `configs/` | Every tunable value (`default.yaml`) and the event list (`events.yaml`) |
| `notebooks/` | Thin Colab drivers, committed with their outputs |
| `tests/` | pytest suite (CPU-only, no network) |
| `docs/superpowers/` | Design spec and implementation plans |

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
2. `02_train_pix2pix.ipynb`
   - Runs an overfit sanity check, then GAN training.
3. `03_train_diffusion.ipynb`
   - Trains the diffusion model (about 3–4 h).
4. `04_evaluate.ipynb`
   - Computes all metrics and figures, the Stable Diffusion comparison and the survey.
5. `05_demo.ipynb`
   - Opens the interactive map: draw an area, pick the model, hazard and rainfall, then
     crossfade through the severities.

Both training notebooks resume from the last epoch when re-run. Tiles, checkpoints and outputs
live in `MyDrive/kerala_hazard_vision_ai/`. Copy `outputs/` and the executed notebooks back into
the repo to publish the results.

## Limitations

- **Resolution:** at 10 m, individual buildings are not visible. Only area-scale damage is
  visible, such as landslide scars and flooded fields.
- **Few events:** with 10 events, results describe a small sample, and overlapping tiles make
  the effective sample size smaller than the tile count.
- **Floods:** flood water recedes before the next dry season, and the 2018 event predates
  global Sentinel-2 L2A coverage, so floods may fail the go/no-go check. See `01_data.ipynb`.
- **Soil:** soil texture is a class number treated as a continuous value.
- **Demo rainfall:** the demo uses a constant rainfall scenario instead of a forecast.
- **Area check:** the demo checks the Kerala bounding rectangle, not the state border.
- **Extrapolation:** severities above the training maximum are extrapolation and are labelled
  so in the demo.
- **Diffusion speed:** diffusion needs 50 denoising steps per image, so it is much slower than
  pix2pix's single pass.

## Data credits

- Copernicus Sentinel-2 and Copernicus DEM GLO-30 (ESA / European Union)
- CHIRPS daily rainfall (UCSB Climate Hazards Center)
- OpenLandMap soil texture (EnvirometriX, CC BY-SA 4.0)
- All accessed through Google Earth Engine
- Stable Diffusion v1.5 (CreativeML OpenRAIL-M licence)

## Methods

- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021
- Saharia et al., *Palette: Image-to-Image Diffusion Models*, SIGGRAPH 2022
