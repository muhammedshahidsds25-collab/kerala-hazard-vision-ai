# Post-Disaster Image Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python package plus Colab notebooks that train a pix2pix cGAN to generate
post-disaster Sentinel-2 images of Kerala from a pre-disaster image, conditioning layers and a
severity value. It is evaluated against an identity baseline, a pretrained Stable Diffusion model
and human raters.

**Architecture:**
- `src/disastergen` holds all logic: config, GEE export, tiling, severity, dataset, models,
  training, evaluation, diffusion baseline, human evaluation and demo helpers.
- Notebooks are thin drivers that run on Colab.
- Pure-numpy and torch code is unit-tested on CPU. Code that calls Earth Engine or the GPU is
  exercised only in the notebooks.

**Tech Stack:** Python 3.10+, PyTorch, torchmetrics (with torch-fidelity for FID), numpy, PyYAML,
earthengine-api, geemap, rasterio, diffusers, pytest, ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-08-disaster-gen-design.md`

## Global Constraints

- Marks depend on code quality:
  - type hints and docstrings on public functions
  - `logging`, not `print`, in the package
  - no magic numbers outside `configs/*.yaml` and `config.py`
  - `ruff check` and `pytest` must pass before every commit
- The laptop has no GPU. Every test runs on CPU in under 60 seconds total, with no network calls.
- Band order everywhere: `("B2", "B3", "B4", "B8")` (blue, green, red, NIR).
- Model input is 9 channels in this order:
  1–4. the four before bands
  5. slope
  6. rain
  7. soil
  8. severity
  9. hazard
- Model output is 4 channels.
- Tiles are 256×256.
- Hazard codes: `flood = 0`, `landslide = 1`.
- pix2pix settings: L1 weight 100, Adam lr 2e-4, betas (0.5, 0.999), batch 16, about 100 epochs.
- Train/test split is by event. One whole event is held out.
- Report numbers come only from `outputs/metrics.json` and generated figures.
- Notebooks are committed with their outputs, because there is no presentation.
- GitHub remote: `https://github.com/muhammedshahidsds25-collab/disaster-gen`.
- Build order follows the task numbers. There are no dates.

## Review Focus

1. Cloud-masked pixels (NaN) inside a tile: drop the tile if its NaN fraction is above
   `max_nodata_frac`, otherwise fill the gaps. It must never produce a NaN loss.
   Test: Task 3 `test_tiles_with_nodata_are_dropped_or_filled`.
2. Severity outside [0, 1], from the demo slider or a bad label: raise `ValueError`.
   Test: Task 5 `test_build_input_rejects_bad_severity`.
3. Colab disconnects mid-run: `--resume` continues from the last saved epoch with the same
   optimizer state. Test: Task 7 `test_checkpoint_roundtrip_resumes_epoch`.
4. A `test_event` name with a typo or not found in the tiles: raise `ValueError`. Otherwise every
   tile would silently go to training and the test set would be empty.
   Test: Task 5 `test_split_rejects_unknown_test_event`.
5. A demo area outside Kerala, or larger than `max_demo_area_km2`: raise `ValueError` with a
   clear message. Test: Task 10 `test_validate_area_rejects_outside_kerala_and_too_large`.

---

### Task 1: Repo skeleton, config, CI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md` (stub), `configs/default.yaml`,
  `configs/events.yaml`, `src/disastergen/__init__.py`, `src/disastergen/config.py`,
  `.github/workflows/ci.yml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> Config` and frozen dataclasses:
  - `Config(data: DataConfig, severity: SeverityConfig, model: ModelConfig, train: TrainConfig, demo: DemoConfig, events: list[Event])`
  - `Event(name: str, hazard: str, date: date, bbox: tuple[float, float, float, float])`
    - `bbox` order is (min_lon, min_lat, max_lon, max_lat)
  - `DataConfig(tile_size=256, stride=256, max_nodata_frac=0.2, pre_window_days=120, post_window_days=150, post_offset_days=90, max_cloud_pct=40, scale_m=10, rain_days_before=7, drive_folder="disastergen", test_event="pettimudi_2020")`
  - `SeverityConfig(ndvi_drop=0.25, ndwi_rise=0.2, eps=1e-6)`
  - `ModelConfig(in_channels=9, out_channels=4, ngf=64, ndf=64, num_downs=8)`
  - `TrainConfig(epochs=100, batch_size=16, lr=2e-4, beta1=0.5, beta2=0.999, l1_weight=100.0, seed=42, severe_threshold=0.05, val_frac=0.1, checkpoint_dir="checkpoints")`
  - `DemoConfig(kerala_bbox=(74.85, 8.17, 77.42, 12.80), max_demo_area_km2=25.0, sweep=(0.1, 0.3, 0.6, 1.0), train_severity_max=1.0)`
  - `train_severity_max` is overwritten after the data step from the observed maximum.

- [ ] **Step 1: `git init`, add `.gitignore`**
  - Ignore: `.venv/`, `__pycache__/`, `checkpoints/`, `data/`, `*.npz`, `*.tif`, `.ipynb_checkpoints/`.
  - Do not ignore: `outputs/metrics.json`, `outputs/figures/`.
- [ ] **Step 2: Write the failing tests in `tests/test_config.py`**

```python
def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.in_channels == 9 and cfg.model.out_channels == 4
    assert cfg.train.l1_weight == 100.0 and cfg.data.tile_size == 256
    assert {e.hazard for e in cfg.events} <= {"flood", "landslide"}
    assert cfg.data.test_event in {e.name for e in cfg.events}

def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"; p.write_text("model:\n  bogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(p)
```

- [ ] **Step 3: Run `pytest tests/test_config.py -v`. Expected: FAIL (module missing).**
- [ ] **Step 4: Implement `config.py`, `pyproject.toml` and both YAML files**
  - `pyproject.toml`:
    - src layout, package `disastergen`
    - core deps: `numpy`, `torch`, `torchmetrics[image]`, `pyyaml`
    - extras `gee = [earthengine-api, geemap, rasterio]`
    - extras `diffusion = [diffusers, transformers, accelerate]`
    - extras `dev = [pytest, ruff]`
    - ruff: line-length 100, rules `E,F,I,UP,B,D` with the Google docstring convention
  - `load_config` reads `default.yaml` and `events.yaml` (sibling file) and rejects unknown keys.
  - `events.yaml`, six events. The centres are approximate, so the bbox is about ±0.03° and is
    checked visually in Task 4:

    | name | hazard | date | centre (lat, lon) |
    |---|---|---|---|
    | `kerala_floods_2018` | flood | 2018-08-16 | 10.30, 76.33 (Chalakudy) |
    | `kavalappara_2019` | landslide | 2019-08-08 | 11.33, 76.30 |
    | `puthumala_2019` | landslide | 2019-08-08 | 11.50, 76.18 |
    | `pettimudi_2020` | landslide | 2020-08-06 | 10.17, 77.02 |
    | `kokkayar_2021` | landslide | 2021-10-16 | 9.60, 76.85 |
    | `wayanad_2024` | landslide | 2024-07-30 | 11.47, 76.13 |

- [ ] **Step 5: Add `.github/workflows/ci.yml`**
  - Triggers on push and pull request, Ubuntu, Python 3.11.
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - `pip install -e .[dev]`
  - `ruff check .`
  - `pytest`
- [ ] **Step 6: Set up the local environment and run the tests. Expected: PASS.**
  - `python -m venv .venv`
  - `.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - `.venv\Scripts\pip install -e .[dev]`
  - `.venv\Scripts\ruff check .`
  - `.venv\Scripts\pytest -v`
- [ ] **Step 7: Commit with message `chore: project skeleton, typed config, CI`.**

### Task 2: Severity

**Files:**
- Create: `src/disastergen/data/__init__.py`, `src/disastergen/data/severity.py`
- Test: `tests/test_severity.py`

**Interfaces:**
- Consumes: `SeverityConfig`
- Produces:
  - `ndvi(bands: np.ndarray) -> np.ndarray`
  - `ndwi(bands: np.ndarray) -> np.ndarray`
    - `bands` has shape (4, H, W) in `BANDS` order.
  - `change_mask(before, after, hazard: str, cfg: SeverityConfig) -> np.ndarray[bool]`
  - `severity(before, after, hazard: str, cfg: SeverityConfig) -> float` in [0, 1]
    - Defined as the fraction of finite pixels that are flagged as changed.
  - `BANDS: tuple[str, ...]`, `HAZARD_CODES = {"flood": 0, "landslide": 1}`

- [ ] **Step 1: Write the failing tests**

```python
def test_no_change_is_zero(): assert severity(x, x, "landslide", cfg) == 0.0
def test_full_vegetation_loss_is_one():  # before: NIR high/red low; after: NIR≈red
    assert severity(veg, bare, "landslide", cfg) == pytest.approx(1.0)
def test_flood_uses_ndwi_rise(): assert severity(dry, wet, "flood", cfg) == pytest.approx(1.0)
def test_half_tile_change_is_half(): assert severity(veg, half_bare, "landslide", cfg) == pytest.approx(0.5)
def test_zero_reflectance_no_nan(): assert np.isfinite(ndvi(np.zeros((4, 8, 8))) ).all()
def test_unknown_hazard_raises():
    with pytest.raises(ValueError): severity(x, x, "tsunami", cfg)
```

- [ ] **Step 2: Run `pytest tests/test_severity.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement the functions**
  - `ndvi = (B8 - B4) / (B8 + B4 + eps)`
  - `ndwi = (B3 - B8) / (B3 + B8 + eps)`
  - Landslide change: `ndvi_before - ndvi_after > ndvi_drop`
  - Flood change: `ndwi_after - ndwi_before > ndwi_rise`
  - NaN pixels are excluded from both the numerator and the denominator.
- [ ] **Step 4: Run the tests. Expected: PASS. Run `ruff check .`.**
- [ ] **Step 5: Commit with message `feat: change-based severity labels`.**

### Task 3: Tiling and tile files

**Files:**
- Create: `src/disastergen/data/tiling.py`
- Test: `tests/test_tiling.py`

**Interfaces:**
- Consumes: `severity`, `HAZARD_CODES`, `DataConfig`, `SeverityConfig`
- Produces:
  - `split_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
    - Input shape (11, H, W), in the order: before 4 bands, after 4 bands, slope, rain, soil.
    - Returns `(before (4,H,W), after (4,H,W), cond (3,H,W))`.
  - `iter_tiles(stack, size: int, stride: int, max_nodata_frac: float) -> Iterator[tuple[int, int, np.ndarray]]`
  - `write_event_tiles(stack, event: Event, out_dir: Path, cfg: Config) -> int`
    - Writes one `.npz` per tile with keys `before, after, cond, hazard, severity, event`.
    - Returns the tile count.

- [ ] **Step 1: Write the failing tests**

```python
def test_iter_tiles_count(): assert len(list(iter_tiles(np.zeros((11, 512, 768)), 256, 256, 0.2))) == 6
def test_tiles_with_nodata_are_dropped_or_filled():
    s = np.ones((11, 256, 512)); s[:, :, :256] = np.nan; s[:, 0, 300] = np.nan
    tiles = list(iter_tiles(s, 256, 256, 0.2))
    assert len(tiles) == 1 and np.isfinite(tiles[0][2]).all()
def test_write_event_tiles_roundtrip(tmp_path):
    n = write_event_tiles(stack, event, tmp_path, cfg)
    z = np.load(next(tmp_path.glob("*.npz")))
    assert n == 1 and z["before"].shape == (4, 256, 256) and z["cond"].shape == (3, 256, 256)
    assert 0.0 <= float(z["severity"]) <= 1.0 and int(z["hazard"]) == 1
```

- [ ] **Step 2: Run the tests. Expected: FAIL.**
- [ ] **Step 3: Implement**
  - Remaining NaNs in a kept tile are filled with the per-channel median of that tile.
  - File name: `f"{event.name}_r{row}_c{col}.npz"`, saved with `np.savez_compressed`.
  - The stored arrays are float32.
- [ ] **Step 4: Run the tests. Expected: PASS. Run ruff.**
- [ ] **Step 5: Commit with message `feat: tiling with nodata handling`.**

### Task 4: GEE export and the data notebook

**Files:**
- Create: `src/disastergen/data/gee_export.py`, `notebooks/01_data.ipynb`
- Test: `tests/test_gee_windows.py`, covering the pure date logic only (no `ee` import at module top level)

**Interfaces:**
- Consumes: `Event`, `DataConfig`, `write_event_tiles`
- Produces:
  - `composite_windows(event_date: date, cfg: DataConfig) -> tuple[tuple[date, date], tuple[date, date]]`
    - pre window: `[event_date - pre_window_days, event_date - 1 day]`
    - post window: `[event_date + post_offset_days, + post_window_days]`
  - `s2_composite(aoi: ee.Geometry, start: date, end: date, max_cloud_pct: int) -> ee.Image`
    - Uses `COPERNICUS/S2_SR_HARMONIZED`, masks with the SCL classes 3, 8, 9, 10 and takes the
      median over `BANDS`.
  - `conditioning(aoi: ee.Geometry, event: Event, cfg: DataConfig) -> ee.Image`
    - Three bands: slope from `COPERNICUS/DEM/GLO30` via `ee.Terrain.slope`, rain as the sum of
      `UCSB-CHG/CHIRPS/DAILY` over `rain_days_before`, and soil from
      `OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02` band `b0`.
  - `event_stack(event: Event, cfg: DataConfig) -> ee.Image`
    - 11 bands in `split_stack` order, at `scale_m`.
  - `cloud_free_fraction(event: Event, cfg: DataConfig) -> dict[str, float]`
    - Keys `pre` and `post`: the share of the AOI that has a valid composite pixel.
  - `export_event(event: Event, cfg: DataConfig) -> ee.batch.Task`
    - A GeoTIFF export to Drive folder `drive_folder`.

- [ ] **Step 1: Write the failing test**

```python
def test_composite_windows():
    (ps, pe), (qs, qe) = composite_windows(date(2020, 8, 6), cfg.data)
    assert pe < date(2020, 8, 6) < qs and (pe - ps).days == cfg.data.pre_window_days - 1
```

- [ ] **Step 2: Run the test. Expected: FAIL. Implement `composite_windows`. Run again. Expected: PASS.**
- [ ] **Step 3: Implement the `ee` functions**
  - `import ee` happens inside the functions or under `TYPE_CHECKING`, so CI never needs
    earthengine-api.
- [ ] **Step 4: Write `notebooks/01_data.ipynb` with these cells:**
  1. Mount Drive.
  2. `pip install -e .[gee]`.
  3. `ee.Authenticate(); ee.Initialize(project=<id>)`.
  4. Use geemap to show the before and after composites for each event. Check each bbox by eye
     and fix `events.yaml` if it is off.
  5. Print the `cloud_free_fraction` table and drop events where pre or post is below 0.6.
  6. Export each event and wait for the tasks.
  7. Run `write_event_tiles` for each GeoTIFF read with rasterio.
  8. Print tile counts and the severity histogram per event.
  9. Write the observed maximum severity to the config as `train_severity_max`.
- [ ] **Step 5: Run the notebook on Colab (the owner, with the GEE project ID) and commit it with its outputs.**
  - The go/no-go result for floods is recorded in a markdown cell.
- [ ] **Step 6: Commit with message `feat: Earth Engine export pipeline and data notebook`.**

### Task 5: Dataset and split

**Files:**
- Create: `src/disastergen/data/dataset.py`
- Test: `tests/test_dataset.py`, using small synthetic `.npz` files written to `tmp_path` with size 64

**Interfaces:**
- Consumes: the tile `.npz` format from Task 3, `TrainConfig`
- Produces:
  - `split_by_event(paths: list[Path], test_events: set[str], val_frac: float) -> dict[str, list[Path]]`
    - Keys: `train`, `val`, `test`.
    - Test = every tile of the test events.
    - Val = a spatial block of each remaining event: the tiles with the largest column index,
      making up `val_frac` of that event's tiles.
    - Raises if a test event has no tiles.
  - `ChannelStats(mean: np.ndarray, std: np.ndarray)`
    - Covers the 7 image and conditioning channels. Severity and hazard are not normalized.
    - Saved and loaded with `.save(path)` and `.load(path)` (JSON).
  - `compute_stats(paths: list[Path]) -> ChannelStats`
  - `build_input(before, cond, severity: float, hazard: int, stats: ChannelStats) -> torch.Tensor`
    - Returns a (9, H, W) tensor. Raises if severity is outside [0, 1].
  - `TileDataset(paths, stats, augment: bool)`
    - Each item is `(input (9,H,W), target (4,H,W))`.
    - The target is scaled to [-1, 1] with `stats`, then clipped to ±3σ and divided by 3.
  - `to_reflectance(target: torch.Tensor, stats) -> torch.Tensor`, the inverse of that scaling.
  - `weighted_sampler(ds: TileDataset, threshold: float) -> WeightedRandomSampler`
    - Tiles above the threshold get weight 1/(share of severe tiles).

- [ ] **Step 1: Write the failing tests**

```python
def test_item_shapes(): x, y = ds[0]; assert x.shape == (9, 64, 64) and y.shape == (4, 64, 64)
def test_severity_and_hazard_channels_constant(): x, _ = ds[0]; assert x[7].unique().numel() == 1 and x[8].unique().numel() == 1
def test_augment_applies_same_transform():  # tile where after == before
    x, y = ds_aug[0]; assert torch.allclose(unscale_in(x[:4]), unscale_out(y), atol=1e-5)
def test_target_in_range(): _, y = ds[0]; assert y.min() >= -1 and y.max() <= 1
def test_stats_from_train_only(): assert compute_stats(split["train"]).mean.shape == (7,)
def test_split_is_disjoint_and_complete():
    s = split_by_event(paths, {"a"}, 0.2)
    assert {p.name.split("_r")[0] for p in s["test"]} == {"a"}
    assert not set(s["train"]) & set(s["val"]) and sum(map(len, s.values())) == len(paths)
def test_split_rejects_unknown_test_event():
    with pytest.raises(ValueError): split_by_event(paths, {"typo"}, 0.2)
def test_build_input_rejects_bad_severity():
    with pytest.raises(ValueError): build_input(b, c, 1.5, 1, stats)
def test_sampler_oversamples_severe(): assert sampler_weights_for_severe > sampler_weights_for_calm
```

  `unscale_in` and `unscale_out` are tiny test helpers that use `stats`. They let the test check
  that the input before-bands and the target were flipped and rotated identically.
- [ ] **Step 2: Run the tests. Expected: FAIL.**
- [ ] **Step 3: Implement**
  - Augmentation: a random `k` for `torch.rot90` and a random flip, drawn once per item and
    applied to both the input and the target.
- [ ] **Step 4: Run the tests. Expected: PASS. Run ruff.**
- [ ] **Step 5: Commit with message `feat: tile dataset, event split, weighted sampling`.**

### Task 6: Models

**Files:**
- Create: `src/disastergen/models/__init__.py`, `models/generator.py`, `models/discriminator.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `ModelConfig`
- Produces:
  - `UNetGenerator(in_ch: int, out_ch: int, ngf: int, num_downs: int)`
    - pix2pix U-Net: skip connections, dropout 0.5 in the 3 innermost decoder blocks, tanh output.
    - The input side must be divisible by `2**num_downs`.
  - `PatchDiscriminator(in_ch: int, ndf: int, n_layers: int = 3)`
    - 70×70 PatchGAN. Its input is the condition and the image concatenated.
  - `init_weights(module, std=0.02) -> None`
  - `build_models(cfg: ModelConfig) -> tuple[UNetGenerator, PatchDiscriminator]`
    - The discriminator `in_ch` is `in_channels + out_channels`, which is 13.

- [ ] **Step 1: Write the failing tests**

```python
def test_generator_shape_and_range():
    g = UNetGenerator(9, 4, ngf=8, num_downs=8); y = g(torch.randn(1, 9, 256, 256))
    assert y.shape == (1, 4, 256, 256) and y.abs().max() <= 1
def test_small_generator_for_tests(): assert UNetGenerator(9, 4, 8, 6)(torch.randn(2, 9, 64, 64)).shape == (2, 4, 64, 64)
def test_discriminator_patch_shape(): assert PatchDiscriminator(13, 8)(torch.randn(1, 13, 256, 256)).shape == (1, 1, 30, 30)
def test_bad_input_size_raises():
    with pytest.raises(ValueError): UNetGenerator(9, 4, 8, 8)(torch.randn(1, 9, 100, 100))
```

- [ ] **Step 2: Run the tests. Expected: FAIL.**
- [ ] **Step 3: Implement, following Isola et al. 2017**
  - Encoder blocks: Conv(4, stride 2), then BatchNorm, then LeakyReLU(0.2).
  - Decoder blocks: ConvTranspose, then BatchNorm, then ReLU.
  - The outermost and innermost blocks have no norm.
  - Build the U-Net as a list of encoder and decoder blocks in `forward`, not with recursion, so
    a reader can follow it.
- [ ] **Step 4: Run the tests. Expected: PASS. Run ruff.**
- [ ] **Step 5: Commit with message `feat: pix2pix U-Net generator and PatchGAN discriminator`.**

### Task 7: Training loop and the training notebook

**Files:**
- Create: `src/disastergen/train.py`, `notebooks/02_train.ipynb`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `build_models`, `TileDataset`, `weighted_sampler`, `split_by_event`, `compute_stats`, `Config`
- Produces:
  - `set_seed(seed: int) -> None`
  - `train_step(g, d, opt_g, opt_d, x, y, l1_weight: float) -> dict[str, float]`
    - Keys `d_loss`, `g_gan`, `g_l1`. Uses `BCEWithLogitsLoss` and the pix2pix label convention.
  - `save_checkpoint(path, epoch, g, d, opt_g, opt_d) -> None`
  - `load_checkpoint(path, g, d, opt_g, opt_d) -> int` (returns the next epoch)
  - `fit(cfg: Config, tile_dir: Path, resume: bool) -> Path`
    - Returns the path of the best generator by validation L1.
    - Logs losses to `checkpoints/history.csv` and saves `checkpoints/stats.json`.
  - CLI: `python -m disastergen.train --config configs/default.yaml --tiles <dir> [--resume]`

- [ ] **Step 1: Write the failing tests (tiny models: ngf 8, num_downs 6, 64×64 inputs)**

```python
def test_train_step_finite_losses():
    out = train_step(g, d, og, od, torch.randn(2, 9, 64, 64), torch.rand(2, 4, 64, 64) * 2 - 1, 100.0)
    assert set(out) == {"d_loss", "g_gan", "g_l1"} and all(math.isfinite(v) for v in out.values())
def test_train_step_reduces_l1_on_fixed_batch():  # 30 steps on one batch
    assert last["g_l1"] < first["g_l1"]
def test_checkpoint_roundtrip_resumes_epoch(tmp_path):
    save_checkpoint(tmp_path / "c.pt", 4, g, d, og, od)
    assert load_checkpoint(tmp_path / "c.pt", g2, d2, og2, od2) == 5
    assert all(torch.equal(a, b) for a, b in zip(g.state_dict().values(), g2.state_dict().values()))
def test_set_seed_reproducible(): set_seed(1); a = torch.rand(3); set_seed(1); assert torch.equal(a, torch.rand(3))
```

- [ ] **Step 2: Run the tests. Expected: FAIL.**
- [ ] **Step 3: Implement**
  - `fit` saves `last.pt` every epoch and `best_g.pt` when validation L1 improves.
  - Uses the GPU if one is available, with `torch.autocast` mixed precision on CUDA.
- [ ] **Step 4: Run the tests. Expected: PASS. Run ruff.**
- [ ] **Step 5: Write `notebooks/02_train.ipynb` with these cells:**
  1. Mount Drive; clone the repo; `pip install -e .`
  2. Overfit sanity check: 20 tiles for 200 steps. The L1 loss must drop clearly.
  3. Full run with `fit`, `checkpoint_dir` on Drive and `resume=True`.
  4. Plot the loss curves from `history.csv` into `outputs/figures/losses.png`.
- [ ] **Step 6: Train on Colab (the owner), commit the notebook with its outputs plus `losses.png`.**
- [ ] **Step 7: Commit with message `feat: training loop with resumable checkpoints`.**

### Task 8: Evaluation, figures, diffusion comparison

**Files:**
- Create: `src/disastergen/evaluate.py`, `src/disastergen/diffusion_baseline.py`, `notebooks/03_evaluate.ipynb`
- Modify: `configs/default.yaml` and `src/disastergen/config.py`
  - Add `EvalConfig(prompts: list[str], sd_model: str, sd_strength: float, sd_steps: int)` as
    `Config.eval`.
  - YAML values:
    - `prompts`: 3–4 templates using `{hazard}`
    - `sd_model: "stable-diffusion-v1-5/stable-diffusion-v1-5"`
    - `sd_strength: 0.5`
    - `sd_steps: 30`
  - Extend `test_default_config_loads` with `assert cfg.eval.prompts`.
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `TileDataset`, `to_reflectance`, `severity`, `UNetGenerator`, `Config`
- Produces:
  - `rgb_uint8(bands4: torch.Tensor) -> torch.Tensor`
    - Takes B4, B3, B2, applies a 2–98 percentile stretch and returns uint8 RGB for FID and
      figures.
  - `image_metrics(pred, target) -> dict[str, float]`, with keys `ssim` and `psnr`
  - `fid(pred_rgb, target_rgb) -> float`, via torchmetrics `FrechetInceptionDistance(feature=2048)`
  - `severity_consistency(requested: np.ndarray, measured: np.ndarray) -> dict[str, float]`, with
    keys `pearson_r` and `mae`
  - `evaluate_model(g, ds, cfg, sweep) -> dict`
  - `evaluate_identity(ds, cfg) -> dict`, the identity baseline (prediction = before)
  - `save_metrics(results: dict, path=Path("outputs/metrics.json")) -> None`
  - `plot_triplets(...)` and `plot_severity_sweep(...)`, which save to `outputs/figures/`
  - `render_prompts(templates: list[str], hazard: str) -> list[str]`
  - `sd_img2img(rgb: list[PIL.Image], prompt: str, cfg) -> list[PIL.Image]`
    - Lazy `diffusers` import, fp16, fixed generator seed.

- [ ] **Step 1: Write the failing tests**

```python
def test_identical_images_perfect(): m = image_metrics(t, t); assert m["ssim"] == pytest.approx(1.0) and m["psnr"] > 40
def test_severity_consistency_perfect(): assert severity_consistency(np.array([.1, .5, .9]), np.array([.1, .5, .9]))["pearson_r"] == pytest.approx(1.0)
def test_rgb_uint8_shape_dtype(): r = rgb_uint8(torch.rand(2, 4, 64, 64)); assert r.dtype == torch.uint8 and r.shape == (2, 3, 64, 64)
def test_render_prompts(): assert all("landslide" in p for p in render_prompts(["after a {hazard}"], "landslide"))
def test_save_metrics_roundtrip(tmp_path): save_metrics({"a": 1.0}, tmp_path / "m.json"); assert json.loads((tmp_path / "m.json").read_text()) == {"a": 1.0}
```

  FID is not unit-tested, because it downloads Inception weights. It runs only in the notebook.
- [ ] **Step 2: Run the tests. Expected: FAIL. Implement. Run again. Expected: PASS. Run ruff.**
- [ ] **Step 3: Write `notebooks/03_evaluate.ipynb`**
  - Score the held-out event three ways: pix2pix, the identity baseline, and SD img2img with
    each prompt.
  - Severity sweep at `demo.sweep`, using `severity()` on the outputs.
  - Write the results table to `outputs/metrics.json` and save the figures.
  - Optional ablation: retrain with the conditioning channels zeroed, if time allows.
- [ ] **Step 4: Run on Colab (the owner). Commit the notebook with its outputs, `metrics.json` and the figures.**
- [ ] **Step 5: Commit with message `feat: evaluation, baselines and diffusion comparison`.**

### Task 9: Human evaluation

**Files:**
- Create: `src/disastergen/human_eval.py`, `human_eval/` (survey images, `key.csv`, `answers.csv`)
- Test: `tests/test_human_eval.py`

**Interfaces:**
- Consumes: RGB images from Task 8
- Produces:
  - `build_survey(real: list[Path], fake: list[Path], n: int, out_dir: Path, seed: int) -> Path`
    - Copies `n` images, half real and half generated, shuffled and renamed `q01.png` and so on.
    - Writes `key.csv` with columns `item` and `is_generated`, and returns its path.
  - `score_survey(answers_csv: Path, key_csv: Path) -> dict[str, float]`
    - `answers.csv` has columns `rater`, `item` and `answer` (`real` or `generated`).
    - Returns `accuracy`, `fooled_rate` (the share of generated images labelled real) and `n_raters`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_survey_balanced(tmp_path): key = read(build_survey(real, fake, 20, tmp_path, 0)); assert sum(key.is_generated) == 10
def test_score_all_fooled(tmp_path): assert score_survey(all_real_answers, key)["fooled_rate"] == 1.0
def test_score_rejects_unknown_item(tmp_path):
    with pytest.raises(ValueError): score_survey(answers_with_q99, key)
```

- [ ] **Step 2: Run the tests. Expected: FAIL. Implement. Run again. Expected: PASS. Run ruff.**
- [ ] **Step 3: The owner shares the 20 images (for example in a Google Form), collects 5–10 raters into `answers.csv` and runs `score_survey`.**
  - The result is merged into `metrics.json` under `human_eval`.
- [ ] **Step 4: Commit with message `feat: real-vs-generated human evaluation`.**

### Task 10: Demo

**Files:**
- Create: `src/disastergen/demo.py`, `notebooks/04_demo.ipynb`
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes: `gee_export.event_stack`-style fetch (reused: `s2_composite` and `conditioning` on the
  user's AOI), `build_input`, `to_reflectance`, `rgb_uint8`, `UNetGenerator`, `ChannelStats`
- Produces:
  - `validate_area(bbox, cfg: DemoConfig) -> None`
    - Raises `ValueError` if the area is outside `kerala_bbox` or larger than `max_demo_area_km2`.
  - `generate_sweep(g, before, cond, hazard: int, severities: Sequence[float], stats) -> list[np.ndarray]`
    - Returns RGB uint8 frames. Tiles larger AOIs into 256×256 chunks and stitches them back.
  - `is_extrapolation(sev: float, cfg: DemoConfig) -> bool`
    - True when `sev` is above `train_severity_max`.

- [ ] **Step 1: Write the failing tests**

```python
def test_validate_area_rejects_outside_kerala_and_too_large():
    with pytest.raises(ValueError, match="Kerala"): validate_area((80.0, 13.0, 80.1, 13.1), cfg.demo)
    with pytest.raises(ValueError, match="km"): validate_area((76.0, 10.0, 76.5, 10.5), cfg.demo)
    validate_area((76.13, 11.46, 76.15, 11.48), cfg.demo)  # Wayanad, ok
def test_generate_sweep_frames(): fr = generate_sweep(tiny_g, b, c, 1, [0.1, 0.6], stats); assert len(fr) == 2 and fr[0].dtype == np.uint8
def test_generate_sweep_non_multiple_size(): assert generate_sweep(tiny_g, b_300x300, c_300x300, 1, [0.5], stats)[0].shape[:2] == (300, 300)
def test_is_extrapolation(): assert is_extrapolation(0.95, replace(cfg.demo, train_severity_max=0.8))
```

- [ ] **Step 2: Run the tests. Expected: FAIL. Implement. Run again. Expected: PASS. Run ruff.**
  - For a size that is not a multiple of 256, use reflect padding, then crop back.
- [ ] **Step 3: Write `notebooks/04_demo.ipynb`**
  - A geemap map with a draw control and a place-search box.
  - An `ipywidgets` severity slider and a hazard toggle.
  - Shows the before image and the generated image, plus a crossfade over `demo.sweep`
    (`ipywidgets.Play` driving the image).
  - Shows an extrapolation warning when needed.
  - Run it for Wayanad and Pettimudi and commit it with its outputs.
- [ ] **Step 4: Commit with message `feat: interactive Kerala demo`.**

### Task 11: README, publish, report

**Files:**
- Modify: `README.md`
- Create: the report as a Claude Doc, exported to PDF for Moodle

- [ ] **Step 1: Write `README.md`**
  - One-paragraph summary.
  - How it works, as a pipeline diagram in text.
  - Setup (laptop tests and the Colab notebooks).
  - How to reproduce, in notebook order.
  - Results table copied from `metrics.json`, with the triplet and sweep figures.
  - Limitations: 10 m resolution, few events, flood status, extrapolation.
  - Data credits and licences: Copernicus, CHIRPS, OpenLandMap.
- [ ] **Step 2: Final checks: `ruff check .` and `pytest` pass, and `git status` is clean.**
- [ ] **Step 3: The owner creates an empty public repo `disaster-gen` on GitHub, then runs:**
  - `git remote add origin https://github.com/muhammedshahidsds25-collab/disaster-gen.git`
  - `git push -u origin main`
  - Confirm the CI check passes on GitHub.
- [ ] **Step 4: Write the report (under 10 pages)**
  - Sections: title, introduction, methodology, data, results, discussion.
  - Reuse the introduction and literature survey from the topic PDF.
  - Every number comes from `metrics.json`.
  - Include the repo link.
  - The owner decides the author list.
- [ ] **Step 5: Export to PDF and upload to Moodle before 2026-11-15 00:00.**
