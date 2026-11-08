# Upgrade Plan: GAN vs Diffusion and More Training Events

> Implements spec §7b (`docs/superpowers/specs/2026-11-08-kerala-hazard-vision-ai-design.md`).
> Executed inline with TDD. Global constraints from the main plan still apply.

**Goal:** Add a Palette-style conditional diffusion model trained on the same tiles as pix2pix,
score both with the same evaluation code, and add four Western Ghats landslide events as extra
training data.

## Tasks

1. **Events.** Add the 4 events to `configs/events.yaml`:
   - `taliye_2021` (2021-07-22)
   - `ambeghar_2021` (2021-07-22)
   - `irshalwadi_2023` (2023-07-19)
   - `shiroor_2024` (2024-07-16)

   Test: `test_test_event_is_in_kerala` (the held-out event lies inside `demo.kerala_bbox`).

2. **Shared training helpers.**
   - Generic `save_checkpoint(path, epoch, objects: dict[str, Any])` and
     `load_checkpoint(path, objects) -> int`.
   - `prepare_data(cfg, tile_dir, ckpt_dir, batch_size) -> (train_loader, val_loader, stats)`.
   - `History(path, fields)` for the CSV log.
   - pix2pix `fit` uses all of them.
   - Tests: the existing train tests, updated.

3. **Diffusion model.** `models/diffusion.py`:
   - `ConditionalDiffusion(nn.Module)` built from `DiffusionConfig` and `ModelConfig`:
     - `loss(x, y) -> Tensor` (MSE on the predicted noise)
     - `sample(x, steps, seed) -> Tensor` (DDIM, output in [-1, 1])
   - Tests: loss is a finite scalar, sample shape and range, sample is deterministic for a seed,
     loss drops on a fixed batch.

4. **Diffusion training.** `train_diffusion.py`:
   - `fit_diffusion(cfg, tile_dir, resume) -> Path` (best EMA weights).
   - fp16 autocast and GradScaler on CUDA, EMA via `torch.optim.swa_utils.AveragedModel`.
   - Validation loss uses fixed noise.
   - Tests: runs and resumes on tiny tiles.

5. **Evaluation via predictors.**
   - `Predict = Callable[[Tensor], Tensor]`, built by `gan_predict(g, device)` and
     `diffusion_predict(model, device, steps, seed)`.
   - `evaluate_model(predict, ds, cfg, with_fid)` also reports `sec_per_tile`.
   - Tests updated, plus a diffusion predictor test.

6. **Human evaluation with two models.**
   - `build_survey(real, fakes: Mapping[str, Sequence[Path]], n_per_source, out_dir, seed)`.
   - `score_survey` returns `fooled_rate` per model.
   - Tests updated.

7. **Demo.** `generate_sweep(predict, ...)`. The notebook gets a model toggle.

8. **Notebooks and docs.**
   - Notebooks: `02_train_pix2pix`, `03_train_diffusion`, `04_evaluate`, `05_demo`.
   - Update the README.
