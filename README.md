# COMP560 Final Project — MANet pet segmentation, evaluated on cats

Trains the MANet segmentation model from my wildfire-spread research on the
Oxford-IIIT Pet dataset (all 37 breeds, cats + dogs) to predict a pixel mask of
the animal, then evaluates and visualizes it on cats.

## Files

| File | What it does |
|---|---|
| `manet_lib.py` | Models (`MANet`, `MANet_small`, `MANet_small_small`, `BasicCNN`), dataset, loss, metrics, flip TTA, plotting helpers |
| `01_explore_data.ipynb` | Dataset counts, breeds, example trimaps, pet-coverage stats, augmentation preview |
| `02_train_manet.ipynb` | Trains full MANet on all pets for 80 epochs |
| `02b_train_basic_cnn.ipynb` | Trains the basic CNN baseline with identical settings |
| `03_predict_cats.ipynb` | Compares all runs on the test set (with/without flip TTA), CNN vs MANet per cat and per breed, best/worst/random cats for each model, threshold sweep, your own photos |
| `archive/02_train_manet_40ep_run.ipynb` | The first MANet run (40 epochs), kept for comparison |
| `my_cats/` | Drop your own cat photos here, then re-run the last cell of notebook 03 |
| `outputs/` | Saved figures, training histories, test predictions, cat cutouts |
| `checkpoints/` | `basiccnn_e80`, `manet_e40`, `manet_e80` best checkpoints (not in git) |

## Experiments

All runs use the same train/val/test images, augmentation, loss, optimizer and schedule.

| Run | Params | Notes |
|---|---|---|
| `BasicCNN`, 80 epochs | 0.28M | Baseline: 1 conv + ReLU per stage, 3 max-pools, 3 transposed convs. No skips, no normalization, no attention |
| `MANet`, 40 epochs | 33.4M | First run |
| `MANet`, 80 epochs | 33.4M | Same model trained twice as long |

Each is evaluated with and without **horizontal-flip test-time augmentation**:
predict on the image and its mirror, flip the second prediction back, average the probabilities.

### Results (official test split, 1,181 cats with non-empty masks)

| Run | best epoch | val Dice | cat Dice | cat Dice + flip TTA | cat IoU + TTA | dog Dice + TTA |
|---|---|---|---|---|---|---|
| Basic CNN | 75 | 0.833 | 0.837 | 0.840 | 0.739 | 0.847 |
| MANet, 40 ep | 38 | 0.892 | 0.896 | 0.897 | 0.826 | 0.907 |
| MANet, 80 ep | 30 | 0.899 | 0.899 | **0.903** | **0.836** | **0.913** |

- MANet beats the basic CNN on 87% of cats (mean +0.063 Dice), on every breed; the biggest gaps are on Russian Blue, Egyptian Mau, Bombay.
- The 80-epoch MANet run **diverged at epoch 36** (val Dice 0.89 → 0.70, still at LR ≈ 9e-4) and
  early-stopped at epoch 45; the reported checkpoint is from epoch 30, before the blow-up.
  It never reached the low-LR end of the OneCycle schedule.

## Setup

- Data: `data/oxford-iiit-pet/` (`images/` + `annotations/`), copied from
  `dev/Research/Presentation/data`. Not in git.
- Environment: the `prs` conda env from the research project
  (`dev/Research/Presentation/prs`, PyTorch nightly + CUDA 12.8 for the RTX 5080).
  Select it as the notebook kernel.

## Task

- Input: RGB image resized to 256×256, ImageNet-normalized.
- Target: binary mask, pet (trimap 1) + pet border (trimap 3) vs background (trimap 2).
- Split: official `trainval` split → 85% train / 15% val (stratified by breed);
  official `test` split for evaluation (1,183 cats, 2,486 dogs).
- Loss: BCE + soft Dice. AdamW + OneCycle LR, bf16 autocast, early stopping on val Dice.

## Model notes

Same MANet blocks as `dev/manet/centered64_lib.py` (GroupNorm DoubleConvs,
PAB self-attention at the bottleneck, MFAB multi-scale attention on skips), with
two changes:

1. `n_classes` is passed through to the output layer in every variant.
2. MFAB's bilinear upsampling of the pooled 1×1…6×6 maps is written as two
   matmuls (`bilinear_upsample`). It is numerically identical to
   `F.interpolate(..., mode="bilinear", align_corners=False)` but ~6× faster to
   train at 256×256: the `F.interpolate` backward pass spent 85% of GPU time
   accumulating gradients into those tiny maps.
