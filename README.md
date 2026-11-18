# COMP560 Final Project — MANet pet segmentation, evaluated on cats

To download the dataset used, go to: https://www.robots.ox.ac.uk/~vgg/data/pets/ (the dataset was too large to keep in github).

In this project we compare how MA-NET performs on masking pets compared to a CNN using the
Oxford-IIIT Pet dataset (all 37 breeds, cats + dogs) to predict a pixel mask of
the pet.

## Files

| File | What it does |
|---|---|
| `manet_lib.py` | Models (`MANet`, `MANet_small`, `MANet_small_small`, `BasicCNN`), dataset, loss, metrics, flip TTA, plotting helpers |
| `01_explore_data.ipynb` | Dataset counts, breeds, example trimaps, pet-coverage stats, augmentation preview |
| `02_train_manet.ipynb` | Trains full MANet on all pets for 80 epochs |
| `02b_train_basic_cnn.ipynb` | Trains the basic CNN baseline with identical settings |
| `03_predict_cats.ipynb` | Compares all runs on the test set (with/without flip TTA), CNN vs MANet per cat and per breed, best/worst/random cats for each model, threshold sweep, your own photos |
| `04_my_cats_compare.ipynb` | Loads the best basic-CNN and MANet checkpoints and shows them side by side on your own photos (original, CNN, MANet). Set `IMAGE_PATH` for one photo, or leave it `None` to run everything in `my_cats/` |
| `archive/02_train_manet_40ep_run.ipynb` | The first MANet run (40 epochs), kept for comparison |
| `archive/02_train_manet_80ep_lr1e-3_run.ipynb` | An earlier 80-epoch MANet run at peak LR 1e-3 that diverged at epoch 36, replaced by the LR 3e-4 run |
| `my_cats/` | Pictures of my own cat I uploaded to test the model (didn't push to git) |
| `outputs/` | Saved figures, training histories, test predictions |


## Methods

All runs use the same train/val/test images, augmentation, loss, optimizer and schedule
(the 80-epoch MANet run uses a lower peak learning rate, see below).

| Run | Params | Notes |
|---|---|---|
| `BasicCNN`, 80 epochs | 0.28M | Baseline: 1 conv + ReLU per stage, 3 max-pools, 3 transposed convs. No skips, no normalization, no attention |
| `MANet`, 40 epochs | 33.4M | First run |
| `MANet`, 80 epochs | 33.4M | Same model trained twice as long, peak LR 3e-4 |

Each is evaluated with and without **horizontal-flip test-time augmentation**:
predict on the image and its mirror, flip the second prediction back, average the probabilities.

### Results (official test split, 1,181 cats with non-empty masks)

| Run | best epoch | val Dice | cat Dice | cat Dice + flip TTA | cat IoU + TTA | dog Dice + TTA |
|---|---|---|---|---|---|---|
| Basic CNN | 75 | 0.833 | 0.837 | 0.840 | 0.739 | 0.847 |
| MANet, 40 ep | 38 | 0.892 | 0.896 | 0.897 | 0.826 | 0.907 |
| MANet, 80 ep | 71 | 0.935 | 0.935 | **0.937** | **0.890** | **0.946** |

- MANet beats the basic CNN on 96% of cats (mean +0.097 Dice), on every breed; the biggest gaps are on Egyptian Mau, Bombay, Bengal.
- The 80-epoch MANet run uses a peak learning rate of 3e-4. A first attempt at 1e-3 (the setting the other runs use)
  **diverged at epoch 36** (val Dice 0.89 → 0.70) and early-stopped at epoch 45 with a best val Dice of 0.899;
  it is kept in `archive/02_train_manet_80ep_lr1e-3_run.ipynb`.

## Task

- Input: RGB image resized to 256×256, ImageNet-normalized.
- Target: binary mask, pet (trimap 1) + pet border (trimap 3) vs background (trimap 2).
- Split: official `trainval` split → 85% train / 15% val (stratified by breed);
  official `test` split for evaluation (1,183 cats, 2,486 dogs).
- Loss: BCE + soft Dice. AdamW + OneCycle LR, bf16 autocast, early stopping on val Dice.


## Limitations / Future Directions

1. The pets data set is not very feature rich, which mitigates the effectiveness of MA-NET's attention mechanisms.
2. Implementing MA-NET on a dataset with more features could reveal greater differences between MA-NET and a CNN.
3. The dataset was also relatively small for a model as complex as MA-NET; a larger dataset could allow the model to learn through more epochs.
