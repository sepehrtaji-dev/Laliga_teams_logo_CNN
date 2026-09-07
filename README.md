# LaLiga Team Logo CNN

Fine-tuned **ResNet18** classifier for LaLiga club crests, plus a small Tkinter app that predicts a team from an uploaded logo.

Latest change on `main`: pretrained ImageNet ResNet18 with a custom head and heavier augmentation, aimed at logos that are rotated, cropped, or photographed in the wild.

## What is in the repo

| File | Role |
| --- | --- |
| `model.py` | Training: dataset loaders, ResNet18, AdamW, early stopping |
| `use.py` | Inference GUI (`tkinter`) |
| `test_train_split.ipynb` | Train / val / test split helper |
| `best_laliga_model.pth` | Best validation checkpoint |
| `laliga_logo_cnn_final.pth` | Final weights after training |
| `training_history.csv` / `.png` | Loss and accuracy curves |

Expected data layout (not committed):

```text
split_data/
  train/<team>/*.png
  val/<team>/*.png
  test/<team>/*.png
```

## Classes (20 clubs)

Athletic Bilbao, Atletico Madrid, Barcelona, Celta Vigo, Deportivo Alaves, Espanyol, Getafe, Girona, Las Palmas, Leganes, Mallorca, Osasuna, Rayo Vallecano, Real Betis, Real Madrid, Real Sociedad, Real Valladolid, Sevilla, Valencia, Villarreal.

These names must match folder names under `split_data/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU is used automatically when CUDA is available.

## Train

```bash
python model.py
```

Highlights from the current trainer:

- Input size **224×224** (ResNet default)
- ImageNet-normalized tensors
- Strong train-time augmentation: crop, flip, rotation, affine, color jitter, perspective, blur, random erasing
- Frozen early ResNet layers; trainable last blocks + new head
- Head: `Dropout(0.4) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 20)`
- `CrossEntropyLoss` with label smoothing `0.1`
- AdamW `1e-4`, weight decay `1e-4`
- `ReduceLROnPlateau` on val loss
- Early stopping after 7 epochs without a val-accuracy improvement

Checkpoints store `model_state_dict`, `class_to_idx`, `idx_to_class`, and `num_classes`.

## Predict

```bash
python use.py
```

Browse an image; the app shows the predicted club and softmax confidence. It prefers `best_laliga_model.pth` and falls back to `laliga_logo_cnn_final.pth`. Both old raw `state_dict` checkpoints and the new dict format are supported.

## Training notes from the saved history

`training_history.csv` currently covers 14 epochs. Validation accuracy climbs quickly and then sits at 1.0 while train accuracy is still ~0.87. That usually means the **val set is too easy** (clean official logos) compared with the augmented train set. Fine for a first pass; not a guarantee on messy real-world photos.

Worth doing next:

1. Hold out more diverse val/test images (phone photos, kits, watermarks).
2. Unfreeze more of ResNet18 once the head is stable.
3. Add a confusion matrix and per-class recall.
4. Git LFS for the `~18 MB` `.pth` files.
5. Drop `verbose=True` on `ReduceLROnPlateau` (removed in recent PyTorch).

## License / data

Club crests are owned by the clubs and LaLiga. Use this project for personal learning only unless you have rights to the images.
