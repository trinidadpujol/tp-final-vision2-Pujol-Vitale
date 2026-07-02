# Implementation plan — Bovine identification by muzzle replication

Spec for building with Claude Code. Goal: **replicate the results of Li, Erickson & Xiong (2022)**,
"Individual Beef Cattle Identification Using Muzzle Images and Deep Learning Techniques"
(*Animals* 12(11):1453), on the Zenodo Muzzle DB dataset. This is the baseline requirement of the
project (replicate a published paper on bovine muzzle identification). Later phases
(cross-dataset + domain adaptation) build on top of this.

---

## 0. Context and goal

- **Task:** closed-set classification. 268 cattle = 268 classes. Given a muzzle image, predict
  which individual it belongs to.
- **Dataset:** 4923 muzzle images of 268 feedlot cattle (USA), already downloaded locally.
- **Result to match:** best paper accuracy = **98.7%** (VGG16_BN model with cross-entropy +
  data augmentation). The second best stable model was VGG19_BN with weighted cross-entropy.
- **Success criterion for the replication:** test accuracy in the range **~96–98%+**. Matching
  98.7% to the decimal is unlikely due to the randomness of the reshuffle; landing in that range
  and **reproducing the trend** (that weighted CE and data augmentation help classes with few
  images) counts as a valid replication.

> Important note: the paper reports that **VGG16_BN is the best model, not ResNet-50**. To
> replicate faithfully we must use VGG16_BN. We also train ResNet-50 because it is the backbone
> we will reuse in the domain adaptation phases.

---

## 1. The exact recipe from the paper (target, do not improvise)

| Component | Paper value |
|---|---|
| Input resolution | **300×300** px (NOT 224×224) |
| Normalization | channel intensities to **[0,1]** (divide by 255). Does NOT use ImageNet mean/std |
| Split | **65% train / 15% val / 20% test**, random, reshuffle, **per image** (all 268 classes present in each split) |
| Transfer learning | pretrained on ImageNet; **fine-tune ONLY the fully-connected layers** (convolutional backbone frozen) |
| Epochs | 50 |
| Optimizer | SGD, momentum 0.9 |
| Learning rate | initial 0.001, decayed by factor 0.1 every 7 epochs (StepLR step_size=7, gamma=0.1) |
| Base loss | Cross-Entropy |
| Replicates | 5 runs with the same seed → report **mean test accuracy** |
| Metric | top-1 global accuracy + per-class accuracy. Optional: processing speed (ms/image) |

**Class imbalance (important):** images per animal range from **4 to 70**. The 4 cattle with
only 4 images (IDs 2100, 4549, 5355, 5925) are the ones that score 0% without optimization.

**Imbalance optimization (what boosts from 98.4% to 98.7%):**
- **Weighted Cross-Entropy (WCE):** per-class weight `w_i = N_max / N_i`, with `N_max = 70`
  (images of the most-sampled animal), `N_i` = images of class i.
- **Data augmentation** (train only): horizontal flip; brightness factor between 0.2 and 0.5;
  random rotation between −15° and +15°; Gaussian blur with kernel between 1 and 5.

**Values not specified in the paper (decide and document):**
- Batch size: use 32 (adjust if GPU runs out of memory with 300×300).
- Seeds: fix 5 explicit seeds (e.g. 0,1,2,3,4) for the replicates.

---

## 2. Project structure to generate

```
cattle-reid/
├── plan.md                  # this file
├── README.md                # how to run each phase
├── requirements.txt
├── config.py                # paths, hyperparameters, seeds (single source of truth)
├── data/
│   └── (DO NOT commit the dataset; only point DATA_DIR from config)
├── src/
│   ├── dataset.py           # Dataset + DataLoader, stratified split
│   ├── transforms.py        # preprocessing + paper data augmentation
│   ├── models.py            # VGG16_BN and ResNet-50 builder (freeze backbone / full finetune)
│   ├── losses.py            # CE and Weighted CE
│   ├── train.py             # training + validation loop, 1 run
│   ├── evaluate.py          # global and per-class accuracy on test
│   └── utils.py             # seeds, logging, checkpoint/metric saving
├── scripts/
│   ├── 00_inspect_data.py   # verifies dataset structure and reports stats
│   ├── 01_make_splits.py    # generates and saves splits (json with paths+labels)
│   ├── 02_train_vgg.py      # replication: VGG16_BN, 5 replicates, 3 variants
│   └── 03_train_resnet.py   # own backbone: ResNet-50
├── outputs/
│   ├── splits/              # saved splits (reproducibility)
│   ├── checkpoints/         # saved weights
│   └── results/             # csv/json metrics + summary table
└── notebooks/
    └── colab_runner.ipynb   # wrapper to run on Colab/Kaggle with GPU
```

---

## 3. Tasks by phase

### Phase 0 — Data inspection (`scripts/00_inspect_data.py`)
**Before writing any training code.** The dataset is already downloaded but we must confirm
its real structure.
1. Receive `DATA_DIR` from `config.py`.
2. List subdirectories; confirm there are ~268 folders (one per animal).
3. Count images per folder. Report: number of classes, total images (expected ~4923), min/max/mean
   images per class, simple histogram.
4. Verify that min is 4 and max is 70 (sanity check against the paper).
5. Detect image extensions present (.jpg/.png), corrupt or unreadable images.
6. Print a report. **Do not advance to Phase 1 until the report is correct.** If the structure
   differs (e.g. a single folder level, or a csv of labels), adapt `dataset.py` accordingly.

### Phase 1 — Dataset, splits, and transforms (`dataset.py`, `transforms.py`, `01_make_splits.py`)
1. **Stratified per-image split** 65/15/20 with a fixed seed. Stratify by class so that all 268
   appear in train, val, and test. Save splits as JSON (list of `(path, label)`) in
   `outputs/splits/` for reproducibility — **do not re-split per run**.
2. `LabelEncoder` folder→integer 0..267; save the mapping.
3. **Base transforms (val/test):** resize 300×300 → ToTensor (this already scales to [0,1]).
   **Do not** apply ImageNet normalization (the paper uses raw [0,1]). Leave ImageNet
   normalization as a configurable flag for later experiments.
4. **Train transforms (data augmentation variant):** resize 300×300 + RandomHorizontalFlip +
   ColorJitter(brightness=(0.2,0.5)) + RandomRotation(15) + GaussianBlur(kernel ∈ {1,3,5})
   → ToTensor.
5. PyTorch `Dataset` reading from split JSONs. `DataLoader` with configurable `num_workers`.

### Phase 2 — Models and losses (`models.py`, `losses.py`)
1. `build_model(name, num_classes=268, freeze_backbone=True)`:
   - `vgg16_bn`: load `torchvision.models.vgg16_bn(weights=IMAGENET)`, replace the last
     classifier layer with `Linear(..., 268)`. If `freeze_backbone`, freeze `features` and train
     only `classifier`.
   - `resnet50`: load pretrained, replace `fc` with `Linear(2048, 268)`. Support
     `freeze_backbone=True` (FC only) and `False` (full fine-tune).
2. `losses.py`:
   - Standard CE.
   - **Weighted CE:** compute weights `w_i = 70 / N_i` from train split counts; pass to
     `nn.CrossEntropyLoss(weight=...)`.

### Phase 3 — Training and evaluation (`train.py`, `evaluate.py`, `02_train_vgg.py`)
1. `train.py`: standard loop — SGD(momentum=0.9, lr=0.001), StepLR(step=7, gamma=0.1), 50
   epochs, track val accuracy per epoch, save best checkpoint by val acc.
2. `evaluate.py`: load best checkpoint, compute **top-1 global accuracy** and **per-class
   accuracy** on test. Save to CSV. (Optional: ms/image.)
3. `02_train_vgg.py` runs the **full replication**:
   - Model VGG16_BN, `freeze_backbone=True`.
   - **3 variants** × **5 seeds**:
     - (a) CE alone, no augmentation
     - (b) CE + data augmentation
     - (c) Weighted CE (no augmentation)
   - Report mean ± std test accuracy per variant.
   - **Success validation:** the best variant must fall ~96–98%+, and (b)/(c) must improve the
     accuracy of classes with few images vs (a). Generate a summary table in `outputs/results/`.

### Phase 4 — Own backbone ResNet-50 (`03_train_resnet.py`)
1. Same recipe, ResNet-50 model.
2. Run **two modes**: `freeze_backbone=True` (as in the paper) and `freeze_backbone=False`
   (full fine-tune).
3. Save the best run weights in `outputs/checkpoints/` — **this is the model reused in
   domain adaptation**.
4. Report its accuracy. Not expected to exactly match VGG16_BN.

---

## 4. Gotchas (verify explicitly)

- **Split per image, NOT per animal.** In closed-set classification all 268 classes must be in
  train/val/test. Never leave an animal only in test (would break the task). The per-animal /
  disjoint split only applies in the future re-identification phase.
- **[0,1] normalization, not ImageNet.** The paper scales to raw [0,1]. `ToTensor` already does
  this. Adding `Normalize(mean,std)` from ImageNet produces different numbers than the paper.
  Leave it as a flag, default off.
- **Frozen backbone.** The paper trains only the FC layers. If the full network is fine-tuned,
  the result is not comparable with the paper. For replication: `freeze_backbone=True`.
- **The 4 classes with 4 images.** If global accuracy looks good but lower than the paper,
  check the per-class accuracy of IDs 2100/4549/5355/5925 — they are likely dragging down the
  average.
- **GPU memory with 300×300 + VGG16_BN.** VGG is heavy on VRAM. If OOM, reduce batch size
  before changing resolution (resolution is part of the recipe).
- **Reproducibility.** Fix seeds for `random`, `numpy`, `torch`, and `torch.cuda` in
  `utils.py`. Save splits to disk and reuse them.

---

## 5. Execution environment: Kaggle Notebooks

**Chosen platform: Kaggle.** Free GPU (P100 or T4×2), 30 h/week, no driver setup or
availability queues. The P100 is the same GPU used in the paper.
(Google Cloud is reserved for the future domain adaptation phase.)

### 5.1. Dependencies
- Python 3.10+, PyTorch + torchvision, scikit-learn (stratified split), Pillow, numpy, pandas,
  tqdm. The Kaggle image already includes most of these; only install what is missing.
- Maintain a `requirements.txt` with pinned versions for local reproducibility, but on Kaggle
  rely on the base image + targeted pip installs.

### 5.2. Upload the dataset to Kaggle
The dataset (~643 MB) is already downloaded locally. Upload it as a **private Kaggle Dataset**
(Create → New Dataset → upload the .zip or folder). Once created, attach it to the notebook
with "Add Input" and it will be mounted read-only at:
```
/kaggle/input/<dataset-slug>/
```
Confirm the real structure inside in Phase 0 (there may be an extra top-level folder depending
on how it was zipped).

### 5.3. Upload the code
Two options, in order of preference:
1. **GitHub repo** → in the notebook, with internet enabled, `git clone`. Cleanest for iteration
   and versioning.
2. Upload `src/` as a **Kaggle Dataset of type "utility script"** and attach it.

### 5.4. Notebook configuration
- **Accelerator:** Settings → Accelerator → **GPU P100** (or T4×2; for training one model at
  a time a single GPU is enough, no need to complicate with multi-GPU).
- **Internet: ON.** Needed for `git clone`, `pip install`, and — critically — for **torchvision
  to download ImageNet pretrained weights** the first time. Requires a phone-verified Kaggle
  account.

### 5.5. Paths (map in `config.py`)
```
DATA_DIR   = /kaggle/input/<dataset-slug>/...   # read-only
OUTPUT_DIR = /kaggle/working/outputs            # writable + persists when saving a version
```
Important: **`/kaggle/working/` is the only writable directory that persists** (up to 20 GB,
saved when doing "Save Version"). All checkpoints, splits, and results go there. Anything
outside `/kaggle/working/` is lost when the session ends.

### 5.6. Session limits (plan the sweep accordingly)
- Interactive session: cuts off at ~12 h; idle timeout ~20–40 min (session dies if no activity).
- Background execution ("Save & Run All" / commit): runs up to ~12 h without needing to stay
  connected. **Use this mode for the long sweep** (3 variants × 5 seeds).
- Quota: **30 GPU hours per week**.
- The budget is more than enough: with `freeze_backbone=True` only the FC head is trained, so
  each run is fast. Still, **validate the pipeline with 1 seed and few epochs first**, then
  launch the full sweep in background. Do not burn quota while debugging.

### 5.7. Startup checklist in the notebook
1. `nvidia-smi` → confirm the GPU.
2. `python -c "import torch; print(torch.cuda.is_available())"` → must print `True`.
3. `git clone` the repo (or attach the code dataset).
4. Set `DATA_DIR` to the `/kaggle/input/...` path and run `scripts/00_inspect_data.py`.
5. Verify that the report matches (268 classes, ~4923 images, min 4 / max 70) before training.

---

## 6. Deliverable for this stage

1. Reproducible pipeline that, pointing at `DATA_DIR` (mounted at `/kaggle/input/...`), runs
   Phases 0→4 end-to-end in a Kaggle GPU notebook.
2. Summary table in `outputs/results/` with mean ± std accuracy per variant (VGG16_BN:
   CE / CE+aug / WCE) and for ResNet-50.
3. ResNet-50 checkpoint saved for reuse.
4. README with execution instructions.

---

## 7. Future phases (out of scope now, design the code to accommodate them)

Do not implement yet, but design `models.py` and `dataset.py` to extend to:
- **Embedding extractor:** take the trained ResNet-50 backbone, strip the classification head,
  expose embeddings.
- **Gallery/probe protocol** with disjoint identities, Rank-1 and mAP metrics, for cross-dataset
  evaluation (Pakistan/Ahmed, etc.).
- **Domain adaptation:** DANN with Gradient Reversal Layer, and self-training with pseudo-labels
  via clustering.

---

## Reference

Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification Using
Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453.
DOI: 10.3390/ani12111453. Dataset: Zenodo record 6324361.
