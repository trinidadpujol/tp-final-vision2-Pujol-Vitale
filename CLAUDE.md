# CLAUDE.md — Project context

> This file is the persistent context for Claude Code. Read it at the start of each session,
> together with `plan.md`. If anything here conflicts with a specific instruction from the user,
> ask before proceeding.

---

## What this project is

Final project for **Advanced Computer Vision** (Universidad de San Andrés). The goal of this
stage is to **replicate a published paper on individual bovine identification from muzzle images**
and, building on that, progressively construct a cross-dataset benchmark and domain adaptation
experiments.

**Paper to replicate:** Li, Erickson & Xiong (2022), *Individual Beef Cattle Identification
Using Muzzle Images and Deep Learning Techniques*, Animals 12(11):1453.
DOI 10.3390/ani12111453.

**Dataset:** Zenodo Muzzle DB (record 6324361). 4923 muzzle images of 268 cattle, organized by
individual. Already downloaded.

**Task (this stage):** closed-set classification, 268 cattle = 268 classes. Given a muzzle
image, predict the individual.

**What success means:** test accuracy ~96–98%+ (the paper reports 98.7% with VGG16_BN) **and**
reproducing the trend that weighted cross-entropy and data augmentation help the classes with few
images. Matching the exact decimal is NOT the goal and is not expected.

---

## Status and roadmap

- ✅ **Planning** — complete. Detailed spec in `plan.md`.
- ✅ **Stage 1 (phases 0–4):** data inspection, splits, dataset/transforms, models (VGG16_BN to
  replicate + ResNet-50 as our backbone), training, evaluation.
- ✅ **Stage 2 (phases 5–6):** embedding extractor, gallery/probe protocol (Rank-1/mAP),
  cross-dataset and cross-modality gap experiments, ImageNet baseline comparison.

**`plan.md` is the source of truth for "what to do."** Do not re-implement the paper's recipe
from memory: it is all there (resolution, split, optimizer, losses, augmentation, hyperparameters).
If Claude Code needs a value, go to `plan.md`.

---

## How to work in this repo

1. **Read `plan.md` before writing code.** Work phase by phase, in order. Do not skip Phase 0
   (data inspection): the dataset may not have the structure we assumed.
2. **Small, reviewable changes.** One commit per logical unit of work, with a clear message.
   No mega-commits.
3. **Ask before deviating from the paper's recipe** or making non-trivial architectural
   decisions. Deviations are documented (see below).
4. **Validate before scaling.** Test the pipeline with 1 seed and few epochs before launching
   the full sweep (3 variants × 5 seeds). Do not burn GPU quota while debugging.
5. Keep `README.md` updated with instructions for running each phase.

---

## Stack and structure

- **Language/frameworks:** Python 3.10+, PyTorch + torchvision, scikit-learn, Pillow, numpy,
  pandas, tqdm.
- **Execution:** Kaggle Notebooks with GPU (P100 / T4). See section 5 of `plan.md` for details
  on data mounting, paths, and session limits.
- **Folder structure:** defined in `plan.md` section 2 (`src/`, `scripts/`, `outputs/`,
  `config.py` as single source of truth for hyperparameters and paths).

---

## Key commands

> Update this list as the project grows. Keep it current.

```bash
# data inspection (always run first)
python scripts/00_inspect_data.py

# generate and save splits
python scripts/01_make_splits.py

# VGG16_BN replication (3 variants × N seeds)
python scripts/02_train_vgg.py

# own backbone ResNet-50
python scripts/03_train_resnet.py

# Stage 2: inspect CMPD300 + generate splits
python scripts/00_inspect_cmpd300.py

# Stage 2: train source encoder on CMPD300
python scripts/05_train_source.py

# Stage 2: re-ID harness (gap + ImageNet baseline)
python scripts/06_eval_reid.py --target-dir /path/to/target --compare-imagenet

# GPU check (on Kaggle)
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Code conventions

- **`config.py` is the only source of hyperparameters, paths, and seeds.** Do not hardcode
  loose values in scripts.
- Functions with type hints; short docstrings where they add value.
- Readable logging per run: print/save the full config at the start of each training run.
- No notebooks as source of truth for logic: logic lives in `src/`, notebooks only orchestrate.
- Determinism: fix seeds for `random`, `numpy`, `torch`, and `torch.cuda` in a single place
  (`utils.py`).

---

## Responsibility principles (IMPORTANT)

This is an academic replication project. The integrity of results comes first.

- **Never fabricate, manually adjust, or hardcode metrics to match the paper.** Report real
  numbers, even if they are worse. If we cannot reproduce, we document why — that is a valid
  result, not a failure to hide.
- **Document every deviation from the paper's recipe.** If for any reason we change resolution,
  normalization, optimizer, etc., it goes into `DEVIATIONS.md` with a justification. The
  replication is evaluated by fidelity, not just by the number.
- **Real reproducibility:** splits saved to disk and reused (no re-splitting per run), fixed
  seeds, pinned library versions, config logged with each run.
- **Clearly separate "what the paper says" from "what we decided"** in comments and docs.
- **Honest per-class results, not just the aggregate.** Report per-class accuracy (especially
  the 8 cattle with 4 images); do not hide the worst case behind the average.

---

## Repository hygiene

- **DO NOT commit the dataset** or heavy binaries (images, large checkpoints). Use `.gitignore`.
  The dataset lives on Kaggle (mounted at `/kaggle/input/...`), not in git.
- **DO NOT commit credentials, tokens, or API keys.** No Kaggle/GCP credentials in the repo.
- `outputs/` (checkpoints, results) out of git except for final metric tables/CSVs, which are
  worth versioning.
- `requirements.txt` with pinned versions.
- Atomic, descriptive commits.

---

## Domain facts not to get wrong

- **It is MUZZLE (not face).** The muzzle pattern is like an individual fingerprint. Do not
  confuse with bovine facial recognition (a different modality).
- **268 classes**, not 256.
- **The best model in the paper is VGG16_BN, not ResNet-50.** To replicate 98.7% use
  VGG16_BN. ResNet-50 is also trained as our own backbone for domain adaptation, but it is
  not the model that replicates the paper's number.
- **Stage 1 is closed-set classification: the split is PER IMAGE, all 268 classes in
  train/val/test.** The per-animal / disjoint-identity split (gallery/probe, Rank-1/mAP)
  only applies in Stage 2's re-identification protocol. Do not mix the two protocols.
- **Frozen backbone** for a faithful replication (the paper only fine-tunes the FC layers).
- **Raw [0,1] normalization**, not ImageNet mean/std (what the paper uses).

---

## What NOT to do

- Do not start training without having run Phase 0 and confirmed the real dataset structure.
- Do not introduce heavy frameworks (PyTorch Lightning, Hydra, etc.) without agreement:
  keep the stack simple and readable.
- Do not optimize prematurely (multi-GPU, mixed precision, etc.) until the baseline is running
  and validated.
- Do not "improve" the paper's recipe during replication. Replicate faithfully first;
  improvements come separately and afterwards.

---

## Reference

Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification Using
Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453.
DOI: 10.3390/ani12111453. Dataset: Zenodo record 6324361.
