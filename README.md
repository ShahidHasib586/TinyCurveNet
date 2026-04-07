# 🌊 TinyCurveNet: Physics-Based Underwater Image Enhancement

TinyCurveNet is a lightweight, physics-inspired neural network for underwater image enhancement.  
Instead of predicting pixel-wise outputs, the model estimates global image transformation parameters:

- Per-channel gain
- Gamma correction
- Color correction matrix (CCM)
- Channel bias

This design makes it highly efficient, interpretable, and suitable for real-time robotic applications.

---

## 📂 Repository Structure

```text

phys_curve_prog5/
├── checkpoints/ # trained models (best.pt, global_best.pt)
├── logs/ # training logs (slurm outputs)
├── configs/ # training configs (if available)
├── datasets/ # (NOT included)
├── train.py
├── eval_metrics.py
├── test_custom_data.py
└── ...

```

---

## ⚙️ Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

```
## Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

This project uses CSV manifest files:
```text
input_path,gt_path
```

Example:
```text
/data/input/img1.jpg,/data/gt/img1.jpg
```
Make sure:

Paths are valid on your system
Images are aligned (paired datasets)

## 🏋️ Training

Run training:
```bash
python3 train.py \
  --train_csv manifests/train_pairs_train.csv \
  --val_csv manifests/train_pairs_val.csv \
  --out runs/phys_curve_prog5 \
  --epochs 100 \
  --batch 8 \
  --crop 256 \
  --lr 2e-4
```
## 📈 Evaluation

Evaluate model performance:

```bash
python3 metrics.py \
  --model checkpoints/global_best.pt \
  --csv manifests/test_pairs.csv
```

## ⚡ Fast Testing on Custom Images

You can quickly test the model on your own images:

```bash
python3 test_custom_data.py \
  --model checkpoints/global_best.pt \
  --input_dir path/to/your/images \
  --output_dir outputs/

```
