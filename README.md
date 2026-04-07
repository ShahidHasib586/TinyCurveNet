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
## 🖼️ Qualitative Results

The figure below shows example enhancement results produced by TinyCurveNet.

<p align="center">
  <img src="https://github.com/ShahidHasib586/TinyCurveNet/blob/main/Outputs/tinynet_output.png?raw=true" alt="TinyCurveNet qualitative results" width="100%">
</p>

## 📊 Evaluation Metrics

TinyCurveNet was evaluated using standard full-reference image enhancement metrics.

| Metric | Value | Interpretation |
|--------|------:|----------------|
| **PSNR** | **22.06** | Measures reconstruction fidelity |
| **SSIM** | **0.9096** | Measures structural similarity |
| **MS-SSIM** | **0.9390** | Multi-scale structural similarity |
| **L1** | **0.0698** | Pixel-wise reconstruction error |
| **LPIPS** | **0.1404** | Perceptual similarity error |

Higher is better for **PSNR**, **SSIM**, and **MS-SSIM**.  
Lower is better for **L1** and **LPIPS**.

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
