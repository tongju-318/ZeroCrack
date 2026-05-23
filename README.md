# ZeroCrack

<p align="center">
  <strong>SAM3-UNet baseline for crack segmentation</strong>
</p>

<p align="center">
  <a href="README.zh-CN.md">中文文档</a>
  ·
  <a href="#quick-start">Quick Start</a>
  ·
  <a href="#model-weights">Model Weights</a>
  ·
  <a href="#visual-results">Visual Results</a>
  ·
  <a href="#training">Training</a>
  ·
  <a href="#inference">Inference</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA%20Ready-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/Task-Crack%20Segmentation-00A67E?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-See%20LICENSE.txt-555555?style=for-the-badge">
</p>

ZeroCrack is a lightweight SAM3-based U-Net segmentation baseline for crack detection. It keeps the SAM3 ViT image encoder as the visual backbone, inserts parameter-efficient adapter blocks, and uses a compact decoder to predict binary crack masks. The repository includes training, inference, evaluation, and GIF visualization examples for open-source reproduction.

## Highlights

| Capability | Description |
| --- | --- |
| SAM3 backbone | Uses a frozen pretrained SAM3 ViT image encoder as the visual backbone. |
| Adapter tuning | Adds parameter-efficient adapter blocks for downstream crack segmentation. |
| Compact decoder | Implements a lightweight U-Net-style decoder in `ZeroCrack.py`. |
| Full workflow | Includes training, inference, evaluation, and visualization scripts. |
| Visual outputs | Saves binary masks and blue overlay visualizations for inspection. |

## Visual Results

The GIFs below are stored in `visimage/` and show composed crack segmentation results.

<table>
  <tr>
    <td align="center"><strong>Example 1</strong></td>
    <td align="center"><strong>Example 2</strong></td>
  </tr>
  <tr>
    <td><img src="visimage/1all.gif" alt="ZeroCrack result example 1" width="100%"></td>
    <td><img src="visimage/2all.gif" alt="ZeroCrack result example 2" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Example 3</strong></td>
    <td align="center"><strong>Example 4</strong></td>
  </tr>
  <tr>
    <td><img src="visimage/3all.gif" alt="ZeroCrack result example 3" width="100%"></td>
    <td><img src="visimage/4all.gif" alt="ZeroCrack result example 4" width="100%"></td>
  </tr>
</table>

## Quick Start

```bash
conda create -n zerocrack python=3.10 -y
conda activate zerocrack
pip install -r requirements.txt
```

Install PyTorch and TorchVision according to your CUDA version from the official PyTorch instructions before installing the remaining dependencies if needed.

<details>
<summary><strong>Checkpoint requirement</strong></summary>

Download the SAM3 pretrained checkpoint before training. The training script expects this checkpoint through `--zerocrack_path`.

</details>

## Model Weights

The SAM3 weight and the ZeroCrack pretrained weight are available from Baidu Netdisk:

<p>
  <a href="https://pan.baidu.com/s/1zXr7P68CTS-_lyQ_id3GJA?pwd=zero">
    <img alt="Download weights" src="https://img.shields.io/badge/Baidu%20Netdisk-Download%20Weights-1677FF?style=for-the-badge">
  </a>
</p>

```text
Link: https://pan.baidu.com/s/1zXr7P68CTS-_lyQ_id3GJA?pwd=zero
Code: zero
```

Suggested local layout:

```text
weights/
`-- sam3.pt

checkpoint/
`-- ZeroCrack.pth
```

## Repository Structure

```text
ZeroCrack/
|-- ZeroCrack.py              # ZeroCrack model definition
|-- dataset.py                # training and testing datasets
|-- train.py                  # training entry point
|-- test.py                   # 1024x1024 resize inference
|-- test1024.py               # aspect-ratio-preserving 1024 inference
|-- eval.py                   # mask evaluation with py_sod_metrics
|-- train.sh                  # training command template
|-- test.sh                   # inference command template
|-- eval.sh                   # evaluation command template
|-- requirements.txt          # Python dependencies
|-- dataset/                  # expected dataset layout
|-- visimage/                 # GIF visualizations
`-- ZeroCrack/                # SAM3-related modules and utilities
```

## Dataset Format

The training and testing pipeline expects paired images and masks.

```text
dataset/
|-- train/
|   |-- image/
|   `-- mask/
|-- val/
|   |-- image/
|   `-- mask/
`-- test/
    |-- image/
    `-- mask/
```

Supported image extensions are `.jpg`, `.jpeg`, and `.png`. Training masks are loaded as single-channel binary masks. Keep image and mask filenames sorted consistently so that `dataset.py` can pair them correctly.

## Training

Edit `train.sh` or run the command directly:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --zerocrack_path /path/to/sam3.pt \
  --train_image_path dataset/train/image \
  --train_mask_path dataset/train/mask \
  --save_path checkpoints/zerocrack \
  --epoch 20 \
  --lr 0.0002 \
  --batch_size 12
```

Training uses `336x336` inputs by default and saves snapshots as:

```text
ZeroCrack-5.pth
ZeroCrack-10.pth
ZeroCrack-15.pth
ZeroCrack-20.pth
```

## Inference

<details open>
<summary><strong>Use square 1024x1024 resize</strong></summary>

`test.py` resizes each image to `1024x1024` before inference and restores the prediction to the original image size.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --checkpoint checkpoints/zerocrack/ZeroCrack-20.pth \
  --test_image_path dataset/test/image \
  --test_gt_path dataset/test/mask \
  --save_path results/zerocrack \
  --threshold 0.5 \
  --alpha 0.3
```

</details>

<details>
<summary><strong>Preserve aspect ratio with padding</strong></summary>

`test1024.py` keeps the original aspect ratio by resizing with padding before inference.

```bash
CUDA_VISIBLE_DEVICES=0 python test1024.py \
  --checkpoint checkpoints/zerocrack/ZeroCrack-20.pth \
  --test_image_path dataset/test/image \
  --save_path results/zerocrack_1024 \
  --threshold 0.5 \
  --alpha 0.3
```

</details>

Both inference scripts save:

```text
results/
|-- masks/      # binary mask predictions
`-- overlays/   # blue overlay visualizations on original images
```

## Evaluation

After generating prediction masks, run:

```bash
python eval.py \
  --dataset_name CrackDataset \
  --pred_path results/zerocrack/masks \
  --gt_path dataset/test/mask
```

The script reports mDice, mIoU, S-measure, weighted F-measure, F-measure, E-measure, and MAE.

## Notes

- `train.py` builds `ZeroCrack(args.zerocrack_path, 336)` and freezes the SAM3 ViT backbone before adapter fine-tuning.
- `test.py` and `test1024.py` build `ZeroCrack(img_size=1024)` for high-resolution inference.
- If `rankseg` is not installed and you do not use the commented RankSEG post-processing branch in `test.py`, remove or comment out the `from rankseg import RankSEG` import.
- For large inputs, use a CUDA GPU with sufficient memory.

## Acknowledgement

This project builds on ideas and components from [SAM3](https://github.com/facebookresearch/sam3) and [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet). Thanks to the authors for their open-source contributions.

## License

This repository follows the license terms provided in `LICENSE.txt`.
