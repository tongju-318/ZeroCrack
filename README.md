# ZeroCrack: SAM3-UNet for Crack Segmentation

ZeroCrack is a lightweight SAM3-based U-Net segmentation baseline for crack detection. It keeps the SAM3 ViT image encoder as the visual backbone, inserts parameter-efficient adapter blocks, and uses a compact decoder to predict binary crack masks. The repository includes training, inference, evaluation, and GIF visualization examples for open-source reproduction.

## Highlights

- SAM3 ViT backbone with frozen pretrained weights.
- Adapter-based fine-tuning for downstream crack segmentation.
- Lightweight U-Net-style decoder implemented in `ZeroCrack.py`.
- Training script for paired image/mask datasets.
- Inference scripts that save both binary masks and blue overlay visualizations.
- Evaluation script based on `py_sod_metrics`.

## Visual Results

The following GIFs are stored in `visimage/` and show the composed crack segmentation results.

| Example 1 | Example 2 |
| --- | --- |
| <img src="visimage/1all.gif" alt="ZeroCrack result example 1" width="100%"> | <img src="visimage/2all.gif" alt="ZeroCrack result example 2" width="100%"> |

| Example 3 | Example 4 |
| --- | --- |
| <img src="visimage/3all.gif" alt="ZeroCrack result example 3" width="100%"> | <img src="visimage/4all.gif" alt="ZeroCrack result example 4" width="100%"> |

## Repository Structure

```text
SAM3-UNet/
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
`-- zerocrack/                # SAM3-related modules and utilities
```

## Installation

Create a Python environment first. Python 3.10 or newer is recommended.

```bash
conda create -n zerocrack python=3.10 -y
conda activate zerocrack
```

Install PyTorch and TorchVision according to your CUDA version from the official PyTorch instructions, then install the project dependencies:

```bash
pip install -r requirements.txt
```

Download the SAM3 pretrained checkpoint following the official SAM3 release instructions. The training script expects this checkpoint through `--zerocrack_path`.

## Dataset Format

The simple training and testing pipeline expects paired images and masks. A recommended layout is:

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

Training uses 336x336 inputs by default and saves snapshots as:

```text
ZeroCrack-5.pth
ZeroCrack-10.pth
ZeroCrack-15.pth
ZeroCrack-20.pth
```

## Inference

`test.py` resizes each image to 1024x1024 before inference and restores the prediction to the original image size:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --checkpoint checkpoints/zerocrack/ZeroCrack-20.pth \
  --test_image_path dataset/test/image \
  --test_gt_path dataset/test/mask \
  --save_path results/zerocrack \
  --threshold 0.5 \
  --alpha 0.3
```

`test1024.py` keeps the original aspect ratio by resizing with padding before inference:

```bash
CUDA_VISIBLE_DEVICES=0 python test1024.py \
  --checkpoint checkpoints/zerocrack/ZeroCrack-20.pth \
  --test_image_path dataset/test/image \
  --save_path results/zerocrack_1024 \
  --threshold 0.5 \
  --alpha 0.3
```

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
