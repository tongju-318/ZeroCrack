# ZeroCrack

<p align="center">
  <strong>面向裂缝分割的 SAM3-UNet 轻量基线</strong>
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="#可视化结果">可视化结果</a>
  ·
  <a href="#训练">训练</a>
  ·
  <a href="#推理">推理</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA%20Ready-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/Task-Crack%20Segmentation-00A67E?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-See%20LICENSE.txt-555555?style=for-the-badge">
</p>

ZeroCrack 是一个基于 SAM3 的轻量级 U-Net 裂缝分割基线。项目保留 SAM3 ViT 图像编码器作为视觉主干，在主干网络中加入参数高效的 Adapter 模块，并通过紧凑解码器预测二值裂缝掩码。仓库包含训练、推理、评估和 GIF 可视化示例，方便复现实验流程。

## 项目亮点

| 能力 | 说明 |
| --- | --- |
| SAM3 主干 | 使用冻结的 SAM3 ViT 预训练图像编码器作为视觉特征提取器。 |
| Adapter 微调 | 通过参数高效的 Adapter 模块适配裂缝分割任务。 |
| 轻量解码器 | 在 `ZeroCrack.py` 中实现紧凑的 U-Net 风格解码器。 |
| 完整流程 | 提供训练、推理、指标评估和可视化脚本。 |
| 结果可读 | 推理阶段同时保存二值掩码和蓝色叠加可视化图。 |

## 可视化结果

以下 GIF 位于 `visimage/`，展示裂缝分割的组合可视化结果。

<table>
  <tr>
    <td align="center"><strong>示例 1</strong></td>
    <td align="center"><strong>示例 2</strong></td>
  </tr>
  <tr>
    <td><img src="visimage/1all.gif" alt="ZeroCrack 裂缝分割示例 1" width="100%"></td>
    <td><img src="visimage/2all.gif" alt="ZeroCrack 裂缝分割示例 2" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>示例 3</strong></td>
    <td align="center"><strong>示例 4</strong></td>
  </tr>
  <tr>
    <td><img src="visimage/3all.gif" alt="ZeroCrack 裂缝分割示例 3" width="100%"></td>
    <td><img src="visimage/4all.gif" alt="ZeroCrack 裂缝分割示例 4" width="100%"></td>
  </tr>
</table>

## 快速开始

```bash
conda create -n zerocrack python=3.10 -y
conda activate zerocrack
pip install -r requirements.txt
```

如需 GPU 训练或推理，请先根据本机 CUDA 版本安装匹配的 PyTorch 和 TorchVision，再安装 `requirements.txt` 中的其他依赖。

<details>
<summary><strong>预训练权重说明</strong></summary>

请按照 SAM3 官方发布说明下载 SAM3 预训练 checkpoint。训练脚本通过 `--zerocrack_path` 参数读取该权重文件。

</details>

## 仓库结构

```text
ZeroCrack/
|-- ZeroCrack.py              # ZeroCrack 模型定义
|-- dataset.py                # 训练和测试数据集
|-- train.py                  # 训练入口
|-- test.py                   # 1024x1024 resize 推理
|-- test1024.py               # 保持比例并 padding 的 1024 推理
|-- eval.py                   # 基于 py_sod_metrics 的掩码评估
|-- train.sh                  # 训练命令模板
|-- test.sh                   # 推理命令模板
|-- eval.sh                   # 评估命令模板
|-- requirements.txt          # Python 依赖
|-- dataset/                  # 推荐数据集目录
|-- visimage/                 # GIF 可视化结果
`-- ZeroCrack/                # SAM3 相关模块和工具
```

## 数据集格式

训练和测试流程默认读取成对的图像与掩码，推荐目录如下：

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

支持的图像扩展名包括 `.jpg`、`.jpeg` 和 `.png`。训练掩码会以单通道二值掩码读取。请保持图像和掩码文件名排序一致，便于 `dataset.py` 正确配对。

## 训练

可以编辑 `train.sh`，也可以直接运行：

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

训练默认使用 `336x336` 输入，并按以下格式保存模型快照：

```text
ZeroCrack-5.pth
ZeroCrack-10.pth
ZeroCrack-15.pth
ZeroCrack-20.pth
```

## 推理

<details open>
<summary><strong>使用 1024x1024 resize 推理</strong></summary>

`test.py` 会先将每张图像缩放到 `1024x1024`，完成推理后再把预测结果恢复到原图尺寸。

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
<summary><strong>保持原始宽高比推理</strong></summary>

`test1024.py` 会先按比例缩放并 padding 到 1024 尺寸，再进行推理。

```bash
CUDA_VISIBLE_DEVICES=0 python test1024.py \
  --checkpoint checkpoints/zerocrack/ZeroCrack-20.pth \
  --test_image_path dataset/test/image \
  --save_path results/zerocrack_1024 \
  --threshold 0.5 \
  --alpha 0.3
```

</details>

两个推理脚本都会保存：

```text
results/
|-- masks/      # 二值预测掩码
`-- overlays/   # 叠加在原图上的蓝色可视化结果
```

## 评估

生成预测掩码后运行：

```bash
python eval.py \
  --dataset_name CrackDataset \
  --pred_path results/zerocrack/masks \
  --gt_path dataset/test/mask
```

脚本会输出 mDice、mIoU、S-measure、weighted F-measure、F-measure、E-measure 和 MAE 等指标。

## 使用提示

- `train.py` 会构建 `ZeroCrack(args.zerocrack_path, 336)`，并在 Adapter 微调前冻结 SAM3 ViT 主干。
- `test.py` 和 `test1024.py` 会构建 `ZeroCrack(img_size=1024)`，用于高分辨率推理。
- 如果未安装 `rankseg`，且不使用 `test.py` 中已注释的 RankSEG 后处理分支，可以移除或注释 `from rankseg import RankSEG`。
- 大尺寸输入建议使用显存充足的 CUDA GPU。

## 致谢

本项目参考并使用了 [SAM3](https://github.com/facebookresearch/sam3) 和 [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) 的相关思想与组件，感谢原作者的开源贡献。

## 许可证

本仓库遵循 `LICENSE.txt` 中提供的许可证条款。
