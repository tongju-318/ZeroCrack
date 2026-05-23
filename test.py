import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from ZeroCrack import ZeroCrack
from dataset import TestDataset
from torch.utils.data import DataLoader
from rankseg import RankSEG


def load_zerocrack_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {
        (k.replace("sam3_vit.", "zerocrack_vit.", 1) if k.startswith("sam3_vit.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def save_prediction(binary_mask, raw_image_path, save_path, name, overlay_alpha=0.3):
    """
    保存黑白掩码图，并生成仅在 Mask 区域着色的蓝色覆膜图。
    背景区域保持原图质量，不进行任何处理。
    """
    # 1. 保存掩码图 (Binary Mask)
    mask_dir = os.path.join(save_path, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    cv2.imwrite(os.path.join(mask_dir, name), binary_mask)

    # 2. 保存覆膜图 (Overlay)
    overlay_dir = os.path.join(save_path, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # 读取原图
    original = cv2.imread(raw_image_path)
    if original is None:
        # 针对路径中可能存在的特殊编码进行兼容处理
        original = np.array(Image.open(raw_image_path).convert('RGB'))
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    # 创建输出图像的副本
    overlay = original.copy()

    # 获取掩码的布尔索引 (只有白色区域为 True)
    mask_bool = binary_mask > 0

    # 定义叠加颜色：蓝色 (BGR 格式: [蓝色, 绿色, 红色])
    color_bgr = np.array([255, 0, 0], dtype=np.uint8)

    # 关键逻辑：仅针对 mask_bool 为 True 的像素点进行线性融合
    # 公式：结果 = 原图像素 * (1 - alpha) + 蓝色 * alpha
    overlay[mask_bool] = (
        original[mask_bool].astype(float) * (1 - overlay_alpha) +
        color_bgr.astype(float) * overlay_alpha
    ).astype(np.uint8)

    cv2.imwrite(os.path.join(overlay_dir, name), overlay)


def main():
    parser = argparse.ArgumentParser(description="ZeroCrack Test Script")
    parser.add_argument("--checkpoint", type=str, required=False, help="模型权重路径")
    parser.add_argument("--test_image_path", type=str, required=False, help="测试图文件夹")
    parser.add_argument("--test_gt_path", type=str, default=None, help="GT文件夹(可选)")
    parser.add_argument("--save_path", type=str, required=False, help="结果保存路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    parser.add_argument("--alpha", type=float, default=0.3, help="蓝色区域透明度 (0.1-0.9)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定推理尺寸
    input_size = 1024
    test_set = TestDataset(args.test_image_path, input_size, args.test_gt_path)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # 加载模型
    model = ZeroCrack(img_size=input_size).to(device)
    load_zerocrack_checkpoint(model, args.checkpoint, device)
    model.eval()

    print(f"已加载模型: {args.checkpoint}")
    print(f"模式: 仅对 Mask 区域上色 (颜色: 蓝色, 透明度: {args.alpha})")

    for i, (image, name, w_tensor, h_tensor) in enumerate(test_loader):
        name = name[0]
        original_w = int(w_tensor[0].item())
        original_h = int(h_tensor[0].item())

        with torch.no_grad():
            image = image.to(device)
            output = model(image)

            # 1. 获取概率图并缩放到原图尺寸
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = cv2.resize(prob_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            # prob_map = torch.sigmoid(output)
            # rankseg = RankSEG(metric='Dice', solver='RMA')
            # prob_map = rankseg.predict(prob_map)
            # prob_map = prob_map.squeeze().cpu().numpy()
            # prob_map = cv2.resize(prob_map, (original_w, original_h),
            #                       interpolation=cv2.INTER_LINEAR)

            # 2. 生成二值化掩码
            binary_mask = (prob_map > args.threshold).astype(np.uint8) * 255

            # 3. 处理并保存图像
            img_full_path = os.path.join(args.test_image_path, name)
            save_prediction(binary_mask, img_full_path, args.save_path, name, overlay_alpha=args.alpha)

            if (i + 1) % 5 == 0:
                print(f"处理进度: [{i + 1}/{len(test_set)}] - {name}")

    print(f"\n所有结果已保存至: {args.save_path}")


if __name__ == "__main__":
    main()
