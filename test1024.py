import argparse
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from ZeroCrack import ZeroCrack


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


def resize_and_pad(image_np, target_size=1024, pad_value=0):
    """
    保持纵横比将图像resize到最大内接矩形，然后补边到目标尺寸
    :param image_np: 原始图像numpy数组 (H, W, C) 或 (H, W)
    :param target_size: 目标尺寸（正方形）
    :param pad_value: 补边填充值
    :return: 处理后的图像, 变换参数 (scale, pad_top, pad_bottom, pad_left, pad_right)
    """
    h, w = image_np.shape[:2]
    
    # 计算缩放比例，保持纵横比
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 缩放图像
    resized_image = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 计算需要补的边（上下左右均分补边，保证图像居中）
    pad_w = max(target_size - new_w, 0)
    pad_h = max(target_size - new_h, 0)
    
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # 执行补边（兼容单通道/三通道）
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_value
    )
    
    return padded_image, (scale, pad_top, pad_bottom, pad_left, pad_right)


def crop_and_resize_back(image_np, pad_params, original_size):
    """
    裁剪掉补边区域，然后缩放到原始尺寸
    :param image_np: 补边后的图像numpy数组
    :param pad_params: 变换参数 (scale, pad_top, pad_bottom, pad_left, pad_right)
    :param original_size: 原始图像尺寸 (original_w, original_h)
    :return: 恢复到原始尺寸的图像
    """
    scale, pad_top, pad_bottom, pad_left, pad_right = pad_params
    h, w = image_np.shape[:2]
    original_w, original_h = original_size

    # 1. 裁剪掉补边区域
    crop_top = pad_top
    crop_bottom = h - pad_bottom
    crop_left = pad_left
    crop_right = w - pad_right
    cropped_image = image_np[crop_top:crop_bottom, crop_left:crop_right]
    
    # 2. 缩放到原始尺寸
    # 注意：如果scale < 1，cropped_image的尺寸会小于原始尺寸，需要放大；否则需要缩小
    result_image = cv2.resize(cropped_image, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    return result_image


def main():
    parser = argparse.ArgumentParser(description="ZeroCrack Test Script (保持纵横比resize + 补边)")
    parser.add_argument("--checkpoint", type=str, required=False, help="模型权重路径")
    parser.add_argument("--test_image_path", type=str, required=False, help="测试图文件夹")
    parser.add_argument("--save_path", type=str, required=False, help="结果保存路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    parser.add_argument("--alpha", type=float, default=0.3, help="蓝色区域透明度 (0.1-0.9)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定推理尺寸
    input_size = 1024
    
    # 加载模型
    model = ZeroCrack(img_size=input_size).to(device)
    load_zerocrack_checkpoint(model, args.checkpoint, device)
    model.eval()

    print(f"已加载模型: {args.checkpoint}")
    print(f"模式: 保持纵横比resize + 补边推理 + Mask区域蓝色上色 (透明度: {args.alpha})")
    print(f"推理目标尺寸: {input_size}x{input_size}")
    
    # 获取所有测试图像路径
    image_paths = sorted([os.path.join(args.test_image_path, f) for f in os.listdir(args.test_image_path)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    # 定义归一化参数，明确指定为float32类型
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for i, img_path in enumerate(image_paths):
        name = os.path.basename(img_path)
        
        # -------------------------- 1. 直接加载原始图像 --------------------------
        # 使用PIL加载图像，保持原始尺寸
        raw_image = Image.open(img_path).convert('RGB')
        original_w, original_h = raw_image.size
        
        # 转换为numpy数组 (H, W, C)
        image_np = np.array(raw_image)
        
        # -------------------------- 2. 保持纵横比resize + 补边 --------------------------
        # 保持纵横比将图像resize到最大内接矩形，然后补边到目标尺寸
        padded_image, pad_params = resize_and_pad(image_np, target_size=input_size)
        
        # -------------------------- 3. 图像预处理 --------------------------
        # 归一化：(image - mean) / std
        padded_image_norm = padded_image.astype(np.float32) / 255.0
        padded_image_norm = (padded_image_norm - mean) / std
        
        # 转换为tensor并添加batch维度 (1, C, H, W)，确保是float32类型
        padded_tensor = torch.from_numpy(padded_image_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            # -------------------------- 4. 模型推理 --------------------------
            # 开始计时：从送入模型开始
            start_time = time.time()
            output = model(padded_tensor)
            
            # -------------------------- 5. 概率图处理 --------------------------
            # 获取概率图并squeeze
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # 结束计时：得到1024x1024的mask
            end_time = time.time()
            inference_time = end_time - start_time
            
            # -------------------------- 6. 恢复到原始尺寸 --------------------------
            # 裁剪掉补边区域，然后缩放到原始尺寸
            prob_map = crop_and_resize_back(prob_map, pad_params, (original_w, original_h))

            # -------------------------- 7. 生成二值化掩码 --------------------------
            binary_mask = (prob_map > args.threshold).astype(np.uint8) * 255

            # -------------------------- 8. 处理并保存图像 --------------------------
            save_prediction(binary_mask, img_path, args.save_path, name, overlay_alpha=args.alpha)

            # 输出进度和推理时长
            if (i + 1) % 5 == 0:
                print(f"处理进度: [{i + 1}/{len(image_paths)}] - {name} (原始尺寸: {original_w}x{original_h}, 推理时长: {inference_time:.3f}秒)")
            else:
                print(f"处理进度: [{i + 1}/{len(image_paths)}] - {name} (推理时长: {inference_time:.3f}秒)")

    print(f"\n所有结果已保存至: {args.save_path}")
    print("推理完成，所有掩码已恢复原始图像尺寸！")


if __name__ == "__main__":
    main()
