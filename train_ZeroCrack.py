import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM3UNet import SAM3UNet

# 训练配置参数
parser = argparse.ArgumentParser("SAM3-UNet Training with Validation")
parser.add_argument("--sam3_path", default="/root/autodl-tmp/SAM3-UNet/weight/sam3.pt", type=str)

# --- 接续训练的权重路径 ---
parser.add_argument("--resume_path", default="/root/autodl-tmp/SAM3-UNet/4650ckpt/SAM3-UNet-5.pth", type=str, help="第5轮的权重路径")

# 训练集路径
parser.add_argument("--train_image_path", default="/root/autodl-tmp/input/4650all_data/4650image", type=str)
parser.add_argument("--train_mask_path", default="/root/autodl-tmp/input/4650all_data/4650mask", type=str)

# 验证集路径
parser.add_argument("--val_image_path", default="/root/autodl-tmp/input/deep_crack_500/884_image_deep_crack_500", type=str)
parser.add_argument("--val_mask_path", default="/root/autodl-tmp/input/deep_crack_500/884_mask_deep_crack_500", type=str)

parser.add_argument('--save_path', default="/root/autodl-tmp/SAM3-UNet/4650ckpt", type=str)
parser.add_argument("--epoch", type=int, default=30, help="总训练轮数")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--img_size", default=1024, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)

args = parser.parse_args()


def structure_loss(pred, mask):
    """
    结构化损失函数：结合加权的 BCE 和 Dice 损失
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    # 1. 加权 BCE 损失
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
    # 2. 加权 Dice 损失
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wdice = 1 - (2. * inter + smooth) / (union + smooth)
    
    return (wbce + wdice).mean()


def validate(model, dataloader, device):
    """
    增强版评估函数：计算 MAE, Dice Score 和 IoU
    """
    model.eval()
    mae_sum = 0
    dice_sum = 0
    iou_sum = 0
    eps = 1e-6
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            output = torch.sigmoid(model(x))
            pred_bin = (output > 0.5).float()
            
            mae_sum += torch.mean(torch.abs(output - target)).item()
            inter = (pred_bin * target).sum()
            union = (pred_bin + target).sum()
            dice = (2. * inter + eps) / (union + eps)
            iou = (inter + eps) / (union - inter + eps)
            dice_sum += dice.item()
            iou_sum += iou.item()
            
    num_batches = len(dataloader)
    return {
        'mae': mae_sum / num_batches,
        'dice': dice_sum / num_batches,
        'iou': iou_sum / num_batches
    }


def main(args):    
    # 初始化训练集和验证集
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, args.img_size, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, args.img_size, mode='test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # 设备与模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM3UNet(args.sam3_path, args.img_size)
    
    # --- 核心修改：加载第5轮的权重 ---
    if os.path.exists(args.resume_path):
        print(f"==> Resuming from checkpoint: {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path, map_location=device))
    else:
        print(f"==> Warning: No checkpoint found at {args.resume_path}, starting from scratch.")
        
    model.to(device)
    

    optim = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    
  
    for _ in range(5):
        scheduler.step()
    
    os.makedirs(args.save_path, exist_ok=True)
    best_dice = 0.0 
    

    start_epoch = 0 
    print(f"Starting Training from Epoch {start_epoch + 1} with BCE-Dice")
    
    for epoch in range(start_epoch, args.epoch):
        model.train()
        
        for i, batch in enumerate(train_loader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            optim.zero_grad()
            pred = model(x)
            loss = structure_loss(pred, target)
            
            loss.backward()
            optim.step()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch:[{epoch+1}/{args.epoch}] Batch:[{i+1}/{len(train_loader)}] Loss:{loss.item():.4f}")
        
        # 验证
        metrics = validate(model, val_loader, device)
        current_dice = metrics['dice']
        print(f"--- Epoch {epoch+1} | Dice: {current_dice:.4f} | IoU: {metrics['iou']:.4f} ---")
        
        # 保存 Best 模型
        if current_dice > best_dice:
            best_dice = current_dice
            best_path = os.path.join(args.save_path, 'SAM3-UNet-best.pth')
            torch.save(model.state_dict(), best_path)
                
        scheduler.step()
        
        # 定期保存权重
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            save_name = os.path.join(args.save_path, f'SAM3-UNet-{epoch+1}.pth')
            torch.save(model.state_dict(), save_name)


if __name__ == "__main__":
    main(args)