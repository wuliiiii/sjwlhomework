"""
Experiment 7: Grad-CAM Visualization (PyTorch版本)
Experiment 7: Grad-CAM Visualization (PyTorch Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import os

class GradCAM(nn.Module):
    """Grad-CAM实现类"""
    
    def __init__(self, model, target_layer):
        super(GradCAM, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子函数
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """保存前向传播的激活值"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None,device=torch.device('cpu')):
        """生成Grad-CAMHeatmap"""
        # 前向传播
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        # 计算权重
        gradients = (self.gradients).to(device)
        activations = (self.activations).to(device)
        
        # 全局平均池化得到权重
        weights = torch.mean(gradients, dim=[2, 3]).to(device)
        
        # 加权求和
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # ReLU激活
        cam = F.relu(cam)
        
        # 归一化到0-1
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().detach().numpy()


def visualize_gradcam(model, data_loader, class_names,device,logger=None, num_samples=8):
    """VisualizationGrad-CAMResults"""
    if logger is not None:
        logger.info(f"Visualizing Grad-CAM Results...")
    else:
        print(f"\n🔍 生成Grad-CAMVisualization...")
        print(f"device {device}")
    
    # 选择目标层（ResNet18的最后一个卷积层）
    target_layer = model.layer4[-1].conv2
    
    # 创建Grad-CAM对象
    grad_cam = GradCAM(model, target_layer).to(device)
    
    # 收集样本
    samples = []
    for i, (image, label) in enumerate(data_loader):
        if len(samples) >= num_samples:
            break
        
        # Prediction
        with torch.no_grad():
            image = image.to(device)

            output = model(image)
            predicted = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1).max().item()
        
        # 生成Grad-CAM
        cam = grad_cam.generate_cam(image, predicted,device)
        
        # 反归一化图像用于显示
        img_display = image.squeeze().permute(1, 2, 0).cpu()
        # 正确代码：将mean/std移到img_display所在设备（自动适配）
        mean = torch.tensor([0.485, 0.456, 0.406]).to(img_display.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(img_display.device)
        img_display = img_display * std + mean
        img_display = torch.clamp(img_display, 0, 1)
        
        samples.append({
            'image': img_display.numpy(),
            'cam': cam,
            'true_label': label.item(),
            'pred_label': predicted,
            'confidence': confidence
        })
    
    # Visualization结果
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i, sample in enumerate(samples):
        row = i // 2
        col = (i % 2) * 2
        
        # Original image
        axes[row, col].imshow(sample['image'])
        axes[row, col].set_title(f"Original Image\nTrue: {class_names[sample['true_label']]}\n"
                                f"Pred: {class_names[sample['pred_label']]}\n"
                                f"Confidence: {sample['confidence']:.3f}")
        axes[row, col].axis('off')

        # Grad-CAM热力图
        cam_resized = cv2.resize(sample['cam'], (224, 224))
        axes[row, col+1].imshow(sample['image'])
        axes[row, col+1].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[row, col+1].set_title(f"Grad-CAM Overlay\nAttention Visualization")
        axes[row, col+1].axis('off')

    save_dir = "./output/pic"
    save_path = os.path.join(save_dir, "gradcam_visualization.png")

    plt.suptitle('Grad-CAM Visualization Results', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if logger is not None:
        logger.info(f"Grad-CAMVisualization已保存:{save_path}")
    else:
        print(f"Grad-CAMVisualization已保存:{save_path}")


def analyze_attention_patterns(model, data_loader, class_names,device,logger=None):
    """Analysis不同Class的注意力模式"""
    if logger is not None:
        logger.info(f"Analyzing Grad-CAM Results...")
    else:
        print(f"Analysis不同Class的注意力模式...")
    
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer).to(device)
    
    # 收集每个类别的样本
    class_samples = {0: [], 1: []}
    
    for image, label in data_loader:
        if len(class_samples[0]) >= 5 and len(class_samples[1]) >= 5:
            break
        image = image.to(device)
        label_idx = label.item()
        if len(class_samples[label_idx]) < 5:
            # Prediction
            with torch.no_grad():
                output = model(image)
                predicted = output.argmax(dim=1).item()
            
            # 只分析正确预测的样本
            if predicted == label_idx:
                cam = grad_cam.generate_cam(image, predicted)
                class_samples[label_idx].append(cam)
    
    # 计算每个类别的平均注意力图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for class_idx in range(2):
        if len(class_samples[class_idx]) > 0:
            # 平均注意力图
            avg_cam = np.mean(class_samples[class_idx], axis=0)
            
            # 显示平均注意力图
            axes[class_idx, 0].imshow(avg_cam, cmap='jet')
            axes[class_idx, 0].set_title(f'{class_names[class_idx]}\nAverage Attention Map')
            axes[class_idx, 0].axis('off')

            # 显示注意力强度分布
            axes[class_idx, 1].hist(avg_cam.flatten(), bins=50, alpha=0.7)
            axes[class_idx, 1].set_title(f'{class_names[class_idx]}\nAttention Intensity Distribution')
            axes[class_idx, 1].set_xlabel('Attention Intensity')
            axes[class_idx, 1].set_ylabel('Frequency')

            # 显示高注意力区域
            threshold = np.percentile(avg_cam, 80)  # 前20%的高注意力区域
            high_attention = avg_cam > threshold
            axes[class_idx, 2].imshow(high_attention, cmap='Reds')
            axes[class_idx, 2].set_title(f'{class_names[class_idx]}\nHigh Attention Regions\n(Top 20%)')
            axes[class_idx, 2].axis('off')

    save_dir = "./output/pic"
    save_path = os.path.join(save_dir, "attention_pattern_analysis.png")
    plt.suptitle('Attention Pattern Analysis by Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if logger is not None:
        logger.info(f"注意力Analysis已保存:{save_path}")
    else:
        print(f"注意力Analysis已保存:{save_path}")

def compare_correct_vs_wrong_predictions(model, data_loader, class_names,device,logger=None):
    """比较正确预测和ErrorPrediction的注意力模式"""
    if logger is not None:
        logger.info(f"Comparing Grad-CAM Results...")
    else:
        print(f"Comparing Grad-CAM Results...")
    
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer).to(device)
    
    correct_samples = []
    wrong_samples = []
    
    for image, label in data_loader:
        if len(correct_samples) >= 3 and len(wrong_samples) >= 3:
            break
        image = image.to(device)
        # Prediction
        with torch.no_grad():
            output = model(image)
            predicted = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1).max().item()
        
        # 生成Grad-CAM
        cam = grad_cam.generate_cam(image, predicted)
        
        # 反归一化图像
        img_display = image.squeeze().permute(1, 2, 0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(img_display.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(img_display.device)
        img_display = img_display * std + mean
        img_display = torch.clamp(img_display, 0, 1).numpy()
        
        sample = {
            'image': img_display,
            'cam': cam,
            'true_label': label.item(),
            'pred_label': predicted,
            'confidence': confidence
        }
        
        if predicted == label.item() and len(correct_samples) < 3:
            correct_samples.append(sample)
        elif predicted != label.item() and len(wrong_samples) < 3:
            wrong_samples.append(sample)
    
    # Visualization比较
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    
    # 正确预测
    for i, sample in enumerate(correct_samples):
        axes[0, i*2].imshow(sample['image'])
        axes[0, i*2].set_title(f"Correct Prediction {i+1}\nTrue: {class_names[sample['true_label']]}\n"
                              f"Pred: {class_names[sample['pred_label']]}")
        axes[0, i*2].axis('off')

        cam_resized = cv2.resize(sample['cam'], (224, 224))
        axes[0, i*2+1].imshow(sample['image'])
        axes[0, i*2+1].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[0, i*2+1].set_title(f"Grad-CAM\nConfidence: {sample['confidence']:.3f}")
        axes[0, i*2+1].axis('off')

    # Error预测
    for i, sample in enumerate(wrong_samples):
        axes[1, i*2].imshow(sample['image'])
        axes[1, i*2].set_title(f"Wrong Prediction {i+1}\nTrue: {class_names[sample['true_label']]}\n"
                              f"Pred: {class_names[sample['pred_label']]}")
        axes[1, i*2].axis('off')

        cam_resized = cv2.resize(sample['cam'], (224, 224))
        axes[1, i*2+1].imshow(sample['image'])
        axes[1, i*2+1].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[1, i*2+1].set_title(f"Grad-CAM\nConfidence: {sample['confidence']:.3f}")
        axes[1, i*2+1].axis('off')

    save_dir = "./output/pic"
    save_path = os.path.join(save_dir, "correct_vs_wrong.png")

    plt.suptitle('Correct vs Wrong Predictions - Attention Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')


    if logger is not None:
        logger.info(f'Comparing Grad-CAM Results:{save_path}')
    else:
        print(f'Comparing Grad-CAM Results:{save_path}')

