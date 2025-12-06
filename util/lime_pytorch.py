"""
Experiment 8: LIME Interpretability Analysis (PyTorch版本)
Experiment 8: LIME Explainability Analysis (PyTorch Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import warnings
warnings.filterwarnings('ignore')
import os


def create_prediction_function(model, transform):
    """创建LIME需要的Prediction函数"""
    def predict_fn(images):
        """
        LIME预测函数
        输入: numpy数组形式的ImageBatch (batch_size, height, width, channels)
        输出: Prediction概率 (batch_size, num_classes)
        """
        batch_predictions = []
        
        for img in images:
            # 转换为PIL图像格式
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # 转换为tensor并应用变换
            img_tensor = transform(transforms.ToPILImage()(img)).unsqueeze(0)
            
            # Prediction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                batch_predictions.append(probabilities.numpy()[0])
        
        return np.array(batch_predictions)
    
    return predict_fn

def explain_with_lime(model, data_loader, class_names, transform, num_samples=4,logger=None):
    """使用LIME解释ModelPrediction"""
    if logger is not None:
        logger.info("使用LIME生成解释...")
    else:
        print("使用LIME生成解释...")


    model = model.cpu()
    
    # 创建预测函数
    predict_fn = create_prediction_function(model, transform)
    
    # 创建LIME解释器
    explainer = lime_image.LimeImageExplainer()
    
    # 收集样本进行解释
    samples = []
    explanations = []
    
    for i, (image, label) in enumerate(data_loader):
        if len(samples) >= num_samples:
            break
        
        # 转换为numpy数组
        img_array = image.squeeze().permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)  # 0-1 → 0-255，转为uint8

        # Prediction
        img_for_pred = transform(transforms.ToPILImage()(img_array)).unsqueeze(0)
        with torch.no_grad():
            output = model(img_for_pred)
            predicted = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1).max().item()

        if logger is not None:
            logger.info(f"正在解释Sample {i+1}/{num_samples}...")
        else:
            print(f"正在解释Sample {i+1}/{num_samples}...")
        
        # 生成LIME解释
        explanation = explainer.explain_instance(
            img_array,
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=1000
        )
        
        samples.append({
            'image': img_array,
            'true_label': label.item(),
            'pred_label': predicted,
            'confidence': confidence
        })
        explanations.append(explanation)
    
    return samples, explanations

def visualize_lime_explanations(samples, explanations, class_names,logger):
    """VisualizationLIME解释Results"""
    logger.info(f"VisualizationLIME解释Results...")
    
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, (sample, explanation) in enumerate(zip(samples, explanations)):
        # Original image
        axes[i, 0].imshow(sample['image'])
        axes[i, 0].set_title(f"Original Image\nTrue: {class_names[sample['true_label']]}\n"
                            f"Pred: {class_names[sample['pred_label']]}\n"
                            f"Confidence: {sample['confidence']:.3f}")
        axes[i, 0].axis('off')

        # 获取预测类别的解释
        pred_label = sample['pred_label']

        # 正面特征（支持预测的区域）
        temp_pos, mask_pos = explanation.get_image_and_mask(
            pred_label, positive_only=True, num_features=10, hide_rest=False
        )
        axes[i, 1].imshow(mark_boundaries(temp_pos, mask_pos))
        axes[i, 1].set_title(f"Supporting Regions\n({class_names[pred_label]})")
        axes[i, 1].axis('off')

        # 负面特征（反对预测的区域）
        temp_neg, mask_neg = explanation.get_image_and_mask(
            pred_label, positive_only=False, num_features=10, hide_rest=False
        )
        axes[i, 2].imshow(mark_boundaries(temp_neg, mask_neg))
        axes[i, 2].set_title(f"Opposing Regions\n({class_names[pred_label]})")
        axes[i, 2].axis('off')

        # 超像素分割
        segments = explanation.segments
        axes[i, 3].imshow(mark_boundaries(sample['image'], segments))
        axes[i, 3].set_title(f"Superpixel Segmentation\n({len(np.unique(segments))} regions)")
        axes[i, 3].axis('off')

    save_dir = "./output/pic"
    save_path = os.path.join(save_dir, "lime_explanations.png")

    plt.suptitle('LIME Explainability Analysis Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"LIME解释已保存:{save_path}")

def analyze_feature_importance(samples, explanations, class_names,logger):
    """AnalysisFeature importance"""
    logger.info(f"AnalysisFeature importance...")
    
    # 收集每个类别的特征重要性
    class_importance = {0: [], 1: []}
    
    for sample, explanation in zip(samples, explanations):
        pred_label = sample['pred_label']
        
        # 获取特征重要性分数
        importance_scores = explanation.local_exp[pred_label]
        
        # 计算正面和负面特征的数量和强度
        positive_features = [score for _, score in importance_scores if score > 0]
        negative_features = [score for _, score in importance_scores if score < 0]
        
        class_importance[pred_label].append({
            'positive_count': len(positive_features),
            'negative_count': len(negative_features),
            'positive_strength': sum(positive_features) if positive_features else 0,
            'negative_strength': sum(negative_features) if negative_features else 0,
            'total_features': len(importance_scores)
        })
    
    # Visualization特征重要性统计
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 准备数据
    classes = list(class_importance.keys())
    pos_counts = []
    neg_counts = []
    pos_strengths = []
    neg_strengths = []
    
    for class_idx in classes:
        if class_importance[class_idx]:
            avg_pos_count = np.mean([item['positive_count'] for item in class_importance[class_idx]])
            avg_neg_count = np.mean([item['negative_count'] for item in class_importance[class_idx]])
            avg_pos_strength = np.mean([item['positive_strength'] for item in class_importance[class_idx]])
            avg_neg_strength = np.mean([abs(item['negative_strength']) for item in class_importance[class_idx]])
            
            pos_counts.append(avg_pos_count)
            neg_counts.append(avg_neg_count)
            pos_strengths.append(avg_pos_strength)
            neg_strengths.append(avg_neg_strength)
        else:
            pos_counts.append(0)
            neg_counts.append(0)
            pos_strengths.append(0)
            neg_strengths.append(0)
    
    # 特征数量对比
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, pos_counts, width, label='Supporting Features', color='green', alpha=0.7)
    axes[0, 0].bar(x + width/2, neg_counts, width, label='Opposing Features', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Average Feature Count')
    axes[0, 0].set_title('Supporting vs Opposing Feature Count')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([class_names[i] for i in classes])
    axes[0, 0].legend()

    # 特征强度对比
    axes[0, 1].bar(x - width/2, pos_strengths, width, label='Supporting Strength', color='green', alpha=0.7)
    axes[0, 1].bar(x + width/2, neg_strengths, width, label='Opposing Strength', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Average Feature Strength')
    axes[0, 1].set_title('Supporting vs Opposing Feature Strength')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([class_names[i] for i in classes])
    axes[0, 1].legend()
    
    # Feature importance分布
    all_scores = []
    all_labels = []
    
    for sample, explanation in zip(samples, explanations):
        pred_label = sample['pred_label']
        importance_scores = [score for _, score in explanation.local_exp[pred_label]]
        all_scores.extend(importance_scores)
        all_labels.extend([class_names[pred_label]] * len(importance_scores))
    
    # 按类别分组绘制直方图
    for i, class_idx in enumerate(classes):
        class_scores = [score for score, label in zip(all_scores, all_labels) 
                       if label == class_names[class_idx]]
        if class_scores:
            axes[1, i].hist(class_scores, bins=20, alpha=0.7, color=f'C{i}')
            axes[1, i].set_title(f'{class_names[class_idx]}\nFeature Importance Distribution')
            axes[1, i].set_xlabel('Importance Score')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].axvline(x=0, color='black', linestyle='--', alpha=0.5)

    save_dir = "./output/pic"
    save_path = os.path.join(save_dir, "feature_importance.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Feature importanceAnalysis已保存:{save_path}")


def compare_lime_predictions(samples, explanations, class_names,logger):
    """比较LIME解释的Prediction consistency"""
    logger.info(f"AnalysisLIME解释的Prediction consistency...")
    
    consistent_predictions = 0
    total_predictions = len(samples)
    
    for i, (sample, explanation) in enumerate(zip(samples, explanations)):
        original_pred = sample['pred_label']
        
        # 获取LIME的预测（基于扰动样本）
        lime_pred = explanation.top_labels[0]
        
        if original_pred == lime_pred:
            consistent_predictions += 1
        
        logger.info(f"Sample {i+1}: 原始Prediction={class_names[original_pred]}, LIME预测={class_names[lime_pred]},一致性={'Yes' if original_pred == lime_pred else 'No'}")
    
    consistency_rate = consistent_predictions / total_predictions
    logger.info(f"\n  Prediction consistency: {consistent_predictions}/{total_predictions} ({consistency_rate:.2%})")
    
    return consistency_rate

