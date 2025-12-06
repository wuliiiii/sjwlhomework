from exp.exp_basic import Exp_Basic
from torch.optim import lr_scheduler
import torch.nn as nn
from data_provider.data_factory import data_provider
from util.tools import EarlyStopping, adjust_learning_rate
from util.instructor_solution_guide import get_data_transforms_solution
from util.gradcam_pytorch import visualize_gradcam,analyze_attention_patterns,compare_correct_vs_wrong_predictions
from util.lime_pytorch import explain_with_lime,visualize_lime_explanations,analyze_feature_importance,compare_lime_predictions
from torch import optim
import torch
import os
import time
import warnings
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,roc_curve, auc
import seaborn as sns
from torchvision import transforms, datasets, models
import sys

# Sample Visualization Functions
def plot_training_history_solution(history,logger=None):
    """Sample training history plotting"""

    # 定义保存目录
    save_dir = "./output/pic"
    # 自动创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 拼接完整的保存路径
    save_path = os.path.join(save_dir, "history")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if logger is not None:
        logger.info(f"训练图已成功保存至: {os.path.abspath(save_path)}")


def plot_confusion_matrix_solution(cm, class_names,logger=None):
    """Sample confusion matrix plotting"""

    # 定义保存目录
    save_dir = "./output/pic"
    # 自动创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 拼接完整的保存路径
    save_path = os.path.join(save_dir, "confusion_matrix")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if logger is not None:
        logger.info(f"混淆矩阵图已成功保存至: {os.path.abspath(save_path)}")

def plot_roc_curve_solution( y_true, y_prob, class_names, positive_class_idx=1,logger=None):
        """绘制并保存ROC曲线，标注AUC值"""
        save_dir = "./output/pic"
        save_path = os.path.join(save_dir, "roc_curve.png")

        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=positive_class_idx)
        roc_auc = auc(fpr, tpr)
        positive_class_name = class_names[positive_class_idx]

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve for {positive_class_name} Classification')
        plt.legend(loc="lower right")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if logger:
            logger.info(f"ROC曲线已保存至: {os.path.abspath(save_path)}")
            logger.info(f"ROC曲线AUC值: {roc_auc:.4f} (正类: {positive_class_name})")

class Exp_ResNetClassifier(Exp_Basic):
    def __init__(self, args):
        super(Exp_ResNetClassifier, self).__init__(args)
        self.logger = self._setup_logger()
        self.logger.info(f"Exp_ResNetClassifier init")

    def _build_model(self):
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _setup_logger(self):
        """设置日志器，配置日志格式和级别（修复中文乱码）"""
        logger_name = f"{self.__class__.__name__}_{id(self)}"
        logger = logging.getLogger(logger_name)

        # 避免重复设置处理器
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL

            # 定义统一的日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'  # 补充时间格式，可选
            )

            # ========== 修复控制台中文乱码 ==========
            console_handler = logging.StreamHandler(sys.stdout)  # 指定stdout，避免stderr编码问题
            console_handler.setLevel(logging.INFO)  # 控制台输出INFO及以上级别
            # 强制设置控制台编码为UTF-8（Windows关键）
            if hasattr(console_handler.stream, 'reconfigure'):
                console_handler.stream.reconfigure(encoding='utf-8')
            else:
                # 兼容Python 3.7以下版本
                import io
                console_handler.stream = io.TextIOWrapper(
                    console_handler.stream.buffer, encoding='utf-8'
                )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # ========== 修复日志文件中文乱码 ==========
            file_handler = None  # 初始化，避免未定义报错
            if hasattr(self.args, 'log_path') and self.args.log_path:  # 增加非空判断
                # 核心：指定encoding='utf-8'，mode='a'追加模式（默认也是a，显式指定更清晰）
                file_handler = logging.FileHandler(
                    self.args.log_path,
                    mode='a',
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG及以上级别
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            # 防止日志传播到根日志器
            logger.propagate = False

            # 可选：Windows CMD下切换编码（如果需要）
            if os.name == 'nt':  # 判断是否为Windows系统
                os.system('chcp 65001 > nul')  # 静默切换CMD编码为UTF-8

        return logger

    def _get_data(self, flag,use_lime=False):
        data_set, data_loader = data_provider(self.args, flag,self.logger,use_lime=use_lime)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting, debug=False):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        model,history=self.train_model_solution(self.model,train_loader,vali_loader,self.args.train_epochs,self.device)

        plot_training_history_solution(history,self.logger)

        return model


    def train_model_solution(self, model, train_loader, val_loader, num_epochs, device):
        """
        加入多种正则化技术的训练函数
        包含：权重衰减、Dropout（模型内）、标签平滑、梯度裁剪、早停、学习率调度
        """
        # 计算类别权重处理数据不平衡
        class_weights = torch.tensor([647 / 950, 303 / 950], dtype=torch.float32).to(device)

        # 1. 标签平滑正则化（替代普通CrossEntropyLoss）
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # 标签平滑系数0.1
        # 2. 优化器：保留权重衰减（L2正则化），这是最基础的正则化
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight_decay即L2正则
        # 学习率调度器（学习率衰减，防止过拟合）
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        # 初始化训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # 3. 早停（Early Stopping）相关参数：防止过拟合，监控验证损失
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience = 10  # 容忍多少轮验证损失不下降
        patience_counter = 0
        min_delta = 1e-4  # 验证损失下降的最小阈值

        # 梯度裁剪（Gradient Clipping）参数：防止梯度爆炸，也是一种正则化手段
        clip_norm = 1.0  # 梯度的最大范数

        self.logger.info(
            f"训练初始化完成 | 正则化配置：标签平滑(0.1)、权重衰减(1e-4)、梯度裁剪({clip_norm})、早停(patience={patience})")
        self.logger.info(f"训练轮数：{num_epochs} | 设备：{device}")

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # 4. 梯度裁剪：限制梯度的L2范数，防止梯度爆炸，同时缓解过拟合
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                optimizer.step()

                # 统计训练指标
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # 计算平均指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            # 更新历史记录
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            # 更新学习率
            scheduler.step()

            # 早停逻辑：监控验证损失（也可监控验证准确率）
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0  # 重置计数器
            else:
                patience_counter += 1  # 验证损失未下降，计数器+1

            # 保存最佳模型（新增：同时考虑准确率和损失，且epoch>5避免早期过拟合）
            if val_acc > best_val_acc and epoch > 5:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'{self.args.checkpoints}/best_model.pth')

            # 打印日志
            log_msg = (
                f'Epoch [{epoch + 1}/{num_epochs}] | '
                f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                f'Patience Counter: {patience_counter}/{patience} | '
                f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )
            self.logger.info(log_msg)

            # 早停触发：如果计数器超过容忍度，提前终止训练
            if patience_counter >= patience:
                self.logger.info(f"早停触发！验证损失连续{patience}轮未下降，提前结束训练")
                break

        # 训练结束
        self.logger.info(f"训练完成 | 最终最佳验证准确率：{best_val_acc:.2f}%")
        return model, history

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        class_names = ["ICAS", "Non-ICAS"]

        if test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(f'{self.args.checkpoints}/best_model.pth'))

        # 展示训练
        metrics = self.evaluate_model_solution(self.model,test_loader,self.device)
        self.logger.info(metrics)

        # gradcam
        self._gradcam()

        # line
        self._lime()


    def _gradcam(self):
        test_data, test_loader = self._get_data(flag='test')
        class_names = ["ICAS", "Non-ICAS"]

        # Step2: 生成Grad-CAM可视化
        visualize_gradcam(self.model.model.backbone, test_loader, class_names, device=self.device, logger=self.logger)

        # step3:分析注意力机制
        analyze_attention_patterns(self.model.model.backbone, test_loader, class_names, device=self.device,
                                   logger=self.logger)

        # Step4: 比较正确和错误预测
        compare_correct_vs_wrong_predictions(self.model.model.backbone, test_loader, class_names, device=self.device,
                                             logger=self.logger)


    def _lime(self):
        lime_test_data, lime_test_loader = self._get_data(flag='test', use_lime=True)
        class_names = ["ICAS", "Non-ICAS"]
        # 使用Lime
        # 用于LIME的变换（不包含归一化）
        model_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 用于模型预测的变换（包含归一化）
        line_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Step2: 使用LIME生成解释
        samples, explanations = explain_with_lime(self.model.model.backbone, lime_test_loader, class_names,
                                                  model_transform,num_samples=4, logger=self.logger)

        # Step3: 可视化LIME解释
        visualize_lime_explanations(samples, explanations, class_names, logger=self.logger)

        # Step4: 分析特征重要性
        analyze_feature_importance(samples, explanations, class_names, logger=self.logger)

        # Step5: 比较预测一致性
        consistency_rate = compare_lime_predictions(samples, explanations, class_names, logger=self.logger)

        self.logger.info(f"  Prediction consistency: {consistency_rate:.2%}")
        self.logger.info(f"  LIME通过扰动输入Image的超像素区域来解释Model决策")
        self.logger.info(f"  绿色区域支持预测，红色区域反对Prediction")


    # Sample Evaluation Function Implementation
    def evaluate_model_solution(self,model, test_loader, device):
        """Sample evaluation implementation"""

        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)
        cm = confusion_matrix(all_labels, all_predictions)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }

        # 绘制并保存混淆矩阵和ROC曲线
        class_names = ["ICAS", "Non-ICAS"]
        plot_confusion_matrix_solution(cm, class_names,logger=self.logger)
        plot_roc_curve_solution(all_labels, all_probabilities, class_names,logger=self.logger)

        return metrics
