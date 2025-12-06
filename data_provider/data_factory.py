import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split  # 用于分层划分
from util.instructor_solution_guide import get_data_transforms_solution

# 假设ThermalDataset的定义（若已存在可忽略，此处为补全代码）
class ThermalDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_samples()  # 存储(样本路径, 标签)的列表

    def _load_samples(self):
        # 需根据实际数据集结构实现，此处为示例（假设data_path下有non_icas/icas两个子文件夹）
        samples = []
        for label, cls_name in enumerate(['non_icas', 'icas']):
            cls_path = os.path.join(self.data_path, cls_name)
            if not os.path.exists(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                samples.append((img_path, label))
        return samples

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 此处需补充图像读取逻辑（如PIL.Image.open），并应用transform
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

def data_provider(args, flag,logger,use_lime=False):
    """
    数据集加载与划分函数：70%训练集、15%研究集、15%测试集（分层划分）
    Args:
        args: 参数对象，需包含data_path（数据集路径）、batch_size（批大小）
        flag: 取值为'train'/'research'/'test'，指定返回哪个子集的数据集和加载器
    Returns:
        dataset: 指定子集的Dataset对象
        data_loader: 指定子集的DataLoader对象
    """
    # 校验flag合法性
    assert flag in ['train', 'val', 'test'], f"flag must be 'train'/'val'/'test', got {flag}"
    data_path = args.data_path

    # 检查数据集路径是否存在
    if not os.path.exists(data_path):
        logger.info(f"Error: Dataset path {data_path} does not exist!")
        return False

    # 定义图像变换（与原代码一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if use_lime:
        # 用于模型预测的变换（包含归一化）
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    # 获取数据变换：训练集用增强，研究/测试集用预处理
    # train_transform, val_transform = get_data_transforms_solution()
    # transform = train_transform if flag == 'train' else val_transform


    # 加载完整数据集
    full_dataset = ThermalDataset(data_path, transform=transform)
    total_samples = len(full_dataset)
    logger.info(f"Dataset loaded successfully! Total samples: {total_samples}")

    # 提取所有样本的标签（用于分层划分）
    labels = [label for _, label in full_dataset.samples]

    # 第一步：划分70%训练集 和 30%临时集（研究+测试）
    train_idx, temp_idx = train_test_split(
        range(total_samples),
        test_size=0.3,  # 30%为临时集
        random_state=args.fix_seed,  # 固定随机种子，确保划分结果可复现
        stratify=labels  # 分层划分，保证类别比例一致
    )

    # 提取临时集的标签（用于第二步划分）
    temp_labels = [labels[idx] for idx in temp_idx]

    # 第二步：将30%临时集划分为15%研究集 和 15%测试集（临时集的50%/50%）
    research_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # 临时集的50%为测试集，最终占总数据15%
        random_state=args.fix_seed,
        stratify=temp_labels  # 分层划分
    )

    # 建立flag与子集索引的映射
    idx_map = {
        'train': train_idx,
        'val': research_idx,
        'test': test_idx
    }
    subset_idx = idx_map[flag]

    # 创建子集数据集
    subset_dataset = Subset(full_dataset, subset_idx)

    # 统计原数据集和各子集的类别分布
    def count_class(dataset, indices=None):
        """统计数据集/子集的类别分布"""
        counts = {'non_icas': 0, 'icas': 0}
        if indices is None:  # 统计完整数据集
            samples = dataset.samples
        else:  # 统计子集
            samples = [dataset.samples[idx] for idx in indices]
        for _, label in samples:
            if label == 0:
                counts['non_icas'] += 1
            else:
                counts['icas'] += 1
        return counts

    # 打印类别分布日志
    full_counts = count_class(full_dataset)
    subset_counts = count_class(full_dataset, subset_idx)
    logger.info("=" * 60)
    logger.info(f"Full dataset class distribution:")
    logger.info(f"  Non-ICAS: {full_counts['non_icas']} samples ({full_counts['non_icas']/total_samples*100:.1f}%)")
    logger.info(f"  ICAS: {full_counts['icas']} samples ({full_counts['icas']/total_samples*100:.1f}%)")
    logger.info(f"{flag.capitalize()} set ({len(subset_idx)} samples, {len(subset_idx)/total_samples*100:.1f}% of total):")
    logger.info(f"  Non-ICAS: {subset_counts['non_icas']} samples ({subset_counts['non_icas']/len(subset_idx)*100:.1f}%)")
    logger.info(f"  ICAS: {subset_counts['icas']} samples ({subset_counts['icas']/len(subset_idx)*100:.1f}%)")
    logger.info("=" * 60)

    # 创建数据加载器（训练集shuffle=True，研究/测试集shuffle=False）
    shuffle = True if flag == 'train' else False
    batch_size = 4 if flag == 'train' else 1

    data_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return subset_dataset, data_loader
