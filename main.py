from util.test_data_loading import test_data_loading
from util.instructor_solution_guide import ResNetClassifier
from data_provider.data_factory import data_provider
from exp.exp_ResNetClassifier import Exp_ResNetClassifier
import torch
import argparse
import random
import numpy as np
import logging
import os
import sklearn
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='classifer')

# basic config
parser.add_argument('--is_training', type=int,  default=1, help='status')
parser.add_argument('--model', type=str, default='ResNetClassifier',help='model name, options: [ResNetClassifier]')
parser.add_argument('--data_path', type=str, default='./dataset/thermal_classification_cropped/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--num_class', type=int, default=2, help='data num_class')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--drop_last', type=bool, default=True, help='drop last')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')

# 默认cpu

parser.add_argument('--use_gpu', action='store_true', help='use gpu (default: False)')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)

#other
parser.add_argument('--fix_seed', type=int, default=2023, help='number of seed')

def seed_all(fix_seed=2023):
    sklearn.random.seed(fix_seed)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)  # 多GPU
    np.random.seed(fix_seed)

def main():
    args = parser.parse_args()
    print(args.use_gpu)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = args.fix_seed
    seed_all(fix_seed)

    print(args.learning_rate, fix_seed)

    if args.use_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 生成日志文件路径
    mode = "train" if args.is_training else "test"
    log_path = './log/{}_seed_{}_{}_epochs_{}.log'.format(
        args.model,
        fix_seed,
        mode,
        args.train_epochs
    )

    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            pass

    args.seed = fix_seed
    args.log_path = log_path
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    args.logger = logger

    args.logger.info('Args in experiment:')
    args.logger.info(args)

    Exp = Exp_ResNetClassifier

    if args.is_training:
        setting = '{}_{}'.format(args.model,  args.fix_seed)

        exp = Exp(args)  # set experiments
        args.logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        args.logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting,test=1)
        torch.cuda.empty_cache()

    else:

        setting = '{}_{}'.format(args.model, args.fix_seed)

        exp = Exp(args)  # set experiments
        args.logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()



# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
