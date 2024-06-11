# encoding=utf8

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))

from battery.averaging.tools.dataset import Dataset
from tools.train import Train
from tools.eval import Eval

warnings.filterwarnings('ignore')


class Main:
    """
    算法主程序类
    """

    def __init__(self, args):
        """
        初始化
        :param args: 初始化信息
        """
        self.args = args

    def run(self):
        """
        运行算法主程序
        """
        # Full
        model_cfg = self.args.model_cfg
        dataset = Dataset(self.args)
        features = dataset.get_feature()

        mode_full = Train(self.args)
        mode_full.regression(features, model_cfg=model_cfg)

        Eval(self.args, model=mode_full.model).evaluation(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--config_path', type=str,
                        default='../config/model_merge.json')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)
    print(args)
    Main(args).run()
