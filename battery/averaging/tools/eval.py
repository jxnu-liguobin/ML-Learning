# encoding=utf8

import argparse
import json
import os
import sys
import warnings

from battery.averaging.tools.dataset import Dataset

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
from battery.averaging.tools.averaging_model import *

warnings.filterwarnings('ignore')


class Eval:
    """
    验证类
    """

    def __init__(self, args, model=None):
        self.args = args
        self.model = model

    def evaluation(self, datasets, load_model=False):
        # get three sets
        x_val, y_val = datasets.get("val")
        x_test, y_test = datasets.get("test")
        if load_model:
            # regr = joblib.load(f"./model/model_regression.pkl")
            regr = AveragingModels([])
            regr.load("./model/model_merge")
        else:
            regr = self.model

        # predict values/cycle life for all three sets
        pred_val = regr.predict(x_val)
        pred_test = regr.predict(x_test)

        # mean percentage error (same as paper)
        error_val = mean_absolute_percentage_error(y_val, pred_val) * 100
        error_test = mean_absolute_percentage_error(y_test, pred_test) * 100

        print(f"Regression Error (validation (primary) test): {error_val}%")
        print(f"Regression Error batch 3 (test (secondary)): {error_test}%")

    def run_evaluation(self):
        """
        验证回归模型主参数
        """
        features = Dataset(self.args).get_feature()
        self.evaluation(features, load_model=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--config_path', type=str,
                        default='../config/model_merge.json')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)

    Eval(args).run_evaluation()
