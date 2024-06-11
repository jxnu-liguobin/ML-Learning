# encoding=utf8

import argparse
import json
import os
import random
import sys
import warnings

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from xgboost import XGBRegressor

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
from battery.averaging.tools.dataset import Dataset
from battery.averaging.tools.averaging_model import *

warnings.filterwarnings('ignore')

__MODEL__AVAILABLE__ = {
    "ElasticNet": ElasticNet,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "KernelRidge": KernelRidge,
    "AdaBoostRegressor": AdaBoostRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "LinearRegression": LinearRegression,
    "XGBRegressor": XGBRegressor
}


class Train:
    """
    训练类
    """

    def __init__(self, args):
        self.args = args
        self.model = None

    def manual_seed(self, seed_value):
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    def regression(self, datasets, model_cfg=None, save_model=False):
        """
        回归函数
        :param datasets: 数据集
        :param alpha_train: 惩罚项系数
        :param l1_ratio: L1模型权重
        :param regression_type: 回归模型类型
        :param log_target: 是否进行log变换
        :param model: 使用模型
        """
        # get three sets
        self.manual_seed(4)
        x_train, y_train = datasets.get("train")
        y_scaler = Dataset.get_label_scaler(y_train)

        n_models = len(model_cfg)
        model_selects = []
        for i in range(n_models):
            model_name = model_cfg[i]["model_name"]
            if model_name == "KerasRegressor":
                model_selects.append(
                    OptionalNnModels(
                        KerasRegressor(build_fn=build_nn,
                                       epochs=model_cfg[i]["epochs"],
                                       batch_size=model_cfg[i]["batch_size"],
                                       verbose=0),
                        target_scaler=y_scaler))
            else:
                model = __MODEL__AVAILABLE__.get(model_name, None)
                if model is None:
                    raise ValueError("model_name is not available")
                model_selects.append(
                    OptionalModel(
                        model(**model_cfg[i][model_name]),
                        log_target=model_cfg[i]["log_target"]
                    )
                )
        aver_model = AveragingModels(model_selects)

        # fit regression model
        aver_model.fit(x_train, y_train)
        # predict values/cycle life for all three sets
        pred_train = aver_model.predict(x_train)

        # mean percentage error (same as paper)
        error_train = mean_absolute_percentage_error(y_train, pred_train) * 100
        if save_model:
            # joblib.dump(aver_model, f"./model/model_regression.pkl")
            if not os.path.exists('./model_merge'):
                os.mkdir('./model_merge')
            aver_model.save("./model_merge")
        else:
            self.model = aver_model
        print(f"Regression Error (Train): {error_train}%")

    def run_regression(self):
        """
        训练回归模型主参数
        """
        model_cfg = self.args.model_cfg
        features = Dataset(self.args).get_feature()
        self.regression(features, model_cfg=model_cfg, save_model=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--config_path', type=str,
                        default='../config/model_merge.json')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)

    Train(args).run_regression()
