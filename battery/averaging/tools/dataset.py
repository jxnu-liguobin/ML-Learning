# encoding=utf8

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from battery.averaging.tools.preprocess import Preprocess

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
warnings.filterwarnings('ignore')


class Dataset:
    """
    数据集类
    """

    def __init__(self, args):
        self.args = args

    def load_batches_to_dict(self, bat_dict1, bat_dict2, bat_dict3):
        """
        构建字典，用来记录三个数据集的信息
        """
        batches_dict = {}

        # Replicating Load Data logic
        print("Loading batches ...")

        batch1 = bat_dict1
        # remove batteries that do not reach 80% capacity
        del batch1['b1c8']
        del batch1['b1c10']
        del batch1['b1c12']
        del batch1['b1c13']
        del batch1['b1c22']

        # updates/replaces the values of dictionary with the new dictionary)
        batches_dict.update(batch1)

        batch2 = bat_dict2
        # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
        # and put it with the correct cell from batch1
        batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
        batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
        add_len = [662, 981, 1060, 208, 482]

        for i, bk in enumerate(batch1_keys):
            batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
            for j in batch1[bk]['summary'].keys():
                if j == 'cycle':
                    batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j],
                                                          batch2[batch2_keys[i]]['summary'][j] + len(
                                                              batch1[bk]['summary'][j])))
                else:
                    batch1[bk]['summary'][j] = np.hstack(
                        (batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
            last_cycle = len(batch1[bk]['cycles'].keys())
            for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
                batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

        del batch2['b2c7']
        del batch2['b2c8']
        del batch2['b2c9']
        del batch2['b2c15']
        del batch2['b2c16']

        # All keys have to be updated after the reordering.
        batches_dict.update(batch1)
        batches_dict.update(batch2)

        batch3 = bat_dict3
        # remove noisy channels from batch3
        del batch3['b3c37']
        del batch3['b3c2']
        del batch3['b3c23']
        del batch3['b3c32']
        del batch3['b3c38']
        del batch3['b3c39']

        batches_dict.update(batch3)

        print("Done loading batches")
        return batches_dict

    def build_feature_df(self, batch_dict):
        """
        建立一个DataFrame，包含加载的批处理字典中所有最初使用的特性
        注意: cell["cycle"]["100"] == cell["summary"][100]
        """

        print("Start building features ...")

        # 124 cells (3 batches)
        n_cells = len(batch_dict.keys())

        # Initializing feature vectors:
        # numpy vector with 124 zeros
        cycle_life = np.zeros(n_cells)
        # 1. delta_Q_100_10(V)
        minimum_dQ_100_10 = np.zeros(n_cells)
        variance_dQ_100_10 = np.zeros(n_cells)
        skewness_dQ_100_10 = np.zeros(n_cells)
        kurtosis_dQ_100_10 = np.zeros(n_cells)

        dQ_100_10_2 = np.zeros(n_cells)
        # 2. Discharge capacity fade curve features
        slope_lin_fit_2_100 = np.zeros(
            n_cells)  # Slope of the linear fit to the capacity fade curve, cycles 2 to 100
        intercept_lin_fit_2_100 = np.zeros(
            n_cells)  # Intercept of the linear fit to capavity face curve, cycles 2 to 100
        discharge_capacity_2 = np.zeros(n_cells)  # Discharge capacity, cycle 2
        # Difference between max discharge capacity and cycle 2
        diff_discharge_capacity_max_2 = np.zeros(n_cells)
        discharge_capacity_100 = np.zeros(n_cells)  # for Fig. 1.e
        slope_lin_fit_95_100 = np.zeros(n_cells)  # for Fig. 1.f
        # 3. Other features
        # Average charge time, cycle 1 to 5
        mean_charge_time_2_6 = np.zeros(n_cells)
        minimum_IR_2_100 = np.zeros(n_cells)  # Minimum internal resistance

        # Internal resistance, difference between cycle 100 and cycle 2
        diff_IR_100_2 = np.zeros(n_cells)
        integral_temperature_2_100 = np.zeros(n_cells)

        # Classifier features
        minimum_dQ_5_4 = np.zeros(n_cells)
        variance_dQ_5_4 = np.zeros(n_cells)
        cycle_550_clf = np.zeros(n_cells)

        # iterate/loop over all cells.
        for i, cell in enumerate(batch_dict.values()):
            cycle_life[i] = cell['cycle_life']
            # 1. delta_Q_100_10(V)
            c10 = cell['cycles']['10']
            c100 = cell['cycles']['100']
            dQ_100_10 = c100['Qdlin'] - c10['Qdlin']

            minimum_dQ_100_10[i] = np.log10(np.abs(np.min(dQ_100_10)))
            variance_dQ_100_10[i] = np.log(np.abs(np.var(dQ_100_10)))
            skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
            kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))

            Qdlin_100_10 = cell['cycles']['100']['Qdlin'] - \
                           cell['cycles']['10']['Qdlin']
            dQ_100_10_2[i] = np.var(Qdlin_100_10)

            # 2. Discharge capacity fade curve features
            # Compute linear fit for cycles 2 to 100:
            # discharge cappacities; q.shape = (99, 1);
            q = cell['summary']['QD'][2:101].reshape(-1, 1)
            # Cylce index from 2 to 100; X.shape = (99, 1)
            X = cell['summary']['cycle'][2:101].reshape(-1, 1)
            linear_regressor_2_100 = LinearRegression()
            linear_regressor_2_100.fit(X, q)

            slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
            intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
            discharge_capacity_2[i] = q[0][0]
            diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]

            discharge_capacity_100[i] = q[-1][0]

            q95_100 = cell['summary']['QD'][95:101].reshape(-1, 1)
            # discharge cappacities; q.shape = (99, 1);
            q95_100 = q95_100 * 1000
            X95_100 = cell['summary']['cycle'][95:101].reshape(-1,
                                                               1)  # Cylce index from 95 to 100; X.shape = (99, 1)
            linear_regressor_95_100 = LinearRegression()
            linear_regressor_95_100.fit(X95_100, q95_100)
            slope_lin_fit_95_100[i] = linear_regressor_95_100.coef_[0]

            # 3. Other features
            mean_charge_time_2_6[i] = np.mean(
                cell['summary']['chargetime'][2:7])
            minimum_IR_2_100[i] = np.min(cell['summary']['IR'][2:101])
            diff_IR_100_2[i] = cell['summary']['IR'][100] - \
                               cell['summary']['IR'][2]
            integral_temperature_2_100[i] = np.sum(
                cell["summary"]["Tavg"][2:101])

            # Classifier features
            c4 = cell['cycles']['4']
            c5 = cell['cycles']['5']
            dQ_5_4 = c5['Qdlin'] - c4['Qdlin']
            minimum_dQ_5_4[i] = np.log10(np.abs(np.min(dQ_5_4)))
            variance_dQ_5_4[i] = np.log10(np.var(dQ_5_4))
            cycle_550_clf[i] = cell['cycle_life'] >= 550

        # combining all featues in one big matrix where rows are the cells and colums are the features
        # note last two variables below are labels/targets for ML i.e cycle life and cycle_550_clf
        features_df = pd.DataFrame({
            "cell_key": np.array(list(batch_dict.keys())),  # 0
            "minimum_dQ_100_10": minimum_dQ_100_10,  # 1
            "variance_dQ_100_10": variance_dQ_100_10,  # 2
            "skewness_dQ_100_10": skewness_dQ_100_10,  # 3
            "kurtosis_dQ_100_10": kurtosis_dQ_100_10,  # 4
            "slope_lin_fit_2_100": slope_lin_fit_2_100,  # 5
            "intercept_lin_fit_2_100": intercept_lin_fit_2_100,  # 6
            "discharge_capacity_2": discharge_capacity_2,  # 7
            "diff_discharge_capacity_max_2": diff_discharge_capacity_max_2,  # 8
            "mean_charge_time_2_6": mean_charge_time_2_6,  # 9
            "minimum_IR_2_100": minimum_IR_2_100,  # 10
            "diff_IR_100_2": diff_IR_100_2,  # 11
            "minimum_dQ_5_4": minimum_dQ_5_4,  # 12
            "variance_dQ_5_4": variance_dQ_5_4,  # 13
            "integral_temperature_2_100": integral_temperature_2_100,  # 14
            "slope_lin_fit_95_100": slope_lin_fit_95_100,  # 15
            "cycle_life": cycle_life,  # 16
            "cycle_550_clf": cycle_550_clf  # 17
        })

        print("Done building features")
        return features_df

    def train_val_split(self, features_df, model="regression", remove_exceptional_cells=True):
        """
        划分train&test数据集，注意：数据集要按照指定方式划分
        :param features_df: 包含最初使用的特性dataframe
        :param model: 使用模型的flag
        """

        # get the features for the model version (full, variance, discharge)
        feature_indices = ["minimum_dQ_100_10", "variance_dQ_100_10",
                           "variance_dQ_5_4", "diff_IR_100_2", "slope_lin_fit_2_100",
                           "discharge_capacity_2"]
        # get all cells with the specified features
        model_features = features_df[feature_indices]
        # get last two columns (cycle life and classification)
        labels = features_df.iloc[:, -2:]
        # labels are (cycle life ) for regression other wise (0/1) for classsification
        labels = labels.iloc[:,
                 0] if model == "regression" else labels.iloc[:, 1]

        # split data in to train/primary_test/and secondary test
        train_cells = np.arange(1, 84, 2)
        val_cells = np.arange(0, 84, 2).tolist()
        if remove_exceptional_cells:
            val_cells.remove(42)
        test_cells = np.arange(84, 124, 1)

        # get cells and their features of each set and convert to numpy for further computations
        x_train = model_features.loc[train_cells]
        x_val = model_features.iloc[val_cells]
        x_test = model_features.iloc[test_cells]

        # target values or labels for training
        y_train = labels.iloc[train_cells]
        y_val = labels.iloc[val_cells]
        y_test = labels.iloc[test_cells]

        # return 3 sets
        return {"train": [x_train, y_train], "val": [x_val, y_val], "test": [x_test, y_test]}

    def data_normalize(self, battery_dataset):
        data_normalize = battery_dataset.copy()
        # s = StandardScaler().fit(data_normalize.iloc[train_cells])
        # s = Normalizer().fit(data_normalize.iloc[train_cells])
        s = MinMaxScaler().fit(data_normalize["train"][0])
        data_normalize["train"][0] = s.transform(data_normalize["train"][0])
        data_normalize["val"][0] = s.transform(data_normalize["val"][0])
        data_normalize["test"][0] = s.transform(data_normalize["test"][0])

        return data_normalize

    @staticmethod
    def get_label_scaler(y):
        s = MinMaxScaler().fit(y.values.reshape(-1, 1))
        return s

    def get_feature(self):
        """
        类主函数，返回可用于训练的数据集
        """
        pre_dataset = Preprocess(args=self.args)
        bat_dict1, bat_dict2, bat_dict3 = pre_dataset.read()
        # calling function to load from disk
        all_batches_dict = self.load_batches_to_dict(bat_dict1, bat_dict2, bat_dict3)
        # function to build features for ML
        features_df = self.build_feature_df(all_batches_dict)
        battery_dataset = self.train_val_split(features_df)
        battery_dataset = self.data_normalize(battery_dataset)
        return battery_dataset
