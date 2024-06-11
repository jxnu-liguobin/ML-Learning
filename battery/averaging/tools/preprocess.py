# encoding=utf8

import h5py
import numpy as np
import warnings
import pickle

warnings.filterwarnings('ignore')


class Preprocess:
    """
    数据预处理类
    """

    def __init__(self, args):
        self.args = args

    def write(self):
        bat_dict1, bat_dict2, bat_dict3 = self.data_preprocess()
        data = {"bat_dict1": bat_dict1, "bat_dict2": bat_dict2, "bat_dict3": bat_dict3}
        for key, value in data.items():
            print("saving {} to disk ...".format(key))
            with open(key + ".pkl", "wb") as f:
                pickle.dump(value, f)

    def read(self):
        import os
        pickle_files = ['bat_dict1.pkl', 'bat_dict2.pkl', 'bat_dict3.pkl']
        if not all(os.path.exists(file) for file in pickle_files):
            print("pkl not exists, create one")
            self.write()
            print("pkl created")
        data = {}
        print("Loading pkl from disk ...")
        for file_name in pickle_files:
            with open(file_name, "rb") as f:
                data[file_name] = pickle.load(f)
        return data["bat_dict1.pkl"], data["bat_dict2.pkl"], data["bat_dict3.pkl"]

    def data_preprocess(self):
        """
        数据预处理主函数，返回完成预处理数据集，3个batch
        """

        # batch 1
        f = h5py.File(self.args.matFilename1)
        batch = f['batch']
        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        for i in range(num_cells):
            cl = f[batch['cycle_life'][i, 0]][()]
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                       'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]][()]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
                T = np.hstack((f[cycles['T'][j, 0]][()]))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][()]))
                V = np.hstack((f[cycles['V'][j, 0]][()]))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
                t = np.hstack((f[cycles['t'][j, 0]][()]))
                cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[str(j)] = cd
            cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
            key = 'b1c' + str(i)
            bat_dict[key] = cell_dict

        bat_dict1 = bat_dict

        # batch 2
        f = h5py.File(self.args.matFilename2)

        batch = f['batch']

        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        for i in range(num_cells):
            cl = f[batch['cycle_life'][i, 0]][()]
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                       'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]][()]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
                T = np.hstack((f[cycles['T'][j, 0]][()]))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][()]))
                V = np.hstack((f[cycles['V'][j, 0]][()]))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
                t = np.hstack((f[cycles['t'][j, 0]][()]))
                cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[str(j)] = cd

            cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
            key = 'b2c' + str(i)
            bat_dict[key] = cell_dict

        bat_dict2 = bat_dict

        # batch 3 
        f = h5py.File(self.args.matFilename3)

        batch = f['batch']

        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        for i in range(num_cells):
            cl = f[batch['cycle_life'][i, 0]][()]
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                       'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]][()]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
                T = np.hstack((f[cycles['T'][j, 0]][()]))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][()]))
                V = np.hstack((f[cycles['V'][j, 0]][()]))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
                t = np.hstack((f[cycles['t'][j, 0]][()]))
                cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[str(j)] = cd

            cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
            key = 'b3c' + str(i)
            bat_dict[key] = cell_dict

        bat_dict3 = bat_dict

        return bat_dict1, bat_dict2, bat_dict3
