import lightgbm as lgb
import numpy as np
import pandas as pd
import utils
import tqdm
from copy import deepcopy

"""
Base tuned configs from https://github.com/catboost/catboost/tree/master/catboost/benchmarks/quality_benchmarks/notebooks
"""
BASE_CONFIGS = {
    "adult": {'num_leaves': 16, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
              'min_data_in_leaf': 2, 'bagging_fraction': 0.8231258721410332,
              'min_sum_hessian_in_leaf': 0.0023527955721999202, 'feature_fraction_seed': 0,
              'lambda_l1': 0.0009040060631182545, 'bagging_freq': 1, 'lambda_l2': 0, 'objective': 'binary',
              'drop_seed': 0, 'learning_rate': 0.01623096612907187, 'feature_fraction': 0.6153146698674108},
    'amazon': {'num_leaves': 35, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
               'min_data_in_leaf': 43, 'bagging_fraction': 0.7533430554699614,
               'min_sum_hessian_in_leaf': 5, 'feature_fraction_seed': 0, 'lambda_l1': 0,
               'bagging_freq': 1, 'lambda_l2': 0, 'objective': 'binary', 'drop_seed': 0,
               'learning_rate': 10 * 0.0023062503242047053, 'feature_fraction': 0.8942720034091154}, # changed learning rate to speed up convergence
    'appet': {# 'num_leaves': 6, 'verbose': -1, 'bagging_seed': 0, 'metric': 'binary_logloss', 'data_random_seed': 0,
              # 'min_data_in_leaf': 95, 'bagging_fraction': 0.8956806282244631,
              # 'min_sum_hessian_in_leaf': 0.05112201732261002, 'feature_fraction_seed': 0,
              # 'lambda_l1': 0.013807846588423104, 'bagging_freq': 1, 'lambda_l2': 0, 'objective': 'binary',
              # 'drop_seed': 0, 'learning_rate': 0.012538826956019477, 'feature_fraction': 0.5112693049121645},
              'metric': 'auc', 'objective': 'binary', 'bagging_freq': 1, 'feature_fraction': 0.5, 'bagging_fraction' : 0.99, 'learning_rate' : 0.05, "verbose": -1
              },
    'click': {'num_leaves': 45, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
              'min_data_in_leaf': 8, 'bagging_fraction': 0.8420658067233723,
              'min_sum_hessian_in_leaf': 0.004549958246639956, 'feature_fraction_seed': 0,
              'lambda_l1': 0.7677413533309505, 'bagging_freq': 1, 'lambda_l2': 0.10721553735236211,
              'objective': 'binary', 'drop_seed': 0, 'learning_rate': 2 * 0.029263299622061, #increased learing rate
              'feature_fraction': 0.670232967018771},
    'internet': {'num_leaves': 8, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
                 'min_data_in_leaf': 16, 'bagging_fraction': 0.9046010662322765,
                 'min_sum_hessian_in_leaf': 1.271575758364032e-05, 'feature_fraction_seed': 0,
                 'lambda_l1': 0.016012980104284796, 'bagging_freq': 1, 'lambda_l2': 0, 'objective': 'binary',
                 'drop_seed': 0, 'learning_rate': 0.006879696425995826, 'feature_fraction': 0.9957686774658463},
    'kdd98' : {'num_leaves': 5, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
               'min_data_in_leaf': 5, 'bagging_fraction': 0.7801172267397591,
               'min_sum_hessian_in_leaf': 132.9945857111621, 'feature_fraction_seed': 0,
               'lambda_l1': 0.0022903323397730152, 'bagging_freq': 1, 'lambda_l2': 0, 'objective': 'binary',
               'drop_seed': 0, 'learning_rate': 0.029609632447460447, 'feature_fraction': 0.7235330841303137},
    'kddchurn': {'num_leaves': 19, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
                 'min_data_in_leaf': 2, 'bagging_fraction': 0.7790653522214104,
                 'min_sum_hessian_in_leaf': 1.7132439204757702e-06, 'feature_fraction_seed': 0,
                 'lambda_l1': 4.31311677117027e-06, 'bagging_freq': 1, 'lambda_l2': 3.289808221335635,
                 'objective': 'binary', 'drop_seed': 0, 'learning_rate': 0.002468127789150438,
                 'feature_fraction': 0.655578595572939},
    "kick" : {'num_leaves': 157, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
              'min_data_in_leaf': 4, 'bagging_fraction': 0.7618075391294016,
              'min_sum_hessian_in_leaf': 3.776016815653153e-06, 'feature_fraction_seed': 0, 'lambda_l1': 0,
              'bagging_freq': 1, 'lambda_l2': 0.07371594809127686, 'objective': 'binary', 'drop_seed': 0,
              'learning_rate': 0.019483006590789694, 'feature_fraction': 0.7515374545303946}, #increased learing rate
    "upsel" : {'num_leaves': 11, 'verbose': -1, 'bagging_seed': 0, 'metric': 'auc', 'data_random_seed': 0,
               'min_data_in_leaf': 109, 'bagging_fraction': 0.8965102673774312,
               'min_sum_hessian_in_leaf': 1.0201871963154937e-05, 'feature_fraction_seed': 0,
               'lambda_l1': 0.006143582721183149, 'bagging_freq': 1, 'lambda_l2': 0, 'objective': 'binary',
               'drop_seed': 0, 'learning_rate': 0.004440192867499371, 'feature_fraction': 0.5804298705452275}
}


class BaseExperiment(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def run(self):
        history = []
        for params, dataset, cat_names in self.generate_configs():
            log = lgb.cv(params, dataset, num_boost_round=5000, early_stopping_rounds=250)
            history.append({
                'params': params,
                'log' : log
            })
        return history

    def generate_configs(self):
        raise NotImplementedError()


class BaselineExperiment(BaseExperiment):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def generate_configs(self):
        df, cat_names = utils.read_dataset_by_name(self.dataset_name)
        dataset = lgb.Dataset(df.drop(['Target'], axis=1), label=df['Target'])
        base_config = deepcopy(BASE_CONFIGS[self.dataset_name])
        base_config['force_row_wise'] = True
        yield base_config, dataset, cat_names
        return


class SGBExperiment(BaseExperiment):
    def __init__(self, dataset_name, bagging_fraction_space = np.linspace(0.05, 0.95, 9)):
        super().__init__(dataset_name)
        self.bagging_fraction_space = bagging_fraction_space

    def generate_configs(self):
        df, cat_names = utils.read_dataset_by_name(self.dataset_name)
        dataset = lgb.Dataset(df.drop(['Target'], axis=1), label=df['Target'],)
        for f in tqdm.tqdm(self.bagging_fraction_space):
            base_config = deepcopy(BASE_CONFIGS[self.dataset_name])
            base_config['bagging_fraction'] = f
            base_config['force_row_wise'] = True
            yield base_config, dataset, cat_names
        return


class MVSExperiment(SGBExperiment):
    DATASET_LAMBDA = {
            'adult': 1e1,
            'amazon': 1e-5,
            'click': 1.,
            'internet': 1e-2,
            'kick': 10.
            }
    def __init__(self, dataset_name, bagging_fraction_space = np.linspace(0.05, 0.95, 9),
                 mvs_adaptive=True, mvs_lambda=1e-2):
        super().__init__(dataset_name, bagging_fraction_space)
        self.mvs_adptive = mvs_adaptive
        self.mvs_lambda = self.DATASET_LAMBDA[dataset_name]

    def generate_configs(self):
        df, cat_names = utils.read_dataset_by_name(self.dataset_name)
        dataset = lgb.Dataset(df.drop(['Target'], axis=1), label=df['Target'])
        for f in tqdm.tqdm(self.bagging_fraction_space):
            base_config = deepcopy(BASE_CONFIGS[self.dataset_name])
            base_config['boosting'] = 'mvs'
            base_config['bagging_fraction'] = f
            base_config["mvs_max_sequential_size"] = 25000
            base_config['mvs_adaptive'] = self.mvs_adptive
            base_config['mvs_lambda'] = self.mvs_lambda
            base_config['force_row_wise'] = True
            yield base_config, dataset, cat_names
        return


class GOSSExperiment(BaseExperiment):
    DATASET_RAITO = {
            'adult': (1, 5),
            'amazon': (5, 1),
            'click': (1, 5),
            'internet': (1, 5),
            'kick': (1, 5)
            }

    def __init__(self, dataset_name, bagging_fraction_space = np.linspace(0.05, 0.95, 9), top=1, other=5):
        super().__init__(dataset_name)
        self.bagging_fraction_space = bagging_fraction_space
        self.top = self.DATASET_RAITO[dataset_name][0]
        self.other = self.DATASET_RAITO[dataset_name][1]

    def generate_configs(self):
        df, cat_names = utils.read_dataset_by_name(self.dataset_name)
        dataset = lgb.Dataset(df.drop(['Target'], axis=1), label=df['Target'])
        for f in tqdm.tqdm(self.bagging_fraction_space):
            base_config = deepcopy(BASE_CONFIGS[self.dataset_name])
            base_config['boosting'] = 'goss'
            # base_config['force_row_wise'] = True
            base_config['top_rate'] = f * self.top / float(self.top + self.other)
            base_config['other_rate'] = f * self.other / float(self.top + self.other)
            base_config['bagging_freq'] = 0
            base_config['bagging_fraction'] = 1.0
            base_config['force_row_wise'] = True
            yield base_config, dataset, cat_names
        return
