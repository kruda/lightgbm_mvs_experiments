import pandas as pd
import numpy as np
from scipy.io import arff
import re
import sys
import os
from io import StringIO


def read_click(*args, **kwargs):
    with open("./data/click/track2/subsampling_idx.txt") as fin:
        ids = map(int, fin.readlines())
    unique_ids = set(ids)
    data_strings = {}
    with open('./data/click/track2/training.txt') as fin:
        for i, string in enumerate(fin):
            if i in unique_ids:
                data_strings[i] = string
    data_rows = [v for _,v in data_strings.items()]

    data = pd.read_table(StringIO("".join(data_rows)), header=None, delim_whitespace=True).apply(np.float64)
    colnames = ['click',
                'impression',
                'url_hash',
                'ad_id',
                'advertiser_id',
                'depth',
                'position',
                'query_id',
                'keyword_id',
                'title_id',
                'description_id',
                'user_id']
    data.columns = colnames
    data["Target"] = data["click"].apply(lambda x: 1 if x == 0 else -1)
    data.drop(["click"], axis=1, inplace=True)
    categorical_features = {1, 2, 3, 6, 7, 8, 9, 10}
    def clean_string(s):
        return "v_" + re.sub('[^A-Za-z0-9]+', "_", str(s))

    for i in categorical_features:
        data[data.columns[i]] = data[data.columns[i]].apply(clean_string)
        data[data.columns[i]] = data[data.columns[i]].astype('category')
    return data, []


def read_kick():
    data = pd.read_csv("./data/kick/training.csv")
    target = data["IsBadBuy"].apply(lambda x: 1.0 if x == 0 else -1.0)
    data["PurchYear"] = pd.DatetimeIndex(data['PurchDate']).year
    data["PurchMonth"] = pd.DatetimeIndex(data['PurchDate']).month
    data["PurchDay"] = pd.DatetimeIndex(data['PurchDate']).day
    data["PurchWeekday"] = pd.DatetimeIndex(data['PurchDate']).weekday
    categorical_features = set([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 23, 24, 25, 26, 27, 29, 31, 32, 33, 34])
    data.drop(["RefId", "IsBadBuy", "PurchDate"], axis=1, inplace=True)
    def clean_string(s):
        return re.sub('[^A-Za-z0-9]+', "_", str(s))

    for i in categorical_features:
        data[data.columns[i]] = data[data.columns[i]].apply(clean_string)

    columns_to_impute = []
    for i, column in enumerate(data.columns):
        if i not in categorical_features and pd.isnull(data[column]).any():
            columns_to_impute.append(column)
    for column_name in columns_to_impute:
        data[column_name + "_imputed"] = pd.isnull(data[column_name]).astype(float)
        data[column_name].fillna(0, inplace=True)
    for i, column in enumerate(data.columns):
        if i not in categorical_features:
            data[column] = data[column].astype(float)
        else:
            data[column] = data[column].astype('category')

    data["Target"] = target

    return data, []

def read_adult(data_path='./data/adult/adult.data', names_path='./data/adult/adult.names'):
    with open(names_path) as f:
        lines = list(filter(lambda l: not l.startswith('|') and l.strip(), f.readlines()))
    categorial = []
    feature_name = []
    for line in lines:
        split = line.split(':')
        if len(split) == 2:
            categorial.append(not split[1].startswith('continuous'))
            feature_name.append(split[0])
    feature_name.append('Target')
    categorial.append(True)
    df = pd.read_csv(data_path, names=feature_name)
    for n, c in zip(feature_name, categorial):
        if c:
            df[n] = df[n].astype('category').cat.codes
    return df, [n for n,c in zip(feature_name, categorial) if c and n != 'Target']


def read_internet(data_path='./data/internet/kdd_internet_usage.arff'):
    data, meta = arff.loadarff(data_path)
    df = pd.DataFrame(data)
    df['Target'] = df['Who_Pays_for_Access_Work'].apply(lambda x:int(x.decode()))
    df.drop(["Who_Pays_for_Access_Work", "Willingness_to_Pay_Fees", "Years_on_Internet", "who"], axis=1, inplace=True)
    for c in df.columns:
        if c == 'Target':
            continue
        df[c] = df[c].apply(lambda x: x.decode()).astype('category')
    return df, [c for c in df.columns if c !='Target']


def _prepare_kdd(data_path):
    def to_float_str(element):
        try:
            return str(float(element))
        except ValueError:
            return element
    data = pd.read_csv(data_path, sep='\t')
    
    categorical_features = { 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228 }

    for i in categorical_features:
        data[data.columns[i]].fillna("?", inplace=True)
        data[data.columns[i]] = data[data.columns[i]].astype('category')
    columns_to_impute = []
    for i, column in enumerate(data.columns):
        if i not in categorical_features and pd.isnull(data[column]).any():
            columns_to_impute.append(column)
    for column_name in columns_to_impute:
        data[column_name].fillna(0, inplace=True)
    for i, column in enumerate(data.columns):
        if i not in categorical_features:
            data[column] = data[column].astype(float)
    return data


def read_appet(data_path='./data/appet/orange_small_train.data/orange_small_train.data',
        label_path='./data/appet/orange_small_train.data/orange_small_train_appetency.labels'):
    data = _prepare_kdd(data_path)
    with open(label_path) as f:
        data["Target"] = list(map(lambda x: (int(x) + 1)/2, f.readlines()))
    return data, []


def read_kddchurn(data_path='./data/appet/orange_small_train.data/orange_small_train.data',
        label_path='./data/appet/orange_small_train.data/orange_large_train_churn.labels'):
    data = _prepare_kdd(data_path)
    with open(label_path) as f:
        data["Target"] = list(map(lambda x: (int(x) + 1)/2, f.readlines()))
    return data, []


def read_upsel(data_path='./data/appet/orange_small_train.data/orange_small_train.data',
        label_path='./data/appet/orange_small_train.data/orange_small_train_upselling.labels'):
    data = _prepare_kdd(data_path)
    with open(label_path) as f:
        data["Target"] = list(map(lambda x: (int(x) + 1)/2, f.readlines()))
    return data, []


def read_amazon(data_path='./data/amazon/train.csv'):
    df = pd.read_csv(data_path)
    print([(c, df[c].nunique()) for c in df.columns])
    df['Target'] = df['ACTION']
    df.drop(["ACTION", "RESOURCE"], axis=1, inplace=True)
    return df, []


def read_dataset_by_name(dataset_name, *args, **kwargs):
    return {
        'adult': read_adult,
        'internet': read_internet,
        'amazon': read_amazon,
        # 'appet' : read_appet,
        # 'kddchurn': read_kddchurn,
        # 'upsel': read_upsel
        'click': read_click,
        'kick': read_kick
    }[dataset_name](*args, **kwargs)
