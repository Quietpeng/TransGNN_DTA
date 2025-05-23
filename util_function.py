import pandas as pd
import numpy as np
import os
import json
import random
import pickle
from itertools import repeat
from collections import OrderedDict
from preprocess import drug_encoder, target_encoder
import torch
from torch.utils.data import Dataset, DataLoader

def convert_y_unit(y, from_, to_):
    """
    转换Kd/pKd单位

    参数:
    y (int或float或numpy.array): 需要转换的Kd值或数组
    from_ (str): 原始单位，可以是'nM'或'p'
    to_ (str): 目标单位，可以是'nM'或'p'

    返回:
    转换后的Kd值或数组
    """
    array_flag = False
    if isinstance(y, (int, float)):
        y = np.array([y])
        array_flag = True
    y = y.astype(float)    
    # 基础单位为纳摩尔每升(nM)
    if from_ == 'nM':
        y = y
    elif from_ == 'p':
        y = 10 ** (-y) / 1e-9
        
    if to_ == 'p':
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10  # 避免对数运算中的0值
        y = -np.log10(y * 1e-9)
    elif to_ == 'nM':
        y = y
        
    if array_flag:
        return y[0]
    return y


def length_func(list_or_tensor):
    """
    获取列表或张量的长度

    参数:
    list_or_tensor (list或tensor): 需要获取长度的列表或张量

    返回:
    int: 列表或张量的长度
    """
    if type(list_or_tensor) == list:
        return len(list_or_tensor)
    return list_or_tensor.shape[0]


def load_davis_dataset():
    """
    加载DAVIS基准回归数据集，该数据集用于药物与靶点亲和力的回归任务。

    函数将从文件中读取训练集和测试集的划分信息、药物的SMILES字符串、靶点蛋白序列以及亲和力值。
    然后，将亲和力值转换为对数值，并构建包含SMILES字符串、靶点蛋白序列和亲和力值的数据集。
    最终，函数返回包含训练集和测试集的数据列表。

    返回:
    train_test_dataset (list): 包含训练集和测试集的数据列表，每个元素是一个包含多个字典的列表，
                             每个字典包含一个样本的'smiles'、'protein'和'aff'键值对。
    """
    # 读取训练集划分文件
    trainn_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'folds', 'train_fold_setting1.txt')))
    train_fold = []
    for e in zip(*trainn_fold):
        for ee in e:
            train_fold.append(ee)
    
    test_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'folds', 'test_fold_setting1.txt')))
    
    ligands = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'ligands_can.txt')),
        object_pairs_hook=OrderedDict)
    proteins = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'proteins.txt')),
        object_pairs_hook=OrderedDict)
    
    affinity = pickle.load(open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'Y'), 
                                'rb'), encoding='latin1')
    smiles_lst, protein_lst = [], []

    for k in ligands.keys():
        smiles = ligands[k]
        smiles_lst.append(smiles)
    for k in proteins.keys():
        protein_lst.append(proteins[k])

    affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    
    os.makedirs(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'processed'), exist_ok=True)
    train_test_dataset = []
    for split in ['train', 'test']:
        split_dir = os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'processed', split)
        os.makedirs(split_dir, exist_ok=True)
        fold = train_fold if split == 'train' else test_fold
        rows, cols = np.where(np.isnan(affinity) == False)
        rows, cols = rows[fold], cols[fold]
        
        data_lst = [[] for _ in range(1)]
        for idx in range(len(rows)):
            data = {}
            data['smiles'] = smiles_lst[rows[idx]]
            data['protein'] = protein_lst[cols[idx]]
            af = affinity[rows[idx], cols[idx]]
            data['aff'] = af

            data_lst[idx % 1].append(data)
        random.shuffle(data_lst)
        train_test_dataset.append(data_lst[0])
    return train_test_dataset


def load_kiba_dataset():
    """
    加载Kiba基准回归数据集

    返回:
    train_test_dataset (list): 包含训练集和测试集的数据列表
    """
    trainn_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'folds', 'train_fold_setting1.txt')))
    train_fold = []
    for e in zip(*trainn_fold):
        for ee in e:
            train_fold.append(ee)
    test_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'folds', 'test_fold_setting1.txt')))
    ligands = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'ligands_can.txt')),
        object_pairs_hook=OrderedDict)
    proteins = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'proteins.txt')),
        object_pairs_hook=OrderedDict)
    
    affinity = pickle.load(open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'Y'), 
                                'rb'), encoding='latin1')
    smiles_lst, protein_lst = [], []

    for k in ligands.keys():
        smiles = ligands[k]
        smiles_lst.append(smiles)
    for k in proteins.keys():
        protein_lst.append(proteins[k])

    affinity = np.asarray(affinity)
    
    os.makedirs(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'processed'), exist_ok=True)
    train_test_dataset = []
    for split in ['train', 'test']:
        split_dir = os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'processed', split)
        os.makedirs(split_dir, exist_ok=True)
        fold = train_fold if split == 'train' else test_fold
        rows, cols = np.where(np.isnan(affinity) == False)
        rows, cols = rows[fold], cols[fold]
        
        data_lst = [[] for _ in range(1)]
        for idx in range(len(rows)):
            data = {}
            data['smiles'] = smiles_lst[rows[idx]]
            data['protein'] = protein_lst[cols[idx]]
            af = affinity[rows[idx], cols[idx]]
            data['aff'] = af

            data_lst[idx % 1].append(data)
        random.shuffle(data_lst)
        train_test_dataset.append(data_lst[0])
    return train_test_dataset


def load_DAVIS(convert_to_log=True):
    """
    加载原始DAVIS数据集

    参数:
    convert_to_log (bool): 是否将亲和力值转换为对数值，默认为True

    返回:
    SMILES (numpy.array): 药物的SMILES字符串数组
    Target_seq (numpy.array): 靶点序列数组
    y (numpy.array): 亲和力值数组
    """
    affinity = pd.read_csv('./dataset/regression/DAVIS/affinity.txt', header=None, sep=' ')
    with open('./dataset/regression/DAVIS/target_seq.txt') as f1:
        target = json.load(f1)
    with open('./dataset/regression/DAVIS/SMILES.txt') as f2:
        drug = json.load(f2)
        
    target = list(target.values())
    drug = list(drug.values())
    
    SMILES = []
    Target_seq = []
    y = []
    
    for i in range(len(drug)):
        for j in range(len(target)):
            SMILES.append(drug[i])
            Target_seq.append(target[j])
            y.append(affinity.values[i, j])
            
    if convert_to_log:
        y = convert_y_unit(np.array(y), 'nM', 'p')
    else:
        y = y
    return np.array(SMILES), np.array(Target_seq), np.array(y)


def load_KIBA():
    """
    加载原始KIBA数据集

    返回:
    SMILES (numpy.array): 药物的SMILES字符串数组
    Target_seq (numpy.array): 靶点序列数组
    y (numpy.array): 亲和力值数组
    """
    affinity = pd.read_csv('./dataset/regression/KIBA/affinity.txt', header=None, sep='\t')
    affinity = affinity.fillna(-1)
    with open('./dataset/regression/KIBA/target_seq.txt') as f:
        target = json.load(f)
    with open('./dataset/regression/KIBA/SMILES.txt') as f:
        drug = json.load(f)
        
    target = list(target.values())
    drug = list(drug.values())
    
    SMILES = []
    Target_seq = []
    y = []
    
    for i in range(len(drug)):
        for j in range(len(target)):
            if affinity.values[i, j] != -1:
                SMILES.append(drug[i])
                Target_seq.append(target[j])
                y.append(affinity.values[i, j])

    y = y
    return np.array(SMILES), np.array(Target_seq), np.array(y)

def load_ChEMBL_pkd():
    """
    加载原始的ChEMBL数据集，其中包含pKd值。

    该函数读取ChEMBL数据集中的亲和力值、靶点序列和药物SMILES字符串，然后进行预处理，包括填充缺失值、
    将亲和力值从pKd转换为nM（如果需要），并返回处理后的数据。

    返回:
    np.array: 处理后的药物SMILES字符串数组。
    np.array: 处理后的靶点序列数组。
    np.array: 处理后的亲和力值数组。
    """
    affinity = pd.read_csv('./dataset/regression/ChEMBL/Chem_Affinity.txt', header=None)
    affinity = affinity.fillna(-1)
    target = pd.read_csv('./dataset/regression/ChEMBL/ChEMBL_Target_Sequence.txt', header=None)
    drug = pd.read_csv('./dataset/regression/ChEMBL/Chem_SMILES_only.txt', header=None)
    
    SMILES = []
    Target = []
    y = []
    drugcnt = []
    
    for i in range(len(target)):
        Target.append(target[0][i])
        y.append(affinity[0][i])
        SMILES.append(drug[0][i])

    # 处理亲和力值和SMILES字符串
    aff = []
    total = []
    for i in range(len(SMILES)):
        drugcnt.append(len(SMILES[i].split()))
    for i in aff:
        total += i
    smile = []
    for segments in SMILES:
        for x in segments.split():
            smile.extend(x)
    
    smiles_res = []
    y_tmp = []
    target_res = []
    tmp = []
    
    for i in range(len(drugcnt)):
        tmp.extend(repeat(Target[i], drugcnt[i]))
    for i in range(len(total)):
        if total[i] != '-1':
            y_tmp.append(total[i])
            smiles_res.append(smile[i])
            target_res.append(tmp[i])

    y_res = [float(i) for i in y_tmp]
    return np.array(smiles_res), np.array(target_res), np.array(y_res)


def load_ChEMBL_kd():
    """
    加载原始的ChEMBL数据集，其中包含Kd值。

    该函数读取ChEMBL数据集中的亲和力值、靶点序列和药物SMILES字符串，然后进行预处理，包括填充缺失值、
    将亲和力值从Kd转换为pKd（如果需要），并返回处理后的数据。

    返回:
    np.array: 处理后的药物SMILES字符串数组。
    np.array: 处理后的靶点序列数组。
    np.array: 处理后的亲和力值数组。
    """
    affinity = pd.read_csv('./dataset/regression/ChEMBL/Chem_Kd_nM.txt', header=None)
    target = pd.read_csv('./dataset/regression/ChEMBL/ChEMBL_Target_Sequence.txt', header=None)
    drug = pd.read_csv('./dataset/regression/ChEMBL/Chem_SMILES_only.txt', header=None)
    
    SMILES = []
    Target = []
    y = []
    drugcnt = []
    
    for i in range(len(target)):
        Target.append(target[0][i])
        y.append(affinity[0][i])
        SMILES.append(drug[0][i])

    aff = []
    total = []
    for i in range(len(SMILES)):
        drugcnt.append(len(SMILES[i].split()))
    for i in range(len(y)):
        aff.insert(i, y[i].split(" "))
    for i in aff:
        total += i
    smile = []
    for segments in SMILES:
        for x in segments.split():
            smile.extend(x)
    
    smiles_res = []
    y_tmp = []
    target_res = []
    tmp = []
    
    for i in range(len(drugcnt)):
        tmp.extend(repeat(Target[i], drugcnt[i]))
    for i in range(len(total)):
        if total[i] != '-1':
            y_tmp.append(total[i])
            smiles_res.append(smile[i])
            target_res.append(tmp[i])

    y_res = [float(i) for i in y_tmp]
    y_res = convert_y_unit(np.array(y_res), 'nM', 'p')
    return np.array(smiles_res), np.array(target_res), np.array(y_res)


def load_BindingDB_kd():
    """
    加载原始的BindingDB数据集，其中包含Kd值。

    该函数读取BindingDB数据集中的亲和力值、靶点序列和药物SMILES字符串，然后进行预处理，包括填充缺失值、
    将亲和力值从Kd转换为pKd（如果需要），并返回处理后的数据。

    返回:
    np.array: 处理后的药物SMILES字符串数组。
    np.array: 处理后的靶点序列数组。
    np.array: 处理后的亲和力值数组。
    """
    affinity = pd.read_csv('./dataset/regression/BindingDB/BindingDB_Kd.txt', header=None)
    target = pd.read_csv('./dataset/regression/BindingDB/BindingDB_Target_Sequence_new.txt', header=None)
    drug = pd.read_csv('./dataset/regression/BindingDB/BindingDB_SMILES_new.txt', header=None)
    
    SMILES = []
    Target = []
    y = []
    drugcnt = []
    
    for i in range(len(target)):
        Target.append(target[0][i])
        y.append(affinity[0][i])
        SMILES.append(drug[0][i])

    aff = []
    total = []
    for i in range(len(SMILES)):
        drugcnt.append(len(SMILES[i].split()))
    for i in range(len(y)):
        aff.insert(i, y[i].split(" "))
    for i in aff:
        total += i
    smile = []
    for segments in SMILES:
        for x in segments.split():
            smile.extend(x)
    
    smiles_res = []
    y_tmp = []
    target_res = []
    tmp = []
    
    for i in range(len(drugcnt)):
        tmp.extend(repeat(Target[i], drugcnt[i]))
    for i in range(len(total)):
        if total[i] != '-1':
            y_tmp.append(total[i])
            smiles_res.append(smile[i])
            target_res.append(tmp[i])

    y_res = [float(i) for i in y_tmp]
    y_res = convert_y_unit(np.array(y_res), 'nM', 'p')
    return np.array(smiles_res), np.array(target_res), np.array(y_res)


def data_process(X_drug, X_target, y, frac, drug_encoding='Transformer', target_encoding='Transformer', 
                 split_method='protein_split', random_seed=1, sample_frac=1, mode='DTI'):
    """
    对原始数据进行预处理。

    参数:
    X_drug (list或numpy.array): 药物SMILES字符串列表或数组。
    X_target (list或numpy.array): 靶点序列列表或数组。
    y (list或numpy.array): 亲和力值列表或数组。
    frac (tuple): 训练集、验证集和测试集的比例。
    drug_encoding (str): 药物编码方式，默认为'Transformer'。
    target_encoding (str): 靶点编码方式，默认为'Transformer'。
    split_method (str): 数据集划分方法，默认为'protein_split'。
    random_seed (int): 随机种子，默认为1。
    sample_frac (float): 样本抽取比例，默认为1。
    mode (str): 模式，默认为'DTI'。

    返回:
    train (DataFrame): 训练集DataFrame。
    val (DataFrame): 验证集DataFrame。
    test (DataFrame): 测试集DataFrame。
    """
    if isinstance(X_target, str):
        X_target = [X_target]
    if len(X_target) == 1:
        X_target = np.tile(X_target, (length_func(X_drug), ))

    df_data = pd.DataFrame(zip(X_drug, X_target, y))
    df_data.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)
    
    if sample_frac != 1:
        df_data = df_data.sample(frac = sample_frac).reset_index(drop = True)
    
    # 删除重复项
    df_data = df_data.drop_duplicates()
    # 仅保留唯一的蛋白质+靶点对，保留标签值最大的项
    d_t = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).index.tolist())
    label = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).tolist())
    df_data = pd.concat([d_t, label], 1)
    df_data.columns = ['Target Sequence', 'SMILES', 'Label']

    # 对药物和靶点应用BPE编码
    df_data['drug_encoding'] = df_data['SMILES'].apply(drug_encoder)
    df_data['target_encoding'] = df_data['Target Sequence'].apply(target_encoder)
    
    # 数据集划分
    if split_method == 'random_split': 
        train, val, test = random_split_dataset(df_data, random_seed, frac)
    elif split_method == 'drug_split':
        train, val, test = drug_split_dataset(df_data, random_seed, frac)
    elif split_method == 'protein_split':
        train, val, test = protein_split_dataset(df_data, random_seed, frac)
    elif split_method == 'no_split':
        return df_data.reset_index(drop=True)
    else:
        raise AttributeError("请选择一种划分方法：random, cold_drug, cold_target!")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def data_process_whole(X_drug, X_target, y, drug_encoding='Transformer',
    target_encoding='Transformer', mode='DTI'):
    """
    对原始测试数据进行预处理。
    参数:
    X_drug (list或numpy.array): 药物SMILES字符串列表或数组。
    X_target (list或numpy.array): 靶点序列列表或数组。
    y (list或numpy.array): 亲和力值列表或数组。
    drug_encoding (str): 药物编码方式，默认为'Transformer'。
    target_encoding (str): 靶点编码方式，默认为'Transformer'。
    mode (str): 模式，默认为'DTI'。

    返回:
    DataFrame: 预处理后的DataFrame。
    """
    if isinstance(X_target, str):
        X_target = [X_target]
    if len(X_target) == 1:
        X_target = np.tile(X_target, (length_func(X_drug), ))
        
    df_data = pd.DataFrame(zip(X_drug, X_target, y))
    df_data.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)

    # 删除重复项
    df_data = df_data.drop_duplicates()
    # 仅保留唯一的蛋白质+靶点对，保留标签值最大的项
    d_t = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).index.tolist())
    label = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).tolist())
    df_data = pd.concat([d_t, label], 1)
    df_data.columns = ['Target Sequence', 'SMILES', 'Label']

    # 对药物和靶点应用BPE编码
    df_data['drug_encoding'] = df_data['SMILES'].apply(drug_encoder)
    df_data['target_encoding'] = df_data['Target Sequence'].apply(target_encoder)

    return df_data.reset_index(drop=True)


def random_split_dataset(df, fold_seed, frac):
    """
    随机划分数据集。
    参数:
    df (DataFrame): 原始DataFrame。
    fold_seed (int): 随机种子。
    frac (tuple): 训练集、验证集和测试集的比例。

    返回:
    train (DataFrame): 训练集DataFrame。
    val (DataFrame): 验证集DataFrame。
    test (DataFrame): 测试集DataFrame。
    """
    _, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
    train = train_val[~train_val.index.isin(val.index)]

    return train, val, test


def drug_split_dataset(df, fold_seed, frac):
    """
    按药物划分数据集。
    参数:
    df (DataFrame): 原始DataFrame。
    fold_seed (int): 随机种子。
    frac (tuple): 训练集、验证集和测试集的比例。

    返回:
    train (DataFrame): 训练集DataFrame。
    val (DataFrame): 验证集DataFrame。
    test (DataFrame): 测试集DataFrame。
    """
    _, val_frac, test_frac = frac
    drug_drop = df['SMILES'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values
    test = df[df['SMILES'].isin(drug_drop)]
    train_val = df[~df['SMILES'].isin(drug_drop)]

    drug_drop_val = train_val['SMILES'].drop_duplicates().sample(frac=val_frac / (1 - test_frac), 
                        replace=False, random_state=fold_seed).values
    val = train_val[train_val['SMILES'].isin(drug_drop_val)]
    train = train_val[~train_val['SMILES'].isin(drug_drop_val)]

    return train, val, test

