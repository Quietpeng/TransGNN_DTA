from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE

# 设置全局变量，药物最大位置，目标最大位置
D_MAX = 50
T_MAX = 545

# 药物词汇路径
drug_vocab_path = './vocabulary/drug_bpe_chembl_freq_100.txt'
drug_codes_bpe = codecs.open(drug_vocab_path)
drug_bpe = BPE(drug_codes_bpe, merges=-1, separator='')
drug_temp = pd.read_csv('./vocabulary/subword_list_chembl_freq_100.csv')
drug_index2word = drug_temp['index'].values
drug_idx = dict(zip(drug_index2word, range(0, len(drug_index2word))))

# 目标词汇路径
target_vocab_path = './vocabulary/target_bpe_uniprot_freq_500.txt'
target_codes_bpe = codecs.open(target_vocab_path)
target_bpe = BPE(target_codes_bpe, merges=-1, separator='')
target_temp = pd.read_csv('./vocabulary/subword_list_uniprot_freq_500.csv')
target_index2word = target_temp['index'].values
target_idx = dict(zip(target_index2word, range(0, len(target_index2word))))

def drug_encoder(input_smiles):
    """
    药物编码器

    参数:
        input_smiles: 输入的药物序列。

    返回:
        v_d: 填充后的药物序列。
        temp_mask_d: 掩码后的药物序列。
    """
    # 使用BPE处理药物序列并分割成词汇
    temp_d = drug_bpe.process_line(input_smiles).split()
    try:
        idx_d = np.asarray([drug_idx[i] for i in temp_d])
    except:
        idx_d = np.array([0])

    # 计算序列长度标志
    flag = len(idx_d)
    if flag < D_MAX:
        v_d = np.pad(idx_d, (0, D_MAX - flag), 'constant', constant_values=0)
        temp_mask_d = [1] * flag + [0] * (D_MAX - flag)
    else:
        v_d = idx_d[:D_MAX]
        temp_mask_d = [1] * D_MAX
    
    return v_d, np.asarray(temp_mask_d)

def target_encoder(input_seq):
    """
    目标编码器

    参数:
        input_seq: 输入的目标序列。

    返回:
        v_t: 填充后的目标序列。
        temp_mask_t: 掩码后的目标序列。
    """
    # 使用BPE处理目标序列并分割成词汇
    temp_t = target_bpe.process_line(input_seq).split()
    try:
        idx_t = np.asarray([target_idx[i] for i in temp_t])
    except:
        idx_t = np.array([0])

    # 计算序列长度标志
    flag = len(idx_t)
    if flag < T_MAX:
        v_t = np.pad(idx_t, (0, T_MAX - flag), 'constant', constant_values=0)
        temp_mask_t = [1] * flag + [0] * (T_MAX - flag)
    else:
        v_t = idx_t[:T_MAX]
        temp_mask_t = [1] * T_MAX

    return v_t, np.asarray(temp_mask_t)

def concordance_index1(y, f):
    """
    计算一致性指数（CI）

    参数:
        y (ndarray): 1维ndarray，代表地面真实Kd值。
        f (ndarray): 1维ndarray，代表模型预测的Kd值。

    返回:
        ci (float): 一致性指数。
    """
    # 对真实Kd值进行排序
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    # 初始化索引和累加器
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    # 计算一致性指数
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

class DataEncoder(Dataset):
    """
    数据编码器
    """
    def __init__(self, ids, label, dti_data):
        """
        初始化方法
        """
        super(DataEncoder, self).__init__()
        self.ids = ids
        self.label = label
        self.data = dti_data

    def __len__(self):
        """
        获取数据集大小
        """
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        获取药物和目标的嵌入向量，标签
        """
        idx = self.ids[idx]
        # 获取SMILES序列和目标序列
        d_input = self.data.iloc[idx]['SMILES']
        t_input = self.data.iloc[idx]['Target Sequence']
        res = []

        # 对药物序列进行编码和掩码
        d_out, mask_d_out = drug_encoder(d_input)
        res.append(d_out)
        res.append(mask_d_out)
        # 对目标序列进行编码和掩码
        t_out, mask_t_out = target_encoder(t_input)
        res.append(t_out)
        res.append(mask_t_out)

        # 获取标签
        labels = self.label[idx]
        res.append(labels)
        return res

class DataEncoderTest(Dataset):
    """
    测试数据编码器
    """
    def __init__(self, ids, dti_data):
        """
        初始化方法
        """
        super(DataEncoderTest, self).__init__()
        self.ids = ids
        self.data = dti_data

    def __len__(self):
        """
        获取数据集大小
        """
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        获取药物和目标的嵌入向量
        """
        idx = self.ids[idx]
        # 获取SMILES序列和目标序列
        d_input = self.data.iloc[idx]['SMILES']
        t_input = self.data.iloc[idx]['Target Sequence']
        res = []

        # 对药物序列进行编码和掩码
        d_out, mask_d_out = drug_encoder(d_input)
        res.append(d_out)
        res.append(mask_d_out)
        # 对目标序列进行编码和掩码
        t_out, mask_t_out = target_encoder(t_input)
        res.append(t_out)
        res.append(mask_t_out)
        return res