import torch
from torch import nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adj, x):
        """
        执行GNN消息传递。
        
        参数:
        adj (torch.sparse.FloatTensor): 邻接矩阵。
        x (torch.Tensor): 节点特征矩阵。
        
        返回:
        torch.Tensor: 消息传递后的节点特征矩阵。
        """

        # 确保 x 的尺寸匹配 adj
        if x.size(0) != adj.size(1):
            if x.size(0) < adj.size(1):
                # 填充特征矩阵
                padding = torch.zeros((adj.size(1) - x.size(0), x.size(1)), device=x.device)
                x = torch.cat([x, padding], dim=0)
            else:
                raise ValueError(f"Expected x size(0) to match adj size(1), but got {x.size(0)} and {adj.size(1)}")

        # 确保 adj 和 x 都是二维张量
        if adj.dim() != 2:
            raise ValueError(f"Expected adj to be a 2D matrix, but got {adj.dim()}D tensor")
        if x.dim() != 2:
            raise ValueError(f"Expected x to be a 2D matrix, but got {x.dim()}D tensor")

        # 执行稀疏矩阵乘法
        h = torch.spmm(adj, x)

        # 调整线性层的输入维度
        if h.size(1) != self.linear.in_features:
            self.linear = nn.Linear(h.size(1), self.linear.out_features).to(h.device)

        # 通过线性层
        h = self.linear(h)

        return F.relu(h)