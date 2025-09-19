import torch
import math
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj_weight = nn.Linear(embed_dim, embed_dim)
        self.k_proj_weight = nn.Linear(embed_dim, embed_dim)
        self.v_proj_weight = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        b, l, d = x.shape
        q, k, v = self.q_proj_weight(x), self.k_proj_weight(x), self.v_proj_weight(x)

        # q,k 计算内积，并除以sqrt(d)
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        # 应用mask
        if mask is not None:
            x = x.masked_fill(mask == 0, float('-inf'))

        # softmax 得到概率分布
        x = F.softmax(x, dim=-1)
        x = self.dropout(x)
        x = torch.matmul(x, v)
        return self.output(x)


if __name__ == '__main__':

    attn = SelfAttention(embed_dim=8)
    x = torch.randn(2, 10, 8)
    print(x)
    print(attn(x).shape)
