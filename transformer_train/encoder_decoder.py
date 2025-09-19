import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        # 自注意力块
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)

        # post层归一化
        self.lnl = nn.LayerNorm(embed_dim)
        # 前馈层
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=ffn_hidden_dim, out_features=embed_dim),
            nn.Dropout(dropout)
        )
        # post层归一化
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 自注意力
        attn_output, _ = self.self_attn.forward(
            query=x,
            key=x,
            value=x,
            attn_mask=mask
        )
        x = x + self.dropout(attn_output)
        # 层归一化
        x = self.lnl(x)
        # 前馈
        ffn_out = self.ffn(x).forward(x)
        x = x + self.dropout(ffn_out)
        # 残差连接
        x = self.ln2(x)
        return x


if __name__ == '__main__':
    encoder_block = EncoderBlock(embed_dim=512, num_heads=8, ffn_hidden_dim=2048)
    x = torch.randn(2, 10, 512)
    encoder_output = encoder_block(x)
    print(encoder_output.shape)