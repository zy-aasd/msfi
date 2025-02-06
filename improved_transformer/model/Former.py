import math

import torch
from timm.models.vision_transformer import _cfg, Mlp, Block
from torch import nn
class MultiCrossAttentionBlock(nn.Module):
    def __init__(self, num_patches, dim_s, dim_m, dim_l, qkv_bias=False):
        super().__init__()
        # 定义 Q, K, V 权重矩阵
        self.qkv_bias = qkv_bias
        # 对 CLS_s 和 CLS_m 进行投影，f(.) 将 CLS 投影到对应的 patch 维度
        self.f_s = nn.Linear(dim_s, dim_m)  # 将 CLS^s 投影到 P_m 的维度 384-》192
        self.f_m = nn.Linear(dim_m, dim_l)  # 将 CLS^m 投影到 P_l 的维度 192-》96
        # 定义 W_q 和 W_k，用于 CLS 和 Patch tokens 之间的交互
        self.W_q1 = nn.Linear(dim_m, dim_m, bias=qkv_bias)  # 查询 (CLS_s -> P_m 维度)
        self.W_k1 = nn.Linear(dim_m, dim_m, bias=qkv_bias)  # 键 (CLS_s 和 P_m)
        self.W_q2 = nn.Linear(dim_l, dim_l, bias=qkv_bias)  # 查询 (CLS_m -> P_l 维度)
        self.W_k2 = nn.Linear(dim_l, dim_l, bias=qkv_bias)  # 键 (CLS_m 和 P_l)
        self.d_k1 = int(math.sqrt(dim_m))
        self.d_k2 = int(math.sqrt(dim_l))
        self.d_k3 = int(math.sqrt(dim_s))
        # 保持attn1和attn2在同一维度
        self.attention_proj_1 = nn.Linear(num_patches[1] + 1, num_patches[0] + 2)  # 将 attention_1 从 1 + patch_num_m 投影到 1 + patch_num_l
        # 定义用于 V 的线性层
        self.W_v = nn.Linear(dim_l, dim_l, bias=qkv_bias)  # 值
        # softmax 和加权系数
        self.softmax = nn.Softmax(dim=-1)
        self.weight = [0.3, 0.7, 1.0]  # 加权系数
        # g^s(.) 特征融合函数
        self.g_s = nn.Linear(dim_l, dim_s) # 96 -》384

    def forward(self, x):
        # 分别获取三个尺度的 CLS 和 Patch tokens
        p_l = x[0]  # B, patch_num_l, dim_l
        cls_m, p_m = x[1][:, 0:1, :], x[1][:, 1:, :]  # B, 1, dim_m 和 B, patch_num_m, dim_m
        cls_s, p_s = x[2][:, 0:1, :], x[2][:, 1:, :]  # B, 1, dim_s 和 B, patch_num_s, dim_s

        # Step 1: 将 CLS_s 投影到 P_m 的维度
        cls_s_proj = self.f_s(cls_s)  # 将 CLS_s 投影到 P_m 的维度，B, 1, dim_m

        # Step 2: CLS_s 作为 W_q1 的查询，CLS_s 与 P_m 连接作为 W_k1 的键
        w_q1 = self.W_q1(cls_s_proj)  # B, 1, dim_m --- CLS_s 投影后的查询
        w_k1 = self.W_k1(torch.cat([cls_s_proj, p_m], dim=1))  # B, (1 + patch_num_m), dim_m ----- CLS_s 与 P_m 的键
        attention_1 = self.softmax(torch.bmm(w_q1, w_k1.transpose(1, 2)) / self.d_k1)  # B, 1, (1 + patch_num_m)
        attention_1 = self.attention_proj_1(attention_1) # B, 1, (1 + patch_num_l)
        # Step 3: 将 CLS_m 投影到 P_l 的维度
        cls_m_proj = self.f_m(cls_m)  # 将 CLS_m 投影到 P_l 的维度，B, 1, dim_l

        # Step 4: CLS_m 作为 W_q2 的查询，CLS_m 与 P_l 连接作为 W_k2 的键 wv的值
        w_q2 = self.W_q2(cls_m_proj)  # B, 1, dim_l --- CLS_m 投影后的查询
        w_k2 = self.W_k2(torch.cat([cls_m_proj, p_l], dim=1))  # B, (1 + patch_num_l), dim_l，----- CLS_m 与 P_l 的键
        attention_2 = self.softmax(torch.bmm(w_q2, w_k2.transpose(1, 2)) / self.d_k2)  # B, 1, (1 + patch_num_l)

        # Step 5: 计算值 (Value)
        w_v = self.W_v(torch.cat([cls_m_proj, p_l], dim=1))  # B, (1 + patch_num_l), dim_l

        # Step 6: 使用注意力权重 进行加权
        weighted_attn = self.weight[0] * attention_1 + self.weight[1] * attention_2 # B, 1, (1 + patch_num_l)
        output = torch.bmm(weighted_attn, w_v * self.weight[2]) # 最终的 cls 输出  B 1 dim_l

        # Step 7: g^s(.) 特征融合
        g_s = self.g_s(output)  # B, 1, dim_s

        # 将p_s作为查询，g_s 作为k和v
        w_q = p_s  # B, N, C
        w_k = w_v = g_s  # B, 1, C
        p_s = self.softmax(torch.bmm(w_q, w_k.transpose(1, 2) / self.d_k3)) # dim_s = 384 = B,N,1
        p_s = torch.bmm(p_s, w_v) # B,N,1 * B,1,C = B,N,C

        output = torch.cat((g_s, p_s), dim=1)  # 融合后的cls token输出 B, patch_num_s+1, dim_s
        return output

if __name__ == '__main__':
    cross = MultiCrossAttentionBlock(num_patches=[3136,784,192], dim_s=384, dim_m=192,
                                          dim_l=96, qkv_bias=True)
    tensors = []
    tensors.append(torch.randn(8, 3137, 96))
    tensors.append(torch.randn(8, 785, 192))
    tensors.append(torch.randn(8, 197, 384))
    print(cross(tensors).shape)


class Former(nn.Module):
    def __init__(self, qkv_bias=False, num_patches=None, drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        # 位置编码embed
        self.num_branches = 3
        embed_dims = [96, 192, 384]
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dims[i]))
                                           for i in range(self.num_branches)])
        for i in range(len(embed_dims)):
            self.norms.append(norm_layer(embed_dims[i]))
        self.cross = MultiCrossAttentionBlock(num_patches = num_patches, dim_s=embed_dims[2], dim_m=embed_dims[1], dim_l=embed_dims[0], qkv_bias=qkv_bias)
        self.mlp = Mlp(embed_dims[2], embed_dims[2] * 3)
        self.drop = nn.Dropout(drop)
    def forward(self, x_sml):
        for i in range(self.num_branches):
            x_sml[i] += self.pos_embed[i]
            x_sml[i] = self.norms[i](x_sml[i])
        x = self.cross(x_sml)
        x = self.mlp(x) + x
        x = self.drop(x)
        x = self.norms[-1](x)
        return x
