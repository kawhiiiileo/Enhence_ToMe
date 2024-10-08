import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP, CLIPEncoderLayer, CLIPVisionTransformer, \
    CLIPEncoder
# from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
# from tome.utils import parse_r
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import math
from typing import Callable, Tuple, List


def do_nothing(x, mode=None):
    return x


a = 0


def bipartite_soft_matching(
        metric: torch.Tensor,
        r: int,
        class_token: bool = False,
        distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        print(f'这里是新循环')
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        print(f'scores.shape is {scores.shape}')

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        ###################################
        print(f'score is {scores}')
        node_topk, node_topk_idx = scores.topk(3, dim=-1)
        node_mean = node_topk.mean(dim=-1)  # # [A, token_even]
        # 将得分进行降序排列
        edge_idx = node_mean.argsort(dim=-1, descending=True)[..., None]  # # [A, token_even，1]

        # 使用 edge_idx 对 node_topk_idx 进行重新排序，以保持与 node_mean 排序一致
        sorted_topk_idx = node_topk_idx.gather(dim=-2, index=edge_idx.expand(-1, -1, node_topk_idx.shape[
            -1]))  # [A, token_even, top_k]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = sorted_topk_idx.gather(dim=-2,
                                         index=src_idx.expand(-1, -1, sorted_topk_idx.shape[-1]))  # [A, r, top_k]

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        epsilon = 1e-10
        sum_topk_scores = node_topk[..., :r, :].sum(dim=-1, keepdim=True) + epsilon  # 计算每个 src 对应的 top-k 得分和
        # print(f'sum_topk_scores is {sum_topk_scores}')
        merge_weights = node_topk[..., :r, :] / sum_topk_scores  # 形状为 [A, r, top_k]，即 α_i,k
        for idx_r in range(r):
            current_token = src[:, idx_r, :].unsqueeze(1)  # [n,1,c]
            current_weights = merge_weights[:, idx_r, :].unsqueeze(-1)  # [n, top_k, 1]
            # 计算加权的token
            weighted_tokens = current_token * current_weights  # [n, topk, c]
            # 获取当前token应该合并到的dst_token索引
            indices_to_add = dst_idx[:, idx_r, :]  # [n, top_k]
            for idx_topk in range(indices_to_add.shape[1]):
                dst_indices = indices_to_add[:, idx_topk].unsqueeze(1)  # [n, 1]
                expanded_indices = dst_indices.expand(n, 1, c)  # 扩展索引以匹配嵌入维度 [n, 1, c]
                # src中的token应该被分解的到topk个token应该占有的权重
                other_tensor = weighted_tokens[:, idx_topk, :].unsqueeze(1)
                index_tensor = expanded_indices
                dst.scatter_add_(1, index_tensor, other_tensor)  # 保持 token 维度

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge