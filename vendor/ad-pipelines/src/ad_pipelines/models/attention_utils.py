from typing import Optional, Tuple, Union

import torch


AttentionInterestTokenIdx = Optional[Union[int, Tuple[int, int]]]


def normalize_attention_interest_token_idx(
    interest_token_idx: AttentionInterestTokenIdx,
) -> Optional[Tuple[int, int]]:
    if interest_token_idx is None:
        return None
    if isinstance(interest_token_idx, int):
        return interest_token_idx, interest_token_idx + 1
    if len(interest_token_idx) != 2:
        raise ValueError("attention_interest_token_idx must be an int or a (start, end) tuple")
    start, end = int(interest_token_idx[0]), int(interest_token_idx[1])
    if end <= start:
        raise ValueError("attention_interest_token_idx end must be greater than start")
    return start, end


def compute_interest_token_attention_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    interest_token_idx: AttentionInterestTokenIdx,
    num_prefix_tokens: int,
    scaling: float,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    token_range = normalize_attention_interest_token_idx(interest_token_idx)
    if token_range is None:
        if dtype is not None:
            query = query.to(dtype=dtype)
            key = key.to(dtype=dtype)
        logits = torch.matmul(query, key.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
            logits = logits + attention_mask
        return logits

    start, end = token_range
    query_slice = query[:, :, start:end]
    key_slice = key[:, :, num_prefix_tokens:]
    if dtype is not None:
        query_slice = query_slice.to(dtype=dtype)
        key_slice = key_slice.to(dtype=dtype)
    logits = torch.matmul(query_slice, key_slice.transpose(-1, -2)) * float(scaling)

    if attention_mask is not None:
        mask_slice = attention_mask[:, :, start:end, num_prefix_tokens : key.shape[-2]]
        logits = logits + mask_slice

    return logits
