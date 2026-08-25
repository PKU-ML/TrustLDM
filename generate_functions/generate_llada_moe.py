import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, set_seed
from tqdm import tqdm
# -----------------------------
# Utility
# -----------------------------
def add_gumbel_noise(logits, temperature: float):
    """
    Gumbel-max trick for categorical sampling.
    When temperature == 0, return logits unchanged (i.e., argmax).
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp_min_(1e-12)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int):
    """
    Compute how many tokens to "commit" each step so the total is evenly distributed across `steps`.
    mask_index: bool tensor [B, L_block], indicates which positions in the current block are still MASK.
    Returns: int64 tensor of shape [B, steps].
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        if remainder[i] > 0:
            num_transfer_tokens[i, : remainder[i].item()] += 1
    return num_transfer_tokens


# -----------------------------
# Core generation
# -----------------------------
@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,                 # [1, L_prompt]
    tokenizer,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 156895,
    context: str = "",
    bias: int = 64,
    short: bool = False
):
    """
    Mask-based iterative refinement generation.

    remasking ∈ {'low_confidence','margin','entropy','random','ltr','rtl'}
    - ltr: commit left-to-right within a block
    - rtl: commit right-to-left within a block
    """
    # print(prompt_ids.shape)
    device = next(model.parameters()).device
    Lp = prompt_ids.shape[1]
    if context:
        ctx_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)  # [1, L_ctx]
        Lc = ctx_ids.shape[1]
        # [prompt | MASK*gen_length | context]
        x = torch.full((1, Lp + gen_length + Lc), mask_id, dtype=torch.long, device=device)
        x[:, :Lp] = prompt_ids.clone()
        x[:, Lp + gen_length : Lp + gen_length + Lc] = ctx_ids  # ← write full context using a slice
        prompt_index = (x != mask_id)
    else:
        x = torch.full((1, Lp + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :Lp] = prompt_ids.clone()
        prompt_index = (x != mask_id)
    # print(1)

    assert gen_length % block_length == 0, "block_length must divide gen_length"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks
    # print(2)

    for num_block in range(num_blocks):
        block_start = Lp + num_block * block_length
        block_end = block_start + block_length
        # only consider the current block
        block_mask_index = (x[:, block_start:block_end] == mask_id)  # [B, L_block]
        # shrink steps if remaining MASK tokens are fewer than the default per-step amount
        steps_this_block = min(steps_per_block, int(block_mask_index.sum().item()))
        if steps_this_block == 0:
            continue

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_this_block)  # [B, steps_this_block]

        for i in range(steps_this_block):
            mask_index = (x == mask_id)

            # CFG (unsupervised)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                modelout = model(x_cat)
                # print(modelout)
                logits = modelout.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                modelout = model(x)  # [B, L, V]
                # print(modelout)
                logits = modelout.logits
            # sample/add noise then take argmax
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

            # compute a "score" / confidence (used by some remasking strategies)
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
                score = x0_p
            elif remasking == "margin":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                top2_probs, _ = torch.topk(p, 2, dim=-1)
                score = (top2_probs[..., 0] - top2_probs[..., 1]).to(logits.dtype)  # [B, L]
            elif remasking == "entropy":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                log_p = torch.log(p + 1e-12)
                entropy = -torch.sum(p * log_p, dim=-1)  # [B, L]
                score = (-entropy).to(logits.dtype)      # lower entropy means higher confidence -> use negative entropy
            elif remasking == "random":
                score = torch.rand_like(x0, dtype=logits.dtype)
            elif remasking in ("ltr", "rtl"):
                score = None  # fixed-order strategies do not need a score
            else:
                raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

            # only allow updates within the current block; later blocks are disabled
            if score is not None:
                score[:, block_end:] = -np.inf

            # only write predictions at masked positions
            x0 = torch.where(mask_index, x0, x)

            # select positions to commit in this step
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            if remasking in ("low_confidence", "margin", "entropy", "random"):
                # only positions still masked in the current block are considered for ranking
                valid = mask_index.clone()
                valid[:, :block_start] = False
                valid[:, block_end:] = False
                local_score = torch.where(valid, score, torch.tensor(-np.inf, device=score.device, dtype=score.dtype))

                for b in range(local_score.shape[0]):
                    k = int(num_transfer_tokens[b, i].item())
                    if k <= 0:
                        continue
                    _, idxs = torch.topk(local_score[b], k=k)
                    transfer_index[b, idxs] = True

            else:
                # fixed-order strategies (ignore confidence)
                base_positions = torch.arange(block_start, block_end, device=x.device)
                if remasking == "rtl":
                    base_positions = torch.flip(base_positions, dims=[0])

                for b in range(x.shape[0]):
                    k = int(num_transfer_tokens[b, i].item())
                    if k <= 0:
                        continue
                    still_mask = mask_index[b, base_positions]
                    available = base_positions[still_mask]
                    if available.numel() == 0:
                        continue
                    pick = available[:k]
                    transfer_index[b, pick] = True

            # apply this step's commits
            x[transfer_index] = x0[transfer_index]
        
        if short:
            break
        if tokenizer.eos_token_id in x[:, block_start: block_end]:
            x[:, block_end: ] =tokenizer.eos_token_id
            break

    return x