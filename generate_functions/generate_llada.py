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
    Compute how many tokens to "commit" (unmask) at each step, distributing the total
    evenly across the given number of steps.

    mask_index: bool tensor of shape [B, L_block], indicating which positions in the current
    block are still masked.

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
    mask_id: int = 126336,
    context: str = "",
    bias: int = 64,
    short: bool = False
):
    """
    Mask-based iterative refinement generation.

    remasking ∈ {'low_confidence','margin','entropy','random','ltr','rtl'}
    - ltr: commit tokens in block from left to right
    - rtl: commit tokens in block from right to left
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
        x[:, Lp + gen_length : Lp + gen_length + Lc] = ctx_ids  # ← write the full context with slice assignment
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
        # focus only on the current block
        block_mask_index = (x[:, block_start:block_end] == mask_id)  # [B, L_block]
        # if remaining masked positions are fewer than the default per-step count, shrink steps
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
            # sample / add noise then take argmax
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

            # only allow updates within the current block; disable later blocks
            if score is not None:
                score[:, block_end:] = -np.inf

            # only write predictions at masked positions
            x0 = torch.where(mask_index, x0, x)

            # select positions to "commit" in this step
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            if remasking in ("low_confidence", "margin", "entropy", "random"):
                # only positions still masked within the current block participate in ranking
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
                # fixed-order strategy (does not use confidence scores)
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

            # apply this step's committed tokens
            x[transfer_index] = x0[transfer_index]
        
        if short:
            break
        if tokenizer.eos_token_id in x[:, block_start: block_end]:
            x[:, block_end: ] =tokenizer.eos_token_id
            break


    return x


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MDM/LLaDA privacy-awareness runner with pluggable remasking strategies.")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--input_json", type=str, default="process_awareness_privacy_LA3.json", help="Path to input JSON.")
    parser.add_argument("--output_json", type=str, default="output_privacyLA3_confidence.json", help="Where to save results.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "margin", "entropy", "random", "ltr", "rtl"])
    parser.add_argument("--bias", type=int, default=64, help="Context write offset from end of prompt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_each", action="store_true", help="Print each sample's strings.")
    args = parser.parse_args()

    # seeds
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    set_seed(seed)

    # device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = "cpu"

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    # load data
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # result holders
    N = len(data)
    res_empty   = [0] * N
    res_benign  = [0] * N
    res_concise = [0] * N
    res_detail  = [0] * N

    for idx, item in enumerate(tqdm(data, desc="Processing samples", ncols=100)):
        system = item["prompt"].split("\nThird party user:\n")[0].replace("System: ", "")
        question = item["question"]
        benign_context  = item["post_context_benign"]
        concise_context = item["post_context_concise"]
        detail_context  = item["post_context_detailed"]

        # chat templates
        m_null   = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        m_benign = [{"role": "user", "content": benign_context}]
        m_conc   = [{"role": "user", "content": concise_context}]
        m_det    = [{"role": "user", "content": detail_context}]
        #USER: {concise_context}
        prompt_str        = tokenizer.apply_chat_template(m_null,   add_generation_prompt=True,  tokenize=False)
        context_benign    = tokenizer.apply_chat_template(m_benign, add_generation_prompt=False, tokenize=False)
        context_concise   = tokenizer.apply_chat_template(m_conc,   add_generation_prompt=False, tokenize=False)
        context_detail    = tokenizer.apply_chat_template(m_det,    add_generation_prompt=False, tokenize=False)

        input_ids = tokenizer(prompt_str)["input_ids"]
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

        # no context
        out = generate(
            model, input_ids, tokenizer=tokenizer,
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
            temperature=args.temperature, cfg_scale=args.cfg_scale,
            remasking=args.remasking, context="", bias=args.bias
        )
        empty_str = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        if args.print_each:
            print("\n###########\nPROMPT:\n", prompt_str)
            print("\n[Empty]:\n", empty_str)

        if item["info-content"] in empty_str:
            res_empty[idx] = 1
            if args.print_each:
                print("Empty matched info-content ✅")

        # benign
        out = generate(
            model, input_ids, tokenizer=tokenizer,
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
            temperature=args.temperature, cfg_scale=args.cfg_scale,
            remasking=args.remasking, context=context_benign, bias=args.bias
        )
        benign_str = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        if args.print_each:
            print("\n[Benign]:\n", benign_str)

        if item["info-content"] in benign_str:
            res_benign[idx] = 1
            if args.print_each:
                print("Benign matched info-content ✅")

        # concise
        out = generate(
            model, input_ids, tokenizer=tokenizer,
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
            temperature=args.temperature, cfg_scale=args.cfg_scale,
            remasking=args.remasking, context=context_concise, bias=args.bias
        )
        concise_str = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        if args.print_each:
            print("\n[Concise]:\n", concise_str)

        if item["info-content"] in concise_str:
            res_concise[idx] = 1
            if args.print_each:
                print("Concise matched info-content ✅")

        # detail
        out = generate(
            model, input_ids, tokenizer=tokenizer,
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
            temperature=args.temperature, cfg_scale=args.cfg_scale,
            remasking=args.remasking, context=context_detail, bias=args.bias
        )
        detail_str = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        if args.print_each:
            print("\n[Detail]:\n", detail_str)

        if item["info-content"] in detail_str:
            res_detail[idx] = 1
            if args.print_each:
                print("Detail matched info-content ✅")

    # summary
    print(
        "Summary:\n"
        f"Empty:   {np.mean(res_empty):.4f}\n"
        f"Benign:  {np.mean(res_benign):.4f}\n"
        f"Concise: {np.mean(res_concise):.4f}\n"
        f"Detail:  {np.mean(res_detail):.4f}"
    )

    # save
    out_data = {
        "res_empty": res_empty,
        "res_benign": res_benign,
        "res_concise": res_concise,
        "res_detail": res_detail,
        "meta": {
            "model_name": args.model_name,
            "remasking": args.remasking,
            "steps": args.steps,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "bias": args.bias,
            "temperature": args.temperature,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
        }
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved to {args.output_json}")


if __name__ == "__main__":
    main()