# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import torch.distributions as dists
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Union, Dict, Any, List, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput
import warnings


def top_k_logits(logits, top_k=None):
    """Top-k sampling"""
    if top_k is None:
        return logits
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_k=None, margin_confidence=False, neg_entropy=False):
    """Sample tokens with various strategies"""
    if temperature > 0:
        logits = logits / temperature
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


@torch.no_grad()
def diffusion_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: int = 512,
    steps: int = 512,
    temperature: float = 0.2,
    top_k: Optional[int] = None,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    eps: float = 1e-3,
    output_history: bool = False,
    return_dict_in_generate: bool = True,
    mask_token_id: int = 151666,
    generation_tokens_hook_func = lambda step, x, logits: x,
    generation_logits_hook_func = lambda step, x, logits: logits,
    **kwargs
) -> Union[DreamModelOutput, torch.LongTensor]:
    """
    Dream model diffusion generation implementation
    
    Args:
        model: Dream model instance
        input_ids: Input token IDs
        attention_mask: Attention mask
        max_new_tokens: Maximum number of new tokens to generate
        steps: Number of diffusion steps
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        alg: Algorithm type ('origin', 'maskgit_plus', 'topk_margin', 'entropy')
        alg_temp: Algorithm temperature for soft selection
        eps: Epsilon value for timestep scheduling
        output_history: Whether to output generation history
        return_dict_in_generate: Whether to return dictionary format
        mask_token_id: MASK token ID
        generation_tokens_hook_func: Hook function for token control
        generation_logits_hook_func: Hook function for logits control
    
    Returns:
        Generated sequences or DreamModelOutput
    """
    
    # Calculate max_length and initialize sequence with context support
    context = kwargs.get("context", None)
    # print(1)
    if context is not None and len(context)>0:
        # Get tokenizer from kwargs (provided by upper-level generate() function)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer is required when context is provided")
        
        # Tokenize context
        ctx_result = tokenizer(context, add_special_tokens=False, return_tensors="pt")
        ctx_ids = ctx_result["input_ids"].to(input_ids.device)
        
        # Ensure ctx_ids has the same batch dimension as input_ids
        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)
        
        ctx_length = ctx_ids.shape[1]
        max_length = input_ids.shape[1] + max_new_tokens + ctx_length
        
        # Initialize sequence: [input | gen_length | post_context]
        x = torch.full((input_ids.shape[0], max_length), mask_token_id, dtype=torch.long, device=input_ids.device)
        # Place input sequence
        x[:, :input_ids.shape[1]] = input_ids
        
        # Place context after the generation region
        start_idx = input_ids.shape[1] + max_new_tokens
        x[:, start_idx:start_idx + ctx_length] = ctx_ids
        
        # Mark context region (used for later tok_idx computation)
        context_mask = torch.zeros_like(x, dtype=torch.bool)
        context_mask[:, start_idx:start_idx + ctx_length] = True
        # print(2)
    else:
        max_length = input_ids.shape[1] + max_new_tokens
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        # print(2.5)

    # Initialize histories
    histories = [] if (return_dict_in_generate and output_history) else None
    
    # Handle attention mask
    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # We do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        # print("else!!!")
        tok_idx = None
        attention_mask = "full"
    # print(3)
    
    # Create timesteps
    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
    
    # print(4)
    # Initial token control hook
    x = generation_tokens_hook_func(None, x, None)
    # print(5)

    # early_stop = steps//4 if kwargs["short"] else -1
    
    # Main diffusion loop
    for i in range(steps):
        # print(6.0)
        mask_index = (x == mask_token_id)
        num_masks = mask_index.sum().item()
        # print(6.1)
        # Get model predictions
        logits = model(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        # print(6.2)
        # Logits control hook
        logits = generation_logits_hook_func(i, x, logits)
        # print(6.3)
        
        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
        
        # Skip if no mask tokens
        if mask_logits.numel() == 0:
            continue
        # print(6.4)
        if alg == 'origin':
            # Original algorithm: random transfer
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=model.device, dtype=torch.long) + mask_token_id
            transfer_index_t_s = torch.rand(*x0.shape, device=model.device) < p_transfer
            _, x0[transfer_index_t_s] = sample_tokens(
                mask_logits[transfer_index_t_s], 
                temperature=temperature, 
                top_k=top_k
            )
            x[mask_index] = x0.clone()
        else:
            # Advanced algorithms
            if alg == 'maskgit_plus':
                # print(6.5)
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_k=top_k)
                # print(6.6)
            elif alg == 'topk_margin':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(mask_logits, temperature, top_k=top_k, neg_entropy=True)
            elif alg in ('ltr', 'rtl'):
                # Left-to-right / Right-to-left deterministic decoding order.
                # We still sample tokens for all masked positions, but commit them in a fixed order.
                _, x0 = sample_tokens(mask_logits, temperature=temperature, top_k=top_k)
                confidence = None
            else:
                # Default to entropy if unknown algorithm
                confidence, x0 = sample_tokens(mask_logits, temperature, top_k=top_k, neg_entropy=True)
            
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            number_transfer_tokens = max(1, min(number_transfer_tokens, int(num_mask_token)))  # Ensure valid range

            if alg in ('ltr', 'rtl'):
                if number_transfer_tokens > 0:
                    seq_len = x.size(1)
                    pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
                    if alg == 'ltr':
                        score = torch.where(mask_index, -pos.to(logits.dtype), torch.full_like(pos, float("-inf"), dtype=logits.dtype))
                    else:
                        score = torch.where(mask_index, pos.to(logits.dtype), torch.full_like(pos, float("-inf"), dtype=logits.dtype))
                    _, transfer_pos = torch.topk(score, number_transfer_tokens, dim=-1)
                    transfer_index = torch.zeros_like(mask_index)
                    row_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_pos)
                    transfer_index[row_indices, transfer_pos] = True

                    x_ = torch.zeros_like(x, device=model.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    x[transfer_index] = x_[transfer_index]
            else:
                full_confidence = torch.full_like(x, -torch.inf, device=model.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                # print(6.7)


                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens, dim=-1)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    
                    x_ = torch.zeros_like(x, device=model.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]
                    # print(6.8)
        
        # Token control hook
        x = generation_tokens_hook_func(i, x, logits)
        
        # Store history if requested
        if histories is not None:
            histories.append(x.clone())
        
        # if kwargs["short"] and i >= early_stop:
        #     break
    
    # Return results
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
        )
    else:
        return x


def generate(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 512,
    steps: int = 512,
    temperature: float = 0.2,
    top_k: Optional[int] = None,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    eps: float = 1e-3,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    use_chat_template: bool = True,
    return_dict: bool = True,
    output_history: bool = False,
    mask_token_id: int = 151666,
    **kwargs
) -> Union[str, Dict[str, Any]]:
    """
    Dream model generation function
    
    Args:
        model_path: Path to the Dream model
        prompt: Input prompt text
        max_new_tokens: Maximum number of new tokens to generate
        steps: Number of diffusion steps
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        alg: Algorithm type ('origin', 'maskgit_plus', 'topk_margin', 'entropy')
        alg_temp: Algorithm temperature for soft selection
        eps: Epsilon value for timestep scheduling
        device: Device to run on
        torch_dtype: Model dtype
        trust_remote_code: Whether to trust remote code
        use_chat_template: Whether to use chat template
        return_dict: Whether to return dictionary format
        output_history: Whether to output generation history
        mask_token_id: MASK token ID
        **kwargs: Additional generation parameters
    
    Returns:
        Generated text or dictionary with generation results
    """
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch_dtype, 
        trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=trust_remote_code
    )
    model = model.to(device).eval()
    
    # Get mask_token_id from model's config
    if mask_token_id is None:
        if hasattr(model, 'config') and hasattr(model.config, 'mask_token_id'):
            mask_token_id = model.config.mask_token_id
        elif hasattr(model, 'generation_config') and hasattr(model.generation_config, 'mask_token_id'):
            mask_token_id = model.generation_config.mask_token_id
        else:
            # Use the correct mask_token_id from dream_config.py
            mask_token_id = 151666
    
    # Prepare input
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            return_dict=True, 
            add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device=device)
        attention_mask = inputs.attention_mask.to(device=device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device)
        attention_mask = None
    
    # Generate using our custom diffusion_generate function
    try:
        output = diffusion_generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            alg=alg,
            alg_temp=alg_temp,
            eps=eps,
            output_history=output_history,
            return_dict_in_generate=True,
            mask_token_id=mask_token_id,
            **kwargs
        )
        
        # Decode generated text
        generations = [
            tokenizer.decode(g[len(p):].tolist(), skip_special_tokens=True)
            for p, g in zip(input_ids, output.sequences)
        ]
        
        # Clean up generation (remove EOS token if present)
        generated_text = generations[0]
        if tokenizer.eos_token and tokenizer.eos_token in generated_text:
            generated_text = generated_text.split(tokenizer.eos_token)[0]
        
        if return_dict:
            result = {
                "generated_text": generated_text,
                "full_sequences": output.sequences,
                "input_length": input_ids.shape[1],
                "generation_params": {
                    "max_new_tokens": max_new_tokens,
                    "steps": steps,
                    "temperature": temperature,
                    "top_k": top_k,
                    "alg": alg,
                    "alg_temp": alg_temp,
                    "eps": eps,
                    "mask_token_id": mask_token_id
                }
            }
            
            if output_history and hasattr(output, 'history') and output.history:
                result["generation_history"] = output.history
            
            return result
        else:
            return generated_text
            
    except Exception as e:
        warnings.warn(f"Generation failed with error: {e}")
        if return_dict:
            return {
                "generated_text": "", 
                "error": str(e),
                "generation_params": {
                    "max_new_tokens": max_new_tokens,
                    "steps": steps,
                    "temperature": temperature,
                    "top_k": top_k,
                    "alg": alg,
                    "alg_temp": alg_temp,
                    "eps": eps,
                    "mask_token_id": mask_token_id
                }
            }
        else:
            return ""


def main():
    """Example usage of the Dream generation function"""
    
    # Model configuration
    model_path = "/data1/ybshi/Dream-v0-Instruct-7B/"
    
    # Test prompt
    test_prompt = "Who are you?"
    
    print("=== Dream Model Generation Demo ===\n")
    
    # Single generation example
    print("Generation Example:")
    print("-" * 50)
    
    result = generate(
        model_path=model_path,
        prompt=test_prompt,
        max_new_tokens=256,
        steps=256,
        temperature=0.2,
        top_k=50,
        alg="entropy",
        alg_temp=0.0,
        return_dict=True,
        output_history=False
    )
    
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {result['generated_text']}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Generation params: {result['generation_params']}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()