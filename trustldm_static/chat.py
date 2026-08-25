import time
import warnings
import shortuuid
from tqdm import tqdm
from trustldm_static.utils import timeout
from abc import ABC, abstractmethod
from typing import List, Dict, Union
import requests
import os
from helm.common.request import Request
from trustldm_static.configs.configs import BaseConfig
from helm.proxy.clients.auto_client import AutoClient
from helm.proxy.clients.huggingface_model_registry import HuggingfaceModelQuantizationConfig
from helm.proxy.clients.huggingface_model_registry import ModelLoader, WeightType
from trustldm_static.response import Response


class Chat(ABC):
    def __init__(self, model_name, model_type: str, prompt_price: float, completion_price: float):
        self.model_name = model_name
        self.model_type = model_type
        self.prompt_price = prompt_price
        self.completion_price = completion_price

    @staticmethod
    def from_helm(main_config: BaseConfig, **kwargs):
        if main_config.model_config.model_loader != ModelLoader.HF:
            quant_config = HuggingfaceModelQuantizationConfig(
                model_loader=main_config.model_config.model_loader,
                quant_file=main_config.model_config.quant_file,
                disable_exllama=main_config.model_config.disable_exllama,
                inject_fused_attention=main_config.model_config.inject_fused_attention
            )
        else:
            quant_config = None

        kwargs["api_key"] = main_config.key
        kwargs["quantization_config"] = quant_config
        kwargs["model_dtype"] = main_config.model_config.torch_dtype
        kwargs["conv_template"] = main_config.model_config.conv_template
        kwargs["tokenizer_name"] = main_config.model_config.tokenizer_name
        kwargs["device_map"] = main_config.model_config.device_map

        model_name: str = main_config.model_config.model
        if model_name.startswith("llada/"):
            model_name += ('_' + main_config.model_config.remasking)
            
            print(main_config.model_config.model_path)
            print("0000000000000000000000000000000000")
            model_path = main_config.model_config.model_path
            if not model_path:
                raise ValueError("LLaDA model requires model_path in configuration")
            

            gen_function = main_config.model_config.gen_function
            if not gen_function:
                raise ValueError("LLaDA model requires gen_function in configuration")
            
            if gen_function.startswith("file:"):
                # format: file:/path/to/file.py:function
                parts = gen_function[5:].split(":")
                if len(parts) != 2:
                    raise ValueError("Invalid gen_function format. Expected 'file:/path/to/file.py:function'")
                
                file_path, func_name = parts

                
                import importlib.util
                import sys

                module_name = f"llada_gen_{int(time.time())}"
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                gen = getattr(module, func_name)
                
            else:
                raise NotImplementedError("Not supported filepath")
            
            kwargs.pop("api_key", None)

            if main_config.model_config.steps is not None:
                kwargs["steps"] = main_config.model_config.steps
            if main_config.model_config.gen_length is not None:
                kwargs["gen_length"] = main_config.model_config.gen_length
            if main_config.model_config.block_length is not None:
                kwargs["block_length"] = main_config.model_config.block_length
            if main_config.model_config.add_role is not None:
                kwargs["add_role"] = main_config.model_config.add_role
            if main_config.model_config.cfg_scale is not None:
                kwargs["cfg_scale"] = main_config.model_config.cfg_scale
            
            return LLADAChat(
                model_name=model_name.removeprefix("llada/").rstrip("/"),
                conv_template=kwargs["conv_template"],
                cache=kwargs["cache"],
                model_path=model_path,
                gen=gen,
                remasking=main_config.model_config.remasking,
                **{k: v for k, v in kwargs.items() 
                if k not in ["conv_template", "cache", "model_path", "gen_function"]}
            )
        elif model_name.startswith("dream/"):
            model_name += ('_' + main_config.model_config.remasking)

            print(main_config.model_config.model_path)
            print("0000000000000000000000000000000000")
            model_path = main_config.model_config.model_path
            if not model_path:
                raise ValueError("Dream model requires model_path in configuration")
            
            
            gen_function = main_config.model_config.gen_function
            if not gen_function:
                raise ValueError("Dream model requires gen_function in configuration")
            
            
            if gen_function.startswith("file:"):
                # format: file:/path/to/file.py:function
                parts = gen_function[5:].split(":")
                if len(parts) != 2:
                    raise ValueError("Invalid gen_function format. Expected 'file:/path/to/file.py:function'")
                
                file_path, func_name = parts


                import importlib.util
                import sys
                
                module_name = f"dream_gen_{int(time.time())}"
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                gen = getattr(module, func_name)
                
            else:
                raise NotImplementedError("Not supported filepath")
            
            kwargs.pop("api_key", None)

            if main_config.model_config.steps is not None:
                kwargs["steps"] = main_config.model_config.steps
                model_name += ('_steps' + str(main_config.model_config.steps))
            if main_config.model_config.gen_length is not None:
                kwargs["gen_length"] = main_config.model_config.gen_length
                model_name += ('_len' + str(main_config.model_config.gen_length))
            
            # Create DreamChat instance
            return DreamChat(
                model_name=model_name.removeprefix("dream/").rstrip("/"),
                conv_template=kwargs["conv_template"],
                cache=kwargs["cache"],
                model_path=model_path,
                gen=gen,
                remasking=main_config.model_config.remasking,
                **{k: v for k, v in kwargs.items() 
                if k not in ["conv_template", "cache", "model_path", "gen_function"]}
            )
        elif model_name.startswith("mercury/"):
            model_name += ('_' + main_config.model_config.remasking)

            return MercuryChat(
                model_name=model_name.removeprefix("mercury/").rstrip("/"),
                conv_template=kwargs["conv_template"],
                cache=kwargs["cache"],
                **{k: v for k, v in kwargs.items() 
                if k not in ["conv_template", "cache"]}
            )
        else:
            raise ValueError("Unsupported model organization")

    def calc_price(self, response):
        s = 0
        s += response["usage"]["prompt_tokens"] * self.prompt_price
        s += response["usage"]["completion_tokens"] * self.completion_price
        return s

    def do_classification(self, dataset, task_message, example_prefix=False, dry_run=False, max_tokens=20):
        """
        Do classification (zero-shot or in-context learning by calling `openai.ChatCompletion.create`. Args: dataset
        (`List[Dict]`): test dataset to evaluate. Each item should be a dict containing the following keys: `input`:
        text input of the test instance `label`: label of the instance `option`: candidate label choices of the task
        `examples` (`List[Tuple]`): demonstration examples (text, label). Set as `[]` for zero-shot evaluation.
        Please refer to `example_snli` for more information. task_message (`String`): task description for the test
        dataset (`dataset`). It should contain the answer choice. example_prefix (`Bool`): Whether to put the
        demonstration examples into the `system` messages. Only work for in-context learning. May set
        `example_prefix=True` when evaluating GPT-4.
        """
        cache = []
        acc = 0
        unknown = 0
        cost = 0
        prompt_tokens = 0
        cont_tokens = 0
        try:
            for x in tqdm(dataset):
                # messages = [{"role": "system", "content": system_message}]
                if len(x["examples"]) == 0:
                    messages = [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": task_message + "\n" + x["input"]}]
                else:
                    if example_prefix:
                        messages = [{"role": "system",
                                     "content": "You are a helpful, pattern-following assistant. " + task_message}]
                    else:
                        messages = [{"role": "system", "content": "You are a helpful, pattern-following assistant."},
                                    {"role": "user", "content": task_message},
                                    {"role": "assistant", "content": "Sure, I'd be happy to!"}]

                    for y in x["examples"]:
                        if example_prefix:
                            messages.append({"role": "system", "name": "example_user", "content": y[0]})
                            messages.append(
                                {"role": "system", "name": "example_assistant", "content": y[1].capitalize()}),
                        else:
                            messages.append({"role": "user", "content": y[0]})
                            messages.append({"role": "assistant", "content": y[1].capitalize()}),
                    messages.append({"role": "user", "content": x["input"]})

                response = self.call(messages, dry_run=dry_run, max_tokens=max_tokens)
                cost += self.calc_price(response)
                prompt_tokens += response["usage"]["prompt_tokens"]
                cont_tokens += response["usage"]["completion_tokens"]

                pred = response['choices'][0]['message']['content']
                pred = pred.lower()

                if pred.startswith("answer:"):
                    pred = pred[7:]
                if pred.find("</s>") != -1:
                    pred = pred.split("</s>")[0]
                if pred.find("<|im_end|>") != -1:
                    pred = pred.split("<|im_end|>")[0]
                pred = pred.strip()

                # We consider if the model generates explanations after the answer choice.
                pre = pred.split(".")[0].strip()
                pre = pre.split(",")[0].strip()
                pre = pre.split("\n")[0].strip()
                cache.append((messages, response))
                if pred == x["label"] or pre == x["label"]:
                    acc += 1
                elif pred not in x["option"] and pre not in x["option"]:
                    unknown += 1
        except Exception as e:
            print(e)
            if len(cache) == 0:
                return None, None, 0, []
            else:
                return acc / len(cache), unknown, cost, cache

        return acc / len(dataset), unknown, (cost, prompt_tokens, cont_tokens), cache

    def do_generation(self, dataset, message_constructor, n=1, t=1, max_tokens=150, dry_run=False):
        """
        Do text generation by calling `openai.ChatCompletion.create`
        Args:
            dataset (`List[str]`): test dataset to evaluate. Each item should be a text prompt.
            message_constructor (`MessageConstructor`): format the input prompts tailer for GPT-3.5 and GPT-4
            n (int): number of generations given the same prompt
            t (int): generation temperature
            max_tokens: max number of tokens to generate
        """
        cache = []
        cost = 0
        prompt_tokens = 0
        cont_tokens = 0
        try:
            for i, x in tqdm(enumerate(dataset)):
                if self.model_type == "completion":
                    messages = x
                else:
                    messages = message_constructor.get_message(x)
                response = self.call(messages, max_tokens=max_tokens, n=n, t=t, dry_run=dry_run)
                if dry_run:
                    print(messages)
                    print(response)
                if "message" in response["choices"][0]:
                    continuation = response["choices"][0]["message"]["content"]
                else:
                    continuation = response["choices"][0]["text"]

                is_banned = continuation.find("it contains inappropriate language.") != -1

                cost += self.calc_price(response)
                prompt_tokens += response["usage"]["prompt_tokens"]
                cont_tokens += response["usage"]["completion_tokens"]
                cache.append((messages, continuation, is_banned, x, response))

                if i < 5:
                    print(messages)
                    print(response["choices"])
                    print("=" * 25)
                # print()
                # print("continuation:",continuation)
                # print()
        except Exception as e:
            print(e)
            if len(cache) == 0:
                return (0, 0, 0), []
        return (cost, prompt_tokens, cont_tokens), cache

    @abstractmethod
    def _call(self, messages, t=0, max_tokens=20, n=1):
        pass

    def call(self, messages, t=0, retry=10, max_tokens=20, n=1, dry_run=False):
        """
        A robust implementation for calling `openai.ChatCompletion.create`.
        Args:
            messages: messages conveyed to OpenAI. 
            t: temperature. Set t=0 will make the outputs mostly deterministic.
            n: How many chat completion choices to generate for each input message.
            max_tokens: maximum tokens to generate for chat completion.
            Please look at https://platform.openai.com/docs/api-reference/chat/create for more information.
            [TODO] We may add all arguments of `openai.ChatCompletion.create` here. 
            retry: for sake of Error on OpenAI side, we try `retry + 1` times for a request if we do not get a response.
            dry_run: call the fake api to calculate the prices
        """
        if dry_run:
            return self.dry_run(messages, t, max_tokens, n)

        response = None
        for i in range(retry + 1):
            try:
                response = self._call(messages, t, max_tokens, n)  # .to_dict()
                break
            except TimeoutError:
                print(f"Seemingly openai is frozen, wait {i + 1}s and retry")
                time.sleep(i + 1)
            except Exception as e:
                print("Error:", e)
                print(type(e))
                print(f"wait {i + 1}s and retry")
                time.sleep(i + 1)
        if response is None:
            print(f"try {retry + 1} but still no response, return None")
        return response

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        model = model.replace("openai/", "")
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai
                /openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value, allowed_special={'<|endoftext|>'}))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def dry_run(self, messages, t=0, max_tokens=20, n=1):
        # TODO: Refactor with pydantic
        return Response.from_dict({
            "id": f"chatcmpl-{shortuuid.random()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "dryrun-" + self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": "test " * max_tokens
                    },
                    "finish_reason": "length"
                }
                for i in range(n)
            ],
            "usage": {  # Not implemented for now
                "prompt_tokens": self.num_tokens_from_messages(messages, self.model_name),
                "completion_tokens": max_tokens * n,
                "total_tokens": self.num_tokens_from_messages(messages, self.model_name) + max_tokens * n
            }
        }).to_dict()


class LLADAChat(Chat):
    def __init__(self, model_name: str, conv_template: str,  model_path: str, gen,cache: str=None, **kwargs):
        """
        initialize a LLADAChat object
        
        Args:
            model_name: model name
            conv_template: will not be used
            cache: temporal save
            gen: external generate function, formatted by gen(model, prompt, max_tokens, temperature, n, **kwargs)
            **kwargs: other args, e.g. model, tokenizer, device 
        """
        super().__init__(model_name, model_type=kwargs.get("model_type", "chat"), prompt_price=0, completion_price=0)
        self.add_role = kwargs.get("add_role", None)  
        
        self.gen = gen
        self.remasking = kwargs.get("remasking", "low_confidence")
        self.steps = kwargs.get("steps", 0)
        self.gen_length = kwargs.get("gen_length", 0)
        self.block_length = kwargs.get("block_length", self.gen_length//4)
        self.cfg = kwargs.get("cfg_scale", 0.0)
        # print(self.remasking)
        print("self.model name=:\n",self.model_name)
        print()
            

        # load model and tokenizer locally
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.device_map = kwargs.get("device_map", "auto")
        self.device = "cuda"
        self.model = AutoModel.from_pretrained(model_path,  trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        self.model.to(self.device).eval()
    
    def messages_to_prompt(self, messages: Union[List[Dict], str]) -> str:
        if isinstance(messages, str):
            return messages
        
        context_prompt = None
        messages_new = []
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                warnings.warn("LLaDA does not have support for system prompts.")
                messages_new.append(message)
            elif msg_role == "pre_context":
                messages_new.append({"role": "assistant", "content": message["content"]})
            elif msg_role == "context":
                if context_prompt is not None:
                    raise ValueError("Many-context is not supported.")
                context_prompt = message["content"]
            else:
                messages_new.append(message)
        
        messages = messages_new
        print(messages)
        print()
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=False
        )
        input_ids =  inputs
        return (input_ids, context_prompt)
    
    @timeout(3000)
    def _call(self, messages, t=0, max_tokens=32, n=1):
        """
        Args:
            messages: List[str]
            t: temperature
            max_tokens: max generate tokens
            n: number of generated sequences
        """
        # print("start processing")
        input_ids, context = self.messages_to_prompt(messages)
        if context is None:
            # print("IS NONE!!")
            context = None
        elif self.add_role:
            context = self.tokenizer.apply_chat_template(
                [{"role": self.add_role, "content": context}],
                add_generation_prompt=False, 
                tokenize=False
            )
            # print(context)
        print("context:\n\n", context)
        print()
        
        import torch
        input_ids = torch.tensor(input_ids).to(device=self.device)
        
        if self.steps == 0:
            self.steps = max_tokens
        if self.gen_length == 0:
            self.gen_length = max_tokens
            self.block_length = max_tokens//4
        
        # generation
        generated_texts = []
        for i in range(n):
            try:
                outputs = self.gen(
                        model=self.model,
                        prompt_ids=input_ids,
                        gen_length=self.gen_length,
                        steps=self.steps,
                        block_length=self.block_length,
                        temperature=t,
                        remasking=self.remasking,
                        context=context,
                        tokenizer=self.tokenizer,
                        cfg_scale=self.cfg
                )
            except Exception as e:
                raise RuntimeError(f"External generation function failed: {str(e)}")
            
            # remove inputs and contexts
            generated_text = self.tokenizer.batch_decode(
                outputs[:, input_ids.shape[1]:input_ids.shape[1]+self.gen_length], 
                skip_special_tokens=True
            )[0]
            generated_texts.append(generated_text)
            # print(i,"\n",generated_text)
            # print()
        # print()

        # construct Response format
        return Response.from_dict({
            "id": f"chatcmpl-{shortuuid.random()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"  # modified accordingly
                }
                for i, text in enumerate(generated_texts)
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": max_tokens * n,
                "total_tokens": len(input_ids) + max_tokens * n
            }
        }).to_dict()
    
class DreamChat(Chat):
    def __init__(self, model_name: str, conv_template: str,  model_path: str, gen,cache: str=None, **kwargs):
        """
        initialize a DreamChat object
        
        Args:
            model_name: model name
            conv_template: not in use
            cache: save temperal
            gen: external generating function, formatted by gen(model, prompt, max_tokens, temperature, n, **kwargs)
            **kwargs: other args, e.g. model, tokenizer, device 
        """
        super().__init__(model_name, model_type=kwargs.get("model_type", "chat"), prompt_price=0, completion_price=0)
        
        self.add_role = kwargs.get("add_role", None)
        self.gen = gen
        self.remasking = kwargs.get("remasking", "low_confidence")
        self.steps = kwargs.get("steps", 0)
        self.gen_length = kwargs.get("gen_length", 0)
        self.alg_temp = kwargs.get("alg_temp", 0.0)
        print("self.model name=:\n",self.model_name)
        print()
            

        # load model and tokenizer locally
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.device_map = kwargs.get("device_map", "auto")
        self.device = "cuda"
        self.model = AutoModel.from_pretrained(model_path,  trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        self.model.to(self.device).eval()
    
    def messages_to_prompt(self, messages: Union[List[Dict], str]) -> str:
        if isinstance(messages, str):
            return messages
        
        context_prompt = None
        messages_new = []
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                warnings.warn("Dream does not have support for system prompts.")
                messages_new.append(message)
            elif msg_role == "pre_context":
                messages_new.append({"role": "assistant", "content": message["content"]})
            elif msg_role == "context":
                if context_prompt is not None:
                    raise ValueError("Many-context is not supported.")
                context_prompt = message["content"]
            else:
                messages_new.append(message)
        messages = messages_new
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            return_dict=True, 
            add_generation_prompt=True
        )
        input_ids =  inputs.input_ids
        att = inputs.attention_mask
        return (input_ids, att, context_prompt)
    
    @timeout(3000)
    def _call(self, messages, t=0, max_tokens=32, n=1):
        """
        Args:
            messages: List[Dict[str:str]]
            t: temperature
            max_tokens: max token num
            n: number of generated sequence
        """
        # print("start processing")
        # print(messages)
        input_ids,attention_mask, context = self.messages_to_prompt(messages)
        if context is None:
            context = ""
        elif self.add_role:
            context = self.tokenizer.apply_chat_template(
                [{"role": self.add_role, "content": context}],
                add_generation_prompt=False, 
                tokenize=False
            )
        
        attention_mask = attention_mask.to(device=self.device)
        import torch
        input_ids = torch.tensor(input_ids).to(device=self.device)
        
        if self.steps == 0:
            self.steps = max_tokens
        if self.gen_length == 0:
            self.gen_length = max_tokens
        
        # generation
        generated_texts = []
        for i in range(n):
            try:
                # print("hi")
                # print(input_ids)
                # print(attention_mask)
                output = self.gen(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.gen_length,
                    steps=self.steps,
                    temperature=t,
                    top_k=50,
                    alg=self.remasking,
                    alg_temp=self.alg_temp,
                    return_dict_in_generate=True,
                    context=context,
                    tokenizer=self.tokenizer
                )

                # Decode generated text
                generations = [
                    self.tokenizer.decode(g[len(p):len(p)+self.gen_length].tolist(), skip_special_tokens=True)
                    for p, g in zip(input_ids, output.sequences)
                ]
                
                # Clean up generation (remove EOS token if present)
                generated_text = generations[0]
                if self.tokenizer.eos_token and self.tokenizer.eos_token in generated_text:
                    generated_text = generated_text.split(self.tokenizer.eos_token)[0]
            except Exception as e:
                warnings.warn(f"Generation failed with error: {e}")
                raise RuntimeError(f"External generation function failed: {str(e)}")
            generated_texts.append(generated_text)
            # print(i,"\n",generated_text)
            # print()
        # print()

        #construct Response format
        return Response.from_dict({
            "id": f"chatcmpl-{shortuuid.random()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"  # modified accordingly
                }
                for i, text in enumerate(generated_texts)
            ],
            "usage": {
                "prompt_tokens": len(input_ids) * n,
                "completion_tokens": max_tokens * n,
                "total_tokens": n * len(output.sequences)
            }
        }).to_dict()


class MercuryChat(Chat):
    def __init__(self, model_name: str, conv_template: str, cache: str=None, **kwargs):
        """
        initialize a MercuryChat object for API-based generation
        
        Args:
            model_name: model name
            conv_template: conversation template
            cache: temporal save
            **kwargs: other args
        """
        super().__init__(model_name, model_type=kwargs.get("model_type", "chat"), prompt_price=0, completion_price=0)
        self.gen_length = kwargs.get("gen_length", 50)
        self.temperature = kwargs.get("temperature", 1.0)
        self.api_key = os.getenv("MERCURY_API_KEY")
        if not self.api_key:
            raise ValueError("MERCURY_API_KEY environment variable not set")
        print("self.model name=:\n", self.model_name)
        print()

    def messages_to_prompt(self, messages: Union[List[Dict], str]) -> str:
        if isinstance(messages, str):
            return messages
        
        # Simple conversion to prompt
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        return prompt
    
    @timeout(3000)
    def _call(self, messages, t=0, max_tokens=32, n=1):
        """
        Args:
            messages: List[str]
            t: temperature
            max_tokens: max generate tokens
            n: number of generated sequences
        """
        # Assume Mercury API is OpenAI-compatible
        url = "https://api.mercury.com/v1/chat/completions"  # Placeholder, replace with actual endpoint
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mercury",  # or actual model name
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": t,
            "n": n
        }
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise RuntimeError(f"Mercury API error: {response.status_code} {response.text}")
        
        result = response.json()
        
        # Assume response format similar to OpenAI
        generated_texts = [choice["message"]["content"] for choice in result["choices"]]
        
        # construct Response format
        return Response.from_dict({
            "id": result.get("id", f"chatcmpl-{shortuuid.random()}"),
            "object": "chat.completion",
            "created": result.get("created", int(time.time())),
            "model": self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }
                for i, text in enumerate(generated_texts)
            ],
            "usage": result.get("usage", {
                "prompt_tokens": len(str(messages)),
                "completion_tokens": max_tokens * n,
                "total_tokens": len(str(messages)) + max_tokens * n
            })
        }).to_dict()
    