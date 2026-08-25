import os 
try:
    import litellm
except ImportError:  # Local Mixtral deployments do not require LiteLLM.
    litellm = None
import requests
litellm.set_verbose=True
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model
from loggers import logger
from common import get_api_key

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 16
    API_TIMEOUT = 300

    def __init__(self, model_name):
        super().__init__(model_name)
        self.use_local_mixtral = self.model_name == Model.mixtral
        if self.use_local_mixtral:
            self.api_key = os.environ.get("MIXTRAL_LOCAL_API_KEY", "EMPTY")
            self.local_mixtral_base_url = os.environ.get(
                "MIXTRAL_LOCAL_BASE_URL",
                "http://127.0.0.1:8000/v1/chat/completions",
            )
            self.local_mixtral_model_name = os.environ.get(
                "MIXTRAL_LOCAL_MODEL_NAME",
                "mixtral-local",
            )
        else:
            self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        if litellm is not None:
            litellm.drop_params = True
        self.set_eos_tokens(self.model_name)
        
        # Set LiteLLM logging environment variable instead of using a set_verbose method
        os.environ['LITELLM_LOG'] = 'DEBUG'
        
    def get_litellm_model_name(self, model_name):
        if self.use_local_mixtral:
            self.use_open_source_model = True
            return self.local_mixtral_model_name
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        else:
            self.use_open_source_model =  False
            #if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value 
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        if self.use_local_mixtral:
            # Local Mixtral is served via a plain OpenAI-compatible endpoint, not via LiteLLM's
            # provider registry, so we must not call register_prompt_template with model=mixtral-local.
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
            return
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 
        
        eos_tokens = self.eos_tokens 

        # Add debug logs to check whether any conv messages contain None content
        for i, conv in enumerate(convs_list):
            for j, msg in enumerate(conv):
                if msg.get("content") is None:
                    print(f"Warning: convs_list[{i}][{j}] content is None")
                    # Attempt to fix None values
                    if "content" in msg:
                        msg["content"] = ""
        

        if extra_eos_tokens:
            # Merge and deduplicate stop tokens
            all_eos_tokens = list(set(eos_tokens + extra_eos_tokens))
        else:
            all_eos_tokens = eos_tokens
            
        if self.use_open_source_model:
            self._update_prompt_template()

        if self.use_local_mixtral:
            return self._batched_generate_local_mixtral(
                convs_list=convs_list,
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=all_eos_tokens,
            )
        
        if litellm is None:
            raise ImportError(
                "LiteLLM is required for non-local API attackers. "
                "Install it or use the local Mixtral endpoint."
            )

        outputs = litellm.batch_completion(
            model=self.litellm_model_name, 
            messages=convs_list,
            api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=all_eos_tokens,  # Use de-duplicated stop tokens
        )
        
        responses = [output["choices"][0]["message"].content for output in outputs]

        return responses

    def _batched_generate_local_mixtral(
        self,
        convs_list: list[list[dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str],
    ) -> list[str]:
        responses = []
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        for conv in convs_list:
            normalized_messages = self._normalize_messages_for_local_mixtral(conv)
            payload = {
                "model": self.local_mixtral_model_name,
                "messages": normalized_messages,
                "max_tokens": max_n_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop:
                payload["stop"] = stop

            last_error = None
            for attempt in range(self.API_MAX_RETRY):
                try:
                    response = requests.post(
                        self.local_mixtral_base_url,
                        headers=headers,
                        json=payload,
                        timeout=self.API_TIMEOUT,
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    responses.append(content)
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        f"Local Mixtral request failed on attempt {attempt + 1}/{self.API_MAX_RETRY}: {exc}"
                    )
                    import time
                    time.sleep(30)
            else:
                raise RuntimeError(
                    f"Failed to call local Mixtral endpoint after {self.API_MAX_RETRY} attempts: {last_error}"
                )

        return responses

    def _normalize_messages_for_local_mixtral(self, messages: list[dict]) -> list[dict]:
        """
        The local Mixtral server uses tokenizer.apply_chat_template directly and expects
        strict user/assistant alternation. Convert the OpenAI-style message list into a
        Mistral-friendly sequence by folding any system prompt into the first user turn.
        """
        system_chunks = []
        normalized = []

        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""

            if role == "system":
                if content:
                    system_chunks.append(content)
                continue

            if role not in {"user", "assistant"}:
                continue

            normalized.append({"role": role, "content": content})

        if system_chunks:
            system_prefix = "\n\n".join(system_chunks).strip()
            if normalized and normalized[0]["role"] == "user":
                merged = normalized[0]["content"].strip()
                normalized[0]["content"] = f"{system_prefix}\n\n{merged}" if merged else system_prefix
            else:
                normalized.insert(0, {"role": "user", "content": system_prefix})

        return normalized

# class LocalvLLM(LanguageModel):
    
#     def __init__(self, model_name: str):
#         """Initializes the LLMHuggingFace with the specified model name."""
#         super().__init__(model_name)
#         if self.model_name not in MODEL_NAMES:
#             raise ValueError(f"Invalid model name: {model_name}")
#         self.hf_model_name = HF_MODEL_NAMES[Model(model_name)]
#         destroy_model_parallel()
#         self.model = vllm.LLM(model=self.hf_model_name)
#         if self.temperature > 0:
#             self.sampling_params = vllm.SamplingParams(
#                 temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_n_tokens
#             )
#         else:
#             self.sampling_params = vllm.SamplingParams(temperature=0, max_tokens=self.max_n_tokens)

#     def _get_responses(self, prompts_list: list[str], max_new_tokens: int | None = None) -> list[str]:
#         """Generates responses from the model for the given prompts."""
#         full_prompt_list = self._prompt_to_conv(prompts_list)
#         outputs = self.model.generate(full_prompt_list, self.sampling_params)
#         # Get output from each input, but remove initial space
#         outputs_list = [output.outputs[0].text[1:] for output in outputs]
#         return outputs_list

#     def _prompt_to_conv(self, prompts_list):
#         batchsize = len(prompts_list)
#         convs_list = [self._init_conv_template() for _ in range(batchsize)]
#         full_prompts = []
#         for conv, prompt in zip(convs_list, prompts_list):
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], None)
#             full_prompt = conv.get_prompt()
#             # Need this to avoid extraneous space in generation
#             if "llama-2-7b-chat-hf" in self.model_name:
#                 full_prompt += " "
#             full_prompts.append(full_prompt)
#         return full_prompts

#     def _init_conv_template(self):
#         template = get_conversation_template(self.hf_model_name)
#         if "llama" in self.hf_model_name:
#             # Add the system prompt for Llama as FastChat does not include it
#             template.system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#         return template
    





