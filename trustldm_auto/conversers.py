import copy

from common import get_api_key, conv_template, extract_json, get_path, get_gen
from language_models import APILiteLLM
from config import FASTCHAT_TEMPLATE_NAMES, Model


def load_attack_and_target_models(args, system=""):
    # create attack model and target model
    attackLM = AttackLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        category = args.category,
                        evaluate_locally = args.evaluate_locally,
                        budget_tracker = getattr(args, "budget_tracker", None),
                        )
    
    targetLM = TargetLM(model_name = args.target_model,
                        category = args.category,
                        max_n_tokens = args.target_max_n_tokens,
                        evaluate_locally = args.evaluate_locally,
                        phase = args.jailbreakbench_phase,
                        system = system
                        )
    
    return attackLM, targetLM

def load_indiv_model(model_name, local = False, use_jailbreakbench=True):
    if use_jailbreakbench: 
        if local:
            from jailbreakbench import LLMvLLM
            lm = LLMvLLM(model_name=model_name)
        else:
            from jailbreakbench import LLMLiteLLM
            api_key = get_api_key(Model(model_name))
            lm = LLMLiteLLM(model_name= model_name, api_key = api_key)
    else:
        if local:
            from transformers import AutoModel, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
            lm = (tokenizer, model)
        else:
            lm = APILiteLLM(model_name)
    return lm

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                category: str,
                evaluate_locally: bool,
                perspective: str ="safety",
                budget_tracker=None):
        
        self.model_name = Model(model_name)
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts

        from config import ATTACK_TEMP, ATTACK_TOP_P
        self.temperature = ATTACK_TEMP
        self.top_p = ATTACK_TOP_P

        self.category = category
        self.evaluate_locally = evaluate_locally
        self.model = load_indiv_model(model_name, 
                                      local = evaluate_locally, 
                                      use_jailbreakbench=False # Cannot use JBB as attacker
                                      )
        self.initialize_output = self.model.use_open_source_model
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]
        self.perspective = perspective
        self.budget_tracker = budget_tracker

    @staticmethod
    def _normalize_prompt_content(prompt):
        return prompt if prompt is not None else ""

    def preprocess_conversation(self, convs_list: list, prompts_list: list[str]):
        # For open source models, we can seed the generation with proper JSON
        init_message = ""
        if self.initialize_output:
            
            # Initialize the attack model's generated output to match format
            if len(convs_list[0].messages) <= 1:  # If only system message (or fewer), this is the first iteration
                init_message = '{"improvement": "","pre_context": "'
            else:
                init_message = '{"improvement": "'
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], self._normalize_prompt_content(prompt))
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)
        openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
        
        # Add checks to ensure there are no None values and message structure is valid
        for i, conv in enumerate(openai_convs_list):
            for j, msg in enumerate(conv):
                if msg.get("content") is None:
                    print(f"Fixing error: openai_convs_list[{i}][{j}] content is None")
                    msg["content"] = ""
                # Check that the role is valid
                if msg.get("role") not in ["user", "assistant", "system"]:
                    print(f"Warning: openai_convs_list[{i}][{j}] has invalid role '{msg.get('role')}'")
        
        # Check that the first non-system message is a user message
        for i, conv in enumerate(openai_convs_list):
            # If the first message is system, skip it
            msg_idx = 0
            if len(conv) > 0 and conv[0].get("role") == "system":
                msg_idx = 1
                
            if msg_idx < len(conv) and conv[msg_idx].get("role") != "user":
                print(f"Warning: openai_convs_list[{i}] first non-system message is not a user role: {conv[msg_idx]}")
                
        return openai_convs_list, init_message
        
    def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
        batchsize = len(openai_conv_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        new_adv_prompts = [None] * batchsize
        
        # Continuously generate outputs until all are valid or max_n_attack_attempts is reached
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]
            # Generate outputs 
            outputs_list = self.model.batched_generate(convs_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p,
                                                        extra_eos_tokens=["}"]
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                full_output = init_message + full_output + "}" # Add end brace since we terminate generation on end braces
                attack_dict, json_str = extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    new_adv_prompts[orig_index] = json_str
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output is None for output in valid_outputs]):
            import warnings
            warnings.warn(f"Failed to generate valid output after {self.max_n_attack_attempts} attempts. Terminating.",
                          category=RuntimeWarning)
        return valid_outputs, new_adv_prompts

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        # Stage attacker turns on a copy so failed generations do not pollute the
        # persistent conversation history used in later retries/iterations.
        staged_convs_list = copy.deepcopy(convs_list)
        processed_convs_list, init_message = self.preprocess_conversation(staged_convs_list, prompts_list)
        valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)

        for prompt, jailbreak_prompt, conv in zip(prompts_list, new_adv_prompts, convs_list):
            # Only commit successful generations back to the real conversation state.
            # Failed attempts are discarded entirely so they cannot poison later turns.
            if jailbreak_prompt is None:
                continue
            if self.initialize_output:
                jailbreak_prompt += self.model.post_message

            conv.append_message(conv.roles[0], self._normalize_prompt_content(prompt))
            conv.append_message(conv.roles[1], jailbreak_prompt)
        
        if self.budget_tracker is not None:
            # Keep the historical table definition: count valid attacker outputs.
            self.budget_tracker.record_attack_queries(
                sum(output is not None for output in valid_outputs)
            )

        return valid_outputs

class TargetLM():
    """
        JailbreakBench class for target language models.
        In this case be LLDMs.
    """
    def __init__(self, 
            model_name: str, 
            category: str,
            max_n_tokens : int,
            phase: str,
            evaluate_locally: bool = False,
            use_jailbreakbench: bool = True,
            system: str = ""
            ):
        
        self.model_name = model_name
        self.max_n_tokens = max_n_tokens
        self.phase = phase
        self.use_jailbreakbench = use_jailbreakbench
        self.evaluate_locally = evaluate_locally

        from config import TARGET_TEMP,  TARGET_TOP_P   
        self.temperature = TARGET_TEMP
        self.top_p = TARGET_TOP_P

        # Handle mercury API model separately
        if model_name == "mercury":
            self.model = None
            self.tokenizer = None
            self.gen = None
            self.device = None
            self.api_key = get_api_key(Model("mercury"))
        else:
            # self.model = load_indiv_model(model_name, evaluate_locally, use_jailbreakbench) 
            model_path = get_path(model_name)
            from transformers import AutoModel, AutoTokenizer
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            self.device = "cuda"
            self.model.to(self.device).eval()
            self.gen = get_gen(model_name)
        
        self.system = system if system != "" else "<EMPTY>"
        self.category = category

    def get_response(self, messages, decoding_method, short: bool =False, gen_length: int = 0) -> str:
        """
        get_response: return response of one attack prompt
        
        :param messages: shape like:{["role": "user", "content": "...(str)..."],
                                     ["role": "context", "content": "...(str)..."]}
        
        :param decoding_method: a sampled decoding method
        """
        gen_length = gen_length if gen_length > 0 else self.max_n_tokens

        context_prompt = None
        for message in messages:
            msg_role = message["role"]
            if msg_role == "context":
                if context_prompt is not None:
                    raise ValueError("Many-context is not supported.")
                context_prompt = message["content"]
        messages_new = [
            {k: v for k, v in item.items()}
            for item in messages if item["role"] != "context"
        ]
        messages = messages_new
        if "dream" in self.model_name:
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt",
                return_dict=True, 
                add_generation_prompt=False
            )
            input_ids = inputs.input_ids
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                add_generation_prompt=False
            )
            input_ids =  inputs
        if "dream" in self.model_name:
            attention_mask = inputs.attention_mask
        context = context_prompt
        
        import torch
        input_ids = torch.tensor(input_ids).to(device=self.device)
        
        # Process generated text
        generated_texts = []
        for i in range(1):
            if "llada" in self.model_name:
                try:
                    outputs = self.gen(
                            model=self.model,
                            prompt_ids=input_ids,
                            gen_length=gen_length,
                            steps=gen_length//16 if short else gen_length,
                            block_length=gen_length//4,
                            temperature=0.0,
                            remasking=decoding_method,
                            context=context,
                            tokenizer=self.tokenizer,
                            cfg_scale=0.,
                            short=short
                    )
                    generated_text = self.tokenizer.batch_decode(
                        outputs[:, input_ids.shape[1]:input_ids.shape[1] + self.max_n_tokens],
                        skip_special_tokens=True
                    )[0]
                    generated_texts.append(generated_text)
                except Exception as e:
                    try:
                        # print("Generation failed, retrying once...")
                        # input_ids = input_ids.unsqueeze(0)
                        ctx_ids = self.tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
                        ctx_len = ctx_ids.shape[1]
                        if ctx_len >= gen_length * 2 / 3 :
                            return "The post-context is too long. Sorry, I cannot help you."
                        outputs = self.gen(
                                model=self.model,
                                inputs=input_ids,
                                gen_length=gen_length,
                                steps=gen_length//16 if short else gen_length,
                                block_length=32,
                                temperature=0.0,
                                remasking=decoding_method,
                                context=context,
                                tokenizer=self.tokenizer,
                                cfg_scale=0.,
                                short=short
                        )
                        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    except Exception as e:
                        raise RuntimeError(f"External generation function failed: {str(e)}")
                
                # print(i,"\n",generated_text)
                # print()
            elif "dream" in self.model_name:
                try:
                    # print(input_ids)
                    # print(attention_mask)
                    output = self.gen(
                        model=self.model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_length,
                        steps=gen_length//32 if short else gen_length,
                        temperature=0.2,
                        top_k=50,
                        alg=decoding_method,
                        alg_temp=0.0,
                        return_dict_in_generate=True,
                        context=context,
                        tokenizer=self.tokenizer,
                        short=short
                    )

                    # Decode generated text
                    generations = [
                        self.tokenizer.decode(g[len(p):len(p)+self.max_n_tokens].tolist(), skip_special_tokens=True)
                        for p, g in zip(input_ids, output.sequences)
                    ]
                    
                    # Clean up generation (remove EOS token if present)
                    generated_text = generations[0]
                    if self.tokenizer.eos_token and self.tokenizer.eos_token in generated_text:
                        generated_text = generated_text.split(self.tokenizer.eos_token)[0]
                except Exception as e:
                    raise RuntimeError(f"External generation function failed: {str(e)}")
                generated_texts.append(generated_text)
                # print(i,"\n",generated_text)
                # print()
        # print()

        # construct response
        response = generated_text
        return response
    
    def get_response_with_decoding(self, pre_contexts,post_ocntexts,user_prompt, decoding_methods):
        responses = []
        prompts_list =[]
        
        for i, _ in enumerate(pre_contexts):
            # m_pre = [{"role": "assistant", "content": pre_contexts[i]}]
            # pre_ctx = self.tokenizer.apply_chat_template(m_pre,   add_generation_prompt=True,  tokenize=False)
            message = []
            if self.system != "<EMPTY>":
                message.append({"role": "system", "content": self.system})
            message.extend([{"role": "user", "content": user_prompt},
                       {"role": "assistant", "content": pre_contexts[i]},
                      {"role": "context", "content": post_ocntexts[i]}])
            prompts_list.append(message)
        for i, prompt in enumerate(prompts_list):
            for decoding_method in decoding_methods:
                response = self.get_response(prompt, decoding_method)
                responses.append(response)
        
        return responses
    
    def get_response_with_decoding_and_gen_length(self, pre_context, post_context, user_prompt, decoding_method, gen_length, short=False):
        responses = []
        prompts_list =[]

        if isinstance(pre_context, str):
            pre_context = [pre_context]
        if isinstance(post_context, str):
            post_context = [post_context]
        
        for i, _ in enumerate(pre_context):
            # m_pre = [{"role": "assistant", "content": pre_contexts[i]}]
            # pre_ctx = self.tokenizer.apply_chat_template(m_pre,   add_generation_prompt=True,  tokenize=False)
            message = []
            if self.system != "<EMPTY>":
                message.append({"role": "system", "content": self.system})
            message.extend([{"role": "user", "content": user_prompt},
                       {"role": "assistant", "content": pre_context[i]},
                      {"role": "context", "content": post_context[i]}])
            prompts_list.append(message)
        for i, prompt in enumerate(prompts_list):
            response = self.get_response(prompt, decoding_method, gen_length=gen_length, short=short)
            responses.append(response)
        
        return responses
    
    def get_response_mer(self, pre_context, post_context, user_prompt, gen_length: int = 512) -> str:
        """
        get_response_mer: return response from mercury API for one attack prompt using FIM (Fill-In-Middle) format.
        
        :param pre_context: The context before the generation blank
        :param post_context: The context after the generation blank
        :param user_prompt: The user question
        :param gen_length: Maximum number of tokens to generate
        
        :return: The generated text from mercury API
        """
        import requests
        
        # Construct the prompt and suffix for mercury FIM API
        # Format: prompt contains everything before the blank, suffix contains everything after
        prompt_text = ""
        
        # Add system prompt if present
        if self.system != "<EMPTY>":
            prompt_text += f"System: {self.system}\n"
        
        # Add user prompt and pre-context
        prompt_text += f"User: {user_prompt}\nAssistant: {pre_context}\n"
        suffix_text = post_context
        
        # Call mercury API
        try:
            response = requests.post(
                "https://api.inceptionlabs.ai/v1/fim/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mercury-coder-small",
                    "prompt": prompt_text,
                    "suffix": suffix_text,
                    "max_tokens": gen_length
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Mercury API error: {response.status_code} - {response.text}")
            
            response_json = response.json()
            
            print(response_json)
            # Extract the generated text from the response
            if "choices" in response_json and len(response_json["choices"]) > 0:
                generated_text = response_json["choices"][0].get("text", "")
                return generated_text
            else:
                raise RuntimeError(f"Unexpected response format from Mercury API: {response_json}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call Mercury API: {str(e)}")
