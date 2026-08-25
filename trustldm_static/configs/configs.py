from helm.proxy.clients.huggingface_model_registry import ModelLoader, WeightType
from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import Enum, auto


class ModelType(Enum):
    CHAT = auto()
    COMPLETION = auto()


@dataclass
class FairnessConfig:
    data_dir: Optional[str] = None  # DIR of the chosen dataset
    prompt_file: Optional[str] = None  # PATH of the file of prompts
    gt_file: Optional[str] = None  # PATH of the file of ground truth labels
    sensitive_attr_file: Optional[str] = None  # PATH of the file of sensitive attributes
    dataset: Optional[str] = 'adult'  # dataset
    out_file: Optional[str] = ''
    score_calculation_only: Optional[bool] = False
    max_tokens: Optional[int] = 20

    mode: Optional[str] = None # can be one of: benign, safe_conc, safe_det, harm_conc, harm_det


@dataclass
class PrivacyConfig:
    scenario_name: Optional[str] = None  # Name of the chosen scenario
    data_file: Optional[str] = None  # DIR of the chosen dataset
    out_file: Optional[str] = None  # PATH of the file of output
    template: int = 0  # System prompt template id
    max_tokens: int = 128  # Max number of tokens to generate
    n: int = 1  # Number of generations per API call
    t: int = 1  # Generation temperature
    seed: int = 1  # Random seed

    dataset_size: int = 0  # Number of testing samples
    few_shot_num: int = 0  # Number of demonstrations used in the few-shot setting
    batch_size: int = 0  # Batch size
    question_prompt: int = 0  # Id of question prompt

    personal_infos: List[str] = field(default_factory=list)  # List of personal identifiable information
    prompt_types: List[str] = field(default_factory=list)  # List of prompt types
    privacy_topics: List[str] = field(default_factory=list)  # List of privacy topics

    context_type: List[str] = field(default_factory=list) # can be any subset of ["empty", "benign", "concise", "detailed"]


    


@dataclass
class SafetyConfig:
    data_file: Optional[str] = None # Inpput json file
    out_file: Optional[str] = None # write results
    max_tokens: int = 256 # generate_length
    n: int = 1 # one generation
    t: Optional[float] = 1.0 # temperature
    seed: int = 42 # random_seed
    batch_size: int = 4
    scenario_name: Optional[str] = None # can be adv_bench or jbb

    context_type: List[str] = field(default_factory=list) # can be anys ubset of ["empty","completed", "benign", "concise", "detailed", "safe", "fraud"] 
    openai_api_key: Optional[str] = None # openai key, for LLM judge.


@dataclass
class ModelConfig:
    model: str = "openai/gpt-3.5-turbo-0301" # model name, should start with the model family's name, like "llada//llada_moe". NOT the actual path
    type: ModelType = ModelType.CHAT
    conv_template: Optional[str] = None
    model_loader: ModelLoader = ModelLoader.HF
    torch_dtype: Optional[WeightType] = WeightType.BFLOAT16
    trust_remote_code: bool = True
    use_auth_token: bool = True
    disable_exllama: bool = False  # Configuration from AutoGPTQForCausalLM
    inject_fused_attention: bool = True  # Configuration from AutoGPTQForCausalLM
    quant_file: Optional[str] = None
    tokenizer_name: Optional[str] = None
    device_map: Optional[str] = "auto"

    model_path: Optional[str] = None  # Path to the local model
    gen_function: Optional[str] = None  # Custom generation function. Check in ./generate_functions/...
    remasking: Optional[str] = None

    steps: Optional[int] = None
    gen_length: Optional[int] = None
    block_length: Optional[int] = None # default: gen_length//4, only in use at llada's evaluation
    cfg_scale: Optional[float] = 0.0
    add_role: Optional[str] = None 


@dataclass
class BaseConfig:
    # TODO: Handle device map - change HELM's default behavior
    model_config: ModelConfig
    disable_sys_prompt: Optional[bool] = False

    key: Optional[str] = None  # OpenAI API Key or Huggingface Secret
    dry_run: bool = False

    fairness: Optional[FairnessConfig] = None
    privacy: Optional[PrivacyConfig] = None
    safety: Optional[SafetyConfig] = None
